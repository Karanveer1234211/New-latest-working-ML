#!/usr/bin/env python3
"""
=============================================================================
HORIZON / EV-TARGET COMPARISON  —  3d vs 5d, close-to-close vs open-to-close
=============================================================================

WHY
---
1. ev_target bug: New_model.build_5d_rank_quant_labels defaults to ev_target="cc"
   (close[t]->close[t+h]). You can only ENTER at the next open, so cc rewards the
   overnight gap leg you cannot trade. ev_target="oc" (open[t+1]->close[t+h]) is
   the tradeable label.
2. Horizon: ~78% of ORB breaks happen by T+2 and T+1 breaks had the best Sharpe,
   so a 3-day model may be more efficient than 5-day. But the cc->oc fix removes a
   BIGGER fraction of a 3-day move than a 5-day move, so you must compare 3d vs 5d
   AFTER switching to oc — otherwise you're choosing on an illusion.

This script trains the SAME LightGBM/CV/calibration as New_model.py for each of:
    (horizon in {3, 5}) x (ev_target in {cc, oc})
on a held-out, day-aligned, embargoed OOS slice, and reports — apples-to-apples:
    spearman_IC(prob, ret_adj)      ranking edge
    brier_pseudo                    probability quality
    decile_spread_OC_pct            TRADEABLE: top-decile minus bottom-decile
                                    OPEN-to-close return (not cc!)
    long_basket_net_sharpe          take prob>=80th pct each day, enter at open,
                                    minus cost, daily-basket Sharpe annualised by
                                    the horizon (sqrt(252/h))
    overnight_gap_share_pct         how much of the cc move is the un-tradeable
                                    overnight gap (cc - oc), as % of cc

It IMPORTS your New_model.py so labels/splits/params match training exactly.

USAGE
-----
  python horizon_compare.py --selftest            # synthetic data engine check
  python horizon_compare.py \
      --panel "C:/Users/karanvsi/Desktop/Kite Connect/v3_2_output_full/panel_cache.parquet" \
      --model-file "C:/Users/karanvsi/PyCharmMiscProject/New_model.py" \
      --features "C:/.../features_train.json" \
      --out "C:/Users/karanvsi/Desktop/horizon_compare.csv"
  # add --fast to use 800 trees instead of 3000 (quicker, comparison still valid)

NOTES
-----
* Needs panel_cache.parquet (has features + ret_*_close_pct + ret_*_oc_pct).
  If the oc/cc return columns are missing, they are recomputed PER SYMBOL.
* Needs lightgbm (pip install lightgbm). Same dep as the model.
* This is a SELECTION/label study. Execution conclusions (don't chase, etc.)
  are separate and already done.
=============================================================================
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

IST = "Asia/Kolkata"
COST_PCT = 0.35


# =============================================================================
# Import the user's model module (to reuse exact label/split/param logic)
# =============================================================================

def load_model_module(model_file: Path):
    if not model_file.exists():
        raise SystemExit(f"--model-file not found: {model_file}")
    name = "nm_horizon"
    spec = importlib.util.spec_from_file_location(name, str(model_file))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    for alias in ("New_model", "cpr_fix", "cpr_fix_patched"):
        sys.modules.setdefault(alias, mod)
    spec.loader.exec_module(mod)
    return mod


# =============================================================================
# Ensure forward-return columns exist (per-symbol, leak-safe)
# =============================================================================

def ensure_returns(panel: pd.DataFrame) -> pd.DataFrame:
    df = panel.copy()
    df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    g = df.groupby("symbol", observed=True)
    for h in (3, 5):
        cc = f"ret_{h}d_close_pct"
        oc = f"ret_{h}d_oc_pct"
        if cc not in df.columns:
            df[cc] = (g["close"].shift(-h) / df["close"] - 1) * 100
        if oc not in df.columns:
            # open[t+1] -> close[t+h]
            df[oc] = (g["close"].shift(-h) / g["open"].shift(-1) - 1) * 100
    return df


# =============================================================================
# Build a label for ANY horizon, mirroring build_5d_rank_quant_labels exactly
# =============================================================================

def build_label(nm, panel: pd.DataFrame, horizon: int, ev_target: str) -> pd.DataFrame:
    """Mirror New_model.build_5d_rank_quant_labels but for horizon in {3,5}.
    Returns frame with: date, label (1/0/nan), ret_adj (vol-normalised, used for
    IC + isotonic), ret_cc (close-close %), ret_oc (open-close %, TRADEABLE)."""
    pl = panel.copy()
    pl["date"] = pd.to_datetime(pl["timestamp"]).dt.normalize()
    daily_ret = (pd.to_numeric(pl["close"], errors="coerce")
                 .groupby(pl["symbol"], observed=True).pct_change() * 100.0)
    pl["vol_20"] = (daily_ret.groupby(pl["symbol"], observed=True)
                    .rolling(20, min_periods=10).std().reset_index(level=0, drop=True))
    pl["atr_pct"] = (pd.to_numeric(pl.get("D_atr14"), errors="coerce")
                     / pd.to_numeric(pl["close"], errors="coerce")
                     ).replace([np.inf, -np.inf], np.nan) * 100.0
    vol_basis = pl["vol_20"].fillna(pl["atr_pct"]).replace(0.0, np.nan)

    ret_cc = pd.to_numeric(pl[f"ret_{horizon}d_close_pct"], errors="coerce")
    ret_oc = pd.to_numeric(pl[f"ret_{horizon}d_oc_pct"], errors="coerce")
    ret_for_label = ret_oc if ev_target == "oc" else ret_cc

    pl["ret_adj"] = ret_for_label / vol_basis
    pl["ret_cc"] = ret_cc
    pl["ret_oc"] = ret_oc
    pl["rank_pct"] = pl.groupby("date")["ret_adj"].rank(method="average", pct=True)
    pl["label"] = np.where(pl["rank_pct"] >= 0.80, 1,
                           np.where(pl["rank_pct"] <= 0.20, 0, np.nan))
    return pl


# =============================================================================
# Train one config on the day-aligned embargoed split; return OOS frame
# =============================================================================

def run_config(nm, panel: pd.DataFrame, feats: List[str], horizon: int,
               ev_target: str, fast: bool) -> Optional[pd.DataFrame]:
    import lightgbm as lgb
    from lightgbm import LGBMClassifier

    pl = build_label(nm, panel, horizon, ev_target)
    y_all = pl["label"].astype("float")
    mask = y_all.notna()
    if mask.sum() < 2000:
        print(f"  [{horizon}d/{ev_target}] only {int(mask.sum())} labeled rows; skipping")
        return None

    # exact same feature sanitiser + day-aligned embargoed split as New_model
    X = nm.sanitize_feature_matrix(pl.loc[mask].reindex(columns=feats).copy())
    y = y_all.loc[mask].astype(int)
    t = pl.loc[mask, "timestamp"].values
    emb = getattr(nm, "EMBARGO_DAYS", 5)
    i_tr, i_cal, i_te = nm.split_train_cal_test_by_date(t, 0.70, 0.20, emb)
    if min(len(i_tr), len(i_cal), len(i_te)) == 0:
        i_tr, i_cal, i_te = nm.split_train_cal_test_by_date(t, 0.70, 0.20, 0)

    params = nm._lgbm_cls_params(getattr(nm, "GLOBAL_SEED", 42) + 777)
    if fast:
        params["n_estimators"] = 800
    base = LGBMClassifier(**params)
    cbs = nm._lgb_callbacks(len(i_cal))
    base.fit(X.iloc[i_tr], y.iloc[i_tr],
             eval_set=[(X.iloc[i_cal], y.iloc[i_cal])],
             eval_metric="binary_logloss", callbacks=cbs)
    calib, _ = nm._calibrate_best_brier(base, X.iloc[i_cal], y.iloc[i_cal])
    p_te = calib.predict_proba(X.iloc[i_te])[:, 1]

    pm = pl.loc[mask].iloc[i_te]
    return pd.DataFrame({
        "prob": p_te,
        "ret_adj": pm["ret_adj"].values,
        "ret_cc": pm["ret_cc"].values,
        "ret_oc": pm["ret_oc"].values,
        "rank_pct": pm["rank_pct"].values,
        "date": pd.to_datetime(pm["timestamp"]).dt.normalize().values,
        "label": pm["label"].values,
    })


# =============================================================================
# Metrics
# =============================================================================

def metrics(oos: pd.DataFrame, horizon: int) -> Dict:
    from sklearn.metrics import brier_score_loss
    from scipy.stats import spearmanr
    d = oos.dropna(subset=["prob"]).copy()
    # IC: prob vs vol-adjusted return
    ic, _ = spearmanr(d["prob"], d["ret_adj"], nan_policy="omit")
    # Brier vs pseudo target (top/bottom quintile), matching New_model's OOS report
    tgt = np.where(d["rank_pct"] >= 0.8, 1, np.where(d["rank_pct"] <= 0.2, 0, np.nan))
    m = ~np.isnan(tgt)
    brier = float(brier_score_loss(tgt[m], d["prob"].values[m])) if m.sum() > 30 else np.nan
    # decile spread on TRADEABLE oc return
    d["dec"] = pd.qcut(d["prob"], 10, labels=False, duplicates="drop")
    dec = d.groupby("dec")["ret_oc"].mean()
    dec_spread = float(dec.iloc[-1] - dec.iloc[0]) if len(dec) >= 2 else np.nan
    top_dec_oc = float(dec.iloc[-1]) if len(dec) else np.nan
    # long basket: prob >= 80th percentile each day, enter at open, minus cost
    thr = d["prob"].quantile(0.80)
    longs = d[d["prob"] >= thr]
    daily = longs.groupby("date")["ret_oc"].mean() - COST_PCT
    ann = np.sqrt(252.0 / horizon)
    sharpe = float(ann * daily.mean() / daily.std(ddof=0)) if daily.std(ddof=0) > 0 and len(daily) >= 5 else np.nan
    eq = (1 + daily / 100).cumprod()
    maxdd = float(((eq / eq.cummax() - 1) * 100).min()) if len(daily) else np.nan
    # overnight gap share of cc move (how much you lose by entering at open)
    cc_mean = d["ret_cc"].mean()
    oc_mean = d["ret_oc"].mean()
    gap_share = float((cc_mean - oc_mean) / cc_mean * 100) if cc_mean not in (0, np.nan) and pd.notna(cc_mean) else np.nan
    return {
        "n_oos": int(len(d)),
        "spearman_ic": round(float(ic), 4) if pd.notna(ic) else np.nan,
        "brier_pseudo": round(brier, 4) if pd.notna(brier) else np.nan,
        "decile_spread_oc_pct": round(dec_spread, 3) if pd.notna(dec_spread) else np.nan,
        "top_decile_oc_pct": round(top_dec_oc, 3) if pd.notna(top_dec_oc) else np.nan,
        "long_basket_net_sharpe": round(sharpe, 3) if pd.notna(sharpe) else np.nan,
        "long_basket_maxdd_pct": round(maxdd, 1) if pd.notna(maxdd) else np.nan,
        "overnight_gap_share_pct": round(gap_share, 1) if pd.notna(gap_share) else np.nan,
        "cc_mean_pct": round(float(cc_mean), 3), "oc_mean_pct": round(float(oc_mean), 3),
    }


# =============================================================================
# Self-test (synthetic; needs lightgbm)
# =============================================================================

def selftest(model_file: Optional[Path]):
    print("[selftest] building synthetic panel...")
    rng = np.random.default_rng(0)
    rows = []
    base = pd.Timestamp("2021-01-01", tz=IST)
    n_days, n_syms = 600, 60
    for s in range(n_syms):
        px = 100.0
        # a weak 'feature' that has mild predictive power on forward return
        for dday in range(n_days):
            ts = base + pd.Timedelta(days=dday)
            feat = rng.normal()
            drift = 0.15 * feat
            px *= (1 + rng.normal(drift, 1.5) / 100)
            rows.append({"symbol": f"S{s}", "timestamp": ts, "close": px,
                         "open": px * (1 + rng.normal(0, 0.004)),
                         "high": px * 1.01, "low": px * 0.99, "volume": 1e6,
                         "D_atr14": px * 0.02, "feat1": feat,
                         "feat2": rng.normal(), "feat3": feat + rng.normal(0, 2)})
    panel = pd.DataFrame(rows)
    panel = ensure_returns(panel)
    feats = ["feat1", "feat2", "feat3"]

    if model_file and model_file.exists():
        nm = load_model_module(model_file)
    else:
        print("[selftest] no --model-file; using a minimal inline shim of NM funcs")
        nm = _inline_nm_shim()

    out = []
    for h in (3, 5):
        for ev in ("cc", "oc"):
            oos = run_config(nm, panel, feats, h, ev, fast=True)
            if oos is None:
                continue
            r = {"horizon": h, "ev_target": ev}
            r.update(metrics(oos, h))
            out.append(r)
    res = pd.DataFrame(out)
    print(res.to_string(index=False))
    print("\n[selftest] engine OK (synthetic; numbers meaningless, structure valid).")


def _inline_nm_shim():
    """Minimal stand-ins for the few NM functions, so --selftest works without
    the real model file. The real run should always pass --model-file."""
    import types
    import lightgbm as lgb
    shim = types.SimpleNamespace()
    shim.EMBARGO_DAYS = 5
    shim.GLOBAL_SEED = 42

    def sanitize_feature_matrix(df):
        df = df.copy()
        if df.columns.duplicated().any():
            df = df.loc[:, ~df.columns.duplicated()]
        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        return df

    def split_train_cal_test_by_date(ts_values, train_frac=0.7, cal_frac=0.2, embargo_days=5):
        ts = pd.to_datetime(pd.Series(ts_values)).dt.normalize()
        days = np.sort(ts.unique())
        n = len(days)
        ct, cc = int(train_frac * n), int((train_frac + cal_frac) * n)
        emb = embargo_days
        tr = set(days[:max(0, ct - emb)]); ca = set(days[ct:max(ct, cc - emb)]); te = set(days[cc:])
        a = ts.values
        return (np.where(np.isin(a, list(tr)))[0], np.where(np.isin(a, list(ca)))[0],
                np.where(np.isin(a, list(te)))[0])

    def _lgbm_cls_params(rnd):
        return dict(n_estimators=3000, learning_rate=0.01, num_leaves=31, max_depth=6,
                    feature_fraction=0.7, bagging_fraction=0.7, bagging_freq=1,
                    min_data_in_leaf=100, reg_alpha=0.3, reg_lambda=10.0,
                    n_jobs=-1, random_state=int(rnd), verbosity=-1)

    def _lgb_callbacks(val_size):
        cbs = [lgb.callback.log_evaluation(period=0)]
        if val_size >= 200:
            cbs.insert(0, lgb.callback.early_stopping(stopping_rounds=100))
        return cbs

    def _calibrate_best_brier(est, Xv, yv):
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.metrics import brier_score_loss
        try:
            iso = CalibratedClassifierCV(estimator=est, method="isotonic", cv="prefit").fit(Xv, yv)
        except TypeError:
            iso = CalibratedClassifierCV(base_estimator=est, method="isotonic", cv="prefit").fit(Xv, yv)
        return iso, {}

    shim.sanitize_feature_matrix = sanitize_feature_matrix
    shim.split_train_cal_test_by_date = split_train_cal_test_by_date
    shim._lgbm_cls_params = _lgbm_cls_params
    shim._lgb_callbacks = _lgb_callbacks
    shim._calibrate_best_brier = _calibrate_best_brier
    return shim


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", type=Path, help="panel_cache.parquet")
    ap.add_argument("--model-file", type=Path, help="New_model.py (reuse exact funcs)")
    ap.add_argument("--features", type=Path, default=None, help="features_train.json (else auto-discover)")
    ap.add_argument("--fast", action="store_true", help="800 trees instead of 3000 (quicker)")
    ap.add_argument("--out", type=Path, default=Path("horizon_compare.csv"))
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest(args.model_file); return
    if not args.panel or not args.panel.exists():
        raise SystemExit("Provide --panel panel_cache.parquet (or --selftest).")
    if not args.model_file:
        raise SystemExit("Provide --model-file New_model.py so labels/splits/params match training.")

    nm = load_model_module(args.model_file)
    print(f"[load] panel: {args.panel}")
    panel = pd.read_parquet(args.panel)
    panel["timestamp"] = pd.to_datetime(panel["timestamp"], errors="coerce", utc=True).dt.tz_convert(IST)
    panel = ensure_returns(panel)

    # features
    if args.features and args.features.exists():
        feats = list(json.loads(args.features.read_text())["features"])
    else:
        feats = nm.discover_daily_features(panel)
    feats = [f for f in feats if f in panel.columns]
    print(f"[load] {len(panel):,} rows | {len(feats)} features | "
          f"{panel['timestamp'].min().date()} -> {panel['timestamp'].max().date()}")

    rows = []
    for h in (3, 5):
        for ev in ("cc", "oc"):
            print(f"\n=== training horizon={h}d  ev_target={ev} ===")
            oos = run_config(nm, panel, feats, h, ev, args.fast)
            if oos is None:
                continue
            r = {"horizon": h, "ev_target": ev}
            r.update(metrics(oos, h))
            rows.append(r)
            print(f"  IC={r['spearman_ic']}  Brier={r['brier_pseudo']}  "
                  f"decile_spread_OC={r['decile_spread_oc_pct']}%  "
                  f"net_Sharpe={r['long_basket_net_sharpe']}  "
                  f"overnight_gap_share={r['overnight_gap_share_pct']}%")

    res = pd.DataFrame(rows)
    out = Path(args.out)
    try:
        if out.exists() and out.is_dir():
            out = out / "horizon_compare.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        res.to_csv(out, index=False)
    except (PermissionError, OSError) as e:
        out = Path.cwd() / "horizon_compare.csv"
        print(f"[warn] could not write ({e}); using {out}")
        res.to_csv(out, index=False)

    print("\n" + "=" * 78)
    print("HORIZON / EV-TARGET COMPARISON  (OC = tradeable; pick by net Sharpe + IC)")
    print("=" * 78)
    cols = ["horizon", "ev_target", "n_oos", "spearman_ic", "brier_pseudo",
            "decile_spread_oc_pct", "top_decile_oc_pct", "long_basket_net_sharpe",
            "long_basket_maxdd_pct", "overnight_gap_share_pct", "cc_mean_pct", "oc_mean_pct"]
    print(res[[c for c in cols if c in res.columns]].to_string(index=False))
    print(f"\n[out] {out}")

    # quick verdicts
    if not res.empty:
        oc = res[res.ev_target == "oc"]
        if not oc.empty and oc["long_basket_net_sharpe"].notna().any():
            best = oc.sort_values("long_basket_net_sharpe", ascending=False).iloc[0]
            print(f"\nBest TRADEABLE (oc) config by net Sharpe: {int(best.horizon)}d  "
                  f"Sharpe={best.long_basket_net_sharpe}  IC={best.spearman_ic}")
        # how much edge was the overnight gap?
        for h in (3, 5):
            cc = res[(res.horizon == h) & (res.ev_target == "cc")]
            och = res[(res.horizon == h) & (res.ev_target == "oc")]
            if not cc.empty and not och.empty:
                print(f"  {h}d: IC cc={cc.iloc[0].spearman_ic} -> oc={och.iloc[0].spearman_ic} "
                      f"(overnight gap share of cc move = {och.iloc[0].overnight_gap_share_pct}%)")


if __name__ == "__main__":
    main()
