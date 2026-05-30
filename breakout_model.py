#!/usr/bin/env python3
"""
=============================================================================
BREAKOUT MODEL & EVENT LAB  —  what kind of breakout follows through, and
                              can we predict TRUE breakouts vs FAKEOUTS?
=============================================================================

GOAL
----
1. Define MANY breakout EVENT types (not just "close > prior 20d high"),
   grounded in the features your Daily cache.py already computes.
2. For every event, label it (strict, label-#2):
        TRUE  = closes higher over H days  AND  never falls back below the
                broken level within the window   (a real, held breakout)
        FAKE  = anything else                      (reversal / no follow-through)
   Label uses OPEN-to-close (tradeable: you enter at next open), NOT cc.
3. Two analyses so you can decide WHAT HELPS vs WHAT DOESN'T:
     A) EVENT DIAGNOSTICS  — per event type: how often it fires, raw true-rate,
        mean forward OC return, fakeout rate. Tells you which breakout TYPES are
        worth trading at all.
     B) PREDICTIVE MODEL    — pool all breakout days, features = your daily-cache
        features + which-events-fired flags + a confluence count, label = strict
        true/fake. Reports OOS AUC, Spearman IC, decile lift (does filtering
        breakouts by model prob beat taking every breakout?), gain importance and
        PERMUTATION importance (what features actually matter for true vs fake).

WHY THIS ORDER
--------------
This is the cheap feasibility probe BEFORE a heavy intraday model. It uses only
panel_cache.parquet (daily features you already have). If true-vs-fake has no IC
here, breakout follow-through isn't learnable from daily features and we should
NOT build the intraday version. If it does, this justifies it (and the intraday
build then needs bar_volumes — see breakout_lab.py notes).

EVENT TYPES (extensible — add to build_events())
-------------------------------------------------
  CLOSE_20DH     close breaks prior 20-day high
  CLOSE_50DH     close breaks prior 50-day high
  CLOSE_52WH     close breaks prior 252-day (52w) high
  HIGH_20DH      intraday high breaks prior 20d high (cache D_breakout_high_20)
  DONCH20_POS    Donchian-20 position crosses to the top (D_donch_pos_20 ~1)
  BB_UPPER       close above upper Bollinger band (D_bb_pctB_20 > 1)
  NR_EXPANSION   prior day NR/coiled + today range expands + closes up
  VOL_SURGE_UP   volume surge (D_vol_surge_20) + closes up
  CPR_R1         close breaks above CPR resistance-1 (D_resistance1)
  GAP_UP_HOLD    gaps up >1% and closes green (holds the gap)
  CONFLUENCE_K   >=K of the atomic breakouts fire the same day (the "multiple
                 features align" idea) — K configurable (default 2 and 3)

USAGE
-----
  python breakout_model.py --selftest
  python breakout_model.py \
      --panel "C:/.../panel_cache.parquet" \
      --model-file "C:/.../New_model.py" \
      --features "C:/.../features_train.json" \
      --horizon 5 --out-dir "C:/Users/karanvsi/Desktop/breakout_out"
  # flags: --min-ret 0.0 (true needs oc>=this %), --hold-tol 0.0 (level breach
  #         tolerance %), --fast (800 trees), --horizon 3|5

OUTPUTS (in --out-dir)
----------------------
  event_diagnostics.csv          per event type: fire rate, true rate, returns
  breakout_feature_importance.csv gain + permutation AUC-drop per feature
  breakout_oos_report.json       AUC / IC / decile lift / base rate
  breakout_decile_lift.csv       model-prob decile vs realised true-rate & OC ret
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
# Import the user's model module (reuse exact split/params/calibration)
# =============================================================================

def load_model_module(model_file: Optional[Path]):
    if model_file and model_file.exists():
        name = "nm_breakout"
        spec = importlib.util.spec_from_file_location(name, str(model_file))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        for alias in ("New_model", "cpr_fix", "cpr_fix_patched"):
            sys.modules.setdefault(alias, mod)
        spec.loader.exec_module(mod)
        return mod
    return _inline_nm_shim()


# =============================================================================
# Forward outcome columns (per symbol, leak-safe): OC return + forward low min
# =============================================================================

def add_forward_outcomes(panel: pd.DataFrame, horizon: int) -> pd.DataFrame:
    df = panel.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    g = df.groupby("symbol", observed=True)
    # open[t+1] -> close[t+H]
    df["oc_ret_H"] = (g["close"].shift(-horizon) / g["open"].shift(-1) - 1) * 100.0
    # forward MIN low over t+1..t+H  and forward MAX high (row-wise extremum of shifts)
    lows = pd.concat([g["low"].shift(-k) for k in range(1, horizon + 1)], axis=1)
    highs = pd.concat([g["high"].shift(-k) for k in range(1, horizon + 1)], axis=1)
    df["fwd_low_min_H"] = lows.min(axis=1)
    df["fwd_high_max_H"] = highs.max(axis=1)
    df["entry_open"] = g["open"].shift(-1)
    return df


# =============================================================================
# Breakout EVENT definitions  -> dict[name] = (mask: bool Series, level: Series)
# `level` = the price level that was broken (used for the strict hold test).
# All levels/conditions use ONLY information available at bar t (point-in-time).
# =============================================================================

def _num(s):
    return pd.to_numeric(s, errors="coerce")

def build_events(panel: pd.DataFrame, conf_ks=(2, 3)) -> Dict[str, Tuple[pd.Series, pd.Series]]:
    df = panel
    g = df.groupby("symbol", observed=True)
    close = _num(df["close"]); high = _num(df["high"]); low = _num(df["low"]); open_ = _num(df["open"])
    prev_close = g["close"].shift(1)
    prev_high = g["high"].shift(1)

    # prior-N highs (shifted so they exclude today) — recomputed unambiguously
    def prior_high(n):
        return g["high"].shift(1).rolling(n, min_periods=max(5, n // 2)).max().reset_index(level=0, drop=True)
    hi20 = prior_high(20); hi50 = prior_high(50); hi252 = prior_high(252)

    events: Dict[str, Tuple[pd.Series, pd.Series]] = {}

    # --- atomic close/high breaks ---
    events["CLOSE_20DH"] = ((close > hi20) & hi20.notna(), hi20)
    events["CLOSE_50DH"] = ((close > hi50) & hi50.notna(), hi50)
    events["CLOSE_52WH"] = ((close > hi252) & hi252.notna(), hi252)
    events["HIGH_20DH"] = ((high > hi20) & (close > prev_close) & hi20.notna(), hi20)

    # --- cache-feature-based breaks (guarded by column presence) ---
    if "D_donch_pos_20" in df.columns:
        dp = _num(df["D_donch_pos_20"]); dp_prev = g["D_donch_pos_20"].shift(1)
        events["DONCH20_POS"] = ((dp >= 0.98) & (_num(dp_prev) < 0.98), hi20)
    if "D_bb_pctB_20" in df.columns:
        bb = _num(df["D_bb_pctB_20"])
        # upper band level = close where pctB=1 ~ reconstruct via sma20+2std unavailable;
        # use prev_high as the hold level (conservative) when bb>1
        events["BB_UPPER"] = ((bb > 1.0) & (close > prev_close), prev_high)
    if "D_nr" in df.columns and "D_nr_expand" in df.columns:
        nr_prev = g["D_nr"].shift(1)
        events["NR_EXPANSION"] = ((_num(nr_prev) >= 1) & (_num(df["D_nr_expand"]) == 1)
                                  & (close > open_), prev_high)
    if "D_vol_surge_20" in df.columns:
        events["VOL_SURGE_UP"] = ((_num(df["D_vol_surge_20"]) == 1) & (close > prev_close), prev_high)
    if "D_resistance1" in df.columns:
        r1 = _num(df["D_resistance1"])
        events["CPR_R1"] = ((close > r1) & (prev_close <= r1) & r1.notna(), r1)
    # gap-up hold
    gap = (open_ / prev_close - 1) * 100.0
    events["GAP_UP_HOLD"] = ((gap > 1.0) & (close > open_), prev_close)

    # --- confluence: >=K atomic breakouts fire the same day ---
    atomic = ["CLOSE_20DH", "BB_UPPER", "VOL_SURGE_UP", "HIGH_20DH", "NR_EXPANSION", "DONCH20_POS"]
    atomic = [a for a in atomic if a in events]
    if atomic:
        count = sum(events[a][0].astype(int) for a in atomic)
        for K in conf_ks:
            events[f"CONFLUENCE_{K}of{len(atomic)}"] = ((count >= K), hi20)
    # expose the confluence count as a feature later
    panel["_breakout_confluence_count"] = (sum(events[a][0].astype(int) for a in atomic)
                                           if atomic else 0)
    return events


# =============================================================================
# Strict label (#2): TRUE = closes higher over H AND never falls back below level
# =============================================================================

def strict_label(df: pd.DataFrame, level: pd.Series, min_ret: float, hold_tol: float) -> pd.Series:
    oc = _num(df["oc_ret_H"])
    fwd_low = _num(df["fwd_low_min_H"])
    lvl = _num(level)
    closes_higher = oc >= float(min_ret)
    holds = fwd_low >= lvl * (1.0 - float(hold_tol) / 100.0)
    lab = (closes_higher & holds).astype(float)
    # rows without a full forward window (NaN oc/low) are unlabelable
    lab[oc.isna() | fwd_low.isna() | lvl.isna()] = np.nan
    return lab


# =============================================================================
# Build the modelling dataset: one row per (symbol, date) that fired ANY event
# =============================================================================

def build_dataset(panel: pd.DataFrame, events: Dict, feats: List[str],
                  horizon: int, min_ret: float, hold_tol: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (model_df, event_diag).
    model_df: rows = any-breakout days; cols = feats + EV_* multihot + conf count
              + label + oc_ret + timestamp + symbol.
    event_diag: per-event-type raw diagnostics (fire rate, true rate, returns)."""
    df = panel
    # union mask + per-event flags + per-event label (for diagnostics)
    any_mask = pd.Series(False, index=df.index)
    diag_rows = []
    ev_flags = {}
    # a single 'best level' per row = the highest broken level among fired events
    level_stack = pd.DataFrame(index=df.index)
    for name, (mask, level) in events.items():
        mask = mask.fillna(False)
        ev_flags[f"EV_{name}"] = mask.astype(int)
        any_mask = any_mask | mask
        lvl_for_event = level.where(mask)
        level_stack[name] = lvl_for_event
        # diagnostics on THIS event's fired rows
        lab = strict_label(df, level, min_ret, hold_tol)
        sub = df.loc[mask & lab.notna()]
        lab_sub = lab.loc[mask & lab.notna()]
        if len(sub) >= 30:
            oc = _num(sub["oc_ret_H"])
            diag_rows.append(dict(
                event=name, n_fired=int(mask.sum()), n_labeled=int(len(sub)),
                fire_rate_pct=round(100 * mask.mean(), 3),
                true_rate_pct=round(100 * lab_sub.mean(), 1),
                fake_rate_pct=round(100 * (1 - lab_sub.mean()), 1),
                mean_oc_ret_pct=round(float(oc.mean()), 3),
                median_oc_ret_pct=round(float(oc.median()), 3),
                mean_oc_net_pct=round(float(oc.mean() - COST_PCT), 3),
                win_pct=round(100 * float((oc > 0).mean()), 1),
            ))
    event_diag = pd.DataFrame(diag_rows).sort_values("true_rate_pct", ascending=False)

    # 'broken level' per breakout day = max level across fired events (most conservative hold)
    broken_level = level_stack.max(axis=1)
    label = strict_label(df, broken_level, min_ret, hold_tol)

    rows = df.loc[any_mask].copy()
    rows["_label"] = label.loc[any_mask]
    rows = rows[rows["_label"].notna()].copy()

    # assemble model columns
    keep_feats = [f for f in feats if f in rows.columns]
    for c, s in ev_flags.items():
        rows[c] = s.loc[rows.index]
    ev_cols = list(ev_flags.keys())
    conf_col = ["_breakout_confluence_count"] if "_breakout_confluence_count" in rows.columns else []
    model_cols = keep_feats + ev_cols + conf_col
    out = rows[["timestamp", "symbol", "_label", "oc_ret_H"] + model_cols].copy()
    out = out.rename(columns={"_label": "label"})
    return out, event_diag


# =============================================================================
# Train + evaluate the true-vs-fake classifier
# =============================================================================

def train_eval(nm, model_df: pd.DataFrame, model_cols: List[str], fast: bool) -> dict:
    from lightgbm import LGBMClassifier
    from sklearn.metrics import roc_auc_score, brier_score_loss
    from scipy.stats import spearmanr

    X = nm.sanitize_feature_matrix(model_df.reindex(columns=model_cols).copy())
    y = model_df["label"].astype(int).values
    t = model_df["timestamp"].values
    emb = getattr(nm, "EMBARGO_DAYS", 5)
    i_tr, i_cal, i_te = nm.split_train_cal_test_by_date(t, 0.70, 0.20, emb)
    if min(len(i_tr), len(i_cal), len(i_te)) == 0:
        i_tr, i_cal, i_te = nm.split_train_cal_test_by_date(t, 0.70, 0.20, 0)

    params = nm._lgbm_cls_params(getattr(nm, "GLOBAL_SEED", 42) + 4242)
    if fast:
        params["n_estimators"] = 800
    base = LGBMClassifier(**params)
    cbs = nm._lgb_callbacks(len(i_cal))
    base.fit(X.iloc[i_tr], y[i_tr], eval_set=[(X.iloc[i_cal], y[i_cal])],
             eval_metric="binary_logloss", callbacks=cbs)
    calib, _ = nm._calibrate_best_brier(base, X.iloc[i_cal], y[i_cal])

    Xte = X.iloc[i_te]; yte = y[i_te]
    p = calib.predict_proba(Xte)[:, 1]
    oc_te = _num(model_df.iloc[i_te]["oc_ret_H"]).values

    auc = float(roc_auc_score(yte, p)) if len(set(yte)) > 1 else float("nan")
    brier = float(brier_score_loss(yte, p))
    ic, _ = spearmanr(p, oc_te, nan_policy="omit")
    base_rate = float(np.mean(yte))

    # decile lift: prob decile vs realised true-rate & OC return
    dte = pd.DataFrame({"prob": p, "y": yte, "oc": oc_te})
    dte["dec"] = pd.qcut(dte["prob"], 10, labels=False, duplicates="drop")
    lift = (dte.groupby("dec").agg(n=("y", "size"), true_rate=("y", "mean"),
                                   mean_oc=("oc", "mean")).reset_index())
    lift["true_rate"] = (lift["true_rate"] * 100).round(1)
    lift["mean_oc"] = lift["mean_oc"].round(3)
    top = dte[dte["dec"] == dte["dec"].max()]
    top_true = float(top["y"].mean()); top_oc = float(top["oc"].mean())

    # permutation importance (multi-shuffle) on OOS AUC
    rng = np.random.default_rng(42)
    base_auc = auc
    perm = []
    for col in model_cols:
        if col not in Xte.columns:
            continue
        drops = []
        for _ in range(3):
            Xs = Xte.copy()
            Xs[col] = rng.permutation(Xs[col].values)
            try:
                ps = calib.predict_proba(Xs)[:, 1]
                drops.append(base_auc - roc_auc_score(yte, ps))
            except Exception:
                pass
        if drops:
            perm.append((col, float(np.mean(drops)), float(np.std(drops))))
    perm_df = pd.DataFrame(perm, columns=["feature", "auc_drop_mean", "auc_drop_std"]) \
        .sort_values("auc_drop_mean", ascending=False)

    # gain importance from the base booster
    try:
        gain = base.booster_.feature_importance(importance_type="gain")
        gain_df = pd.DataFrame({"feature": base.booster_.feature_name(), "gain": gain})
        gain_df["gain_pct"] = 100 * gain_df["gain"] / max(gain_df["gain"].sum(), 1)
        imp = perm_df.merge(gain_df[["feature", "gain_pct"]], on="feature", how="outer")
    except Exception:
        imp = perm_df

    return {
        "n_train": int(len(i_tr)), "n_test": int(len(i_te)),
        "base_rate_true": round(base_rate, 4),
        "auc": round(auc, 4) if auc == auc else None,
        "brier": round(brier, 4),
        "ic_prob_vs_oc": round(float(ic), 4) if ic == ic else None,
        "top_decile_true_rate": round(top_true, 4),
        "top_decile_mean_oc_pct": round(top_oc, 3),
        "lift_true_rate_vs_base": round(top_true - base_rate, 4),
        "_lift_table": lift, "_importance": imp,
    }


# =============================================================================
# Self-test (synthetic; needs lightgbm)
# =============================================================================

def selftest(model_file):
    print("[selftest] synthetic panel...")
    rng = np.random.default_rng(0)
    rows = []
    base = pd.Timestamp("2021-01-01", tz=IST)
    for s in range(50):
        px = 100.0
        for dday in range(500):
            ts = base + pd.Timedelta(days=dday)
            ret = rng.normal(0.05, 1.6)
            px *= (1 + ret / 100)
            o = px * (1 + rng.normal(0, 0.004))
            h = max(px, o) * (1 + abs(rng.normal(0, 0.005)))
            l = min(px, o) * (1 - abs(rng.normal(0, 0.005)))
            rows.append({"symbol": f"S{s}", "timestamp": ts, "open": o, "high": h,
                         "low": l, "close": px, "volume": 1e6 * (1 + abs(rng.normal())),
                         "D_atr14": px * 0.02, "D_adx14": rng.uniform(10, 40),
                         "D_rsi14": rng.uniform(30, 80), "D_bb_pctB_20": rng.uniform(-0.2, 1.3),
                         "D_compress_state": rng.uniform(0, 1), "D_vol_surge_20": int(rng.random() < 0.1),
                         "D_donch_pos_20": rng.uniform(0, 1.1), "D_nr": int(rng.random() < 0.15),
                         "D_nr_expand": int(rng.random() < 0.5)})
    panel = pd.DataFrame(rows)
    nm = load_model_module(Path(model_file) if model_file else None)
    feats = ["D_atr14", "D_adx14", "D_rsi14", "D_bb_pctB_20", "D_compress_state",
             "D_vol_surge_20", "D_donch_pos_20"]
    run(panel, nm, feats, horizon=5, min_ret=0.0, hold_tol=0.0, fast=True, out_dir=None)
    print("\n[selftest] engine OK (synthetic; numbers meaningless, structure valid).")


# =============================================================================
# Orchestration
# =============================================================================

def run(panel, nm, feats, horizon, min_ret, hold_tol, fast, out_dir: Optional[Path]):
    panel = add_forward_outcomes(panel, horizon)
    events = build_events(panel)
    model_df, event_diag = build_dataset(panel, events, feats, horizon, min_ret, hold_tol)
    print(f"\n[events] breakout days (any event, labeled): {len(model_df):,}")
    print(f"[events] overall strict TRUE rate: {100*model_df['label'].mean():.1f}%")
    print("\n=== A) EVENT DIAGNOSTICS (raw, no model) — which breakout TYPES follow through ===")
    print(event_diag.to_string(index=False))

    model_cols = [c for c in model_df.columns if c not in ("timestamp", "symbol", "label", "oc_ret_H")]
    res = train_eval(nm, model_df, model_cols, fast)
    print("\n=== B) PREDICTIVE MODEL (true vs fake) ===")
    print(f"  n_test={res['n_test']:,}  base_rate_true={res['base_rate_true']}  "
          f"AUC={res['auc']}  Brier={res['brier']}  IC(prob vs OC)={res['ic_prob_vs_oc']}")
    print(f"  top-decile true-rate={res['top_decile_true_rate']} "
          f"(base {res['base_rate_true']}, LIFT={res['lift_true_rate_vs_base']})  "
          f"top-decile mean OC={res['top_decile_mean_oc_pct']}%")
    print("\n  decile lift (prob decile -> realised true-rate% & mean OC%):")
    print(res["_lift_table"].to_string(index=False))
    print("\n  TOP 20 features by permutation AUC-drop (what HELPS true-vs-fake):")
    imp = res["_importance"]
    cols = [c for c in ["feature", "auc_drop_mean", "auc_drop_std", "gain_pct"] if c in imp.columns]
    print(imp[cols].head(20).to_string(index=False))
    print("\n  BOTTOM 10 (dead / unhelpful for breakout prediction):")
    print(imp[cols].tail(10).to_string(index=False))

    if out_dir is not None:
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        event_diag.to_csv(out_dir / "event_diagnostics.csv", index=False)
        imp.to_csv(out_dir / "breakout_feature_importance.csv", index=False)
        res["_lift_table"].to_csv(out_dir / "breakout_decile_lift.csv", index=False)
        rep = {k: v for k, v in res.items() if not k.startswith("_")}
        rep.update({"horizon": horizon, "min_ret": min_ret, "hold_tol": hold_tol,
                    "n_breakout_days": int(len(model_df)),
                    "overall_true_rate": float(model_df["label"].mean())})
        (out_dir / "breakout_oos_report.json").write_text(json.dumps(rep, indent=2, default=float))
        print(f"\n[out] wrote event_diagnostics.csv, breakout_feature_importance.csv, "
              f"breakout_decile_lift.csv, breakout_oos_report.json -> {out_dir}")


def _inline_nm_shim():
    import types, lightgbm as lgb
    shim = types.SimpleNamespace(EMBARGO_DAYS=5, GLOBAL_SEED=42)
    def sanitize(df):
        df = df.copy()
        if df.columns.duplicated().any():
            df = df.loc[:, ~df.columns.duplicated()]
        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        return df
    def split(ts, tr=0.7, ca=0.2, emb=5):
        s = pd.to_datetime(pd.Series(ts)).dt.normalize()
        days = np.sort(s.unique()); n = len(days)
        ct, cc = int(tr*n), int((tr+ca)*n)
        TR=set(days[:max(0,ct-emb)]); CA=set(days[ct:max(ct,cc-emb)]); TE=set(days[cc:])
        a=s.values
        return (np.where(np.isin(a,list(TR)))[0], np.where(np.isin(a,list(CA)))[0], np.where(np.isin(a,list(TE)))[0])
    def params(rnd):
        return dict(n_estimators=3000, learning_rate=0.01, num_leaves=31, max_depth=6,
                    feature_fraction=0.7, bagging_fraction=0.7, bagging_freq=1,
                    min_data_in_leaf=100, reg_alpha=0.3, reg_lambda=10.0,
                    n_jobs=-1, random_state=int(rnd), verbosity=-1)
    def cbs(v):
        c=[lgb.callback.log_evaluation(period=0)]
        if v>=200: c.insert(0, lgb.callback.early_stopping(stopping_rounds=100))
        return c
    def calib(est, Xv, yv):
        from sklearn.calibration import CalibratedClassifierCV
        try:
            return CalibratedClassifierCV(estimator=est, method="isotonic", cv="prefit").fit(Xv, yv), {}
        except TypeError:
            return CalibratedClassifierCV(base_estimator=est, method="isotonic", cv="prefit").fit(Xv, yv), {}
    shim.sanitize_feature_matrix=sanitize; shim.split_train_cal_test_by_date=split
    shim._lgbm_cls_params=params; shim._lgb_callbacks=cbs; shim._calibrate_best_brier=calib
    return shim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", type=Path)
    ap.add_argument("--model-file", type=Path)
    ap.add_argument("--features", type=Path, default=None)
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--min-ret", type=float, default=0.0, help="TRUE needs oc_ret >= this %% (try COST=0.35)")
    ap.add_argument("--hold-tol", type=float, default=0.0, help="allow forward low to dip this %% below the broken level")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=Path("breakout_out"))
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest(args.model_file); return
    if not args.panel or not args.panel.exists():
        raise SystemExit("Provide --panel panel_cache.parquet (or --selftest).")

    nm = load_model_module(args.model_file)
    panel = pd.read_parquet(args.panel)
    panel["timestamp"] = pd.to_datetime(panel["timestamp"], errors="coerce", utc=True).dt.tz_convert(IST)
    if args.features and args.features.exists():
        feats = list(json.loads(args.features.read_text())["features"])
    elif hasattr(nm, "discover_daily_features"):
        feats = nm.discover_daily_features(panel)
    else:
        feats = [c for c in panel.columns if c.startswith(("D_", "W_", "WQ_", "X_", "M_"))]
    feats = [f for f in feats if f in panel.columns]
    print(f"[load] {len(panel):,} rows | {len(feats)} features | horizon={args.horizon}d | "
          f"min_ret={args.min_ret}% hold_tol={args.hold_tol}%")
    run(panel, nm, feats, args.horizon, args.min_ret, args.hold_tol, args.fast, args.out_dir)


if __name__ == "__main__":
    main()
