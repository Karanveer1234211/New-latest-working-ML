#!/usr/bin/env python3
"""
=============================================================================
WATCHLIST SCANNER  (prob >= 0.65, ORB-break tracking + smarter exits)
=============================================================================

WHAT THIS DOES
--------------
Builds a rolling watchlist of the model's best long candidates over the last
N trading days and tells you, FOR EACH DAY a name is on the list:

    symbol | signal_date | days_ago | regime | probability | calibrated EV
          | exp 5d return / std / Sharpe (from the model's own calibration)
          | ORB-break = YES/NO (did it break its T+1 15-min opening range yet?)
          | break_day (T+1..T+5) + break_price + slippage vs signal close
          | late-day quality score (signal-day intraday close strength etc.)
          | ensemble agreement (prob std across ensemble members)
          | ATR-aware TP / SL bracket + suggested action
          | staleness (is the last bar fresh enough to trade?)

WHY THESE COLUMNS (validated on per_trade_v4.parquet, prob>=0.75 slice)
-----------------------------------------------------------------------
  * RANK BY CALIBRATED PROBABILITY. The model's edge is in the cross-sectional
    rank (OOS Spearman IC ~0.137). Probability >= 0.65 is your gate.
  * ORB-break is a SELECTION / RISK signal, not a standalone strategy.
    Names that go on to break their opening range had +5.7% vs +3.3% mean 5d
    return and HALF the rate of -3%/-5% drawdowns vs names that never broke.
    But entering AT the breakout costs ~1.9% (you pay up), which nearly
    cancels the selection edge in bull regimes. So we SHOW the break as a
    "watch / conviction" flag, not as a forced entry. ~78% of breaks happen
    by T+2, so a 2-3 day watch window captures almost all of them.
  * EXIT IS THE REAL LEVER. A TP+10% / SL-5% bracket on a plain next-open
    entry lifted daily-basket Sharpe from 1.78 -> 3.26 in backtest. We
    translate that into an ATR-scaled bracket per name (clamped to the
    same %), because the biggest, most robust improvement was the exit, not
    the entry timing.
  * REGIME conditioning matters most in bear markets (drawdown control), so
    we surface regime and tighten the suggested stop there.

HOW IT PLUGS INTO YOUR EXISTING CODE
------------------------------------
It IMPORTS the proven feature pipeline + model wrappers from your model file
(default: "New_model.py") so the 137 features are built EXACTLY as in training
-- no re-implementation drift. It loads:
    <out_dir>/models/m5_regime_router.joblib   (or m5_ensemble / m5_classifier)
    <out_dir>/features_train.json
    <out_dir>/calibration_5d_deciles.json
and reads daily bars from your daily cache + 5-min bars from your intraday
cache (the same folders the cachers write to).

USAGE
-----
    python watchlist_scanner.py --out-dir "C:/.../v3_2_output_full" --lookback-days 5
    python watchlist_scanner.py --prob-min 0.65 --watch-window 3
    python watchlist_scanner.py --symbols RELIANCE,TCS,INFY    # debug a few names
    python watchlist_scanner.py --no-intraday                  # skip ORB (daily only, fast)

OUTPUT
------
    <out_dir>/watchlist_scan/watchlist_prob65.csv
    <out_dir>/watchlist_scan/watchlist_prob65.xlsx   (formatted, if openpyxl)
=============================================================================
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

IST = "Asia/Kolkata"

# ----- Default paths (override via CLI / env). Match your other scripts. -----
DEFAULT_OUT_DIR = Path(
    os.environ.get("WL_OUT_DIR", r"C:\Users\karanvsi\Desktop\Kite Connect\v3_2_output_full")
)
DEFAULT_DAILY_CACHE = Path(
    os.environ.get("WL_DAILY_CACHE", r"C:\Users\karanvsi\Desktop\Pycharm\Cache\cache_daily_new")
)
DEFAULT_INTRADAY_CACHE = Path(
    os.environ.get("WL_INTRADAY_CACHE", r"C:\Users\karanvsi\Desktop\Pycharm\Cache\intraday_5min")
)
DEFAULT_MODEL_FILE = Path(
    os.environ.get("WL_MODEL_FILE", str(Path(__file__).with_name("New_model.py")))
)

# ----- Gates / knobs (defaults chosen from the validation, see docstring) ----
PROB_MIN = 0.65
LOOKBACK_DAYS = 5            # how many recent signal-days to include
WATCH_WINDOW = 3            # ORB break is checked over T+1..T+WATCH_WINDOW (78% by T+2)
MIN_CLOSE = 2.0
MIN_AVG20_VOL = 200_000
MAX_STALENESS_DAYS = 5      # drop names whose last bar is older than this
FIRST_15M_BARS = 3          # 3 x 5-min bars = first 15 minutes (the opening range)
HOLDING_DAYS = 5
COST_PCT = 0.35

# ATR-aware bracket: target ~ TP_ATR_MULT * ATR%, clamped to [TP_MIN, TP_MAX] %.
# Defaults reflect the backtest sweet spot (wide TP, moderate SL).
TP_ATR_MULT = 3.0
SL_ATR_MULT = 1.5
TP_MIN_PCT, TP_MAX_PCT = 4.0, 12.0
SL_MIN_PCT, SL_MAX_PCT = 2.0, 6.0
BEAR_SL_TIGHTEN = 0.8       # multiply suggested SL distance by this in bear regimes


# =============================================================================
# Import the existing model module (feature pipeline + model classes)
# =============================================================================

def load_model_module(model_file: Path):
    """Import New_model.py as a module so we can reuse featureize / add_targets /
    _compute_stock_regime and unpickle EnsembleCalibrator / RegimeRouter
    (which must be importable under the SAME module they were pickled from)."""
    if not model_file.exists():
        raise SystemExit(f"FATAL: model file not found: {model_file}\n"
                         f"Pass --model-file pointing at your New_model.py / cpr_fix.py")
    mod_name = "kv_model_module"
    spec = importlib.util.spec_from_file_location(mod_name, str(model_file))
    mod = importlib.util.module_from_spec(spec)
    # Register under several names joblib might have stored at pickle time.
    sys.modules[mod_name] = mod
    for alias in ("New_model", "cpr_fix", "cpr_fix_patched", "__main__"):
        sys.modules.setdefault(alias, mod)
    spec.loader.exec_module(mod)
    return mod


# =============================================================================
# Daily cache loading + feature build
# =============================================================================

def _derive_symbol(p: Path) -> str:
    base = p.name
    for suff in ("_daily.parquet", "_daily.csv", ".parquet", ".csv"):
        if base.endswith(suff):
            return base[: -len(suff)]
    return base


def list_daily_files(daily_cache: Path, symbols: Optional[List[str]]) -> List[Path]:
    files = sorted(glob.glob(str(daily_cache / "*_daily.parquet")))
    files = [Path(f) for f in files if not Path(f).name.startswith("_")]
    if symbols:
        want = {s.upper() for s in symbols}
        files = [f for f in files if _derive_symbol(f).upper() in want]
    return files


def load_daily(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "timestamp" not in df.columns and "date" in df.columns:
        df = df.rename(columns={"date": "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True).dt.tz_convert(IST)
    df = (df.dropna(subset=["timestamp"]).sort_values("timestamp")
            .drop_duplicates("timestamp", keep="last").reset_index(drop=True))
    df["symbol"] = _derive_symbol(path)
    return df


def build_scored_panel(mod, files: List[Path], feature_schema: dict,
                       lookback_days: int) -> Tuple[pd.DataFrame, List[str]]:
    """Load each symbol, build features with the SAME functions used in training,
    compute regime, keep only the last `lookback_days` rows per symbol (we only
    score recent dates). Returns (panel, feature_list)."""
    feats_needed = list(feature_schema["features"])
    rows: List[pd.DataFrame] = []
    n_ok = 0
    for i, f in enumerate(files):
        try:
            d = load_daily(f)
            if len(d) < 220:           # need warmup for 200-day features
                continue
            d = mod.add_targets(d)      # adds ret_*/mfe_/mae_ (harmless here)
            d, _feats = mod.featureize(d)
            # keep a small tail: lookback + a little slack for staleness logic
            d = d.tail(lookback_days + 10).copy()
            rows.append(d)
            n_ok += 1
        except Exception as e:
            continue
        if (i + 1) % 500 == 0:
            print(f"  ...processed {i+1}/{len(files)} files (kept {n_ok})")
    if not rows:
        raise SystemExit("No symbols produced features. Check daily cache path.")
    panel = pd.concat(rows, ignore_index=True)
    panel = mod._compute_stock_regime(panel, regime_lag=getattr(mod, "REGIME_LAG", 1))
    print(f"[panel] {n_ok} symbols, {len(panel):,} recent rows")
    return panel, feats_needed


# =============================================================================
# Model loading + scoring
# =============================================================================

def load_model(out_dir: Path):
    models_dir = out_dir / "models"
    import joblib
    for name in ("m5_regime_router.joblib", "m5_ensemble.joblib", "m5_classifier.joblib"):
        p = models_dir / name
        if p.exists():
            print(f"[model] loading {p.name}")
            return joblib.load(p), name
    raise SystemExit(f"FATAL: no model found in {models_dir}")


def score_panel(model, model_name: str, panel: pd.DataFrame,
                feats: List[str], impute: Dict[str, float]) -> pd.DataFrame:
    """Attach `probability` and `prob_5d_std` (ensemble agreement) to panel."""
    X = panel.reindex(columns=feats).copy()
    for c in feats:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.fillna(pd.Series({c: float(impute.get(c, 0.0)) for c in feats}))

    if model_name.startswith("m5_regime_router") and hasattr(model, "predict_proba_by_regime"):
        regime_vec = panel["stock_regime"] if "stock_regime" in panel.columns \
            else pd.Series(["bull_trend"] * len(panel))
        panel["probability"] = model.predict_proba_by_regime(X, regime_vec)
        members = _gather_members(model)
    else:
        panel["probability"] = model.predict_proba(X)[:, 1]
        members = getattr(model, "members", None)

    # Ensemble agreement: std of member probabilities (low std = members agree).
    if members:
        try:
            member_probs = np.column_stack([
                _project_and_predict(m, X) for m in members
            ])
            panel["prob_5d_std"] = member_probs.std(axis=1)
        except Exception:
            panel["prob_5d_std"] = np.nan
    else:
        panel["prob_5d_std"] = np.nan
    return panel


def _gather_members(router) -> list:
    members = []
    for m in list(getattr(router, "regime_models", {}).values()):
        members.extend(getattr(m, "members", [m]))
    fb = getattr(router, "fallback", None)
    if fb is not None:
        members.extend(getattr(fb, "members", [fb]))
    return members


def _project_and_predict(m, X):
    fl = getattr(m, "feature_list", None)
    Xp = X.reindex(columns=fl) if fl else X
    return m.predict_proba(Xp)[:, 1]


# =============================================================================
# Calibration: prob -> expected 5d return / std / Sharpe
# =============================================================================

class Calibrator:
    def __init__(self, calib_json: Path):
        d = json.loads(Path(calib_json).read_text())
        self.mid = np.asarray(d["prob_mid"], float)
        self.ret = np.asarray(d["avg_ret_5d"], float)
        self.std = np.asarray(d["std_ret_5d"], float)
        self.sharpe = np.asarray(d.get("exp_sharpe_5d",
                                        self.ret / np.where(self.std == 0, np.nan, self.std)), float)

    def _interp(self, p, y):
        return np.interp(np.clip(p, 0, 1), self.mid, y, left=y[0], right=y[-1])

    def expected(self, p: np.ndarray) -> Dict[str, np.ndarray]:
        p = np.asarray(p, float)
        return {
            "exp_ret_5d_pct": self._interp(p, self.ret),
            "exp_std_5d_pct": self._interp(p, self.std),
            "exp_sharpe_5d": self._interp(p, self.sharpe),
        }


# =============================================================================
# Intraday: did the name break its T+1 opening range within the watch window?
# =============================================================================

def _load_intraday(intraday_cache: Path, symbol: str) -> Optional[pd.DataFrame]:
    p = intraday_cache / f"{symbol}.parquet"
    if not p.exists():
        return None
    try:
        d = pd.read_parquet(p)
        d["timestamp"] = pd.to_datetime(d["timestamp"], errors="coerce", utc=True).dt.tz_convert(IST)
        return d.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    except Exception:
        return None


def orb_break_check(intra: pd.DataFrame, signal_date: pd.Timestamp,
                    watch_window: int) -> Dict:
    """Walk forward from the day AFTER signal_date. On the first of those days
    compute the 15-min opening range high (ORH). Report the first day (T+1..)
    on which a 5-min close exceeds the ORH of its OWN day's open range -- i.e.
    a clean intraday breakout-and-hold. Mirrors ORB_15_held_anyday_static."""
    res = {"orb_break": False, "break_day": np.nan, "break_price": np.nan,
           "orh_t1": np.nan, "watch_days_available": 0}
    if intra is None or intra.empty:
        return res
    sig_day = signal_date.normalize()
    fwd = intra[intra["timestamp"].dt.normalize() > sig_day].copy()
    if fwd.empty:
        return res
    fwd["day"] = fwd["timestamp"].dt.normalize()
    days = list(dict.fromkeys(fwd["day"].tolist()))[:watch_window]
    res["watch_days_available"] = len(days)
    for k, day in enumerate(days, start=1):
        bars = fwd[fwd["day"] == day].reset_index(drop=True)
        if len(bars) < FIRST_15M_BARS + 1:
            continue
        orh = float(bars.loc[: FIRST_15M_BARS - 1, "high"].max())
        if k == 1:
            res["orh_t1"] = orh
        after_open = bars.loc[FIRST_15M_BARS:]
        hit = after_open[after_open["close"] > orh]
        if not hit.empty:
            res["orb_break"] = True
            res["break_day"] = k
            res["break_price"] = float(hit.iloc[0]["close"])
            return res
    return res


def late_day_quality(intra: pd.DataFrame, signal_date: pd.Timestamp) -> Dict:
    """Signal-day intraday quality (close strength, last-hour return, above VWAP,
    late volume, tail strength) -> a 0..5 score. Mirrors opportunities.py v4."""
    out = {"lateq_score": np.nan, "lateq_n_eval": 0,
           "sig_close_strength": np.nan, "sig_above_vwap_at_close": np.nan}
    if intra is None or intra.empty:
        return out
    day = intra[intra["timestamp"].dt.normalize() == signal_date.normalize()].reset_index(drop=True)
    if len(day) < 6:
        return out
    hi, lo, cl, op = day["high"], day["low"], day["close"], day["open"]
    vol = pd.to_numeric(day["volume"], errors="coerce").fillna(0.0)
    rng = float(hi.max() - lo.min())
    if rng <= 0:
        return out
    close_strength = (float(cl.iloc[-1]) - float(lo.min())) / rng
    tp = (hi + lo + cl) / 3.0
    vwap = float((tp * vol).sum() / max(vol.sum(), 1e-9))
    above_vwap = float(cl.iloc[-1]) > vwap
    n = len(day)
    last_hr = day.tail(min(12, n))
    first_hr = day.head(min(12, n))
    last_hr_ret = (float(last_hr["close"].iloc[-1]) / float(last_hr["close"].iloc[0]) - 1) * 100 \
        if len(last_hr) > 1 else 0.0
    late_vol_ratio = float(last_hr["volume"].sum()) / max(float(first_hr["volume"].sum()), 1e-9)
    tail_strength = (float(cl.iloc[-1]) - float(last_hr["low"].min())) / rng

    checks = [close_strength > 0.70, last_hr_ret > 0, above_vwap,
              late_vol_ratio > 1.0, tail_strength > 0.50]
    out.update(lateq_score=int(sum(bool(c) for c in checks)), lateq_n_eval=5,
               sig_close_strength=round(close_strength, 3),
               sig_above_vwap_at_close=bool(above_vwap))
    return out


# =============================================================================
# Bracket / action suggestion
# =============================================================================

def suggest_bracket(close: float, atr14: float, regime: str) -> Dict:
    atr_pct = (atr14 / close * 100.0) if (close and atr14 and close > 0) else np.nan
    if not np.isfinite(atr_pct):
        tp_pct, sl_pct = 8.0, 4.0
    else:
        tp_pct = float(np.clip(TP_ATR_MULT * atr_pct, TP_MIN_PCT, TP_MAX_PCT))
        sl_pct = float(np.clip(SL_ATR_MULT * atr_pct, SL_MIN_PCT, SL_MAX_PCT))
    if isinstance(regime, str) and regime.startswith("bear"):
        sl_pct = max(SL_MIN_PCT, sl_pct * BEAR_SL_TIGHTEN)
    return {
        "atr_pct": round(atr_pct, 2) if np.isfinite(atr_pct) else np.nan,
        "tp_pct": round(tp_pct, 1), "sl_pct": round(sl_pct, 1),
        "tp_price": round(close * (1 + tp_pct / 100), 2) if close else np.nan,
        "sl_price": round(close * (1 - sl_pct / 100), 2) if close else np.nan,
        "rr_ratio": round(tp_pct / sl_pct, 2) if sl_pct else np.nan,
    }


def suggest_action(row) -> str:
    """Watch vs enter guidance, combining probability, ORB-break and agreement."""
    p = row["probability"]
    broke = bool(row.get("orb_break", False))
    agree = row.get("prob_5d_std", np.nan)
    high_agree = (pd.notna(agree) and agree <= 0.06)
    if p >= 0.85 and (broke or high_agree):
        return "STRONG: enter at open; trail with bracket"
    if p >= 0.85:
        return "ENTER at open (high prob)"
    if broke:
        return "CONFIRMED: broke ORB -> enter on the break / pullback"
    if row.get("regime", "").startswith("bear"):
        return "WATCH: bear regime -> wait for ORB break before entry"
    return "WATCH: no break yet; naive open entry OK if taking the basket"


# =============================================================================
# MAIN
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description="prob>=0.65 watchlist scanner with ORB-break tracking")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--daily-cache", type=Path, default=DEFAULT_DAILY_CACHE)
    ap.add_argument("--intraday-cache", type=Path, default=DEFAULT_INTRADAY_CACHE)
    ap.add_argument("--model-file", type=Path, default=DEFAULT_MODEL_FILE)
    ap.add_argument("--prob-min", type=float, default=PROB_MIN)
    ap.add_argument("--lookback-days", type=int, default=LOOKBACK_DAYS)
    ap.add_argument("--watch-window", type=int, default=WATCH_WINDOW)
    ap.add_argument("--symbols", type=str, default=None, help="comma list to debug a subset")
    ap.add_argument("--no-intraday", action="store_true", help="skip ORB/late-day (daily only)")
    args = ap.parse_args()

    out_dir = args.out_dir
    scan_dir = out_dir / "watchlist_scan"
    scan_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("WATCHLIST SCANNER")
    print("=" * 70)
    print(f"  out_dir       : {out_dir}")
    print(f"  prob_min      : {args.prob_min}")
    print(f"  lookback_days : {args.lookback_days}")
    print(f"  watch_window  : {args.watch_window} (ORB break checked T+1..T+{args.watch_window})")

    mod = load_model_module(args.model_file)
    feature_schema = json.loads((out_dir / "features_train.json").read_text())
    impute = {k: float(v) for k, v in feature_schema.get("impute", {}).items()}
    calib = Calibrator(out_dir / "calibration_5d_deciles.json")

    symbols = [s.strip().upper() for s in args.symbols.split(",")] if args.symbols else None
    files = list_daily_files(args.daily_cache, symbols)
    print(f"  daily files   : {len(files)}")
    if not files:
        raise SystemExit(f"No *_daily.parquet found in {args.daily_cache}")

    # ---- Build features + score ----
    print("\n[1/4] Building features on recent bars...")
    panel, feats = build_scored_panel(mod, files, feature_schema, args.lookback_days)

    print("[2/4] Scoring...")
    model, model_name = load_model(out_dir)
    panel = score_panel(model, model_name, panel, feats, impute)

    # ---- Tradability + recency gates, then prob gate ----
    panel["date"] = panel["timestamp"].dt.normalize()
    panel["avg20_vol"] = (panel.groupby("symbol")["volume"]
                          .transform(lambda s: s.rolling(20, min_periods=1).mean()))
    today = pd.Timestamp.now(tz=IST).normalize()
    recent_dates = sorted(panel["date"].unique())[-args.lookback_days:]
    cand = panel[
        panel["date"].isin(recent_dates)
        & (pd.to_numeric(panel["close"], errors="coerce") >= MIN_CLOSE)
        & (panel["avg20_vol"] >= MIN_AVG20_VOL)
        & (panel["probability"] >= args.prob_min)
    ].copy()
    cand["days_ago"] = (today - cand["date"]).dt.days
    cand["last_bar_age_days"] = cand.groupby("symbol")["date"].transform("max")
    cand["last_bar_age_days"] = (today - cand["last_bar_age_days"]).dt.days
    cand = cand[cand["last_bar_age_days"] <= MAX_STALENESS_DAYS]
    print(f"[3/4] Candidates with prob>={args.prob_min} in last {args.lookback_days} days: {len(cand)}")

    if cand.empty:
        print("No candidates. Exiting.")
        return

    # ---- Calibrated expectations ----
    exp = calib.expected(cand["probability"].values)
    for k, v in exp.items():
        cand[k] = np.round(v, 3)

    # ---- ORB break + late-day quality (intraday) ----
    rows: List[Dict] = []
    do_intra = not args.no_intraday
    intra_cache: Dict[str, Optional[pd.DataFrame]] = {}
    for _, r in cand.iterrows():
        sym = r["symbol"]
        close = float(pd.to_numeric(r.get("close"), errors="coerce"))
        atr14 = float(pd.to_numeric(r.get("D_atr14"), errors="coerce"))
        rec = {
            "symbol": sym,
            "signal_date": r["date"].date(),
            "days_ago": int(r["days_ago"]),
            "regime": r.get("stock_regime", ""),
            "probability": round(float(r["probability"]), 4),
            "prob_decile": int(np.clip(np.searchsorted(calib.mid, r["probability"]), 0, 9)),
            "exp_ret_5d_pct": r["exp_ret_5d_pct"],
            "exp_std_5d_pct": r["exp_std_5d_pct"],
            "exp_sharpe_5d": r["exp_sharpe_5d"],
            "ensemble_prob_std": round(float(r["prob_5d_std"]), 4) if pd.notna(r.get("prob_5d_std")) else np.nan,
            "close": round(close, 2) if np.isfinite(close) else np.nan,
            "last_bar_age_days": int(r["last_bar_age_days"]),
        }
        if do_intra:
            if sym not in intra_cache:
                intra_cache[sym] = _load_intraday(args.intraday_cache, sym)
            intra = intra_cache[sym]
            sig_ts = r["timestamp"]
            ob = orb_break_check(intra, sig_ts, args.watch_window)
            lq = late_day_quality(intra, sig_ts)
            slippage = ((ob["break_price"] / close - 1) * 100
                        if ob["orb_break"] and np.isfinite(close) else np.nan)
            rec.update({
                "orb_break": "YES" if ob["orb_break"] else "NO",
                "break_day": f"T+{int(ob['break_day'])}" if ob["orb_break"] else "",
                "break_price": round(ob["break_price"], 2) if ob["orb_break"] else np.nan,
                "break_slippage_pct": round(slippage, 2) if pd.notna(slippage) else np.nan,
                "watch_days_avail": ob["watch_days_available"],
                "lateq_score": lq["lateq_score"],
                "sig_close_strength": lq["sig_close_strength"],
            })
        else:
            rec.update({"orb_break": "n/a", "break_day": "", "break_price": np.nan,
                        "break_slippage_pct": np.nan, "watch_days_avail": 0,
                        "lateq_score": np.nan, "sig_close_strength": np.nan})

        rec.update(suggest_bracket(close, atr14, rec["regime"]))
        rows.append(rec)

    wl = pd.DataFrame(rows)
    wl["action"] = wl.apply(suggest_action, axis=1)

    # Net expected EV after cost; rank by it then probability.
    wl["exp_net_ev_pct"] = (pd.to_numeric(wl["exp_ret_5d_pct"], errors="coerce") - COST_PCT).round(3)
    wl = wl.sort_values(["signal_date", "probability"], ascending=[False, False]).reset_index(drop=True)

    col_order = [
        "signal_date", "days_ago", "symbol", "regime", "probability", "prob_decile",
        "exp_ret_5d_pct", "exp_net_ev_pct", "exp_std_5d_pct", "exp_sharpe_5d",
        "orb_break", "break_day", "break_price", "break_slippage_pct", "watch_days_avail",
        "lateq_score", "sig_close_strength", "ensemble_prob_std",
        "close", "atr_pct", "tp_pct", "sl_pct", "tp_price", "sl_price", "rr_ratio",
        "last_bar_age_days", "action",
    ]
    wl = wl[[c for c in col_order if c in wl.columns]]

    out_csv = scan_dir / "watchlist_prob65.csv"
    wl.to_csv(out_csv, index=False)
    print(f"\n[4/4] Wrote {len(wl)} rows -> {out_csv}")
    try:
        with pd.ExcelWriter(scan_dir / "watchlist_prob65.xlsx", engine="openpyxl") as xw:
            wl.to_excel(xw, sheet_name="watchlist", index=False)
    except Exception:
        pass

    # Console preview: most recent day, top 20 by probability.
    latest = wl[wl["signal_date"] == wl["signal_date"].max()]
    show = ["symbol", "regime", "probability", "exp_net_ev_pct",
            "orb_break", "break_day", "lateq_score", "tp_pct", "sl_pct", "action"]
    show = [c for c in show if c in latest.columns]
    print(f"\n=== Latest watchlist ({wl['signal_date'].max()}), top 20 by prob ===")
    print(latest.head(20)[show].to_string(index=False))


if __name__ == "__main__":
    main()
