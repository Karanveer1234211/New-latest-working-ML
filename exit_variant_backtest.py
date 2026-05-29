#!/usr/bin/env python3
"""
=============================================================================
EXIT-VARIANT BACKTEST  —  naive-open HOLD vs ATR-bracket exit
=============================================================================

QUESTION
--------
You currently take every prob-gated signal at the T+1 open and hold to the
T+5 close. Does adding a TP/SL bracket (especially an ATR-aware one) improve
RISK-ADJUSTED return on the SAME signal universe, before you change anything
live?

WHY THIS IS A FAIR TEST
-----------------------
* Same ENTRY for every strategy: NAIVE T+1 open. Only the EXIT differs.
* Same UNIVERSE: every NAIVE_T1_OPEN row in the per-trade parquet.
* Returns are FIRST-TOUCH-CORRECT. The per-trade parquet already stores, for
  each signal, the realised return of a fixed TP/SL grid computed by the OCO
  walk in opportunities.py (which respects intraday bar order — i.e. whether
  the stop or target was hit first). We READ those columns rather than
  re-deriving from MAE/MFE (which cannot know first-touch ordering).
* The ATR-aware bracket maps each name's ATR% to a (TP, SL) pair, then SNAPS
  to the nearest grid level actually simulated, so its return stays
  path-correct. ATR% is taken from panel_cache.parquet if you point at it;
  otherwise that one variant is skipped with a clear note (the fixed-grid
  comparison still runs and is the headline result).

STRATEGIES COMPARED (all = NAIVE T+1 OPEN entry)
------------------------------------------------
  HOLD_T5            hold to T+5 close                         (your current)
  FIXED_TP{n}_SL{m}  static bracket from the grid              (a few presets)
  ATR_BRACKET        TP=clip(TP_MULT*ATR%,..), SL=clip(SL_MULT*ATR%,..),
                     stop tightened in bear regimes, snapped to grid

METRICS (net of round-trip cost) PER STRATEGY x (overall / regime / bucket)
---------------------------------------------------------------------------
  mean, median, win%, daily-basket annualised Sharpe (sqrt(252/HOLD)),
  max drawdown, and a BOOTSTRAP 95% CI on the Sharpe AND on the mean-return
  difference vs HOLD_T5 (so you can see if the gain is real or noise).

USAGE
-----
    python exit_variant_backtest.py --per-trade per_trade_v4.parquet
    python exit_variant_backtest.py --per-trade per_trade_v4.parquet \
        --panel "C:/.../panel_cache.parquet"        # enables ATR_BRACKET
    python exit_variant_backtest.py --entry-variant NAIVE_T1_OPEN --boot 2000

OUTPUT
------
    <dir>/exit_variant_backtest.csv        full grid of metrics
    console: headline table + bootstrap verdict vs your current HOLD_T5
=============================================================================
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# round-trip cost in % (25 bps + 2*5 bps slippage) — matches opportunities.py
COST_PCT = 0.35
HOLDING_DAYS = 5
ANN = np.sqrt(252.0 / HOLDING_DAYS)

# Grid actually simulated in the per-trade parquet (first-touch correct).
TP_GRID = [3.0, 5.0, 7.0, 10.0]
SL_GRID = [2.0, 3.0, 5.0]            # absolute %, stored as sl-2.0 / sl-3.0 / sl-5.0

# Fixed presets to report alongside the baseline.
FIXED_PRESETS = [(10.0, 5.0), (7.0, 3.0), (5.0, 3.0), (3.0, 2.0)]

# ATR-aware bracket config (same as watchlist_scanner.py defaults).
TP_ATR_MULT, SL_ATR_MULT = 3.0, 1.5
TP_MIN, TP_MAX = 4.0, 12.0
SL_MIN, SL_MAX = 2.0, 6.0
BEAR_SL_TIGHTEN = 0.8

RET_HOLD = "fwd_return_to_t5_close_pct"


# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------

def _col(tp: float, sl_abs: float) -> str:
    """Column name for the fixed grid, e.g. tp10_sl-5.0_ret."""
    return f"tp{tp:g}_sl{-sl_abs:.1f}_ret"


def _snap(value: float, grid: List[float]) -> float:
    return float(min(grid, key=lambda g: abs(g - value)))


def daily_sharpe(df: pd.DataFrame, ret_col: str, date_col="signal_date") -> float:
    s = (df.groupby(df[date_col].dt.normalize())[ret_col].mean().sort_index())
    if len(s) < 5 or s.std(ddof=0) == 0:
        return np.nan
    return float(ANN * s.mean() / s.std(ddof=0))


def max_drawdown(df: pd.DataFrame, ret_col: str, date_col="signal_date") -> float:
    d = (df.groupby(df[date_col].dt.normalize())[ret_col].mean().sort_index())
    if d.empty:
        return np.nan
    eq = (1 + d / 100).cumprod()
    return float(((eq / eq.cummax() - 1) * 100).min())


def daily_series(df: pd.DataFrame, ret_col: str, date_col="signal_date") -> pd.Series:
    return df.groupby(df[date_col].dt.normalize())[ret_col].mean().sort_index()


def bootstrap_sharpe_ci(daily: pd.Series, n_boot: int, seed: int = 42
                        ) -> Tuple[float, float]:
    """Block-free i.i.d. bootstrap over trading days (5d horizon ~ weekly
    baskets, so daily-of-basket means are near-independent)."""
    if len(daily) < 10:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    arr = daily.values
    n = len(arr)
    sh = np.empty(n_boot)
    for b in range(n_boot):
        sample = arr[rng.integers(0, n, n)]
        sd = sample.std(ddof=0)
        sh[b] = ANN * sample.mean() / sd if sd > 0 else np.nan
    sh = sh[np.isfinite(sh)]
    if len(sh) == 0:
        return (np.nan, np.nan)
    return (float(np.percentile(sh, 2.5)), float(np.percentile(sh, 97.5)))


def bootstrap_sharpe_diff_ci(daily_a: pd.Series, daily_b: pd.Series,
                             n_boot: int, seed: int = 7) -> Tuple[float, float, float]:
    """CI on Sharpe(daily_a) - Sharpe(daily_b), paired over common dates (we
    resample the SAME day indices for both, preserving their correlation).
    Returns (lo, hi, p_a_gt_b). This is the right test for 'does the exit
    improve RISK-ADJUSTED return', unlike a mean-return diff which a
    variance-reducing bracket will always lose on."""
    idx = daily_a.index.intersection(daily_b.index)
    if len(idx) < 10:
        return (np.nan, np.nan, np.nan)
    a = daily_a.loc[idx].values
    b = daily_b.loc[idx].values
    rng = np.random.default_rng(seed)
    n = len(idx)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        sel = rng.integers(0, n, n)
        sa, sb = a[sel], b[sel]
        sda, sdb = sa.std(ddof=0), sb.std(ddof=0)
        sh_a = ANN * sa.mean() / sda if sda > 0 else np.nan
        sh_b = ANN * sb.mean() / sdb if sdb > 0 else np.nan
        diffs[i] = sh_a - sh_b
    diffs = diffs[np.isfinite(diffs)]
    if len(diffs) == 0:
        return (np.nan, np.nan, np.nan)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return (float(lo), float(hi), float((diffs > 0).mean()))


# -----------------------------------------------------------------------------
# build per-row returns for each strategy
# -----------------------------------------------------------------------------

def add_strategy_returns(trades: pd.DataFrame, panel: Optional[pd.DataFrame]
                         ) -> Tuple[pd.DataFrame, List[str], Dict[str, str]]:
    """Returns (df with *_ret cols, list of strategy names, name->ret_col map)."""
    df = trades.copy()
    strat_cols: Dict[str, str] = {}

    # Baseline: hold to T+5 close
    df["HOLD_T5__ret"] = pd.to_numeric(df[RET_HOLD], errors="coerce")
    strat_cols["HOLD_T5"] = "HOLD_T5__ret"

    # Fixed presets straight from the simulated grid
    for tp, sl in FIXED_PRESETS:
        col = _col(tp, sl)
        if col in df.columns:
            name = f"FIXED_TP{tp:g}_SL{sl:g}"
            df[f"{name}__ret"] = pd.to_numeric(df[col], errors="coerce")
            strat_cols[name] = f"{name}__ret"

    # ATR-aware bracket (needs ATR% — join from panel if provided)
    if panel is not None:
        atr_pct = _attach_atr_pct(df, panel)
        if atr_pct.notna().any():
            tp_raw = np.clip(TP_ATR_MULT * atr_pct, TP_MIN, TP_MAX)
            sl_raw = np.clip(SL_ATR_MULT * atr_pct, SL_MIN, SL_MAX)
            is_bear = df["regime"].astype(str).str.startswith("bear")
            sl_raw = np.where(is_bear, np.maximum(SL_MIN, sl_raw * BEAR_SL_TIGHTEN), sl_raw)

            tp_snap = pd.Series(tp_raw, index=df.index).apply(lambda v: _snap(v, TP_GRID))
            sl_snap = pd.Series(sl_raw, index=df.index).apply(lambda v: _snap(v, SL_GRID))

            atr_ret = np.full(len(df), np.nan)
            for tp in TP_GRID:
                for sl in SL_GRID:
                    col = _col(tp, sl)
                    if col not in df.columns:
                        continue
                    mask = (tp_snap.values == tp) & (sl_snap.values == sl)
                    atr_ret[mask] = pd.to_numeric(df[col], errors="coerce").values[mask]
            # rows with no ATR fall back to hold
            nofill = np.isnan(atr_ret)
            atr_ret[nofill] = df["HOLD_T5__ret"].values[nofill]
            df["ATR_BRACKET__ret"] = atr_ret
            df["atr_pct"] = np.round(atr_pct.values, 2)
            df["atr_tp_pct"] = tp_snap.values
            df["atr_sl_pct"] = sl_snap.values
            strat_cols["ATR_BRACKET"] = "ATR_BRACKET__ret"
            print(f"[atr] ATR_BRACKET enabled on {int(atr_pct.notna().sum()):,} rows "
                  f"(median ATR%={atr_pct.median():.2f}, "
                  f"median TP={tp_snap.median():g}/SL={sl_snap.median():g})")
        else:
            print("[atr] panel provided but no ATR%/close match; skipping ATR_BRACKET")
    else:
        print("[atr] no --panel given; skipping ATR_BRACKET "
              "(fixed-grid comparison still runs)")

    return df, list(strat_cols.keys()), strat_cols


def _attach_atr_pct(trades: pd.DataFrame, panel: pd.DataFrame) -> pd.Series:
    """Join D_atr14 / close from the panel on (symbol, signal_date) -> ATR%."""
    p = panel.copy()
    if "timestamp" in p.columns:
        p["signal_date"] = pd.to_datetime(p["timestamp"]).dt.tz_localize(None) \
            if pd.to_datetime(p["timestamp"]).dt.tz is None \
            else pd.to_datetime(p["timestamp"]).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    cols = [c for c in ("symbol", "signal_date", "D_atr14", "close") if c in p.columns]
    p = p[cols].dropna(subset=["symbol", "signal_date"])
    p["date_key"] = pd.to_datetime(p["signal_date"]).dt.normalize()

    t = trades.copy()
    t["date_key"] = pd.to_datetime(t["signal_date"]).dt.tz_localize(None).dt.normalize() \
        if pd.to_datetime(t["signal_date"]).dt.tz is not None \
        else pd.to_datetime(t["signal_date"]).dt.normalize()
    merged = t.merge(p[["symbol", "date_key", "D_atr14", "close"]],
                     on=["symbol", "date_key"], how="left", suffixes=("", "_pn"))
    atr = pd.to_numeric(merged["D_atr14"], errors="coerce")
    close = pd.to_numeric(merged.get("close_pn", merged.get("close")), errors="coerce")
    return (atr / close * 100.0).replace([np.inf, -np.inf], np.nan)


# -----------------------------------------------------------------------------
# metrics
# -----------------------------------------------------------------------------

def block_metrics(df: pd.DataFrame, name: str, ret_col: str, n_boot: int) -> Dict:
    r = pd.to_numeric(df[ret_col], errors="coerce")
    net = r - COST_PCT
    daily = daily_series(df.assign(_r=net), "_r")
    lo, hi = bootstrap_sharpe_ci(daily, n_boot)
    return {
        "strategy": name,
        "n": int(r.notna().sum()),
        "mean_pct": round(float(r.mean()), 3),
        "net_pct": round(float(net.mean()), 3),
        "median_pct": round(float(r.median()), 3),
        "win_pct": round(float(100 * (r > 0).mean()), 1),
        "sharpe": round(daily_sharpe(df.assign(_r=net), "_r"), 3),
        "sharpe_ci_lo": round(lo, 3) if pd.notna(lo) else np.nan,
        "sharpe_ci_hi": round(hi, 3) if pd.notna(hi) else np.nan,
        "max_dd_pct": round(max_drawdown(df.assign(_r=net), "_r"), 1),
    }


def run_block(df: pd.DataFrame, strat_cols: Dict[str, str], label: str,
              n_boot: int) -> pd.DataFrame:
    rows = [block_metrics(df, name, col, n_boot) for name, col in strat_cols.items()]
    out = pd.DataFrame(rows)
    out.insert(0, "scope", label)

    # bootstrap Sharpe-diff vs HOLD_T5 (the correct risk-adjusted test)
    base_daily = daily_series(df.assign(_r=pd.to_numeric(df[strat_cols["HOLD_T5"]],
                                                          errors="coerce") - COST_PCT), "_r")
    diffs = []
    for name, col in strat_cols.items():
        dd = daily_series(df.assign(_r=pd.to_numeric(df[col], errors="coerce") - COST_PCT), "_r")
        lo, hi, p = bootstrap_sharpe_diff_ci(dd, base_daily, n_boot)
        diffs.append({"strategy": name,
                      "vs_hold_sharpediff_lo": round(lo, 3) if pd.notna(lo) else np.nan,
                      "vs_hold_sharpediff_hi": round(hi, 3) if pd.notna(hi) else np.nan,
                      "p_beats_hold": round(p, 3) if pd.notna(p) else np.nan})
    out = out.merge(pd.DataFrame(diffs), on="strategy", how="left")
    return out


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="naive-open HOLD vs ATR-bracket exit backtest")
    ap.add_argument("--per-trade", type=Path, required=True,
                    help="per_trade parquet from opportunities.py")
    ap.add_argument("--panel", type=Path, default=None,
                    help="panel_cache.parquet (enables ATR_BRACKET via D_atr14)")
    ap.add_argument("--entry-variant", type=str, default="NAIVE_T1_OPEN",
                    help="which entry variant defines the universe (default NAIVE_T1_OPEN)")
    ap.add_argument("--boot", type=int, default=1000, help="bootstrap resamples")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    pt = pd.read_parquet(args.per_trade)
    pt["signal_date"] = pd.to_datetime(pt["signal_date"])
    trades = pt[pt["variant"] == args.entry_variant].copy()
    if trades.empty:
        raise SystemExit(f"No rows for entry variant {args.entry_variant!r}. "
                         f"Available: {sorted(pt['variant'].unique())[:10]}...")
    print("=" * 78)
    print(f"EXIT-VARIANT BACKTEST  |  entry = {args.entry_variant}")
    print("=" * 78)
    print(f"  universe : {len(trades):,} signals "
          f"({trades.signal_date.min().date()} -> {trades.signal_date.max().date()})")
    buckets = sorted(trades['prob_bucket'].dropna().unique())
    print(f"  prob     : {trades.probability.min():.3f}..{trades.probability.max():.3f}  "
          f"buckets={buckets}")
    print(f"  regimes  : {sorted(trades['regime'].dropna().unique())}")
    print(f"  cost     : {COST_PCT}%  | bootstrap = {args.boot}")

    panel = pd.read_parquet(args.panel) if args.panel and args.panel.exists() else None
    df, strat_names, strat_cols = add_strategy_returns(trades, panel)

    all_blocks = [run_block(df, strat_cols, "OVERALL", args.boot)]
    for reg in sorted(df["regime"].dropna().unique()):
        sub = df[df["regime"] == reg]
        if len(sub) >= 50:
            all_blocks.append(run_block(sub, strat_cols, f"regime={reg}", args.boot))
    for b in buckets:
        sub = df[df["prob_bucket"] == b]
        if len(sub) >= 50:
            all_blocks.append(run_block(sub, strat_cols, f"bucket={b}", args.boot))

    result = pd.concat(all_blocks, ignore_index=True)
    out_path = args.out or args.per_trade.with_name("exit_variant_backtest.csv")
    result.to_csv(out_path, index=False)

    # ---- console: overall headline ----
    show = ["strategy", "n", "net_pct", "win_pct", "sharpe",
            "sharpe_ci_lo", "sharpe_ci_hi", "max_dd_pct",
            "vs_hold_sharpediff_lo", "vs_hold_sharpediff_hi", "p_beats_hold"]
    overall = result[result["scope"] == "OVERALL"][show]
    print("\n=== OVERALL (net of cost; Sharpe annualised; CI = 95% bootstrap) ===")
    print(overall.to_string(index=False))

    base_sh = float(overall.loc[overall.strategy == "HOLD_T5", "sharpe"].iloc[0])
    base_net = float(overall.loc[overall.strategy == "HOLD_T5", "net_pct"].iloc[0])
    print(f"\nYour current HOLD_T5 Sharpe = {base_sh:.2f}")
    best = overall.sort_values("sharpe", ascending=False).iloc[0]
    if best.strategy != "HOLD_T5":
        verdict = ("REAL: 95% CI on Sharpe gain excludes 0"
                   if (pd.notna(best.vs_hold_sharpediff_lo) and best.vs_hold_sharpediff_lo > 0)
                   else "PROMISING but Sharpe-gain CI includes 0 (not conclusive)")
        print(f"Best risk-adjusted exit  = {best.strategy}: Sharpe {best.sharpe:.2f} "
              f"(+{best.sharpe-base_sh:.2f}), p(higher Sharpe than hold)={best.p_beats_hold} "
              f"-> {verdict}")
        print(f"  NOTE: brackets cap winners, so MEAN return falls "
              f"({best.net_pct:.2f}% vs {base_net:.2f}%); the case for them is "
              f"variance reduction, not higher mean.")
    else:
        print("HOLD_T5 already has the best Sharpe in this universe.")

    print("\n=== BY REGIME / BUCKET (Sharpe per strategy) ===")
    piv = result.pivot_table(index="scope", columns="strategy",
                             values="sharpe", aggfunc="first")
    print(piv.to_string())
    print(f"\n[out] {out_path}")


if __name__ == "__main__":
    main()
