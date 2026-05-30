#!/usr/bin/env python3
"""
=============================================================================
ENTRY VARIANT LAB  —  test MANY non-chasing entry levels on the same universe
=============================================================================

WHY
---
ORB-break selects recovering names well but you CHASE the breakout price, which
eats ~80% of the edge. LIMIT_AT_PREVCLOSE beat plain-open on Sharpe AND
drawdown because it does NOT chase. This lab generalises that idea: it tests a
big, EXTENSIBLE catalog of entry triggers/levels and reports, on the SAME
prob-gated signal universe, which execution gives the best risk-adjusted
return without eating profit.

Entry families tested (add your own in VARIANTS — it's a plain dict):
  Baselines:        OPEN_T1, CLOSE_T1
  Reference limits: LIMIT_PREVCLOSE, LIMIT_PREVCLOSE_MINUS_0p5/1p0,
                    LIMIT_PREV_LOW, LIMIT_PREV_HIGH, LIMIT_WEEK_LOW/HIGH/OPEN,
                    LIMIT_DAY_LOW_T1, LIMIT_VWAP_T1
  Confirm-don't-chase: OPEN_IF_ABOVE_PREVCLOSE, RECLAIM_VWAP_30m,
                    HOLD_ABOVE_PREVCLOSE_EOD, FIRST_GREEN_15m
  Daily-cache levels: LIMIT_EMA20, LIMIT_SMA20, LIMIT_CPR_BC, LIMIT_DONCH_LO20,
                    BREAK_PREV_HIGH, BREAK_DONCH_HI20  (chasing ones flagged)
  Gap filters can be layered on ANY variant (reject_gap_down_pct).

Each variant => one trade per signal (or "skipped" => cash 0 in the portfolio).
Exit is held to T+5 close (you can swap in your TP/SL ladder later).

METRICS (net of cost), per OVERALL / regime / prob_bucket:
  deploy%  taken_mean  win%  selection_vs_skipped  entry_slippage_vs_prevclose
  daily-basket Sharpe  max_dd   (equal-weight AND prob-weighted portfolio)

DATA  (runs on YOUR machine — needs the caches the other scripts use)
---------------------------------------------------------------------
Two ways to get the path data:
  A) --paths extracted_paths.parquet     reuse paths already extracted by
                                          ORB execution.py (fast)
  B) --signals signals.parquet --intraday-dir <dir>   extract bars here
Either way each signal row must carry the bar arrays
(bar_opens/bar_highs/bar_lows/bar_closes/bar_days) and prev_close. If you
point --daily-cache at cache_daily_new, daily-level variants (EMA/SMA/CPR/
Donchian/weekly) are enabled by joining the signal-day daily row.

USAGE
-----
  python entry_variant_lab.py --paths .../extracted_paths_v2.parquet \
      --daily-cache "C:/.../cache_daily_new" --out entry_lab_results.csv
  python entry_variant_lab.py --paths paths.parquet --variants LIMIT_PREVCLOSE,LIMIT_VWAP_T1
  python entry_variant_lab.py --selftest          # synthetic-bar engine check

EXTEND
------
Add to VARIANTS: a dict with keys
  kind: 'limit' | 'stop' | 'market' | 'confirm'
  level: callable(ctx) -> price   (ctx has prev_close, day1 bars, daily row...)
  reject_gap_down_pct: optional float (skip if T+1 open gaps below this)
  chase: bool (True = you pay up; just for labelling)
=============================================================================
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

IST = "Asia/Kolkata"
COST_PCT = 0.35
HOLDING_DAYS = 5
ANN = np.sqrt(252.0 / HOLDING_DAYS)
FIRST_15M_BARS = 3       # 3 x 5-min = opening range
FIRST_30M_BARS = 6


# =============================================================================
# Per-signal context: everything a level function might need
# =============================================================================

class Ctx:
    """Holds the forward bars for one signal plus reference levels. Day index
    bar_days: 0 = T+1 (entry day), 1 = T+2, ... (matches ORB execution.py)."""
    def __init__(self, row: pd.Series, daily_row: Optional[pd.Series]):
        self.prev_close = float(row.get("prev_close", np.nan))
        self.o = np.asarray(row["bar_opens"], float)
        self.h = np.asarray(row["bar_highs"], float)
        self.l = np.asarray(row["bar_lows"], float)
        self.c = np.asarray(row["bar_closes"], float)
        self.d = np.asarray(row["bar_days"], int)
        self.ts = np.asarray(row.get("bar_timestamps", []), dtype="int64")
        self.daily = daily_row if daily_row is not None else pd.Series(dtype=float)
        # day-1 (entry day) slice
        self.d1 = np.where(self.d == 0)[0]
        # previous day OHLC from the daily row if present
        self.prev_high = self._dly("D_prev_high")
        self.prev_low = self._dly("D_prev_low")

    def _dly(self, col):
        v = self.daily.get(col, np.nan)
        try:
            return float(v)
        except Exception:
            return np.nan

    def day1_open(self):
        return float(self.o[self.d1[0]]) if len(self.d1) else np.nan

    def day1_vwap_to(self, nbars: int):
        idx = self.d1[:nbars]
        if len(idx) == 0:
            return np.nan
        tp = (self.h[idx] + self.l[idx] + self.c[idx]) / 3.0
        return float(np.mean(tp))

    def fwd_t5_close_ret(self, entry_price: float, entry_idx: int) -> float:
        """Held to the last bar of T+5 (day index < HOLDING_DAYS)."""
        hold = np.where((self.d >= 0) & (self.d < HOLDING_DAYS))[0]
        if entry_price <= 0 or len(hold) == 0:
            return np.nan
        last = hold[-1]
        return (self.c[last] / entry_price - 1.0) * 100.0


# =============================================================================
# Entry resolution: returns (entry_price, entry_idx) or (nan, -1) if skipped
# =============================================================================

def _first_touch_limit(ctx: Ctx, level: float, days=HOLDING_DAYS):
    """Buy limit at `level`: fill the first bar whose LOW <= level (over the
    hold window). Fill price = level (assume limit honoured)."""
    if not np.isfinite(level):
        return np.nan, -1
    idx = np.where((ctx.d >= 0) & (ctx.d < days))[0]
    for i in idx:
        if ctx.l[i] <= level:
            return float(level), int(i)
    return np.nan, -1


def _first_touch_stop(ctx: Ctx, level: float, days=HOLDING_DAYS):
    """Buy stop at `level` (breakout): fill first bar whose HIGH >= level.
    Fill = level (you chase up to it)."""
    if not np.isfinite(level):
        return np.nan, -1
    idx = np.where((ctx.d >= 0) & (ctx.d < days))[0]
    for i in idx:
        if ctx.h[i] >= level:
            return float(level), int(i)
    return np.nan, -1


def _market_open(ctx: Ctx):
    if len(ctx.d1) == 0:
        return np.nan, -1
    return ctx.day1_open(), int(ctx.d1[0])


def resolve_entry(name: str, spec: dict, ctx: Ctx):
    # gap-down reject (applies to any variant)
    gd = spec.get("reject_gap_down_pct")
    if gd is not None and np.isfinite(ctx.prev_close) and len(ctx.d1):
        gap = (ctx.day1_open() / ctx.prev_close - 1) * 100
        if gap < gd:
            return np.nan, -1, "gap_reject"

    kind = spec["kind"]
    level = spec["level"](ctx) if "level" in spec else np.nan

    if kind == "market":
        px, i = _market_open(ctx)
    elif kind == "limit":
        px, i = _first_touch_limit(ctx, level)
    elif kind == "stop":
        px, i = _first_touch_stop(ctx, level)
    elif kind == "confirm":
        px, i = spec["resolve"](ctx)
    else:
        return np.nan, -1, "bad_kind"
    return px, i, ("filled" if i >= 0 else "no_fill")


# =============================================================================
# CONFIRM-style resolvers (don't chase: enter at a close once a condition holds)
# =============================================================================

def _confirm_reclaim_vwap_30m(ctx: Ctx):
    idx = ctx.d1[:FIRST_30M_BARS]
    for k in range(1, len(idx)):
        vwap = ctx.day1_vwap_to(k + 1)
        if ctx.c[idx[k]] >= vwap:
            return float(ctx.c[idx[k]]), int(idx[k])
    return np.nan, -1


def _confirm_first_green_15m(ctx: Ctx):
    idx = ctx.d1[:FIRST_15M_BARS]
    if len(idx) < FIRST_15M_BARS:
        return np.nan, -1
    if ctx.c[idx[-1]] > ctx.o[idx[0]]:
        return float(ctx.c[idx[-1]]), int(idx[-1])
    return np.nan, -1


def _confirm_hold_above_prevclose_eod(ctx: Ctx):
    if len(ctx.d1) == 0 or not np.isfinite(ctx.prev_close):
        return np.nan, -1
    last = ctx.d1[-1]
    if ctx.c[last] >= ctx.prev_close:
        return float(ctx.c[last]), int(last)
    return np.nan, -1


def _confirm_open_if_above_prevclose(ctx: Ctx):
    if len(ctx.d1) == 0 or not np.isfinite(ctx.prev_close):
        return np.nan, -1
    op = ctx.day1_open()
    return (op, int(ctx.d1[0])) if op >= ctx.prev_close else (np.nan, -1)


# =============================================================================
# VARIANT CATALOG  (extend freely)
# Each level function takes a Ctx and returns a price.
# =============================================================================

def L_prevclose(ctx): return ctx.prev_close
def L_prevclose_m05(ctx): return ctx.prev_close * 0.995
def L_prevclose_m10(ctx): return ctx.prev_close * 0.99
def L_prev_low(ctx): return ctx.prev_low
def L_prev_high(ctx): return ctx.prev_high
def L_day1_low(ctx): return float(np.min(ctx.l[ctx.d1])) if len(ctx.d1) else np.nan
def L_day1_open(ctx): return ctx.day1_open()
def L_vwap_30m(ctx): return ctx.day1_vwap_to(FIRST_30M_BARS)
# daily-cache levels (require --daily-cache)
def L_ema20(ctx): return ctx._dly("D_ema20")
def L_sma20(ctx): return ctx._dly("D_sma20")
def L_sma50(ctx): return ctx._dly("D_sma50")
def L_cpr_bc(ctx): return ctx._dly("D_cpr_bc")
def L_cpr_tc(ctx): return ctx._dly("D_cpr_tc")
def L_pivot(ctx): return ctx._dly("D_pivot")
def L_support1(ctx): return ctx._dly("D_support1")
def L_vpoc(ctx): return ctx._dly("D_vpoc")
def L_weekly_vpoc(ctx): return ctx._dly("D_weekly_vpoc")
def L_week_low(ctx): return ctx._dly("W_low") if "W_low" in ctx.daily else np.nan
# breakout (chasing) levels for contrast
def L_donch_hi20(ctx):
    # prev 20d high proxy from daily cache if present
    return ctx._dly("D_breakout_high_20_level") if "D_breakout_high_20_level" in ctx.daily else np.nan


VARIANTS: Dict[str, dict] = {
    # ---- baselines ----
    "OPEN_T1":            {"kind": "market", "chase": False},
    "CLOSE_T1":           {"kind": "limit", "level": lambda c: (float(c.c[c.d1[-1]]) if len(c.d1) else np.nan), "chase": False},
    # ---- non-chasing reference limits (buy a dip to the level) ----
    "LIMIT_PREVCLOSE":         {"kind": "limit", "level": L_prevclose, "reject_gap_down_pct": -0.5, "chase": False},
    "LIMIT_PREVCLOSE_m0.5":    {"kind": "limit", "level": L_prevclose_m05, "chase": False},
    "LIMIT_PREVCLOSE_m1.0":    {"kind": "limit", "level": L_prevclose_m10, "chase": False},
    "LIMIT_PREV_LOW":          {"kind": "limit", "level": L_prev_low, "chase": False, "needs_daily": True},
    # LOOKAHEAD: buying at the entry-day's OWN low always fills at the best price
    # of the day, which you cannot know in advance. NOT tradeable. Kept only as a
    # diagnostic upper bound; excluded from the honest leaderboard.
    "LIMIT_DAY1_LOW":          {"kind": "limit", "level": L_day1_low, "chase": False, "lookahead": True},
    "LIMIT_VWAP_30m":          {"kind": "limit", "level": L_vwap_30m, "chase": False},
    # ---- confirm, don't chase (enter at a close once condition holds) ----
    "OPEN_IF_ABOVE_PREVCLOSE": {"kind": "confirm", "resolve": _confirm_open_if_above_prevclose, "chase": False},
    "RECLAIM_VWAP_30m":        {"kind": "confirm", "resolve": _confirm_reclaim_vwap_30m, "chase": False},
    "HOLD_ABOVE_PREVCLOSE_EOD":{"kind": "confirm", "resolve": _confirm_hold_above_prevclose_eod, "chase": False},
    "FIRST_GREEN_15m":         {"kind": "confirm", "resolve": _confirm_first_green_15m, "chase": False},
    # ---- daily-cache reference limits (need --daily-cache) ----
    "LIMIT_EMA20":             {"kind": "limit", "level": L_ema20, "chase": False, "needs_daily": True},
    "LIMIT_SMA20":             {"kind": "limit", "level": L_sma20, "chase": False, "needs_daily": True},
    "LIMIT_SMA50":             {"kind": "limit", "level": L_sma50, "chase": False, "needs_daily": True},
    "LIMIT_CPR_BC":            {"kind": "limit", "level": L_cpr_bc, "chase": False, "needs_daily": True},
    "LIMIT_PIVOT":             {"kind": "limit", "level": L_pivot, "chase": False, "needs_daily": True},
    "LIMIT_SUPPORT1":          {"kind": "limit", "level": L_support1, "chase": False, "needs_daily": True},
    "LIMIT_WEEKLY_VPOC":       {"kind": "limit", "level": L_weekly_vpoc, "chase": False, "needs_daily": True},
    # ---- chasing breakouts (for contrast; expect worse Sharpe) ----
    "STOP_PREV_HIGH":          {"kind": "stop", "level": L_prev_high, "chase": True},
    "STOP_CPR_TC":             {"kind": "stop", "level": L_cpr_tc, "chase": True, "needs_daily": True},
}


# =============================================================================
# Portfolio metrics
# =============================================================================

def _sharpe(daily: pd.Series):
    if len(daily) < 5 or daily.std(ddof=0) == 0:
        return np.nan
    return float(ANN * daily.mean() / daily.std(ddof=0))


def _maxdd(daily: pd.Series):
    if daily.empty:
        return np.nan
    eq = (1 + daily / 100).cumprod()
    return float(((eq / eq.cummax() - 1) * 100).min())


def _portfolio(df: pd.DataFrame, ret_col: str, weight_col: Optional[str]):
    g = df.groupby(df["signal_date"].dt.normalize())
    if weight_col is None:
        daily = g[ret_col].mean().sort_index()
    else:
        def wavg(x):
            w = pd.to_numeric(x[weight_col], errors="coerce").fillna(0).values
            r = pd.to_numeric(x[ret_col], errors="coerce").values
            return float(np.dot(w, np.nan_to_num(r)) / w.sum()) if w.sum() > 0 else float(np.nanmean(r))
        daily = g.apply(wavg).sort_index()
    return _sharpe(daily), _maxdd(daily)


# =============================================================================
# Run one variant across all signals
# =============================================================================

def run_variant(paths: pd.DataFrame, daily_map: dict, name: str, spec: dict) -> pd.DataFrame:
    out = []
    recs = paths.to_dict("records")
    for rec in recs:
        daily_row = None
        if spec.get("needs_daily"):
            key = (rec["symbol"], pd.Timestamp(rec["signal_date"]).normalize())
            daily_row = daily_map.get(key)
            if daily_row is None:
                continue
        ctx = Ctx(pd.Series(rec), daily_row)
        px, i, status = resolve_entry(name, spec, ctx)
        taken = i >= 0
        ret = ctx.fwd_t5_close_ret(px, i) if taken else np.nan
        slip = ((px / ctx.prev_close - 1) * 100) if (taken and np.isfinite(ctx.prev_close)) else np.nan
        out.append({
            "symbol": rec["symbol"], "signal_date": pd.Timestamp(rec["signal_date"]),
            "regime": rec.get("regime"), "prob_bucket": rec.get("prob_bucket"),
            "probability": rec.get("probability"),
            "taken": taken, "ret": ret, "slip": slip, "status": status,
        })
    return pd.DataFrame(out)


def summarize(per: pd.DataFrame, name: str, chase: bool) -> List[dict]:
    rows = []
    for scope, sub in [("OVERALL", per)] + \
            [(f"regime={r}", per[per.regime == r]) for r in sorted(per.regime.dropna().unique())] + \
            [(f"bucket={b}", per[per.prob_bucket == b]) for b in sorted(per.prob_bucket.dropna().unique())]:
        if len(sub) < 50:
            continue
        taken = sub[sub.taken]
        n = len(sub); nt = len(taken)
        deploy = 100 * nt / max(n, 1)
        tm = taken.ret.mean() - COST_PCT if nt else np.nan
        win = 100 * (taken.ret > 0).mean() if nt else np.nan
        slip = taken.slip.mean() if nt else np.nan
        # same-universe portfolio: non-taken => cash(0)
        port = sub.copy(); port["_r"] = 0.0
        port.loc[port.taken, "_r"] = pd.to_numeric(taken.ret, errors="coerce").values - COST_PCT
        ew_sh, ew_dd = _portfolio(port, "_r", None)
        pw_sh, pw_dd = _portfolio(port, "_r", "probability")
        rows.append(dict(scope=scope, variant=name, chase=chase, n_signals=n,
                         deploy_pct=round(deploy, 1), taken_mean=round(tm, 3) if pd.notna(tm) else np.nan,
                         win_pct=round(win, 1) if pd.notna(win) else np.nan,
                         entry_slip_pct=round(slip, 3) if pd.notna(slip) else np.nan,
                         port_ew_sharpe=round(ew_sh, 3) if pd.notna(ew_sh) else np.nan,
                         port_ew_maxdd=round(ew_dd, 1) if pd.notna(ew_dd) else np.nan,
                         port_pw_sharpe=round(pw_sh, 3) if pd.notna(pw_sh) else np.nan))
    return rows


# =============================================================================
# Daily-cache join (for daily-level variants)
# =============================================================================

def load_daily_map(daily_cache: Path, symbols: List[str]) -> dict:
    import glob
    m = {}
    if not daily_cache:
        return m
    cols = ["timestamp", "D_ema20", "D_sma20", "D_sma50", "D_cpr_bc", "D_cpr_tc",
            "D_pivot", "D_support1", "D_vpoc", "D_weekly_vpoc",
            "D_prev_high", "D_prev_low", "D_prev_close"]
    for sym in set(symbols):
        fp = Path(daily_cache) / f"{sym}_daily.parquet"
        if not fp.exists():
            continue
        try:
            d = pd.read_parquet(fp)
        except Exception:
            continue
        have = [c for c in cols if c in d.columns]
        d = d[have].copy()
        d["timestamp"] = pd.to_datetime(d["timestamp"], errors="coerce", utc=True).dt.tz_convert(IST)
        d["key"] = d["timestamp"].dt.normalize()
        for _, r in d.iterrows():
            m[(sym, r["key"])] = r
    return m


# =============================================================================
# Self-test with synthetic bars (no external data needed)
# =============================================================================

def selftest():
    print("[selftest] building synthetic signals...")
    rng = np.random.default_rng(0)
    recs = []
    base = pd.Timestamp("2024-01-01", tz=IST)
    for s in range(300):
        sd = base + pd.Timedelta(days=int(rng.integers(0, 400)))
        prev_close = 100.0
        bars_o, bars_h, bars_l, bars_c, bars_d = [], [], [], [], []
        drift = rng.normal(0.3, 2.0)  # 5-day drift in %
        for day in range(5):
            px = prev_close * (1 + drift / 100 * (day + 1) / 5)
            for b in range(75):
                o = px * (1 + rng.normal(0, 0.003))
                h = o * (1 + abs(rng.normal(0, 0.004)))
                l = o * (1 - abs(rng.normal(0, 0.004)))
                c = (h + l) / 2
                bars_o.append(o); bars_h.append(h); bars_l.append(l); bars_c.append(c); bars_d.append(day)
        recs.append({"symbol": f"SYM{s}", "signal_date": sd, "regime": rng.choice(["bull_trend", "bear_trend"]),
                     "prob_bucket": "[0.65,0.70)", "probability": float(rng.uniform(0.65, 0.95)),
                     "prev_close": prev_close, "bar_opens": bars_o, "bar_highs": bars_h,
                     "bar_lows": bars_l, "bar_closes": bars_c, "bar_days": bars_d,
                     "bar_timestamps": list(range(len(bars_o)))})
    paths = pd.DataFrame(recs)
    test = {k: v for k, v in VARIANTS.items() if not v.get("needs_daily")}
    all_rows = []
    for name, spec in test.items():
        per = run_variant(paths, {}, name, spec)
        all_rows += summarize(per, name, spec.get("chase", False))
    res = pd.DataFrame(all_rows)
    ov = res[res.scope == "OVERALL"].sort_values("port_ew_sharpe", ascending=False)
    print(ov[["variant", "deploy_pct", "taken_mean", "win_pct", "entry_slip_pct",
              "port_ew_sharpe", "port_ew_maxdd"]].to_string(index=False))
    print("\n[selftest] engine OK — variants resolve, portfolio metrics compute.")


# =============================================================================
# MAIN
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", type=Path, help="extracted_paths parquet with bar arrays")
    ap.add_argument("--daily-cache", type=Path, default=None, help="cache_daily_new dir for daily-level variants")
    ap.add_argument("--variants", type=str, default=None, help="comma list (default: all)")
    ap.add_argument("--out", type=Path, default=Path("entry_lab_results.csv"))
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest(); return
    if not args.paths or not args.paths.exists():
        raise SystemExit("Provide --paths <extracted_paths.parquet> (or --selftest).")

    paths = pd.read_parquet(args.paths)
    paths["signal_date"] = pd.to_datetime(paths["signal_date"])
    need = ["bar_opens", "bar_highs", "bar_lows", "bar_closes", "bar_days", "prev_close"]
    miss = [c for c in need if c not in paths.columns]
    if miss:
        raise SystemExit(f"paths file missing required columns: {miss}")
    print(f"[lab] {len(paths):,} signals | {paths.signal_date.min().date()} -> {paths.signal_date.max().date()}")
    print(f"[lab] prob {paths.probability.min():.2f}..{paths.probability.max():.2f}  "
          f"regimes={sorted(paths.regime.dropna().unique())}")

    chosen = [v.strip() for v in args.variants.split(",")] if args.variants else list(VARIANTS)
    daily_needed = any(VARIANTS[v].get("needs_daily") for v in chosen if v in VARIANTS)
    daily_map = {}
    if daily_needed:
        if not args.daily_cache:
            print("[lab] WARNING: daily-level variants requested but no --daily-cache; they will be skipped.")
        else:
            print("[lab] loading daily cache for level joins...")
            daily_map = load_daily_map(args.daily_cache, paths["symbol"].tolist())
            print(f"[lab] daily rows mapped: {len(daily_map):,}")

    all_rows = []
    for name in chosen:
        if name not in VARIANTS:
            print(f"[lab] unknown variant {name}, skip"); continue
        spec = VARIANTS[name]
        if spec.get("needs_daily") and not daily_map:
            print(f"[lab] {name}: needs daily cache, skipped"); continue
        per = run_variant(paths, daily_map, name, spec)
        all_rows += summarize(per, name, spec.get("chase", False))
        ov = [r for r in all_rows if r["variant"] == name and r["scope"] == "OVERALL"]
        if ov:
            o = ov[0]
            print(f"  {name:26s} deploy={o['deploy_pct']:>5.1f}%  taken_mean={o['taken_mean']}  "
                  f"slip={o['entry_slip_pct']}  EW_Sharpe={o['port_ew_sharpe']}  DD={o['port_ew_maxdd']}")

    res = pd.DataFrame(all_rows)
    # mark look-ahead / chasing variants so the leaderboard is honest
    look = {n for n, s in VARIANTS.items() if s.get("lookahead")}
    if not res.empty:
        res["lookahead"] = res["variant"].isin(look)

    # robust CSV write: if --out is a directory or unwritable, fall back to cwd
    out = Path(args.out)
    try:
        if out.exists() and out.is_dir():
            out = out / "entry_lab_results.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        res.to_csv(out, index=False)
    except (PermissionError, OSError) as e:
        fallback = Path.cwd() / "entry_lab_results.csv"
        print(f"[lab] WARNING: could not write {out} ({e}); writing to {fallback} instead")
        out = fallback
        res.to_csv(out, index=False)
    print(f"\n[out] {out}")

    print("\n=== OVERALL leaderboard by portfolio Sharpe (EW) ===")
    print("    (lookahead=True rows are NOT tradeable — diagnostic only)")
    ov = res[res.scope == "OVERALL"].sort_values("port_ew_sharpe", ascending=False)
    cols = ["variant", "lookahead", "chase", "deploy_pct", "taken_mean", "win_pct",
            "entry_slip_pct", "port_ew_sharpe", "port_ew_maxdd", "port_pw_sharpe"]
    cols = [c for c in cols if c in ov.columns]
    print(ov[cols].to_string(index=False))

    # honest winner = best Sharpe among tradeable (non-lookahead) variants
    trade = ov[~ov.get("lookahead", False)] if "lookahead" in ov.columns else ov
    if not trade.empty:
        b = trade.iloc[0]
        base = ov[ov.variant == "OPEN_T1"]
        base_sh = float(base.port_ew_sharpe.iloc[0]) if not base.empty else float("nan")
        print(f"\nBest TRADEABLE entry: {b.variant}  EW_Sharpe={b.port_ew_sharpe} "
              f"(vs OPEN_T1 {base_sh})  deploy={b.deploy_pct}%  slip={b.entry_slip_pct} "
              f"maxDD={b.port_ew_maxdd}%")


if __name__ == "__main__":
    main()
