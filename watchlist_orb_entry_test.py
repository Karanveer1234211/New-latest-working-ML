#!/usr/bin/env python3
"""
=============================================================================
WATCHLIST + ORB-BREAK ENTRY TEST
=============================================================================

HYPOTHESIS (user)
-----------------
Keep a rolling watchlist of every prob>=0.65 name from the last <=5 days.
Do NOT enter at the open. Instead WATCH, and only take the trade if the name
BREAKS its 15-min opening range (ORB) on any of the next 5 days. On the break,
manage with a tight bracket:
    TP in {2%, 3%}
    SL in {-2%, -3%, ORB-low (range_low), 2x-range (vol stop)}   [+ ATR if available]

This script answers, on the SAME signal universe and net of costs:
  1. How many watchlist names actually break ORB, and on which day?
  2. For each (TP x SL) bracket, what is the realised net return, win%,
     SL-hit%, TP-hit% and daily-basket Sharpe of the ORB-break trades?
  3. How does each bracket compare to:
        - HOLD: naive open->T+5 close on the SAME names (no bracket)
        - the SAME-UNIVERSE portfolio where non-breakers sit in CASH(0)
  4. Per regime and per probability bucket.

DATA / FIRST-TOUCH CORRECTNESS
------------------------------
Reads the per-trade parquet from opportunities.py / ORB execution.py. The
ORB variant rows carry, for each signal:
    triggered           -> did it break ORB within 5 days (YES/NO)
    entry_day_offset    -> which day it broke (T+1..T+5)
    fwd_return_to_t5_close_pct -> ORB-entry held to T+5 close (no bracket)
    tp{2,3}_sl{...}_ret -> FIRST-TOUCH bracket outcome (stop checked before
                           target each bar) for the ORB entry. We read these
                           rather than approximate from MAE/MFE.
Stop names: 'slrange_low' = exit at the ORB low (your "orb low" stop),
            'sltwo_range' = entry - 2*(ORH-ORL) volatility stop.

ATR STOP
--------
per_trade_v4.parquet does NOT carry an ATR-based stop column. If your file
has one (e.g. tp2_slatr_ret) it is auto-detected and included; otherwise the
script prints a note and tests the 4 stops that exist. To get a true ATR
stop, re-run ORB execution.py (its SL ladder includes -ATR(14)).

USAGE
-----
    python watchlist_orb_entry_test.py --per-trade per_trade_v4.parquet
    python watchlist_orb_entry_test.py --per-trade per_trade_v4.parquet --compare-windows
    python watchlist_orb_entry_test.py --per-trade .../per_trade.parquet \
        --orb-variant ORB_15_anyday --watch-window 2

WATCH-WINDOW COMPARISON  (--compare-windows)
--------------------------------------------
Sweeps the watch window from T+1..T+5 and reports, for the best general
bracket, how break-rate and Sharpe change as you shorten the window. Answers
"is a 2-day watch as good as 5-day?".

PORTFOLIO SIZING
----------------
The same-universe portfolio (non-breakers in cash) is reported two ways:
  EW = equal weight across the day's breakers
  PW = probability-weighted (size each name by its model prob, normalised
       per day). Tests whether leaning into higher-conviction breaks helps.
=============================================================================
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

COST_PCT = 0.35           # round-trip cost (25bps + 2x5bps slippage)
HOLDING_DAYS = 5
ANN = np.sqrt(252.0 / HOLDING_DAYS)

# The brackets the user asked for: TP {2,3} x SL {2,3, orb-low, 2x-range, atr?}
TP_LEVELS = [2.0, 3.0]
# stop spec: (label, column-suffix). ATR auto-added if present.
STOP_SPECS = [
    ("SL-2%",     "sl-2.0"),
    ("SL-3%",     "sl-3.0"),
    ("ORB-low",   "slrange_low"),
    ("2x-range",  "sltwo_range"),
]
ATR_SUFFIX_CANDIDATES = ["slatr", "slatr14", "sl-atr", "slatr_14"]


def daily_sharpe(df: pd.DataFrame, col: str, date="signal_date") -> float:
    s = df.groupby(df[date].dt.normalize())[col].mean().sort_index()
    if len(s) < 5 or s.std(ddof=0) == 0:
        return np.nan
    return float(ANN * s.mean() / s.std(ddof=0))


def max_dd(df: pd.DataFrame, col: str, date="signal_date") -> float:
    d = df.groupby(df[date].dt.normalize())[col].mean().sort_index()
    if d.empty:
        return np.nan
    eq = (1 + d / 100).cumprod()
    return float(((eq / eq.cummax() - 1) * 100).min())


def _weighted_daily(df: pd.DataFrame, ret_col: str, weight_col: Optional[str],
                    date="signal_date") -> pd.Series:
    """Per-day basket return. If weight_col is None -> equal weight (mean).
    Else weighted average sum(w*r)/sum(w) within each signal_date (prob-weighted
    position sizing). Weights are normalised PER DAY so a day with one name and a
    day with twenty are both fully invested."""
    g = df.groupby(df[date].dt.normalize())
    if weight_col is None:
        return g[ret_col].mean().sort_index()

    def _wavg(x):
        w = pd.to_numeric(x[weight_col], errors="coerce").fillna(0.0).values
        r = pd.to_numeric(x[ret_col], errors="coerce").values
        sw = w.sum()
        return float(np.dot(w, np.nan_to_num(r)) / sw) if sw > 0 else float(np.nanmean(r))

    return g.apply(_wavg).sort_index()


def _sharpe_from_daily(daily: pd.Series) -> float:
    if len(daily) < 5 or daily.std(ddof=0) == 0:
        return np.nan
    return float(ANN * daily.mean() / daily.std(ddof=0))


def _dd_from_daily(daily: pd.Series) -> float:
    if daily.empty:
        return np.nan
    eq = (1 + daily / 100).cumprod()
    return float(((eq / eq.cummax() - 1) * 100).min())


def fmt(x, w=7, p=2):
    return f"{x:>{w}.{p}f}" if pd.notna(x) else f"{'NA':>{w}}"


def analyse(sub: pd.DataFrame, label: str, tp_levels, stops, hold_col):
    n_total = len(sub)
    broke = sub[sub["triggered"].astype(bool)].copy()
    n_break = len(broke)
    break_rate = 100 * n_break / max(n_total, 1)

    print("\n" + "=" * 96)
    print(f"{label}   |   watchlist names={n_total:,}   ORB-broke={n_break:,} "
          f"({break_rate:.1f}%)   never-broke={n_total - n_break:,}")
    print("=" * 96)

    # break-day distribution
    if n_break:
        bd = broke["entry_day_offset"].value_counts().sort_index()
        dist = "  ".join(f"T+{int(k)}:{v}({100*v/n_break:.0f}%)" for k, v in bd.items())
        print(f"  break-day: {dist}")

    # HOLD reference on the SAME broke names (ORB entry, no bracket, to T5)
    if n_break:
        h = pd.to_numeric(broke[hold_col], errors="coerce")
        print(f"  REFERENCE ORB-entry HOLD->T5 (no bracket): "
              f"net={fmt(h.mean()-COST_PCT)}%  win={fmt(100*(h>0).mean(),5,1)}%  "
              f"Sharpe={fmt(daily_sharpe(broke.assign(_r=h-COST_PCT),'_r'),5)}")

    # header
    print(f"\n  {'bracket':16s} {'n':>5} {'net%':>7} {'win%':>6} {'TPhit%':>7} "
          f"{'SLhit%':>7} {'Sharpe':>7} {'maxDD%':>7} | {'EWnet%':>9} {'EW Sh':>7} "
          f"{'EW DD%':>8} | {'PW Sh':>7} {'PW DD%':>8}")
    print("  " + "-" * 120)

    rows = []
    for tp in tp_levels:
        for sname, suf in stops:
            rcol = f"tp{tp:g}_{suf}_ret"
            tcol = f"tp{tp:g}_{suf}_hittp"
            scol = f"tp{tp:g}_{suf}_hitsl"
            if rcol not in sub.columns:
                continue
            # trades = ORB-break names only
            r = pd.to_numeric(broke[rcol], errors="coerce")
            net = r.mean() - COST_PCT
            win = 100 * (r > 0).mean()
            tph = 100 * pd.to_numeric(broke[tcol], errors="coerce").mean() if tcol in broke else np.nan
            slh = 100 * pd.to_numeric(broke[scol], errors="coerce").mean() if scol in broke else np.nan
            sh = daily_sharpe(broke.assign(_r=r - COST_PCT), "_r")
            dd = max_dd(broke.assign(_r=r - COST_PCT), "_r")

            # SAME-UNIVERSE portfolio: non-breakers = CASH(0)
            port = sub.copy()
            port["_pr"] = 0.0
            port.loc[port["triggered"].astype(bool), "_pr"] = (
                pd.to_numeric(broke[rcol], errors="coerce").values - COST_PCT
            )
            # equal-weight portfolio
            eq_daily = _weighted_daily(port, "_pr", None)
            p_net = port["_pr"].mean()
            p_sh = _sharpe_from_daily(eq_daily)
            p_dd = _dd_from_daily(eq_daily)
            # prob-weighted portfolio (size each name by its probability)
            pw_daily = _weighted_daily(port, "_pr", "probability")
            pw_sh = _sharpe_from_daily(pw_daily)
            pw_dd = _dd_from_daily(pw_daily)

            tag = f"TP{tp:g}/{sname}"
            print(f"  {tag:16s} {n_break:>5} {fmt(net)} {fmt(win,6,1)} {fmt(tph,7,1)} "
                  f"{fmt(slh,7,1)} {fmt(sh)} {fmt(dd)} | {fmt(p_net,9)} {fmt(p_sh,7)} "
                  f"{fmt(p_dd,8)} | {fmt(pw_sh,7)} {fmt(pw_dd,8)}")
            rows.append(dict(scope=label, bracket=tag, n_break=n_break, break_rate=break_rate,
                             trade_net=net, trade_win=win, tp_hit=tph, sl_hit=slh,
                             trade_sharpe=sh, trade_maxdd=dd,
                             port_eq_net=p_net, port_eq_sharpe=p_sh, port_eq_maxdd=p_dd,
                             port_pw_sharpe=pw_sh, port_pw_maxdd=pw_dd))
    return rows


def _apply_window(pt_variant: pd.DataFrame, window: int) -> pd.DataFrame:
    """Return a copy where breaks later than T+window are treated as no-break."""
    s = pt_variant.copy()
    if window < 5:
        late = s["triggered"].astype(bool) & (s["entry_day_offset"] > window)
        s.loc[late, "triggered"] = False
    return s


def _run_all_scopes(sub: pd.DataFrame, stops, tag: str = "") -> List[Dict]:
    rows: List[Dict] = []
    rows += analyse(sub, f"OVERALL{tag}", TP_LEVELS, stops, "fwd_return_to_t5_close_pct")
    for reg in sorted(sub["regime"].dropna().unique()):
        s = sub[sub["regime"] == reg]
        if len(s) >= 50:
            rows += analyse(s, f"regime={reg}{tag}", TP_LEVELS, stops, "fwd_return_to_t5_close_pct")
    for b in sorted(sub["prob_bucket"].dropna().unique()):
        s = sub[sub["prob_bucket"] == b]
        if len(s) >= 50:
            rows += analyse(s, f"bucket={b}{tag}", TP_LEVELS, stops, "fwd_return_to_t5_close_pct")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-trade", type=Path, required=True)
    ap.add_argument("--orb-variant", type=str, default=None,
                    help="variant name for the ORB-break engine (auto-detected if omitted)")
    ap.add_argument("--watch-window", type=int, default=5,
                    help="only count breaks on or before T+watch_window (default 5)")
    ap.add_argument("--compare-windows", action="store_true",
                    help="sweep watch-windows 1..5 and print an OVERALL Sharpe/break-rate "
                         "comparison (answers '2-day vs 5-day watch').")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    pt = pd.read_parquet(args.per_trade)
    pt["signal_date"] = pd.to_datetime(pt["signal_date"])

    # pick the ORB variant
    orb_variants = [v for v in pt["variant"].unique() if "ORB" in str(v).upper()]
    if args.orb_variant:
        var = args.orb_variant
    elif orb_variants:
        held = [v for v in orb_variants if "held" in v.lower() or "anyday" in v.lower()]
        var = (held or orb_variants)[0]
    else:
        raise SystemExit(f"No ORB variant found. Variants: {sorted(pt['variant'].unique())}")
    base = pt[pt["variant"] == var].copy()
    print("#" * 96)
    print(f"# WATCHLIST + ORB-BREAK ENTRY TEST   |   ORB variant = {var}")
    print(f"# per-trade: {args.per_trade.name}")
    print("#" * 96)
    print(f"  watchlist universe: {len(base):,} signals "
          f"({base.signal_date.min().date()} -> {base.signal_date.max().date()})")
    print(f"  prob range: {base.probability.min():.3f}..{base.probability.max():.3f}  "
          f"buckets={sorted(base.prob_bucket.dropna().unique())}")

    # detect ATR stop if present
    stops = list(STOP_SPECS)
    for suf in ATR_SUFFIX_CANDIDATES:
        if any(c == f"tp2_{suf}_ret" for c in base.columns):
            stops.append(("ATR", suf))
            print(f"  [atr] found ATR stop column suffix '{suf}' -> included")
            break
    else:
        print("  [atr] no ATR-stop column in this file -> testing SL-2/SL-3/ORB-low/2x-range. "
              "Re-run ORB execution.py for a true ATR(14) stop.")

    print("\nLEGEND: 'trade' metrics = ORB-break trades only (held to T+5, first-touch SL). "
          "EW = equal-weight same-universe portfolio (non-breakers in CASH=0). "
          "PW = probability-weighted sizing of the same portfolio.")

    # ---------------- watch-window sweep ----------------
    if args.compare_windows:
        print("\n" + "#" * 96)
        print("# WATCH-WINDOW SWEEP  (OVERALL): does a shorter watch window keep the edge?")
        print("#" * 96)
        # use the best general bracket (TP3 / 2x-range) if present, else first available
        probe = None
        for tp, (sn, suf) in [(3.0, ("2x-range", "sltwo_range")),
                              (3.0, ("ORB-low", "slrange_low")),
                              (2.0, ("SL-3%", "sl-3.0"))]:
            if f"tp{tp:g}_{suf}_ret" in base.columns:
                probe = (tp, sn, suf)
                break
        print(f"  probe bracket: TP{probe[0]:g}/{probe[1]}")
        print(f"\n  {'window':>8} {'breaks':>7} {'break%':>7} {'trade_net%':>11} "
              f"{'trade_Sh':>9} {'EW_Sh':>7} {'PW_Sh':>7} {'EW_DD%':>8}")
        print("  " + "-" * 74)
        for w in range(1, 6):
            sw = _apply_window(base, w)
            broke = sw[sw["triggered"].astype(bool)]
            nb = len(broke)
            rcol = f"tp{probe[0]:g}_{probe[2]}_ret"
            r = pd.to_numeric(broke[rcol], errors="coerce")
            tsh = daily_sharpe(broke.assign(_r=r - COST_PCT), "_r")
            port = sw.copy(); port["_pr"] = 0.0
            port.loc[port["triggered"].astype(bool), "_pr"] = r.values - COST_PCT
            ew = _weighted_daily(port, "_pr", None)
            pw = _weighted_daily(port, "_pr", "probability")
            print(f"  {('T+'+str(w)):>8} {nb:>7} {fmt(100*nb/len(sw),7,1)} "
                  f"{fmt(r.mean()-COST_PCT,11)} {fmt(tsh,9)} {fmt(_sharpe_from_daily(ew),7)} "
                  f"{fmt(_sharpe_from_daily(pw),7)} {fmt(_dd_from_daily(ew),8)}")
        print("\n  (If T+2 keeps most of the break% and Sharpe, a 2-day watch is enough.)")

    # ---------------- full breakdown at the chosen window ----------------
    sub = _apply_window(base, args.watch_window)
    if args.watch_window < 5:
        dropped = int((base["triggered"].astype(bool)).sum() - (sub["triggered"].astype(bool)).sum())
        print(f"\n[window] watch-window=T+{args.watch_window}: {dropped} late breaks treated as no-break")
    else:
        print(f"\n[window] watch-window=T+5 (full)")

    all_rows = _run_all_scopes(sub, stops)

    res = pd.DataFrame(all_rows)
    out = args.out or args.per_trade.with_name("watchlist_orb_entry_results.csv")
    res.to_csv(out, index=False)
    print(f"\n[out] {out}")

    # headline: best bracket overall by trade Sharpe
    ov = res[res.scope == "OVERALL"].sort_values("trade_sharpe", ascending=False)
    if not ov.empty:
        best = ov.iloc[0]
        print(f"\nBest ORB-break bracket OVERALL by trade Sharpe: {best.bracket} "
              f"(net={best.trade_net:.2f}%, win={best.trade_win:.1f}%, Sharpe={best.trade_sharpe:.2f})")


if __name__ == "__main__":
    main()
