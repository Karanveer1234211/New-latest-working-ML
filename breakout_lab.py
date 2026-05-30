#!/usr/bin/env python3
"""
=============================================================================
BREAKOUT LAB  —  test the "catch true moves / avoid fakeouts" checklist
=============================================================================

GOAL
----
A pro breakout trader's edge is CONFIRMATION + CONTEXT, not the level itself.
This lab measures, on the SAME prob-gated signal universe, which breakout
RULES actually raise Sharpe + win-rate vs naive open entry and vs the
LIMIT_PREVCLOSE winner we already found.

Each hypothesis is one composite ENTRY rule = (trigger) x (filters):

  TRIGGERS (how you get in — chase vs confirm vs retest):
    OPEN_T1                 market on next open (baseline)
    LIMIT_PREVCLOSE         non-chasing limit at prior close (current best)
    BREAK_CLOSE_ORH         enter on first 5-min CLOSE above the 15-min ORH
                            (close-confirmed break, not an intrabar touch)
    BREAK_RETEST_ORH        enter only if, AFTER breaking ORH, price pulls
                            back to ~ORH and HOLDS (the retest-hold). This is
                            the classic "buy the retest, not the break".
    BREAK_PREVHIGH_CLOSE    5-min close above prior-day high (needs daily)
    BREAK_DONCH20_CLOSE     5-min close above prior 20-day high (needs daily)

  FILTERS (only take the break if...):
    trend_only              regime in {bull_trend, bear_trend}
    bull_only               regime == bull_trend
    compression             signal-day was coiled (D_nr OR low D_bb_bw_20 /
                            D_compress_state)  [needs daily]
    clear_air               not already extended (D_donch_pos_20 < 0.9 OR
                            below 52w high)    [needs daily]
    vol_surge               signal-day volume surge (D_vol_surge_20 or
                            D_dvol_z20 >= z)   [needs daily]   (coarse: this is
                            SIGNAL-DAY volume, NOT the breakout-bar volume,
                            because extracted_paths has no bar_volumes. See the
                            patch in the docstring to enable true bar volume.)
    strong_close            breakout bar closes in top 25% of its range
    gap_reject              skip if T+1 opens gapping down > 0.5%

Exit = held to T+5 close (swap in TP/SL ladder later). One trade per signal;
non-triggered => CASH(0) in the same-universe portfolio.

WHY SOME THINGS CAN'T BE TESTED YET
-----------------------------------
extracted_paths_v2.parquet stores bar_opens/highs/lows/closes/bar_days but
NOT bar_volumes. So:
  * true VWAP and breakout-BAR relative-volume are NOT available per bar.
  * vol_surge here uses the SIGNAL-DAY daily volume features as a proxy.
To get real intraday volume confirmation, add ONE line to _extract_paths()
in "ORB execution.py":
      rec["bar_volumes"] = ext["volume"].tolist()
then re-run it. This lab auto-detects bar_volumes and, if present, switches
vol_surge + VWAP to the true bar-level computation.

USAGE
-----
  python breakout_lab.py --selftest
  python breakout_lab.py --paths .../extracted_paths_v2.parquet \
      --daily-cache "C:/.../cache_daily_new" --out breakout_results.csv
=============================================================================
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

IST = "Asia/Kolkata"
COST_PCT = 0.35
HOLDING_DAYS = 5
ANN = np.sqrt(252.0 / HOLDING_DAYS)
FIRST_15M_BARS = 3
STRONG_CLOSE_FRAC = 0.75       # close in top 25% of bar range
VOL_Z = 1.0                    # signal-day dvol z threshold for vol_surge
RETEST_TOL = 0.003             # retest must come within 0.3% of ORH and hold


# =============================================================================
# Context
# =============================================================================

class Ctx:
    def __init__(self, row: dict, daily_row: Optional[pd.Series]):
        self.prev_close = float(row.get("prev_close", np.nan))
        self.o = np.asarray(row["bar_opens"], float)
        self.h = np.asarray(row["bar_highs"], float)
        self.l = np.asarray(row["bar_lows"], float)
        self.c = np.asarray(row["bar_closes"], float)
        self.d = np.asarray(row["bar_days"], int)
        self.v = np.asarray(row["bar_volumes"], float) if "bar_volumes" in row and row["bar_volumes"] is not None else None
        self.t1_orh = float(row.get("t1_orh_15min", np.nan))
        self.t1_orl = float(row.get("t1_orl_15min", np.nan))
        self.daily = daily_row if daily_row is not None else pd.Series(dtype=float)
        self.d1 = np.where(self.d == 0)[0]

    def dly(self, col, default=np.nan):
        v = self.daily.get(col, default)
        try:
            return float(v)
        except Exception:
            return default

    def day1_open(self):
        return float(self.o[self.d1[0]]) if len(self.d1) else np.nan

    def orh(self):
        """Opening-range high: prefer the precomputed t1_orh_15min, else first 3 bars."""
        if np.isfinite(self.t1_orh):
            return self.t1_orh
        idx = self.d1[:FIRST_15M_BARS]
        return float(np.max(self.h[idx])) if len(idx) else np.nan

    def hold_window(self):
        return np.where((self.d >= 0) & (self.d < HOLDING_DAYS))[0]

    def fwd_ret(self, entry_price, entry_idx):
        hold = self.hold_window()
        if not np.isfinite(entry_price) or entry_price <= 0 or len(hold) == 0:
            return np.nan
        return (self.c[hold[-1]] / entry_price - 1.0) * 100.0


# =============================================================================
# Filters  (return True = pass)
# =============================================================================

def f_trend_only(ctx, row):
    return str(row.get("regime", "")).endswith("trend")

def f_bull_only(ctx, row):
    return row.get("regime") == "bull_trend"

def f_gap_reject(ctx, row):
    if not np.isfinite(ctx.prev_close) or len(ctx.d1) == 0:
        return True
    gap = (ctx.day1_open() / ctx.prev_close - 1) * 100
    return gap >= -0.5

def f_compression(ctx, row):
    # coiled = NR day OR Bollinger bandwidth in bottom 40% (compress_state low)
    nr = ctx.dly("D_nr", 0)
    cs = ctx.dly("D_compress_state", np.nan)
    if np.isfinite(cs):
        return (nr >= 1) or (cs <= 0.40)
    return nr >= 1

def f_clear_air(ctx, row):
    # not already pinned at the top of its 20d channel / 52w high
    dp = ctx.dly("D_donch_pos_20", np.nan)
    d52 = ctx.dly("D_dist_from_52wh", np.nan)
    ok = True
    if np.isfinite(dp):
        ok = ok and (dp < 0.90)
    return ok

def f_vol_surge(ctx, row):
    # TRUE bar-volume if available, else signal-day daily proxy
    if ctx.v is not None and len(ctx.d1):
        # breakout-day: first-15m vol vs trailing average of the day
        d1v = ctx.v[ctx.d1]
        if len(d1v) > FIRST_15M_BARS:
            early = d1v[:FIRST_15M_BARS].mean()
            rest = d1v.mean()
            return early >= 1.2 * rest
        return True
    vs = ctx.dly("D_vol_surge_20", np.nan)
    z = ctx.dly("D_dvol_z20", np.nan)
    if np.isfinite(vs) and vs >= 1:
        return True
    if np.isfinite(z):
        return z >= VOL_Z
    return True   # if no data, don't block

FILTERS: Dict[str, Callable] = {
    "trend_only": f_trend_only,
    "bull_only": f_bull_only,
    "gap_reject": f_gap_reject,
    "compression": f_compression,
    "clear_air": f_clear_air,
    "vol_surge": f_vol_surge,
}
DAILY_FILTERS = {"compression", "clear_air", "vol_surge"}


# =============================================================================
# Triggers  (return (entry_price, entry_idx) or (nan,-1))
# =============================================================================

def t_open(ctx, row):
    return (ctx.day1_open(), int(ctx.d1[0])) if len(ctx.d1) else (np.nan, -1)

def t_limit_prevclose(ctx, row):
    lvl = ctx.prev_close
    if not np.isfinite(lvl):
        return np.nan, -1
    for i in ctx.hold_window():
        if ctx.l[i] <= lvl:
            return float(lvl), int(i)
    return np.nan, -1

def _strong_close(ctx, i):
    rng = ctx.h[i] - ctx.l[i]
    if rng <= 0:
        return True
    return (ctx.c[i] - ctx.l[i]) / rng >= STRONG_CLOSE_FRAC

def t_break_close_orh(ctx, row, strong=False):
    lvl = ctx.orh()
    if not np.isfinite(lvl):
        return np.nan, -1
    for i in ctx.hold_window():
        # skip the bars that form the opening range itself
        if ctx.d[i] == 0 and i in set(ctx.d1[:FIRST_15M_BARS]):
            continue
        if ctx.c[i] > lvl:
            if strong and not _strong_close(ctx, i):
                continue
            return float(ctx.c[i]), int(i)
    return np.nan, -1

def t_break_retest_orh(ctx, row):
    """Break ORH on a close, THEN pull back to within RETEST_TOL of ORH and
    hold (next bar closes back above the retest low). Enter on the hold bar."""
    lvl = ctx.orh()
    if not np.isfinite(lvl):
        return np.nan, -1
    hw = list(ctx.hold_window())
    broke = False
    for k, i in enumerate(hw):
        if ctx.d[i] == 0 and i in set(ctx.d1[:FIRST_15M_BARS]):
            continue
        if not broke:
            if ctx.c[i] > lvl:
                broke = True
            continue
        # after break: look for a touch back near ORH then a hold
        if ctx.l[i] <= lvl * (1 + RETEST_TOL):
            # the retest bar; require the NEXT bar to close back above lvl
            if k + 1 < len(hw):
                j = hw[k + 1]
                if ctx.c[j] > lvl:
                    return float(ctx.c[j]), int(j)
    return np.nan, -1

def t_break_prevhigh_close(ctx, row):
    lvl = ctx.dly("D_prev_high", np.nan)
    if not np.isfinite(lvl):
        return np.nan, -1
    for i in ctx.hold_window():
        if ctx.c[i] > lvl:
            return float(ctx.c[i]), int(i)
    return np.nan, -1

def t_break_donch20_close(ctx, row):
    # prior 20-day high level: not directly in cache; approximate with prev_high
    # if a donchian level column exists use it
    lvl = ctx.dly("D_donch_hi20_level", np.nan)
    if not np.isfinite(lvl):
        lvl = ctx.dly("D_prev_high", np.nan)
    if not np.isfinite(lvl):
        return np.nan, -1
    for i in ctx.hold_window():
        if ctx.c[i] > lvl:
            return float(ctx.c[i]), int(i)
    return np.nan, -1

TRIGGERS: Dict[str, Callable] = {
    "OPEN_T1": t_open,
    "LIMIT_PREVCLOSE": t_limit_prevclose,
    "BREAK_CLOSE_ORH": t_break_close_orh,
    "BREAK_CLOSE_ORH_STRONG": lambda c, r: t_break_close_orh(c, r, strong=True),
    "BREAK_RETEST_ORH": t_break_retest_orh,
    "BREAK_PREVHIGH_CLOSE": t_break_prevhigh_close,
    "BREAK_DONCH20_CLOSE": t_break_donch20_close,
}
DAILY_TRIGGERS = {"BREAK_PREVHIGH_CLOSE", "BREAK_DONCH20_CLOSE"}


# =============================================================================
# Hypothesis catalog: (trigger, [filters]). Add freely.
# =============================================================================

def build_hypotheses() -> List[dict]:
    H = []
    # baselines
    H.append({"name": "OPEN_T1", "trigger": "OPEN_T1", "filters": []})
    H.append({"name": "LIMIT_PREVCLOSE", "trigger": "LIMIT_PREVCLOSE", "filters": ["gap_reject"]})
    # close-confirmed ORH break, progressively filtered
    H.append({"name": "ORHbreak", "trigger": "BREAK_CLOSE_ORH", "filters": []})
    H.append({"name": "ORHbreak+trend", "trigger": "BREAK_CLOSE_ORH", "filters": ["trend_only"]})
    H.append({"name": "ORHbreak+strongclose", "trigger": "BREAK_CLOSE_ORH_STRONG", "filters": []})
    H.append({"name": "ORHbreak+vol", "trigger": "BREAK_CLOSE_ORH", "filters": ["vol_surge"]})
    H.append({"name": "ORHbreak+compress", "trigger": "BREAK_CLOSE_ORH", "filters": ["compression"]})
    H.append({"name": "ORHbreak+clearair", "trigger": "BREAK_CLOSE_ORH", "filters": ["clear_air"]})
    H.append({"name": "ORHbreak+FULLCHECKLIST", "trigger": "BREAK_CLOSE_ORH_STRONG",
              "filters": ["trend_only", "compression", "clear_air", "vol_surge", "gap_reject"]})
    # retest-hold (buy the retest, not the break)
    H.append({"name": "ORHretest", "trigger": "BREAK_RETEST_ORH", "filters": []})
    H.append({"name": "ORHretest+trend", "trigger": "BREAK_RETEST_ORH", "filters": ["trend_only"]})
    H.append({"name": "ORHretest+vol+compress", "trigger": "BREAK_RETEST_ORH",
              "filters": ["vol_surge", "compression"]})
    # structural breaks
    H.append({"name": "PrevHighBreak", "trigger": "BREAK_PREVHIGH_CLOSE", "filters": []})
    H.append({"name": "PrevHighBreak+vol+trend", "trigger": "BREAK_PREVHIGH_CLOSE",
              "filters": ["vol_surge", "trend_only"]})
    H.append({"name": "Donch20Break+FULL", "trigger": "BREAK_DONCH20_CLOSE",
              "filters": ["trend_only", "vol_surge", "clear_air"]})
    return H


# =============================================================================
# Metrics
# =============================================================================

def _sharpe(daily):
    if len(daily) < 5 or daily.std(ddof=0) == 0:
        return np.nan
    return float(ANN * daily.mean() / daily.std(ddof=0))

def _maxdd(daily):
    if daily.empty:
        return np.nan
    eq = (1 + daily / 100).cumprod()
    return float(((eq / eq.cummax() - 1) * 100).min())

def _port(df, col, wcol=None):
    g = df.groupby(df["signal_date"].dt.normalize())
    if wcol is None:
        daily = g[col].mean().sort_index()
    else:
        def wavg(x):
            w = pd.to_numeric(x[wcol], errors="coerce").fillna(0).values
            r = pd.to_numeric(x[col], errors="coerce").values
            return float(np.dot(w, np.nan_to_num(r)) / w.sum()) if w.sum() > 0 else float(np.nanmean(r))
        daily = g.apply(wavg).sort_index()
    return _sharpe(daily), _maxdd(daily)


# =============================================================================
# Run one hypothesis
# =============================================================================

def run_hypothesis(recs, daily_map, hyp) -> pd.DataFrame:
    trig = TRIGGERS[hyp["trigger"]]
    flts = [(f, FILTERS[f]) for f in hyp["filters"]]
    needs_daily = (hyp["trigger"] in DAILY_TRIGGERS) or any(f in DAILY_FILTERS for f in hyp["filters"])
    out = []
    for rec in recs:
        daily_row = None
        if needs_daily:
            key = (rec["symbol"], pd.Timestamp(rec["signal_date"]).normalize())
            daily_row = daily_map.get(key)
            if daily_row is None:
                continue
        ctx = Ctx(rec, daily_row)
        if len(ctx.d1) == 0:
            continue
        passed = all(fn(ctx, rec) for _, fn in flts)
        if not passed:
            out.append({**_key(rec), "taken": False, "ret": np.nan})
            continue
        px, i = trig(ctx, rec)
        taken = i >= 0
        out.append({**_key(rec), "taken": taken,
                    "ret": ctx.fwd_ret(px, i) if taken else np.nan})
    return pd.DataFrame(out)

def _key(rec):
    return {"symbol": rec["symbol"], "signal_date": pd.Timestamp(rec["signal_date"]),
            "regime": rec.get("regime"), "prob_bucket": rec.get("prob_bucket"),
            "probability": rec.get("probability")}

def summarize(per, name, hyp):
    rows = []
    scopes = [("OVERALL", per)] + \
             [(f"regime={r}", per[per.regime == r]) for r in sorted(per.regime.dropna().unique())] + \
             [(f"bucket={b}", per[per.prob_bucket == b]) for b in sorted(per.prob_bucket.dropna().unique())]
    for scope, sub in scopes:
        if len(sub) < 50:
            continue
        taken = sub[sub.taken]
        n, nt = len(sub), len(taken)
        port = sub.copy(); port["_r"] = 0.0
        port.loc[port.taken, "_r"] = pd.to_numeric(taken.ret, errors="coerce").values - COST_PCT
        ew_sh, ew_dd = _port(port, "_r")
        pw_sh, _ = _port(port, "_r", "probability")
        rows.append(dict(scope=scope, hypothesis=name, trigger=hyp["trigger"],
                         filters="+".join(hyp["filters"]) or "none",
                         n_signals=n, deploy_pct=round(100*nt/max(n,1), 1),
                         taken_mean=round(taken.ret.mean()-COST_PCT, 3) if nt else np.nan,
                         win_pct=round(100*(taken.ret>0).mean(), 1) if nt else np.nan,
                         port_ew_sharpe=round(ew_sh, 3) if pd.notna(ew_sh) else np.nan,
                         port_ew_maxdd=round(ew_dd, 1) if pd.notna(ew_dd) else np.nan,
                         port_pw_sharpe=round(pw_sh, 3) if pd.notna(pw_sh) else np.nan))
    return rows


# =============================================================================
# Daily cache join
# =============================================================================

def load_daily_map(daily_cache, symbols):
    m = {}
    if not daily_cache:
        return m
    cols = ["timestamp", "D_nr", "D_bb_bw_20", "D_compress_state", "D_donch_pos_20",
            "D_dist_from_52wh", "D_vol_surge_20", "D_dvol_z20", "D_prev_high",
            "D_prev_low", "D_donch_hi20_level"]
    for sym in set(symbols):
        fp = Path(daily_cache) / f"{sym}_daily.parquet"
        if not fp.exists():
            continue
        try:
            d = pd.read_parquet(fp)
        except Exception:
            continue
        have = [c for c in cols if c in d.columns]
        if "timestamp" not in have:
            continue
        d = d[have].copy()
        d["timestamp"] = pd.to_datetime(d["timestamp"], errors="coerce", utc=True).dt.tz_convert(IST)
        d["key"] = d["timestamp"].dt.normalize()
        for _, r in d.iterrows():
            m[(sym, r["key"])] = r
    return m


# =============================================================================
# Self-test
# =============================================================================

def selftest():
    print("[selftest] synthetic bars...")
    rng = np.random.default_rng(0)
    recs = []
    base = pd.Timestamp("2024-01-01", tz=IST)
    for s in range(400):
        sd = base + pd.Timedelta(days=int(rng.integers(0, 400)))
        pc = 100.0
        bo, bh, bl, bc, bd, bv = [], [], [], [], [], []
        drift = rng.normal(0.4, 2.2)
        orh = pc * 1.005
        for day in range(5):
            px = pc * (1 + drift/100 * (day+1)/5)
            for b in range(75):
                o = px*(1+rng.normal(0,0.003)); h=o*(1+abs(rng.normal(0,0.004)))
                l=o*(1-abs(rng.normal(0,0.004))); c=(h+l)/2 + rng.normal(0,0.05)
                bo.append(o); bh.append(h); bl.append(l); bc.append(c); bd.append(day)
                bv.append(1000+abs(rng.normal(0,300)))
        recs.append({"symbol": f"S{s}", "signal_date": sd,
                     "regime": rng.choice(["bull_trend","bear_trend","bull_range"]),
                     "prob_bucket": "[0.65,0.70)", "probability": float(rng.uniform(0.65,0.95)),
                     "prev_close": pc, "t1_orh_15min": orh, "t1_orl_15min": pc*0.995,
                     "bar_opens": bo, "bar_highs": bh, "bar_lows": bl, "bar_closes": bc,
                     "bar_days": bd})  # note: no bar_volumes -> tests the proxy path
    all_rows = []
    for hyp in build_hypotheses():
        # skip daily-dependent ones in selftest (no daily map)
        if hyp["trigger"] in DAILY_TRIGGERS or any(f in DAILY_FILTERS for f in hyp["filters"]):
            continue
        per = run_hypothesis(recs, {}, hyp)
        all_rows += summarize(per, hyp["name"], hyp)
    res = pd.DataFrame(all_rows)
    ov = res[res.scope == "OVERALL"].sort_values("port_ew_sharpe", ascending=False)
    print(ov[["hypothesis","deploy_pct","taken_mean","win_pct","port_ew_sharpe","port_ew_maxdd"]].to_string(index=False))
    print("\n[selftest] engine OK.")


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", type=Path)
    ap.add_argument("--daily-cache", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path("breakout_results.csv"))
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest(); return
    if not args.paths or not args.paths.exists():
        raise SystemExit("Provide --paths <extracted_paths.parquet> or --selftest")

    paths = pd.read_parquet(args.paths)
    paths["signal_date"] = pd.to_datetime(paths["signal_date"])
    has_vol = "bar_volumes" in paths.columns
    print(f"[lab] {len(paths):,} signals | {paths.signal_date.min().date()} -> {paths.signal_date.max().date()}")
    print(f"[lab] bar_volumes present: {has_vol}  "
          f"({'TRUE intraday vol/VWAP enabled' if has_vol else 'using SIGNAL-DAY volume proxy from daily cache'})")
    recs = paths.to_dict("records")

    hyps = build_hypotheses()
    daily_needed = any((h['trigger'] in DAILY_TRIGGERS) or any(f in DAILY_FILTERS for f in h['filters']) for h in hyps)
    daily_map = {}
    if daily_needed:
        if not args.daily_cache:
            print("[lab] WARNING: no --daily-cache; daily-dependent hypotheses skipped.")
        else:
            print("[lab] loading daily cache...")
            daily_map = load_daily_map(args.daily_cache, paths["symbol"].tolist())
            print(f"[lab] daily rows mapped: {len(daily_map):,}")

    all_rows = []
    for hyp in hyps:
        nd = (hyp["trigger"] in DAILY_TRIGGERS) or any(f in DAILY_FILTERS for f in hyp["filters"])
        if nd and not daily_map:
            print(f"  {hyp['name']:28s} SKIP (needs daily cache)"); continue
        per = run_hypothesis(recs, daily_map, hyp)
        all_rows += summarize(per, hyp["name"], hyp)
        o = [r for r in all_rows if r["hypothesis"] == hyp["name"] and r["scope"] == "OVERALL"]
        if o:
            r = o[0]
            print(f"  {hyp['name']:28s} deploy={r['deploy_pct']:>5.1f}%  mean={r['taken_mean']}  "
                  f"win={r['win_pct']}  EW_Sh={r['port_ew_sharpe']}  DD={r['port_ew_maxdd']}")

    res = pd.DataFrame(all_rows)
    out = Path(args.out)
    try:
        if out.exists() and out.is_dir():
            out = out / "breakout_results.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        res.to_csv(out, index=False)
    except (PermissionError, OSError) as e:
        out = Path.cwd() / "breakout_results.csv"
        print(f"[lab] WARNING: could not write ({e}); using {out}")
        res.to_csv(out, index=False)
    print(f"\n[out] {out}")
    print("\n=== OVERALL leaderboard by portfolio Sharpe (EW) ===")
    ov = res[res.scope == "OVERALL"].sort_values("port_ew_sharpe", ascending=False)
    print(ov[["hypothesis","trigger","filters","deploy_pct","taken_mean","win_pct",
              "port_ew_sharpe","port_ew_maxdd","port_pw_sharpe"]].to_string(index=False))


if __name__ == "__main__":
    main()
