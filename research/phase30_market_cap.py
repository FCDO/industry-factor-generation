"""
Phase 30: D000 下游 ratio rank + 市值預過濾 + 純市值對照組

Phase 29 結論
- ratio N=20 timed: α=13.36%, t=4.41, Sharpe 1.63 (微幅擊敗 Full EW timed 11.81/3.84/1.62)
- 但 top picks 含 8067 志旭 (市值 NaN)、3224 三顧 (29 億) 等小型股, 可能 idiosyncratic noise 大

Phase 30 設計
- A. MC filter → ratio: 用市值前 X% 過濾 universe, 然後 ratio top-N
- B. Pure MC top-N: 純市值排序前 N (無 ratio, 對照組)
- C. Phase 29 baseline (重現對比)

變體
- MC filter X% ∈ {30%, 50%, 70%, no_filter}
- N ∈ {10, 15, 20}
- always-on + timed

No-peek
- market_value .shift(1) 一日 buffer (yesterday's close MC)
- ratio matrix 用過去 5y rolling 同 Phase 29

注意
- market_value 從 2013-04 開始, 前期 NaN → 不過濾 (degraded to Phase 29 behavior)
- eval 仍從 2012-05 開始, 但 2013-04 之前 MC filter 不生效

輸出
- phase30_mc_summary.csv     : 各變體 alpha/Sharpe/turnover
- phase30_strategy_returns.csv
- fig_phase30_compare.png    : MC filter sweep
"""
from __future__ import annotations

import os
import pickle
import sys
import time
import warnings
from collections import Counter

import numpy as np
import pandas as pd
import statsmodels.api as sm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
for f in ["Microsoft JhengHei", "Microsoft YaHei", "Noto Sans CJK TC", "SimHei"]:
    if any(f in fn.name for fn in font_manager.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [f]; break
plt.rcParams["axes.unicode_minus"] = False

warnings.filterwarnings("ignore")

ROOT = r"C:\Users\engli\OneDrive\桌面\產業因子生成"
DB = r"C:\Users\engli\finlab_db"
OUT = os.path.join(ROOT, "research", "output")

LOOKBACK = 1250
NW_LAG = 5
COST_RT = 0.0050
SMOOTH_N = 10
MIN_OBS_IN_WIN = 800

DOWNSTREAM_SUBS = ["D600", "D700", "D800", "D900", "DA00", "DB00"]
N_LIST = [10, 15, 20]
MC_TOP_PCT = [None, 0.7, 0.5, 0.3]  # None = 無過濾


def load_daily_ret():
    with open(os.path.join(DB, "etl#adj_close.pickle"), "rb") as f:
        adj = pickle.load(f).set_index("date")
    adj.index = pd.to_datetime(adj.index)
    adj.columns = adj.columns.astype(str)
    keep = [c for c in adj.columns if c.isdigit() and len(c) == 4 and not c.startswith("00")]
    return adj[keep].pct_change(fill_method=None).iloc[1:]


def load_market_cap(daily_index):
    with open(os.path.join(DB, "etl#market_value.pickle"), "rb") as f:
        mv = pickle.load(f).set_index("date")
    mv.index = pd.to_datetime(mv.index)
    mv.columns = mv.columns.astype(str)
    # reindex to daily, ffill (在 missing 日子用前一日)
    mv_daily = mv.reindex(daily_index, method="ffill")
    # 1 day safety shift (用昨日市值)
    mv_daily = mv_daily.shift(1)
    return mv_daily


def screen_joint(r_i_win, lead_win, lead_lag_win, market_lag_win):
    y_lag = r_i_win.shift(1)
    df = pd.concat({"y": r_i_win, "x_now": lead_win, "x_lag": lead_lag_win,
                     "y_lag": y_lag, "m_lag": market_lag_win}, axis=1).dropna()
    if len(df) < MIN_OBS_IN_WIN:
        return None, None, None
    try:
        res = sm.OLS(df["y"],
                     sm.add_constant(df[["x_now", "x_lag", "y_lag", "m_lag"]])).fit(
            cov_type="HAC", cov_kwds={"maxlags": NW_LAG}
        )
        return float(res.tvalues["x_lag"]), float(res.params["x_lag"]), float(res.params["x_now"])
    except Exception:
        return None, None, None


def perf_with_alpha(daily_ret, market, ann=252):
    df = pd.concat([daily_ret.rename("s"), market.rename("b")], axis=1).dropna()
    if len(df) < 60:
        return {}
    res = sm.OLS(df["s"], sm.add_constant(df["b"])).fit(cov_type="HAC", cov_kwds={"maxlags": 60})
    s = df["s"]; cum = (1 + s).cumprod()
    return {
        "n": len(s),
        "ann_ret": s.mean() * ann,
        "ann_vol": s.std() * np.sqrt(ann),
        "sharpe": s.mean() / s.std() * np.sqrt(ann) if s.std() > 0 else np.nan,
        "max_dd": (cum / cum.cummax() - 1).min(),
        "ann_alpha": res.params["const"] * ann,
        "t_alpha": res.tvalues["const"],
        "beta": res.params["b"],
    }


def perf_from_weights(W, daily_ret_uni, cost_rt=COST_RT):
    gross = (W * daily_ret_uni).sum(axis=1)
    turn = W.diff().abs().sum(axis=1)
    turn.iloc[0] = W.iloc[0].abs().sum()
    cost = turn * (cost_rt / 2.0)
    net = gross - cost
    ann_turn = turn.mean() * 252
    return gross, net, ann_turn


def main():
    print("=" * 110)
    print("Phase 30: D000 下游 ratio rank + 市值預過濾 + 純市值對照組")
    print("=" * 110)

    chain = pd.read_csv(os.path.join(ROOT, "industry_chain.csv"), dtype=str)
    daily_ret = load_daily_ret()
    market = daily_ret.mean(axis=1)
    market_lag = market.shift(1)
    mv_daily = load_market_cap(daily_ret.index)
    print(f"\nMarket cap data: {mv_daily.shape}, valid from {mv_daily.dropna(how='all').index.min().date()}")

    # Lead
    d100 = chain[(chain["industry_code"] == "D000") & (chain["sub_code"] == "D100")]["stock_id"].unique().tolist()
    dc00 = chain[(chain["industry_code"] == "D000") & (chain["sub_code"] == "DC00")]["stock_id"].unique().tolist()
    d100_in = [s for s in d100 if s in daily_ret.columns]
    dc00_in = [s for s in dc00 if s in daily_ret.columns]
    lead = 0.5 * daily_ret[d100_in].mean(axis=1) + 0.5 * daily_ret[dc00_in].mean(axis=1)
    lead_lag = lead.shift(1)

    # Universe: D000 下游 6 sub_code unique
    d_down = chain[(chain["industry_code"] == "D000") &
                    (chain["sub_code"].isin(DOWNSTREAM_SUBS))][["stock_id", "name", "sub_code"]]
    d_down = d_down.drop_duplicates("stock_id", keep="first").reset_index(drop=True)
    universe = sorted([s for s in d_down["stock_id"] if s in daily_ret.columns])
    rets_uni = daily_ret[universe]
    name_map = dict(zip(d_down["stock_id"], d_down["name"]))
    sub_map = dict(zip(d_down["stock_id"], d_down["sub_code"]))
    print(f"\nUniverse: {len(universe)} 檔 (D000 下游)")

    first_valid = daily_ret.index[LOOKBACK]
    month_ends = pd.Series(daily_ret.index).groupby([daily_ret.index.year, daily_ret.index.month]).max()
    rebal_dates = pd.DatetimeIndex(month_ends.values)
    rebal_dates = rebal_dates[rebal_dates >= first_valid]
    rebal_list = sorted(rebal_dates)
    print(f"Rebal dates: {len(rebal_list)} ({rebal_list[0].date()} → {rebal_list[-1].date()})")

    # === Compute ratio matrix (重用 Phase 29 邏輯) ===
    print(f"\n[1] 計算 ratio matrix")
    t0 = time.time()
    b_lag_m = pd.DataFrame(np.nan, index=rebal_dates, columns=universe)
    b_now_m = pd.DataFrame(np.nan, index=rebal_dates, columns=universe)

    for j, d in enumerate(rebal_dates):
        win_start_idx = max(0, daily_ret.index.get_loc(d) - LOOKBACK)
        win_start = daily_ret.index[win_start_idx]
        lead_win = lead.loc[win_start:d]
        lead_lag_win = lead_lag.loc[win_start:d]
        market_lag_win = market_lag.loc[win_start:d]
        for sid in universe:
            r_i_win = rets_uni[sid].loc[win_start:d]
            t_lag, b_lag, b_now = screen_joint(r_i_win, lead_win, lead_lag_win, market_lag_win)
            if t_lag is not None:
                b_lag_m.loc[d, sid] = b_lag
                b_now_m.loc[d, sid] = b_now
        if (j + 1) % 24 == 0 or j == len(rebal_dates) - 1:
            print(f"  [{j+1:>3d}/{len(rebal_list)}] {d.date()}  ({time.time()-t0:.0f}s)")

    ratio_m = b_lag_m / b_now_m.where(b_now_m > 0.05)

    # === Diagnostic: 最後一期 universe 的市值分布 + ratio top-15 是否 survive ===
    last_d = rebal_list[-1]
    print(f"\n[Diagnostic] {last_d.date()} ratio top 15 + 市值狀況:")
    s = ratio_m.loc[last_d].dropna()
    s = s[s > 0].sort_values(ascending=False).head(15)
    mv_today = mv_daily.loc[last_d]
    # 計算 universe 內的 mc rank (percentile)
    mv_uni_today = mv_today.loc[[c for c in universe if c in mv_today.index]].dropna()
    mv_pct = mv_uni_today.rank(pct=True)  # 0-1, 1 = largest
    print(f"  universe 中有市值資料: {len(mv_uni_today)}/{len(universe)}")
    print(f"  市值區間: {mv_uni_today.min()/1e8:.1f} 億 ~ {mv_uni_today.max()/1e8:.1f} 億, "
          f"median={mv_uni_today.median()/1e8:.1f} 億")
    print(f"\n{'rank':>4s} {'stock':>5s} {'name':<10s} {'sub':>5s}  {'ratio':>7s}  {'市值(億)':>10s}  {'pct':>6s}  各 MC 門檻")
    for rank_, (sid, score) in enumerate(s.items()):
        mv_v = mv_today.get(sid, np.nan)
        mv_pct_v = mv_pct.get(sid, np.nan) if not pd.isna(mv_v) else np.nan
        passes = []
        for pct_thresh in [0.7, 0.5, 0.3]:  # mc top-30/50/70%
            # mc rank >= (1 - pct_thresh) means in top pct_thresh
            mc_cutoff = 1 - pct_thresh
            if pd.notna(mv_pct_v) and mv_pct_v >= mc_cutoff:
                passes.append(f"top{int(pct_thresh*100)}")
            else:
                passes.append("---")
        passes_str = "  ".join(f"{p:>5s}" for p in passes)
        mv_str = f"{mv_v/1e8:>8.1f}" if pd.notna(mv_v) else "    NaN"
        pct_str = f"{mv_pct_v:>5.2f}" if pd.notna(mv_pct_v) else " NaN "
        print(f"{rank_+1:>4d} {sid:>5s} {name_map.get(sid, ''):<10s} {sub_map.get(sid,''):>5s}  "
              f"{score:>+7.3f}  {mv_str}  {pct_str}  {passes_str}")

    # === Build portfolio for each (mc_filter, N) combination ===
    print(f"\n[2] 構 portfolios")
    weight_dict = {}
    for mc_pct in MC_TOP_PCT:
        for N in N_LIST:
            W = pd.DataFrame(0.0, index=daily_ret.index, columns=universe)
            for i, d in enumerate(rebal_list):
                scores = ratio_m.loc[d].dropna()
                scores = scores[scores > 0]
                if len(scores) == 0:
                    continue
                # MC pre-filter
                if mc_pct is not None:
                    mv_today = mv_daily.loc[d] if d in mv_daily.index else None
                    if mv_today is None or mv_today.dropna().empty:
                        # MC data not available — fallback: no filter (early period 2012-2013)
                        valid_uni = scores.index.tolist()
                    else:
                        # 在 universe 內計算 MC rank
                        mv_uni = mv_today.loc[[c for c in scores.index if c in mv_today.index]].dropna()
                        if len(mv_uni) == 0:
                            valid_uni = scores.index.tolist()
                        else:
                            cutoff = mv_uni.quantile(1 - mc_pct)
                            valid_uni = mv_uni[mv_uni >= cutoff].index.tolist()
                    scores = scores.loc[[s for s in valid_uni if s in scores.index]]

                if len(scores) == 0:
                    continue
                top = scores.sort_values(ascending=False).head(N).index.tolist()
                next_rebal = rebal_list[i + 1] if i + 1 < len(rebal_list) else daily_ret.index.max() + pd.Timedelta(days=1)
                hold_idx = daily_ret.index[(daily_ret.index > d) & (daily_ret.index <= next_rebal)]
                if len(top) > 0:
                    W.loc[hold_idx, top] = 1.0 / len(top)
            weight_dict[("ratio", mc_pct, N)] = W
            n_avg = (W > 0).sum(axis=1).where(lambda x: x > 0).mean()
            tag = "no_filter" if mc_pct is None else f"top{int(mc_pct*100)}%"
            print(f"  ratio + MC {tag}, N={N}: avg active = {n_avg:.1f}")

    # === Pure MC top-N (variant B: 對照組, 無 ratio) ===
    print(f"\n  Pure MC top-N (對照組, 無 ratio):")
    for N in N_LIST:
        W = pd.DataFrame(0.0, index=daily_ret.index, columns=universe)
        for i, d in enumerate(rebal_list):
            mv_today = mv_daily.loc[d] if d in mv_daily.index else None
            if mv_today is None or mv_today.dropna().empty:
                continue
            mv_uni = mv_today.loc[[c for c in universe if c in mv_today.index]].dropna()
            top = mv_uni.sort_values(ascending=False).head(N).index.tolist()
            next_rebal = rebal_list[i + 1] if i + 1 < len(rebal_list) else daily_ret.index.max() + pd.Timedelta(days=1)
            hold_idx = daily_ret.index[(daily_ret.index > d) & (daily_ret.index <= next_rebal)]
            if len(top) > 0:
                W.loc[hold_idx, top] = 1.0 / len(top)
        weight_dict[("pure_mc", None, N)] = W
        n_avg = (W > 0).sum(axis=1).where(lambda x: x > 0).mean()
        print(f"    pure MC top-{N}: avg active = {n_avg:.1f}")

    # === Full EW baseline ===
    W_full = pd.DataFrame(0.0, index=daily_ret.index, columns=universe)
    for d in daily_ret.index[LOOKBACK:]:
        avail = rets_uni.loc[d].dropna().index.tolist()
        if avail:
            W_full.loc[d, avail] = 1.0 / len(avail)

    timing = (lead.ewm(span=SMOOTH_N, adjust=False).mean() > 0).astype(float).shift(1).fillna(0)

    eval_start = rebal_list[0] + pd.Timedelta(days=1)
    print(f"\n[3] 計算 returns (eval start = {eval_start.date()})")

    g_full, n_full, at_full = perf_from_weights(W_full, rets_uni)
    W_full_t = W_full.mul(timing, axis=0).fillna(0)
    g_full_t, n_full_t, at_full_t = perf_from_weights(W_full_t, rets_uni)
    p_full = perf_with_alpha(n_full.loc[eval_start:], market.loc[eval_start:])
    p_full_t = perf_with_alpha(n_full_t.loc[eval_start:], market.loc[eval_start:])

    rows = [
        {"variant": "Full_EW", "mc_filter": "—", "N": "all", "timed": False, **p_full, "ann_turn": at_full},
        {"variant": "Full_EW", "mc_filter": "—", "N": "all", "timed": True, **p_full_t, "ann_turn": at_full_t},
    ]
    daily_returns_dict = {"Full_EW": n_full, "Full_EW_timed": n_full_t}

    for (var_type, mc_pct, N), W in weight_dict.items():
        g, n, at = perf_from_weights(W, rets_uni)
        W_t = W.mul(timing, axis=0).fillna(0)
        g_t, n_t, at_t = perf_from_weights(W_t, rets_uni)
        mc_tag = "no_filter" if mc_pct is None else f"top{int(mc_pct*100)}%"
        key = f"{var_type}_{mc_tag}_N{N}"
        daily_returns_dict[f"{key}"] = n
        daily_returns_dict[f"{key}_timed"] = n_t
        p = perf_with_alpha(n.loc[eval_start:], market.loc[eval_start:])
        p_t = perf_with_alpha(n_t.loc[eval_start:], market.loc[eval_start:])
        rows.append({"variant": var_type, "mc_filter": mc_tag, "N": N, "timed": False, **p, "ann_turn": at})
        rows.append({"variant": var_type, "mc_filter": mc_tag, "N": N, "timed": True, **p_t, "ann_turn": at_t})

    res_df = pd.DataFrame(rows)
    res_df.to_csv(os.path.join(OUT, "phase30_mc_summary.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(daily_returns_dict).to_csv(os.path.join(OUT, "phase30_strategy_returns.csv"),
                                              encoding="utf-8-sig")

    # === Print ===
    print(f"\n" + "=" * 110)
    print(f"策略表現 (扣 50bp RT, eval start = {eval_start.date()})")
    print("=" * 110)
    base_a = res_df[(res_df["variant"] == "Full_EW") & (~res_df["timed"])].iloc[0]
    base_b = res_df[(res_df["variant"] == "Full_EW") & (res_df["timed"])].iloc[0]
    p29_a = res_df[(res_df["variant"] == "ratio") & (res_df["mc_filter"] == "no_filter") & (res_df["N"] == 20) & (~res_df["timed"])].iloc[0]
    p29_b = res_df[(res_df["variant"] == "ratio") & (res_df["mc_filter"] == "no_filter") & (res_df["N"] == 20) & (res_df["timed"])].iloc[0]
    print(f"\nbaselines:")
    print(f"  Full EW always-on:        α={base_a['ann_alpha']:+.4f}, t={base_a['t_alpha']:+.3f}, Sh={base_a['sharpe']:+.3f}")
    print(f"  Full EW timed:            α={base_b['ann_alpha']:+.4f}, t={base_b['t_alpha']:+.3f}, Sh={base_b['sharpe']:+.3f}")
    print(f"  Phase 29 ratio N=20 a-on: α={p29_a['ann_alpha']:+.4f}, t={p29_a['t_alpha']:+.3f}, Sh={p29_a['sharpe']:+.3f}")
    print(f"  Phase 29 ratio N=20 timed:α={p29_b['ann_alpha']:+.4f}, t={p29_b['t_alpha']:+.3f}, Sh={p29_b['sharpe']:+.3f}")

    print(f"\n[A. ratio + MC filter, Always-on] (vs Full EW α={base_a['ann_alpha']:+.4f})")
    print(f"  {'mc_filter':>10s}  {'N':>3s}  {'α':>9s}  {'t_α':>8s}  {'Sh':>7s}  {'turn':>6s}  Δα/Δt vs Full")
    for _, r in res_df[(res_df["variant"] == "ratio") & (~res_df["timed"])].iterrows():
        d_a = (r["ann_alpha"] - base_a["ann_alpha"]) * 100
        d_t = r["t_alpha"] - base_a["t_alpha"]
        d_sh = r["sharpe"] - base_a["sharpe"]
        flag = "✓✓" if d_a > 3 and d_t > 0.5 else ("✓" if d_a > 1 and d_t > 0.2 else ("↑" if d_a > 0 else "↓"))
        print(f"  {r['mc_filter']:>10s}  {r['N']:>3d}  {r['ann_alpha']:>+9.4f}  {r['t_alpha']:>+8.3f}  "
              f"{r['sharpe']:>+7.3f}  {r['ann_turn']:>6.2f}  {d_a:+.2f}pp / {d_t:+.2f} (ΔSh={d_sh:+.3f}) {flag}")

    print(f"\n[A. ratio + MC filter, Timed] (vs Full timed α={base_b['ann_alpha']:+.4f})")
    print(f"  {'mc_filter':>10s}  {'N':>3s}  {'α':>9s}  {'t_α':>8s}  {'Sh':>7s}  {'turn':>6s}  Δα/Δt vs Full timed")
    for _, r in res_df[(res_df["variant"] == "ratio") & (res_df["timed"])].iterrows():
        d_a = (r["ann_alpha"] - base_b["ann_alpha"]) * 100
        d_t = r["t_alpha"] - base_b["t_alpha"]
        d_sh = r["sharpe"] - base_b["sharpe"]
        flag = "✓✓" if d_a > 3 and d_t > 0.5 else ("✓" if d_a > 1 and d_t > 0.2 else ("↑" if d_a > 0 else "↓"))
        print(f"  {r['mc_filter']:>10s}  {r['N']:>3d}  {r['ann_alpha']:>+9.4f}  {r['t_alpha']:>+8.3f}  "
              f"{r['sharpe']:>+7.3f}  {r['ann_turn']:>6.2f}  {d_a:+.2f}pp / {d_t:+.2f} (ΔSh={d_sh:+.3f}) {flag}")

    print(f"\n[B. Pure MC top-N (對照組, 無 ratio), Always-on/Timed]")
    print(f"  {'N':>3s}  {'timed':>6s}  {'α':>9s}  {'t_α':>8s}  {'Sh':>7s}  {'turn':>6s}")
    for _, r in res_df[(res_df["variant"] == "pure_mc")].iterrows():
        print(f"  {r['N']:>3d}  {'Y' if r['timed'] else 'N':>6s}  "
              f"{r['ann_alpha']:>+9.4f}  {r['t_alpha']:>+8.3f}  "
              f"{r['sharpe']:>+7.3f}  {r['ann_turn']:>6.2f}")

    # === MC filter 對 ratio 的邊際影響 ===
    print(f"\n" + "=" * 110)
    print("MC filter 對 ratio rank 的邊際影響 (timed only, vs no_filter same N)")
    print("=" * 110)
    print(f"  {'N':>3s}  {'mc filter':>10s}  {'α':>9s}  {'Sh':>6s}  vs no_filter:  Δα      ΔSh")
    for N in N_LIST:
        nf = res_df[(res_df["variant"] == "ratio") & (res_df["mc_filter"] == "no_filter") &
                     (res_df["N"] == N) & (res_df["timed"])].iloc[0]
        for mc_tag in ["top70%", "top50%", "top30%"]:
            r = res_df[(res_df["variant"] == "ratio") & (res_df["mc_filter"] == mc_tag) &
                       (res_df["N"] == N) & (res_df["timed"])].iloc[0]
            d_a = (r["ann_alpha"] - nf["ann_alpha"]) * 100
            d_sh = r["sharpe"] - nf["sharpe"]
            flag = "✓" if d_a > 0.5 and d_sh > 0 else ("↑" if d_a > 0 else "↓")
            print(f"  {N:>3d}  {mc_tag:>10s}  {r['ann_alpha']:>+9.4f}  {r['sharpe']:>+6.3f}  "
                  f"  Δα={d_a:>+5.2f}pp  ΔSh={d_sh:>+6.3f} {flag}")

    # === Best ===
    print(f"\n" + "=" * 110)
    print("Best (sorted by t_α, timed)")
    print("=" * 110)
    timed = res_df[(res_df["variant"] != "Full_EW") & (res_df["timed"])].sort_values("t_alpha", ascending=False).head(8)
    for _, r in timed.iterrows():
        d_t = r["t_alpha"] - base_b["t_alpha"]
        d_sh = r["sharpe"] - base_b["sharpe"]
        flag = "✓✓" if d_t > 0.3 and d_sh > 0.02 else "↑" if d_t > 0 else "↓"
        print(f"  {r['variant']:>8s}, MC={r['mc_filter']:>10s}, N={r['N']:>3d}: α={r['ann_alpha']:+.4f}, "
              f"t={r['t_alpha']:+.3f}, Sh={r['sharpe']:+.3f}  Δt={d_t:+.2f}, ΔSh={d_sh:+.3f} {flag}")

    # === Plot ===
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    Ns = N_LIST
    cols = {"no_filter": "#0277bd", "top70%": "#388e3c", "top50%": "#f57c00", "top30%": "#c62828"}
    for timed_idx, ax in enumerate(axes):
        timed_v = (timed_idx == 1)
        for mc_tag in ["no_filter", "top70%", "top50%", "top30%"]:
            ts = []
            for N in Ns:
                r = res_df[(res_df["variant"] == "ratio") & (res_df["mc_filter"] == mc_tag) &
                           (res_df["N"] == N) & (res_df["timed"] == timed_v)].iloc[0]
                ts.append(r["t_alpha"])
            ax.plot(Ns, ts, "o-", color=cols[mc_tag], lw=2, ms=8, label=f"MC {mc_tag}")
        baseline = base_b["t_alpha"] if timed_v else base_a["t_alpha"]
        ax.axhline(baseline, color="black", ls="--", alpha=0.5, label=f"Full EW t={baseline:.2f}")
        ax.axhline(1.96, color="gray", ls=":", alpha=0.4)
        ax.set_xlabel("Top-N (by ratio)")
        ax.set_ylabel("t_α")
        ax.set_title(f"({'A' if not timed_v else 'B'}) {'Always-on' if not timed_v else 'Timed'}: ratio + MC filter")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
    plt.suptitle("Phase 30: D000 下游 ratio rank + 市值預過濾 (X=top%)", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig_phase30_compare.png"), dpi=130)
    plt.close()
    print(f"\n圖檔: fig_phase30_compare.png")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
