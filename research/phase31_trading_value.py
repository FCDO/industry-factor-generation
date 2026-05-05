"""
Phase 31: D000 下游 ratio rank + 3M 成交金額預過濾 + 純成交金額對照組

Phase 30 結論
- 市值 (market cap) 過濾全面傷害 (top70/50/30% 全部 -2~-5pp α)
- pure MC top-N 對照組表現平庸 (Sharpe 0.78-0.92)
- 確認 alpha 來源是 ratio 訊號, 不是大市值偏好

Phase 31 改用「3M 成交金額」(更貼近實際流動性)
- 動機: 8067 志旭日均成交金額 0.001 億 (10 萬!) 是真正的流動性陷阱
        1742 台蠟也是 0.001 億, 雖然有市值但根本沒人買賣
- 假設: 成交金額過濾比市值過濾更精準, 可能不會傷害 ratio 訊號
- 但如果結果類似 Phase 30 (中小型股是 ratio 主源), 仍可能傷害

設計
- A. ratio + 3M-avg 成交金額 top X% filter → ratio top-N
- B. Pure 3M-avg 成交金額 top-N (對照組)
- C. Phase 29 baseline (no filter, 重現對比)

3M = 60 trading days, .shift(1) 1 day buffer

輸出
- phase31_tv_summary.csv     : 各變體 alpha/Sharpe/turnover
- phase31_strategy_returns.csv
- fig_phase31_compare.png
"""
from __future__ import annotations

import os
import pickle
import sys
import time
import warnings

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
TV_ROLL = 60   # 3M ≈ 60 trading days

DOWNSTREAM_SUBS = ["D600", "D700", "D800", "D900", "DA00", "DB00"]
N_LIST = [10, 15, 20]
TV_TOP_PCT = [None, 0.7, 0.5, 0.3]


def load_daily_ret():
    with open(os.path.join(DB, "etl#adj_close.pickle"), "rb") as f:
        adj = pickle.load(f).set_index("date")
    adj.index = pd.to_datetime(adj.index)
    adj.columns = adj.columns.astype(str)
    keep = [c for c in adj.columns if c.isdigit() and len(c) == 4 and not c.startswith("00")]
    return adj[keep].pct_change(fill_method=None).iloc[1:]


def load_trading_value(daily_index):
    """3M-avg trading value, daily aligned, 1d shift"""
    with open(os.path.join(DB, "price#成交金額.pickle"), "rb") as f:
        tv = pickle.load(f).set_index("date")
    tv.index = pd.to_datetime(tv.index)
    tv.columns = tv.columns.astype(str)
    # rolling 60d 平均
    tv_avg = tv.rolling(TV_ROLL, min_periods=20).mean()
    # reindex daily, ffill, 1d safety shift
    tv_avg_daily = tv_avg.reindex(daily_index, method="ffill").shift(1)
    return tv_avg_daily


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
    print("Phase 31: D000 下游 ratio rank + 3M 成交金額預過濾 + 純成交金額對照組")
    print("=" * 110)

    chain = pd.read_csv(os.path.join(ROOT, "industry_chain.csv"), dtype=str)
    daily_ret = load_daily_ret()
    market = daily_ret.mean(axis=1)
    market_lag = market.shift(1)
    tv_avg_daily = load_trading_value(daily_ret.index)
    print(f"\n3M-avg trading value: {tv_avg_daily.shape}")

    d100 = chain[(chain["industry_code"] == "D000") & (chain["sub_code"] == "D100")]["stock_id"].unique().tolist()
    dc00 = chain[(chain["industry_code"] == "D000") & (chain["sub_code"] == "DC00")]["stock_id"].unique().tolist()
    d100_in = [s for s in d100 if s in daily_ret.columns]
    dc00_in = [s for s in dc00 if s in daily_ret.columns]
    lead = 0.5 * daily_ret[d100_in].mean(axis=1) + 0.5 * daily_ret[dc00_in].mean(axis=1)
    lead_lag = lead.shift(1)

    d_down = chain[(chain["industry_code"] == "D000") &
                    (chain["sub_code"].isin(DOWNSTREAM_SUBS))][["stock_id", "name", "sub_code"]]
    d_down = d_down.drop_duplicates("stock_id", keep="first").reset_index(drop=True)
    universe = sorted([s for s in d_down["stock_id"] if s in daily_ret.columns])
    rets_uni = daily_ret[universe]
    name_map = dict(zip(d_down["stock_id"], d_down["name"]))
    sub_map = dict(zip(d_down["stock_id"], d_down["sub_code"]))
    print(f"Universe: {len(universe)} 檔 (D000 下游)")

    first_valid = daily_ret.index[LOOKBACK]
    month_ends = pd.Series(daily_ret.index).groupby([daily_ret.index.year, daily_ret.index.month]).max()
    rebal_dates = pd.DatetimeIndex(month_ends.values)
    rebal_dates = rebal_dates[rebal_dates >= first_valid]
    rebal_list = sorted(rebal_dates)

    # === Compute ratio matrix ===
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

    # === Diagnostic: ratio top 15 + trading value 狀況 ===
    last_d = rebal_list[-1]
    print(f"\n[Diagnostic] {last_d.date()} ratio top 15 + 3M-avg 成交金額:")
    s = ratio_m.loc[last_d].dropna()
    s = s[s > 0].sort_values(ascending=False).head(15)
    tv_today = tv_avg_daily.loc[last_d]
    tv_uni_today = tv_today.loc[[c for c in universe if c in tv_today.index]].dropna()
    tv_pct = tv_uni_today.rank(pct=True)
    print(f"  universe 中有成交金額資料: {len(tv_uni_today)}/{len(universe)}")
    print(f"  3M-avg 成交金額: {tv_uni_today.min()/1e6:.2f} 百萬 ~ {tv_uni_today.max()/1e8:.1f} 億, "
          f"median={tv_uni_today.median()/1e8:.2f} 億")

    print(f"\n{'rank':>4s} {'stock':>5s} {'name':<10s} {'sub':>5s}  {'ratio':>7s}  {'TV(億/日)':>10s}  {'pct':>6s}  各 TV 門檻")
    for rank_, (sid, score) in enumerate(s.items()):
        tv_v = tv_today.get(sid, np.nan)
        tv_pct_v = tv_pct.get(sid, np.nan) if not pd.isna(tv_v) else np.nan
        passes = []
        for pct_thresh in [0.7, 0.5, 0.3]:
            tv_cutoff = 1 - pct_thresh
            if pd.notna(tv_pct_v) and tv_pct_v >= tv_cutoff:
                passes.append(f"top{int(pct_thresh*100)}")
            else:
                passes.append("---")
        passes_str = "  ".join(f"{p:>5s}" for p in passes)
        tv_str = f"{tv_v/1e8:>8.3f}" if pd.notna(tv_v) else "    NaN"
        pct_str = f"{tv_pct_v:>5.2f}" if pd.notna(tv_pct_v) else " NaN "
        print(f"{rank_+1:>4d} {sid:>5s} {name_map.get(sid, ''):<10s} {sub_map.get(sid,''):>5s}  "
              f"{score:>+7.3f}  {tv_str}  {pct_str}  {passes_str}")

    # === Build portfolios ===
    print(f"\n[2] 構 portfolios (ratio + TV filter)")
    weight_dict = {}
    for tv_pct in TV_TOP_PCT:
        for N in N_LIST:
            W = pd.DataFrame(0.0, index=daily_ret.index, columns=universe)
            for i, d in enumerate(rebal_list):
                scores = ratio_m.loc[d].dropna()
                scores = scores[scores > 0]
                if len(scores) == 0:
                    continue
                if tv_pct is not None:
                    tv_today = tv_avg_daily.loc[d] if d in tv_avg_daily.index else None
                    if tv_today is None or tv_today.dropna().empty:
                        valid_uni = scores.index.tolist()
                    else:
                        tv_uni = tv_today.loc[[c for c in scores.index if c in tv_today.index]].dropna()
                        if len(tv_uni) == 0:
                            valid_uni = scores.index.tolist()
                        else:
                            cutoff = tv_uni.quantile(1 - tv_pct)
                            valid_uni = tv_uni[tv_uni >= cutoff].index.tolist()
                    scores = scores.loc[[s for s in valid_uni if s in scores.index]]

                if len(scores) == 0:
                    continue
                top = scores.sort_values(ascending=False).head(N).index.tolist()
                next_rebal = rebal_list[i + 1] if i + 1 < len(rebal_list) else daily_ret.index.max() + pd.Timedelta(days=1)
                hold_idx = daily_ret.index[(daily_ret.index > d) & (daily_ret.index <= next_rebal)]
                if len(top) > 0:
                    W.loc[hold_idx, top] = 1.0 / len(top)
            weight_dict[("ratio", tv_pct, N)] = W
            n_avg = (W > 0).sum(axis=1).where(lambda x: x > 0).mean()
            tag = "no_filter" if tv_pct is None else f"top{int(tv_pct*100)}%"
            print(f"  ratio + TV {tag}, N={N}: avg active = {n_avg:.1f}")

    # Pure TV top-N (對照組)
    print(f"\n  Pure TV top-N (對照組, 無 ratio):")
    for N in N_LIST:
        W = pd.DataFrame(0.0, index=daily_ret.index, columns=universe)
        for i, d in enumerate(rebal_list):
            tv_today = tv_avg_daily.loc[d] if d in tv_avg_daily.index else None
            if tv_today is None or tv_today.dropna().empty:
                continue
            tv_uni = tv_today.loc[[c for c in universe if c in tv_today.index]].dropna()
            top = tv_uni.sort_values(ascending=False).head(N).index.tolist()
            next_rebal = rebal_list[i + 1] if i + 1 < len(rebal_list) else daily_ret.index.max() + pd.Timedelta(days=1)
            hold_idx = daily_ret.index[(daily_ret.index > d) & (daily_ret.index <= next_rebal)]
            if len(top) > 0:
                W.loc[hold_idx, top] = 1.0 / len(top)
        weight_dict[("pure_tv", None, N)] = W
        n_avg = (W > 0).sum(axis=1).where(lambda x: x > 0).mean()
        print(f"    pure TV top-{N}: avg active = {n_avg:.1f}")

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
        {"variant": "Full_EW", "tv_filter": "—", "N": "all", "timed": False, **p_full, "ann_turn": at_full},
        {"variant": "Full_EW", "tv_filter": "—", "N": "all", "timed": True, **p_full_t, "ann_turn": at_full_t},
    ]
    daily_returns_dict = {"Full_EW": n_full, "Full_EW_timed": n_full_t}

    for (var_type, tv_pct, N), W in weight_dict.items():
        g, n, at = perf_from_weights(W, rets_uni)
        W_t = W.mul(timing, axis=0).fillna(0)
        g_t, n_t, at_t = perf_from_weights(W_t, rets_uni)
        tv_tag = "no_filter" if tv_pct is None else f"top{int(tv_pct*100)}%"
        key = f"{var_type}_{tv_tag}_N{N}"
        daily_returns_dict[key] = n
        daily_returns_dict[f"{key}_timed"] = n_t
        p = perf_with_alpha(n.loc[eval_start:], market.loc[eval_start:])
        p_t = perf_with_alpha(n_t.loc[eval_start:], market.loc[eval_start:])
        rows.append({"variant": var_type, "tv_filter": tv_tag, "N": N, "timed": False, **p, "ann_turn": at})
        rows.append({"variant": var_type, "tv_filter": tv_tag, "N": N, "timed": True, **p_t, "ann_turn": at_t})

    res_df = pd.DataFrame(rows)
    res_df.to_csv(os.path.join(OUT, "phase31_tv_summary.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(daily_returns_dict).to_csv(os.path.join(OUT, "phase31_strategy_returns.csv"),
                                              encoding="utf-8-sig")

    # === Print ===
    print(f"\n" + "=" * 110)
    print(f"策略表現 (扣 50bp RT, eval start = {eval_start.date()})")
    print("=" * 110)
    base_a = res_df[(res_df["variant"] == "Full_EW") & (~res_df["timed"])].iloc[0]
    base_b = res_df[(res_df["variant"] == "Full_EW") & (res_df["timed"])].iloc[0]
    p29_a = res_df[(res_df["variant"] == "ratio") & (res_df["tv_filter"] == "no_filter") & (res_df["N"] == 20) & (~res_df["timed"])].iloc[0]
    p29_b = res_df[(res_df["variant"] == "ratio") & (res_df["tv_filter"] == "no_filter") & (res_df["N"] == 20) & (res_df["timed"])].iloc[0]
    print(f"\nbaselines:")
    print(f"  Full EW always-on:        α={base_a['ann_alpha']:+.4f}, t={base_a['t_alpha']:+.3f}, Sh={base_a['sharpe']:+.3f}")
    print(f"  Full EW timed:            α={base_b['ann_alpha']:+.4f}, t={base_b['t_alpha']:+.3f}, Sh={base_b['sharpe']:+.3f}")
    print(f"  Phase 29 ratio N=20 a-on: α={p29_a['ann_alpha']:+.4f}, t={p29_a['t_alpha']:+.3f}, Sh={p29_a['sharpe']:+.3f}")
    print(f"  Phase 29 ratio N=20 timed:α={p29_b['ann_alpha']:+.4f}, t={p29_b['t_alpha']:+.3f}, Sh={p29_b['sharpe']:+.3f}")

    print(f"\n[A. ratio + TV filter, Always-on] (vs Full EW α={base_a['ann_alpha']:+.4f})")
    print(f"  {'tv_filter':>10s}  {'N':>3s}  {'α':>9s}  {'t_α':>8s}  {'Sh':>7s}  {'turn':>6s}  Δα/Δt vs Full")
    for _, r in res_df[(res_df["variant"] == "ratio") & (~res_df["timed"])].iterrows():
        d_a = (r["ann_alpha"] - base_a["ann_alpha"]) * 100
        d_t = r["t_alpha"] - base_a["t_alpha"]
        d_sh = r["sharpe"] - base_a["sharpe"]
        flag = "✓✓" if d_a > 3 and d_t > 0.5 else ("✓" if d_a > 1 and d_t > 0.2 else ("↑" if d_a > 0 else "↓"))
        print(f"  {r['tv_filter']:>10s}  {r['N']:>3d}  {r['ann_alpha']:>+9.4f}  {r['t_alpha']:>+8.3f}  "
              f"{r['sharpe']:>+7.3f}  {r['ann_turn']:>6.2f}  {d_a:+.2f}pp / {d_t:+.2f} (ΔSh={d_sh:+.3f}) {flag}")

    print(f"\n[A. ratio + TV filter, Timed] (vs Full timed α={base_b['ann_alpha']:+.4f})")
    print(f"  {'tv_filter':>10s}  {'N':>3s}  {'α':>9s}  {'t_α':>8s}  {'Sh':>7s}  {'turn':>6s}  Δα/Δt vs Full timed")
    for _, r in res_df[(res_df["variant"] == "ratio") & (res_df["timed"])].iterrows():
        d_a = (r["ann_alpha"] - base_b["ann_alpha"]) * 100
        d_t = r["t_alpha"] - base_b["t_alpha"]
        d_sh = r["sharpe"] - base_b["sharpe"]
        flag = "✓✓" if d_a > 3 and d_t > 0.5 else ("✓" if d_a > 1 and d_t > 0.2 else ("↑" if d_a > 0 else "↓"))
        print(f"  {r['tv_filter']:>10s}  {r['N']:>3d}  {r['ann_alpha']:>+9.4f}  {r['t_alpha']:>+8.3f}  "
              f"{r['sharpe']:>+7.3f}  {r['ann_turn']:>6.2f}  {d_a:+.2f}pp / {d_t:+.2f} (ΔSh={d_sh:+.3f}) {flag}")

    print(f"\n[B. Pure TV top-N (對照組, 無 ratio)]")
    print(f"  {'N':>3s}  {'timed':>6s}  {'α':>9s}  {'t_α':>8s}  {'Sh':>7s}  {'turn':>6s}")
    for _, r in res_df[(res_df["variant"] == "pure_tv")].iterrows():
        print(f"  {r['N']:>3d}  {'Y' if r['timed'] else 'N':>6s}  "
              f"{r['ann_alpha']:>+9.4f}  {r['t_alpha']:>+8.3f}  "
              f"{r['sharpe']:>+7.3f}  {r['ann_turn']:>6.2f}")

    # === TV vs MC 對比 ===
    print(f"\n" + "=" * 110)
    print("TV filter (Phase 31) vs MC filter (Phase 30) 邊際影響對比 (timed)")
    print("=" * 110)
    print(f"  Phase 30 MC: top70/50/30% N=20 → -4.17 / -5.11 / -5.39 pp α (全部傷害)")
    print(f"  Phase 31 TV: 看 N=20 結果:")
    nf = res_df[(res_df["variant"] == "ratio") & (res_df["tv_filter"] == "no_filter") & (res_df["N"] == 20) & (res_df["timed"])].iloc[0]
    for tv_tag in ["top70%", "top50%", "top30%"]:
        r = res_df[(res_df["variant"] == "ratio") & (res_df["tv_filter"] == tv_tag) &
                   (res_df["N"] == 20) & (res_df["timed"])].iloc[0]
        d_a = (r["ann_alpha"] - nf["ann_alpha"]) * 100
        d_sh = r["sharpe"] - nf["sharpe"]
        flag = "✓" if d_a > 0.5 and d_sh > 0 else ("↑" if d_a > 0 else "↓")
        print(f"    TV {tv_tag:>7s} N=20: α={r['ann_alpha']:+.4f}, Sh={r['sharpe']:+.3f}  "
              f"Δα={d_a:>+5.2f}pp, ΔSh={d_sh:>+6.3f} {flag}")

    # === Best ===
    print(f"\n" + "=" * 110)
    print("Best (sorted by t_α, timed)")
    print("=" * 110)
    timed = res_df[(res_df["variant"] != "Full_EW") & (res_df["timed"])].sort_values("t_alpha", ascending=False).head(8)
    for _, r in timed.iterrows():
        d_t = r["t_alpha"] - base_b["t_alpha"]
        d_sh = r["sharpe"] - base_b["sharpe"]
        flag = "✓✓" if d_t > 0.3 and d_sh > 0.02 else "↑" if d_t > 0 else "↓"
        print(f"  {r['variant']:>8s}, TV={r['tv_filter']:>10s}, N={r['N']:>3d}: α={r['ann_alpha']:+.4f}, "
              f"t={r['t_alpha']:+.3f}, Sh={r['sharpe']:+.3f}  Δt={d_t:+.2f}, ΔSh={d_sh:+.3f} {flag}")

    # === Plot ===
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    Ns = N_LIST
    cols = {"no_filter": "#0277bd", "top70%": "#388e3c", "top50%": "#f57c00", "top30%": "#c62828"}
    for timed_idx, ax in enumerate(axes):
        timed_v = (timed_idx == 1)
        for tv_tag in ["no_filter", "top70%", "top50%", "top30%"]:
            ts = []
            for N in Ns:
                r = res_df[(res_df["variant"] == "ratio") & (res_df["tv_filter"] == tv_tag) &
                           (res_df["N"] == N) & (res_df["timed"] == timed_v)].iloc[0]
                ts.append(r["t_alpha"])
            ax.plot(Ns, ts, "o-", color=cols[tv_tag], lw=2, ms=8, label=f"TV {tv_tag}")
        baseline = base_b["t_alpha"] if timed_v else base_a["t_alpha"]
        ax.axhline(baseline, color="black", ls="--", alpha=0.5, label=f"Full EW t={baseline:.2f}")
        ax.axhline(1.96, color="gray", ls=":", alpha=0.4)
        ax.set_xlabel("Top-N (by ratio)")
        ax.set_ylabel("t_α")
        ax.set_title(f"({'A' if not timed_v else 'B'}) {'Always-on' if not timed_v else 'Timed'}: ratio + TV filter")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
    plt.suptitle("Phase 31: D000 下游 ratio rank + 3M-avg 成交金額過濾", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig_phase31_compare.png"), dpi=130)
    plt.close()
    print(f"\n圖檔: fig_phase31_compare.png")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
