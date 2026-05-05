"""
Phase 28: D400/D500 中游 ratio rank + universe-level YoY 預過濾

Phase 27 結論
- ratio top → 過濾 YoY ≤ 0 → 由後遞補 (rank-then-filter)
- 結果: filter 把 ratio 最強個股大量排除 (43% 跳過, 含 #1 6208 日揚 YoY -18%)
  由 ratio 弱的個股遞補, 訊號稀釋, N=10 timed Δα -2.19pp / ΔSh -0.286 ↓

Phase 28 改進: universe-level pre-filter (filter-then-rank)
- 把「過去 3 個月 YoY 平均過低」的爛公司先從 universe 剔除
- 然後對剩餘個股做 ratio rank, 取 top-N
- 6208 日揚 雖單月 YoY -18%, 但若過去 3 個月平均仍 > -10%, 表示只是短期回檔, 保留入選

設計
- 預過濾門檻 (掃多個): 3M-avg-YoY < -30%, < -20%, < -10%, < 0%
  測「壞到什麼程度才剔除」
- ratio rank N=10 (Phase 26 winner)
- always-on + timed
- 對照: 無過濾 (Phase 26 重現)

No-peek
- yoy_daily 已 .shift(1) (Phase 27 buffer)
- 3M 平均: 取最近 3 個 publication 點平均, ffill 到 daily

輸出
- phase28_prefilter_summary.csv
- phase28_universe_size_timeline.csv : 每月過濾後 universe 大小
- fig_phase28_threshold_sweep.png
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

N_PICK = 10  # Phase 26 winner
THRESHOLDS = [-100, -30, -20, -10, 0]  # -100 = 無過濾基準
ROLL_M = 3   # 3 個月平均


def load_daily_ret():
    with open(os.path.join(DB, "etl#adj_close.pickle"), "rb") as f:
        adj = pickle.load(f).set_index("date")
    adj.index = pd.to_datetime(adj.index)
    adj.columns = adj.columns.astype(str)
    keep = [c for c in adj.columns if c.isdigit() and len(c) == 4 and not c.startswith("00")]
    return adj[keep].pct_change(fill_method=None).iloc[1:]


def load_yoy_with_avg(daily_index, roll_m=ROLL_M):
    """載入月營收 YoY%, 算 3M 平均, reindex daily, 1d shift"""
    with open(os.path.join(DB, "monthly_revenue#去年同月增減(%).pickle"), "rb") as f:
        yoy = pickle.load(f)
    yoy = yoy.set_index("date")
    yoy.index = pd.to_datetime(yoy.index)
    new_cols = [str(c).split(" ")[0].strip() for c in yoy.columns]
    yoy.columns = new_cols
    yoy = yoy.T.groupby(level=0).first().T

    # 月度 3M 平均 (在 monthly index 上算)
    yoy_avg = yoy.rolling(window=roll_m, min_periods=2).mean()

    # reindex daily, ffill, 1d safety shift
    yoy_avg_daily = yoy_avg.reindex(daily_index, method="ffill").shift(1)
    yoy_raw_daily = yoy.reindex(daily_index, method="ffill").shift(1)
    return yoy_avg_daily, yoy_raw_daily


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


def build_topN_with_universe_filter(rank_matrix, yoy_avg_daily, daily_ret_index,
                                       universe, rebal_list, N, threshold):
    """
    1. universe-level pre-filter: 剔除 yoy_avg_3m < threshold 的個股
    2. 對剩餘個股做 ratio rank, 取 top-N
    """
    W = pd.DataFrame(0.0, index=daily_ret_index, columns=universe)
    n_universe_log = []
    for i, d in enumerate(rebal_list):
        scores = rank_matrix.loc[d].dropna()
        scores = scores[scores > 0]
        if len(scores) == 0:
            n_universe_log.append({"date": d.date(), "threshold": threshold,
                                    "n_eligible": 0, "n_selected": 0})
            continue

        # 取得當下 yoy_avg_3m
        if d in yoy_avg_daily.index:
            yoy_today = yoy_avg_daily.loc[d]
        else:
            valid = yoy_avg_daily.index[yoy_avg_daily.index <= d]
            yoy_today = yoy_avg_daily.loc[valid[-1]] if len(valid) > 0 else pd.Series(dtype=float)

        # universe pre-filter: 剔除過去 3M 平均 < threshold (NaN 也排除以保守)
        if threshold > -99:  # 不是「無過濾」基準
            valid_uni = []
            for sid in scores.index:
                yv = yoy_today.get(sid, np.nan)
                if pd.notna(yv) and yv >= threshold:
                    valid_uni.append(sid)
            scores = scores.loc[valid_uni]

        if len(scores) == 0:
            n_universe_log.append({"date": d.date(), "threshold": threshold,
                                    "n_eligible": 0, "n_selected": 0})
            continue

        scores_sorted = scores.sort_values(ascending=False)
        top = scores_sorted.head(N).index.tolist()
        n_universe_log.append({"date": d.date(), "threshold": threshold,
                                "n_eligible": len(scores), "n_selected": len(top)})

        next_rebal = rebal_list[i + 1] if i + 1 < len(rebal_list) else daily_ret_index.max() + pd.Timedelta(days=1)
        hold_idx = daily_ret_index[(daily_ret_index > d) & (daily_ret_index <= next_rebal)]
        if len(top) > 0:
            W.loc[hold_idx, top] = 1.0 / len(top)
    return W, pd.DataFrame(n_universe_log)


def main():
    print("=" * 110)
    print("Phase 28: D400/D500 ratio rank + universe-level 3M-avg-YoY 預過濾")
    print("=" * 110)
    print(f"  N={N_PICK}  |  thresholds: {THRESHOLDS}  |  rolling window: {ROLL_M}M")

    chain = pd.read_csv(os.path.join(ROOT, "industry_chain.csv"), dtype=str)
    daily_ret = load_daily_ret()
    market = daily_ret.mean(axis=1)
    market_lag = market.shift(1)

    yoy_avg_daily, yoy_raw_daily = load_yoy_with_avg(daily_ret.index)
    print(f"\n3M-avg-YoY daily: {yoy_avg_daily.shape}")

    d100 = chain[(chain["industry_code"] == "D000") & (chain["sub_code"] == "D100")]["stock_id"].unique().tolist()
    dc00 = chain[(chain["industry_code"] == "D000") & (chain["sub_code"] == "DC00")]["stock_id"].unique().tolist()
    d100_in = [s for s in d100 if s in daily_ret.columns]
    dc00_in = [s for s in dc00 if s in daily_ret.columns]
    lead = 0.5 * daily_ret[d100_in].mean(axis=1) + 0.5 * daily_ret[dc00_in].mean(axis=1)
    lead_lag = lead.shift(1)

    d_mid = chain[(chain["industry_code"] == "D000") &
                   (chain["sub_code"].isin(["D400", "D500"]))][["stock_id", "name", "sub_code"]]
    d_mid = d_mid.drop_duplicates("stock_id", keep="first").reset_index(drop=True)
    universe = sorted([s for s in d_mid["stock_id"] if s in daily_ret.columns])
    rets_uni = daily_ret[universe]
    name_map = dict(zip(d_mid["stock_id"], d_mid["name"]))
    print(f"Universe: {len(universe)} 檔")

    first_valid = daily_ret.index[LOOKBACK]
    month_ends = pd.Series(daily_ret.index).groupby([daily_ret.index.year, daily_ret.index.month]).max()
    rebal_dates = pd.DatetimeIndex(month_ends.values)
    rebal_dates = rebal_dates[rebal_dates >= first_valid]
    rebal_list = sorted(rebal_dates)

    # === 計算 ratio 矩陣 ===
    print(f"\n[1] 計算 β_lag/β_now ratio 矩陣")
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
            print(f"  [{j+1:>3d}/{len(rebal_dates)}] {d.date()}  ({time.time()-t0:.0f}s)")

    ratio_m = b_lag_m / b_now_m.where(b_now_m > 0.05)

    # === Diagnostic: 看最後一期 universe 變化 ===
    last_d = rebal_list[-1]
    print(f"\n[Diagnostic] {last_d.date()} universe pre-filter 結果:")
    yoy_today = yoy_avg_daily.loc[last_d]
    yoy_raw_today = yoy_raw_daily.loc[last_d]
    s = ratio_m.loc[last_d].dropna()
    s = s[s > 0].sort_values(ascending=False).head(15)
    print(f"{'rank':>4s} {'stock':>5s} {'name':<10s} {'ratio':>7s} {'YoY單月':>8s} {'YoY 3M-avg':>11s}  各 threshold 是否入選")
    for rank_, (sid, score) in enumerate(s.items()):
        yoy_3m = yoy_today.get(sid, np.nan)
        yoy_now = yoy_raw_today.get(sid, np.nan)
        passes = []
        for th in [-30, -20, -10, 0]:
            if pd.notna(yoy_3m) and yoy_3m >= th:
                passes.append(f"≥{th}")
            else:
                passes.append("---")
        passes_str = "  ".join(f"{p:>4s}" for p in passes)
        print(f"{rank_+1:>4d} {sid:>5s} {name_map.get(sid, ''):<10s} {score:>+7.3f} "
              f"{yoy_now:>+8.2f}% {yoy_3m:>+10.2f}%   {passes_str}")

    # === 構 portfolios for 多 thresholds ===
    print(f"\n[2] 構 portfolios for 多 thresholds")
    weight_dict = {}
    universe_logs = {}
    for th in THRESHOLDS:
        W, log_df = build_topN_with_universe_filter(ratio_m, yoy_avg_daily,
                                                       daily_ret.index, universe,
                                                       rebal_list, N_PICK, th)
        weight_dict[th] = W
        universe_logs[th] = log_df
        n_avg = (W > 0).sum(axis=1).where(lambda x: x > 0).mean()
        ne_med = log_df["n_eligible"].median() if len(log_df) else 0
        ne_min = log_df["n_eligible"].min() if len(log_df) else 0
        tag = "no filter" if th < -99 else f"3M-avg ≥ {th}%"
        print(f"  {tag:>16s}: avg active = {n_avg:.1f}, n_eligible median = {ne_med:.0f} (min {ne_min})")

    # Combined log
    pd.concat(universe_logs.values()).to_csv(os.path.join(OUT, "phase28_universe_size_timeline.csv"),
                                              index=False, encoding="utf-8-sig")

    # === Full EW baseline ===
    W_full = pd.DataFrame(0.0, index=daily_ret.index, columns=universe)
    for d in daily_ret.index[LOOKBACK:]:
        avail = rets_uni.loc[d].dropna().index.tolist()
        if avail:
            W_full.loc[d, avail] = 1.0 / len(avail)

    # Timing
    timing = (lead.ewm(span=SMOOTH_N, adjust=False).mean() > 0).astype(float).shift(1).fillna(0)

    # === Returns ===
    eval_start = rebal_list[0] + pd.Timedelta(days=1)
    print(f"\n[3] 計算 returns (eval start = {eval_start.date()})")

    g_full, n_full, at_full = perf_from_weights(W_full, rets_uni)
    W_full_t = W_full.mul(timing, axis=0).fillna(0)
    g_full_t, n_full_t, at_full_t = perf_from_weights(W_full_t, rets_uni)
    p_full = perf_with_alpha(n_full.loc[eval_start:], market.loc[eval_start:])
    p_full_t = perf_with_alpha(n_full_t.loc[eval_start:], market.loc[eval_start:])

    rows = [
        {"variant": "Full_EW", "threshold": "—", "timed": False, **p_full, "ann_turn": at_full},
        {"variant": "Full_EW", "threshold": "—", "timed": True, **p_full_t, "ann_turn": at_full_t},
    ]

    for th in THRESHOLDS:
        W = weight_dict[th]
        g, n, at = perf_from_weights(W, rets_uni)
        W_t = W.mul(timing, axis=0).fillna(0)
        g_t, n_t, at_t = perf_from_weights(W_t, rets_uni)
        th_label = "no_filter" if th < -99 else f"≥{th}%"
        p = perf_with_alpha(n.loc[eval_start:], market.loc[eval_start:])
        p_t = perf_with_alpha(n_t.loc[eval_start:], market.loc[eval_start:])
        rows.append({"variant": "ratio10", "threshold": th_label, "timed": False, **p, "ann_turn": at})
        rows.append({"variant": "ratio10", "threshold": th_label, "timed": True, **p_t, "ann_turn": at_t})

    res_df = pd.DataFrame(rows)
    res_df.to_csv(os.path.join(OUT, "phase28_prefilter_summary.csv"), index=False, encoding="utf-8-sig")

    # === Print ===
    print(f"\n" + "=" * 110)
    print(f"策略表現 (扣 50bp RT)")
    print("=" * 110)
    base_a = res_df[(res_df["variant"] == "Full_EW") & (~res_df["timed"])].iloc[0]
    base_b = res_df[(res_df["variant"] == "Full_EW") & (res_df["timed"])].iloc[0]
    no_filter_a = res_df[(res_df["threshold"] == "no_filter") & (~res_df["timed"])].iloc[0]
    no_filter_b = res_df[(res_df["threshold"] == "no_filter") & (res_df["timed"])].iloc[0]
    print(f"\nFull EW         always-on: α={base_a['ann_alpha']:+.4f}, t={base_a['t_alpha']:+.3f}, Sh={base_a['sharpe']:+.3f}")
    print(f"Full EW         timed:     α={base_b['ann_alpha']:+.4f}, t={base_b['t_alpha']:+.3f}, Sh={base_b['sharpe']:+.3f}")
    print(f"ratio10 no_filter always: α={no_filter_a['ann_alpha']:+.4f}, t={no_filter_a['t_alpha']:+.3f}, Sh={no_filter_a['sharpe']:+.3f}")
    print(f"ratio10 no_filter timed:   α={no_filter_b['ann_alpha']:+.4f}, t={no_filter_b['t_alpha']:+.3f}, Sh={no_filter_b['sharpe']:+.3f}")

    print(f"\n[Always-on, ratio rank N=10] (vs no_filter α={no_filter_a['ann_alpha']:+.4f}, t={no_filter_a['t_alpha']:+.3f}, Sh={no_filter_a['sharpe']:+.3f})")
    print(f"{'threshold':>11s}  {'α':>9s}  {'t_α':>8s}  {'Sh':>7s}  {'turn':>6s}  Δα/Δt vs no-filter")
    for _, r in res_df[(res_df["variant"] == "ratio10") & (~res_df["timed"])].iterrows():
        d_a = (r["ann_alpha"] - no_filter_a["ann_alpha"]) * 100
        d_t = r["t_alpha"] - no_filter_a["t_alpha"]
        d_sh = r["sharpe"] - no_filter_a["sharpe"]
        flag = "✓✓" if d_a > 1 and d_t > 0.2 else ("✓" if d_a > 0.3 and d_t > 0.05 else ("↑" if d_a > 0 else "↓"))
        print(f"{r['threshold']:>11s}  {r['ann_alpha']:>+9.4f}  {r['t_alpha']:>+8.3f}  "
              f"{r['sharpe']:>+7.3f}  {r['ann_turn']:>6.2f}  {d_a:+.2f}pp / {d_t:+.2f} (ΔSh={d_sh:+.3f}) {flag}")

    print(f"\n[Timed, ratio rank N=10] (vs no_filter α={no_filter_b['ann_alpha']:+.4f}, t={no_filter_b['t_alpha']:+.3f}, Sh={no_filter_b['sharpe']:+.3f})")
    print(f"{'threshold':>11s}  {'α':>9s}  {'t_α':>8s}  {'Sh':>7s}  {'turn':>6s}  Δα/Δt vs no-filter")
    for _, r in res_df[(res_df["variant"] == "ratio10") & (res_df["timed"])].iterrows():
        d_a = (r["ann_alpha"] - no_filter_b["ann_alpha"]) * 100
        d_t = r["t_alpha"] - no_filter_b["t_alpha"]
        d_sh = r["sharpe"] - no_filter_b["sharpe"]
        flag = "✓✓" if d_a > 1 and d_t > 0.2 else ("✓" if d_a > 0.3 and d_t > 0.05 else ("↑" if d_a > 0 else "↓"))
        print(f"{r['threshold']:>11s}  {r['ann_alpha']:>+9.4f}  {r['t_alpha']:>+8.3f}  "
              f"{r['sharpe']:>+7.3f}  {r['ann_turn']:>6.2f}  {d_a:+.2f}pp / {d_t:+.2f} (ΔSh={d_sh:+.3f}) {flag}")

    # === 比較 Phase 27 (rank-then-filter) vs Phase 28 (filter-then-rank) ===
    print(f"\n" + "=" * 110)
    print("關鍵比較: Phase 27 (rank→filter YoY>0) vs Phase 28 (filter universe→rank)")
    print("=" * 110)
    print(f"\n  Phase 26 ratio10 no filter, always-on:    α=10.41%, t=2.58, Sh 1.33")
    print(f"  Phase 27 ratio10 + YoY>0 過濾, always-on: α= 6.41%, t=1.33, Sh 1.06 (-4.00pp)")
    th0_a = res_df[(res_df["threshold"] == "≥0%") & (~res_df["timed"])].iloc[0]
    print(f"  Phase 28 ratio10 + 3M-avg≥0% 預過濾:        α={th0_a['ann_alpha']*100:5.2f}%, "
          f"t={th0_a['t_alpha']:5.2f}, Sh {th0_a['sharpe']:.2f}")
    print(f"  → Phase 28 ≥0% 是否優於 Phase 27?")

    # === Best ===
    print(f"\n" + "=" * 110)
    print("Best (sorted by t_α, timed)")
    print("=" * 110)
    timed_only = res_df[(res_df["variant"] == "ratio10") & (res_df["timed"])].sort_values("t_alpha", ascending=False)
    for _, r in timed_only.iterrows():
        d_t = r["t_alpha"] - base_b["t_alpha"]
        d_sh = r["sharpe"] - base_b["sharpe"]
        flag = "✓✓" if d_t > 0.2 and d_sh > 0 else "↑" if d_t > 0 else "↓"
        print(f"  threshold={r['threshold']:>10s}: α={r['ann_alpha']:+.4f}, t={r['t_alpha']:+.3f}, "
              f"Sh={r['sharpe']:+.3f}, turn={r['ann_turn']:.2f}  (vs Full EW timed Δt={d_t:+.2f}, ΔSh={d_sh:+.3f}) {flag}")

    # === Plot threshold sweep ===
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    th_plot = [-30, -20, -10, 0]
    for timed_idx, ax in enumerate(axes):
        timed = (timed_idx == 1)
        sub = res_df[(res_df["variant"] == "ratio10") & (res_df["timed"] == timed)]
        # 把 threshold 轉 numeric
        x_vals = []
        alphas = []
        ts = []
        shs = []
        for th in [-30, -20, -10, 0]:
            r = sub[sub["threshold"] == f"≥{th}%"]
            if len(r) > 0:
                x_vals.append(th)
                alphas.append(r.iloc[0]["ann_alpha"])
                ts.append(r.iloc[0]["t_alpha"])
                shs.append(r.iloc[0]["sharpe"])
        no_f = sub[sub["threshold"] == "no_filter"].iloc[0]
        ax.plot(x_vals, alphas, "o-", color="#d81b60", lw=2, ms=9, label="α (filter on)")
        ax.axhline(no_f["ann_alpha"], color="#d81b60", ls="--", alpha=0.5, label=f"no filter α={no_f['ann_alpha']:.3f}")
        baseline = base_b["ann_alpha"] if timed else base_a["ann_alpha"]
        ax.axhline(baseline, color="black", ls=":", alpha=0.5, label=f"Full EW α={baseline:.3f}")
        ax.set_xlabel("threshold (3M-avg-YoY ≥ X%)")
        ax.set_ylabel("ann_alpha")
        ax.set_title(f"({'A' if not timed else 'B'}) {'Always-on' if not timed else 'Timed'}")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
    plt.suptitle(f"Phase 28: ratio10 + universe pre-filter (3M-avg-YoY ≥ X%)", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig_phase28_threshold_sweep.png"), dpi=130)
    plt.close()
    print(f"\n圖檔: fig_phase28_threshold_sweep.png")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
