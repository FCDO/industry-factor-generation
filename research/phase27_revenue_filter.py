"""
Phase 27: D400/D500 中游 walk-forward ratio rank + 月營收 YoY > 0 過濾

動機 (Phase 26)
- ratio N=10 always-on α=10.41%, t=2.58, Sharpe 1.33 (selection α 救援成功)
- 但純技術訊號可能挑到「延遲 β 大但基本面惡化」的個股
- 月營收 YoY > 0 是台股最即時、可信的基本面訊號 (每月 10 日公告)
- 假設: ratio rank 挑出技術上有延遲反應的, YoY 過濾掉基本面已轉弱的

過濾邏輯
- 在每月 rebalance 日 d, 對 ratio 排序前 N+buffer 的個股逐一檢查 YoY
- 如果 YoY > 0 → 入選
- 如果 YoY ≤ 0 或 NaN → 跳過, 由排名後面的個股遞補
- 直到湊滿 N 個, 或候選窮盡為止

No-peek
- FinLab YoY index 是 publication date (e.g., 2024-03-10 = 2024-02 月營收, 已公告)
- 把 yoy reindex 到 daily, forward-fill (取最新已公告)
- 額外 .shift(1) 一天 buffer (假設交易日當天不能用當天才公告的)

設計
- 排序: ratio (Phase 26 winner)
- N: 5, 10, 15, 20
- 對照組: 無過濾 (Phase 26 重現) + Full EW (baseline)

輸出
- phase27_revenue_filter_summary.csv
- phase27_strategy_returns.csv
- phase27_selection_timeline.csv : 每月 ratio rank top-30 + YoY 值 + 是否入選
- fig_phase27_filter_compare.png
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

N_LIST = [5, 10, 15, 20]
BUFFER_DAYS = 1   # YoY publication 安全 buffer (.shift(1) trading day)


def load_daily_ret():
    with open(os.path.join(DB, "etl#adj_close.pickle"), "rb") as f:
        adj = pickle.load(f).set_index("date")
    adj.index = pd.to_datetime(adj.index)
    adj.columns = adj.columns.astype(str)
    keep = [c for c in adj.columns if c.isdigit() and len(c) == 4 and not c.startswith("00")]
    return adj[keep].pct_change(fill_method=None).iloc[1:]


def load_yoy(daily_index):
    """載入月營收 YoY%, reindex 到 daily, forward-fill, 1-day safety shift"""
    with open(os.path.join(DB, "monthly_revenue#去年同月增減(%).pickle"), "rb") as f:
        yoy = pickle.load(f)
    yoy = yoy.set_index("date")
    yoy.index = pd.to_datetime(yoy.index)

    # columns 格式: "1101 台泥" — 取第一個 token (stock_id)
    new_cols = []
    for c in yoy.columns:
        sid = str(c).split(" ")[0].strip()
        new_cols.append(sid)
    yoy.columns = new_cols

    # 同 stock_id 多列 (e.g. 1104 環泥/環球水泥), 取第一個非 NaN 的
    yoy = yoy.T.groupby(level=0).first().T

    # reindex 到 daily, ffill
    yoy_daily = yoy.reindex(daily_index, method="ffill")

    # 安全 buffer: 假設交易日當天不能用當天 publication
    yoy_daily = yoy_daily.shift(BUFFER_DAYS)
    return yoy_daily


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


def build_topN_with_filter(rank_matrix, yoy_daily, daily_ret_index, universe, rebal_list, N,
                            apply_filter):
    """ratio 排序前 N, 過濾 YoY > 0 (apply_filter=True), 不夠由後遞補"""
    W = pd.DataFrame(0.0, index=daily_ret_index, columns=universe)
    log_rows = []  # 紀錄每月 selection 過程
    for i, d in enumerate(rebal_list):
        scores = rank_matrix.loc[d].dropna()
        scores = scores[scores > 0]
        if len(scores) == 0:
            continue
        scores_sorted = scores.sort_values(ascending=False)

        # 取得當下可用 YoY (forward-filled, 1-day shift)
        if d in yoy_daily.index:
            yoy_today = yoy_daily.loc[d]
        else:
            # take last available before d
            valid = yoy_daily.index[yoy_daily.index <= d]
            yoy_today = yoy_daily.loc[valid[-1]] if len(valid) > 0 else pd.Series(dtype=float)

        selected = []
        rejected = []
        for sid, score in scores_sorted.items():
            if len(selected) >= N:
                break
            if apply_filter:
                yoy_val = yoy_today.get(sid, np.nan)
                if pd.isna(yoy_val) or yoy_val <= 0:
                    rejected.append((sid, score, yoy_val))
                    continue
            selected.append((sid, score, yoy_today.get(sid, np.nan)))

        # log 該月選股結果
        for rank_, (sid, score, yoy_val) in enumerate(selected):
            log_rows.append({"date": d.date(), "rank": rank_+1, "stock_id": sid,
                              "score": score, "yoy": yoy_val, "status": "selected"})
        for sid, score, yoy_val in rejected:
            log_rows.append({"date": d.date(), "rank": -1, "stock_id": sid,
                              "score": score, "yoy": yoy_val, "status": "rejected_yoy_neg"})

        next_rebal = rebal_list[i + 1] if i + 1 < len(rebal_list) else daily_ret_index.max() + pd.Timedelta(days=1)
        hold_idx = daily_ret_index[(daily_ret_index > d) & (daily_ret_index <= next_rebal)]
        sel_ids = [s for s, _, _ in selected]
        if len(sel_ids) > 0:
            W.loc[hold_idx, sel_ids] = 1.0 / len(sel_ids)
    return W, pd.DataFrame(log_rows)


def main():
    print("=" * 110)
    print("Phase 27: D400/D500 ratio rank + 月營收 YoY > 0 過濾")
    print("=" * 110)

    chain = pd.read_csv(os.path.join(ROOT, "industry_chain.csv"), dtype=str)
    daily_ret = load_daily_ret()
    market = daily_ret.mean(axis=1)
    market_lag = market.shift(1)

    yoy_daily = load_yoy(daily_ret.index)
    print(f"\nYoY daily: {yoy_daily.shape}")
    print(f"  date range: {yoy_daily.index.min().date()} → {yoy_daily.index.max().date()}")
    print(f"  cols sample: {list(yoy_daily.columns[:8])}")
    # 抽樣檢查 — 用實際交易日 (取最後一個 ≤ 2024-03-31 的 trading day)
    sample_d = yoy_daily.index[yoy_daily.index <= pd.Timestamp("2024-03-31")][-1]
    yoy_6208 = yoy_daily.loc[sample_d, '6208'] if '6208' in yoy_daily.columns else float('nan')
    yoy_5434 = yoy_daily.loc[sample_d, '5434'] if '5434' in yoy_daily.columns else float('nan')
    print(f"  在 {sample_d.date()} 取得 6208 日揚 YoY = {yoy_6208:.2f}%")
    print(f"  在 {sample_d.date()} 取得 5434 崇越 YoY = {yoy_5434:.2f}%")

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
    print(f"\nUniverse: {len(universe)} 檔")

    first_valid = daily_ret.index[LOOKBACK]
    month_ends = pd.Series(daily_ret.index).groupby([daily_ret.index.year, daily_ret.index.month]).max()
    rebal_dates = pd.DatetimeIndex(month_ends.values)
    rebal_dates = rebal_dates[rebal_dates >= first_valid]
    rebal_list = sorted(rebal_dates)

    # === 計算 ratio 矩陣 (重用 Phase 26 邏輯) ===
    print(f"\n[1] 計算 β_lag / β_now ratio 矩陣")
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

    # === Diagnostic: 看最後一期 ratio top + YoY ===
    last_d = rebal_list[-1]
    print(f"\n[Diagnostic] {last_d.date()} ratio top 15 + YoY:")
    s = ratio_m.loc[last_d].dropna()
    s = s[s > 0].sort_values(ascending=False).head(15)
    yoy_today = yoy_daily.loc[last_d]
    print(f"{'rank':>4s} {'stock':>5s} {'name':<10s} {'ratio':>7s} {'YoY%':>7s}  status")
    selected_n = 0
    for rank_, (sid, score) in enumerate(s.items()):
        yoy_val = yoy_today.get(sid, np.nan)
        if pd.isna(yoy_val) or yoy_val <= 0:
            status = "❌ 跳過"
        else:
            selected_n += 1
            status = f"✓ 入選 #{selected_n}"
        print(f"{rank_+1:>4d} {sid:>5s} {name_map.get(sid, ''):<10s} {score:>+7.3f} "
              f"{yoy_val:>+7.2f}  {status}")

    # === 構 portfolio (filter + non-filter 對照) ===
    print(f"\n[2] 構 portfolios for N ∈ {N_LIST} (filter on/off)")
    weight_dict = {}
    log_dict = {}
    for N in N_LIST:
        for filt in [False, True]:
            W, log_df = build_topN_with_filter(ratio_m, yoy_daily, daily_ret.index,
                                                 universe, rebal_list, N, filt)
            weight_dict[(N, filt)] = W
            if filt:
                log_dict[N] = log_df
            tag = "filter" if filt else "no_filter"
            n_avg = (W > 0).sum(axis=1).where(lambda x: x > 0).mean()
            print(f"  N={N}, {tag}: avg active = {n_avg:.1f}")

    # 輸出 N=10 with filter 的 selection log
    log_dict[10].to_csv(os.path.join(OUT, "phase27_selection_timeline_N10.csv"),
                         index=False, encoding="utf-8-sig")
    n_rejected = (log_dict[10]["status"] == "rejected_yoy_neg").sum()
    n_selected = (log_dict[10]["status"] == "selected").sum()
    print(f"\n[N=10 filter] 全期間: {n_selected} selections, {n_rejected} rejections "
          f"({n_rejected/(n_selected+n_rejected)*100:.0f}% 因 YoY ≤ 0 跳過)")

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
        {"variant": "Full_EW", "N": "all", "filter": "—", "timed": False, **p_full, "ann_turn": at_full},
        {"variant": "Full_EW", "N": "all", "filter": "—", "timed": True, **p_full_t, "ann_turn": at_full_t},
    ]
    daily_returns_dict = {"Full_EW_net": n_full, "Full_EW_timed_net": n_full_t}

    for N in N_LIST:
        for filt in [False, True]:
            W = weight_dict[(N, filt)]
            g, n, at = perf_from_weights(W, rets_uni)
            W_t = W.mul(timing, axis=0).fillna(0)
            g_t, n_t, at_t = perf_from_weights(W_t, rets_uni)
            tag = "Y" if filt else "N"
            daily_returns_dict[f"ratio{N}_filt{tag}_net"] = n
            daily_returns_dict[f"ratio{N}_filt{tag}_timed_net"] = n_t

            p = perf_with_alpha(n.loc[eval_start:], market.loc[eval_start:])
            p_t = perf_with_alpha(n_t.loc[eval_start:], market.loc[eval_start:])
            rows.append({"variant": "ratio", "N": N, "filter": tag, "timed": False, **p, "ann_turn": at})
            rows.append({"variant": "ratio", "N": N, "filter": tag, "timed": True, **p_t, "ann_turn": at_t})

    res_df = pd.DataFrame(rows)
    res_df.to_csv(os.path.join(OUT, "phase27_revenue_filter_summary.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(daily_returns_dict).to_csv(os.path.join(OUT, "phase27_strategy_returns.csv"),
                                              encoding="utf-8-sig")

    # === Print ===
    print(f"\n" + "=" * 110)
    print(f"策略表現 (扣 50bp RT, eval start = {eval_start.date()})")
    print("=" * 110)
    base_a = res_df[(res_df["variant"] == "Full_EW") & (~res_df["timed"])].iloc[0]
    base_b = res_df[(res_df["variant"] == "Full_EW") & (res_df["timed"])].iloc[0]
    print(f"\nbaseline: Full EW always-on α={base_a['ann_alpha']:+.4f}/t={base_a['t_alpha']:+.3f}/Sh={base_a['sharpe']:+.3f}")
    print(f"          Full EW timed     α={base_b['ann_alpha']:+.4f}/t={base_b['t_alpha']:+.3f}/Sh={base_b['sharpe']:+.3f}")

    print(f"\n[Always-on] (vs Full α={base_a['ann_alpha']:+.4f}, t={base_a['t_alpha']:+.3f}, Sh={base_a['sharpe']:+.3f})")
    print(f"{'N':>3s}  {'filter':>6s}  {'α':>9s}  {'t_α':>8s}  {'Sh':>7s}  {'turn':>6s}  Δα/Δt")
    for _, r in res_df[(res_df["variant"] == "ratio") & (~res_df["timed"])].iterrows():
        d_a = (r["ann_alpha"] - base_a["ann_alpha"]) * 100
        d_t = r["t_alpha"] - base_a["t_alpha"]
        d_sh = r["sharpe"] - base_a["sharpe"]
        flag = "✓✓" if d_a > 3 and d_t > 0.5 else ("✓" if d_a > 1 and d_t > 0.2 else ("↑" if d_a > 0 else "↓"))
        print(f"{r['N']:>3d}  {r['filter']:>6s}  {r['ann_alpha']:>+9.4f}  {r['t_alpha']:>+8.3f}  "
              f"{r['sharpe']:>+7.3f}  {r['ann_turn']:>6.2f}  {d_a:+.2f}pp / {d_t:+.2f} (ΔSh={d_sh:+.3f}) {flag}")

    print(f"\n[Timed] (vs Full timed α={base_b['ann_alpha']:+.4f}, t={base_b['t_alpha']:+.3f}, Sh={base_b['sharpe']:+.3f})")
    print(f"{'N':>3s}  {'filter':>6s}  {'α':>9s}  {'t_α':>8s}  {'Sh':>7s}  {'turn':>6s}  Δα/Δt")
    for _, r in res_df[(res_df["variant"] == "ratio") & (res_df["timed"])].iterrows():
        d_a = (r["ann_alpha"] - base_b["ann_alpha"]) * 100
        d_t = r["t_alpha"] - base_b["t_alpha"]
        d_sh = r["sharpe"] - base_b["sharpe"]
        flag = "✓✓" if d_a > 3 and d_t > 0.5 else ("✓" if d_a > 1 and d_t > 0.2 else ("↑" if d_a > 0 else "↓"))
        print(f"{r['N']:>3d}  {r['filter']:>6s}  {r['ann_alpha']:>+9.4f}  {r['t_alpha']:>+8.3f}  "
              f"{r['sharpe']:>+7.3f}  {r['ann_turn']:>6.2f}  {d_a:+.2f}pp / {d_t:+.2f} (ΔSh={d_sh:+.3f}) {flag}")

    # === filter on vs off 直接對比 ===
    print(f"\n" + "=" * 110)
    print("Filter on vs off 直接對比 (同 N, 同 timed 設定)")
    print("=" * 110)
    print(f"{'N':>3s}  {'timed':>6s}  {'no_filter α/t':>20s}  {'filter α/t':>20s}  {'Δα':>9s}  {'Δt':>7s}  {'ΔSh':>7s}")
    for N in N_LIST:
        for timed in [False, True]:
            r_off = res_df[(res_df["N"] == N) & (res_df["filter"] == "N") & (res_df["timed"] == timed)].iloc[0]
            r_on = res_df[(res_df["N"] == N) & (res_df["filter"] == "Y") & (res_df["timed"] == timed)].iloc[0]
            d_a = (r_on["ann_alpha"] - r_off["ann_alpha"]) * 100
            d_t = r_on["t_alpha"] - r_off["t_alpha"]
            d_sh = r_on["sharpe"] - r_off["sharpe"]
            print(f"{N:>3d}  {'Y' if timed else 'N':>6s}  "
                  f"{r_off['ann_alpha']:+.4f}/{r_off['t_alpha']:+.2f}     "
                  f"{r_on['ann_alpha']:+.4f}/{r_on['t_alpha']:+.2f}     "
                  f"{d_a:>+8.2f}pp  {d_t:>+6.2f}  {d_sh:>+6.3f}")

    # === Best ===
    print(f"\n" + "=" * 110)
    print("Best 策略 (filter on, sorted by t_α)")
    print("=" * 110)
    best_df = res_df[res_df["filter"] == "Y"].sort_values("t_alpha", ascending=False)
    for _, r in best_df.head(6).iterrows():
        timed_tag = "timed" if r["timed"] else "always-on"
        print(f"  N={r['N']:>2d} {timed_tag:>10s}: α={r['ann_alpha']:+.4f}, t={r['t_alpha']:+.3f}, "
              f"Sh={r['sharpe']:+.3f}, turn={r['ann_turn']:.2f}")

    # === Plot ===
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    Ns = N_LIST
    for timed_idx, ax in enumerate(axes):
        timed = (timed_idx == 1)
        for filt, color, marker, label_pre in [
            (False, "#0277bd", "o", "no filter"),
            (True, "#d81b60", "s", "+ YoY > 0 filter"),
        ]:
            tag = "Y" if filt else "N"
            alphas = [res_df[(res_df["N"] == N) & (res_df["filter"] == tag) & (res_df["timed"] == timed)].iloc[0]["ann_alpha"] for N in Ns]
            ax.plot(Ns, alphas, marker + "-", color=color, lw=2, ms=9, label=f"{label_pre} α")
        baseline_a = base_b["ann_alpha"] if timed else base_a["ann_alpha"]
        baseline_t = base_b["t_alpha"] if timed else base_a["t_alpha"]
        ax.axhline(baseline_a, color="black", ls="--", alpha=0.5, label=f"Full EW α={baseline_a:.3f}")
        ax.set_xlabel("Top-N")
        ax.set_ylabel("ann_alpha (filter on/off)")
        ax.set_title(f"({'A' if not timed else 'B'}) {'Always-on' if not timed else 'Timed'} (vs Full EW α={baseline_a:.3f})")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
    plt.suptitle("Phase 27: ratio rank + 月營收 YoY > 0 filter", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig_phase27_filter_compare.png"), dpi=130)
    plt.close()
    print(f"\n圖檔: fig_phase27_filter_compare.png")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
