"""
Phase 26: D400/D500 中游 walk-forward β_lag/β_contemp 延遲份額排序

動機 (Phase 23/25)
- Phase 25 t(β_lag) Top-N rank 改善了 Phase 24 threshold (Sharpe 1.27→1.54)
  但仍未擊敗 Full EW timed (Sharpe 1.59)
- 假設: 排序用 t-stat 沒區分「真延遲反應」vs「同步反應+大樣本」
  例如 3583 辛耘 β_contemp=0.81 但 β_lag=0.00 — 純同步, 沒延遲, 不該入選
- 「延遲份額」= β_lag / β_contemp 才能精準抓「對上游有延遲反應」的個股

設計
- 每月底 rebalance, 過去 5y window:
  跑 r_i = α + β_now·lead(t) + β_lag·lead(t-1) + γ·r_i(t-1) + δ·m(t-1)
  (注意: 同時放 lead(t) 和 lead(t-1), 確保 β_lag 是純延遲, β_now 是純同期)
- 三種排序變體:
  (a) t(β_lag)        — Phase 25 baseline, 用 t-stat
  (b) β_lag (raw)     — 純 β 大小
  (c) β_lag/β_now ratio — 延遲份額
- N ∈ {10, 15, 20}
- 過濾 β_now ≤ 0 (異常, 排除)

No-peek: 同 Phase 24/25

輸出
- phase26_factor_matrices.csv  : t_lag/β_lag/β_now × month × stock
- phase26_summary.csv          : 各 (method, N, timed) alpha
- fig_phase26_compare.png      : 三種排序 vs Phase 25 baseline
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

N_LIST = [10, 15, 20]


def load_daily_ret():
    with open(os.path.join(DB, "etl#adj_close.pickle"), "rb") as f:
        adj = pickle.load(f).set_index("date")
    adj.index = pd.to_datetime(adj.index)
    adj.columns = adj.columns.astype(str)
    keep = [c for c in adj.columns if c.isdigit() and len(c) == 4 and not c.startswith("00")]
    return adj[keep].pct_change(fill_method=None).iloc[1:]


def screen_joint(r_i_win, lead_win, lead_lag_win, market_lag_win):
    """同時放 lead(t) 和 lead(t-1) 跑 joint regression
    回傳 t_lag, beta_lag, beta_now"""
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


def build_topN_weights(rank_matrix, daily_ret_index, universe, rebal_list, N,
                        filter_pos=True):
    """rank_matrix: rebal_date × stock_id 排序值 (大者優先);
       filter_pos: 只取 rank > 0 的 (排除負/異常)"""
    W = pd.DataFrame(0.0, index=daily_ret_index, columns=universe)
    for i, d in enumerate(rebal_list):
        scores = rank_matrix.loc[d].dropna()
        if filter_pos:
            scores = scores[scores > 0]
        if len(scores) == 0:
            continue
        scores_sorted = scores.sort_values(ascending=False)
        top = scores_sorted.head(N).index.tolist()
        next_rebal = rebal_list[i + 1] if i + 1 < len(rebal_list) else daily_ret_index.max() + pd.Timedelta(days=1)
        hold_idx = daily_ret_index[(daily_ret_index > d) & (daily_ret_index <= next_rebal)]
        if len(top) > 0:
            W.loc[hold_idx, top] = 1.0 / len(top)
    return W


def main():
    print("=" * 110)
    print("Phase 26: D400/D500 中游 walk-forward 三種排序變體 (t / β / β_lag/β_now)")
    print("=" * 110)

    chain = pd.read_csv(os.path.join(ROOT, "industry_chain.csv"), dtype=str)
    daily_ret = load_daily_ret()
    market = daily_ret.mean(axis=1)
    market_lag = market.shift(1)

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
    print(f"\nUniverse: {len(universe)} 檔")

    first_valid = daily_ret.index[LOOKBACK]
    month_ends = pd.Series(daily_ret.index).groupby([daily_ret.index.year, daily_ret.index.month]).max()
    rebal_dates = pd.DatetimeIndex(month_ends.values)
    rebal_dates = rebal_dates[rebal_dates >= first_valid]
    rebal_list = sorted(rebal_dates)
    print(f"Rebal dates: {len(rebal_list)} ({rebal_list[0].date()} → {rebal_list[-1].date()})")

    # === 計算三個矩陣 ===
    print(f"\n[1] joint regression (lead(t) + lead(t-1)) on {len(rebal_dates)} months × {len(universe)} stocks")
    t0 = time.time()
    t_lag_m = pd.DataFrame(np.nan, index=rebal_dates, columns=universe)
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
                t_lag_m.loc[d, sid] = t_lag
                b_lag_m.loc[d, sid] = b_lag
                b_now_m.loc[d, sid] = b_now
        if (j + 1) % 24 == 0 or j == len(rebal_dates) - 1:
            print(f"  [{j+1:>3d}/{len(rebal_dates)}] {d.date()}  ({time.time()-t0:.0f}s)")

    # 比率: 排除 β_now ≤ 0.05 (異常或低同期), 否則 ratio 不穩
    ratio_m = b_lag_m / b_now_m.where(b_now_m > 0.05)

    # 儲存
    out_concat = pd.concat({"t_lag": t_lag_m, "b_lag": b_lag_m, "b_now": b_now_m,
                             "ratio": ratio_m}, axis=1)
    out_concat.to_csv(os.path.join(OUT, "phase26_factor_matrices.csv"), encoding="utf-8-sig")

    # === Diagnostics: 檢查最後一期 (2026-04) 三種排序的入選名單差異 ===
    last_d = rebal_list[-1]
    print(f"\n[Diagnostics] {last_d.date()} 三種排序前 15 名:")
    name_map = dict(zip(d_mid["stock_id"], d_mid["name"]))
    for label, mat in [("t_lag", t_lag_m), ("b_lag", b_lag_m), ("ratio", ratio_m)]:
        s = mat.loc[last_d].dropna()
        s = s[s > 0]
        s = s.sort_values(ascending=False).head(15)
        print(f"  {label:>8s}: {[(sid, name_map.get(sid,''), round(v,3)) for sid, v in s.items()][:8]}...")

    # === 各 ranking method × N → 構 weight ===
    print(f"\n[2] 構 portfolio weights for 各排序 × N")
    methods = {
        "t_lag":   t_lag_m,
        "b_lag":   b_lag_m,
        "ratio":   ratio_m,
    }

    weight_dict = {}
    for method, mat in methods.items():
        for N in N_LIST:
            W = build_topN_weights(mat, daily_ret.index, universe, rebal_list, N, filter_pos=True)
            weight_dict[(method, N)] = W
            n_avg = (W > 0).sum(axis=1).where(lambda x: x > 0).mean()
            print(f"  {method:>8s}, N={N}: avg active={n_avg:.1f}")

    # Full EW baseline
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
        {"method": "Full_EW", "N": "all", "timed": False, **p_full, "ann_turn": at_full},
        {"method": "Full_EW", "N": "all", "timed": True, **p_full_t, "ann_turn": at_full_t},
    ]

    for (method, N), W in weight_dict.items():
        g, n, at = perf_from_weights(W, rets_uni)
        W_t = W.mul(timing, axis=0).fillna(0)
        g_t, n_t, at_t = perf_from_weights(W_t, rets_uni)
        p = perf_with_alpha(n.loc[eval_start:], market.loc[eval_start:])
        p_t = perf_with_alpha(n_t.loc[eval_start:], market.loc[eval_start:])
        rows.append({"method": method, "N": N, "timed": False, **p, "ann_turn": at})
        rows.append({"method": method, "N": N, "timed": True, **p_t, "ann_turn": at_t})

    res_df = pd.DataFrame(rows)
    res_df.to_csv(os.path.join(OUT, "phase26_summary.csv"), index=False, encoding="utf-8-sig")

    # === Print summary ===
    print(f"\n" + "=" * 110)
    print(f"策略表現 (扣 50bp RT)")
    print("=" * 110)
    base_a = res_df[(res_df["method"] == "Full_EW") & (~res_df["timed"])].iloc[0]
    base_b = res_df[(res_df["method"] == "Full_EW") & (res_df["timed"])].iloc[0]
    print(f"\nbaselines:")
    print(f"  Full EW always-on: α={base_a['ann_alpha']:+.4f}, t={base_a['t_alpha']:+.3f}, "
          f"Sh={base_a['sharpe']:+.3f}, turn={base_a['ann_turn']:.2f}")
    print(f"  Full EW timed:     α={base_b['ann_alpha']:+.4f}, t={base_b['t_alpha']:+.3f}, "
          f"Sh={base_b['sharpe']:+.3f}, turn={base_b['ann_turn']:.2f}")

    print(f"\n[Always-on] (vs Full α={base_a['ann_alpha']:+.4f}, t={base_a['t_alpha']:+.3f})")
    print(f"{'method':>8s}  {'N':>4s}  {'α':>9s}  {'t_α':>8s}  {'Sh':>7s}  {'turn':>6s}  Δα/Δt")
    for _, r in res_df[(res_df["method"] != "Full_EW") & (~res_df["timed"])].iterrows():
        d_a = (r["ann_alpha"] - base_a["ann_alpha"]) * 100
        d_t = r["t_alpha"] - base_a["t_alpha"]
        flag = "✓" if d_a > 1 and d_t > 0.2 else ("↑" if d_a > 0 else "↓")
        print(f"{r['method']:>8s}  {r['N']:>4d}  {r['ann_alpha']:>+9.4f}  {r['t_alpha']:>+8.3f}  "
              f"{r['sharpe']:>+7.3f}  {r['ann_turn']:>6.2f}  {d_a:+.2f}pp / {d_t:+.2f} {flag}")

    print(f"\n[Timed] (vs Full timed α={base_b['ann_alpha']:+.4f}, t={base_b['t_alpha']:+.3f}, Sh={base_b['sharpe']:+.3f})")
    print(f"{'method':>8s}  {'N':>4s}  {'α':>9s}  {'t_α':>8s}  {'Sh':>7s}  {'turn':>6s}  Δα/Δt")
    for _, r in res_df[(res_df["method"] != "Full_EW") & (res_df["timed"])].iterrows():
        d_a = (r["ann_alpha"] - base_b["ann_alpha"]) * 100
        d_t = r["t_alpha"] - base_b["t_alpha"]
        flag = "✓" if d_a > 1 and d_t > 0.2 else ("↑" if d_a > 0 else "↓")
        print(f"{r['method']:>8s}  {r['N']:>4d}  {r['ann_alpha']:>+9.4f}  {r['t_alpha']:>+8.3f}  "
              f"{r['sharpe']:>+7.3f}  {r['ann_turn']:>6.2f}  {d_a:+.2f}pp / {d_t:+.2f} {flag}")

    # === Best per method ===
    print(f"\n" + "=" * 110)
    print("Best-N (timed) for each method")
    print("=" * 110)
    timed_only = res_df[(res_df["method"] != "Full_EW") & (res_df["timed"])]
    for method in ["t_lag", "b_lag", "ratio"]:
        sub = timed_only[timed_only["method"] == method].sort_values("t_alpha", ascending=False)
        if len(sub) == 0: continue
        best = sub.iloc[0]
        d_a = (best["ann_alpha"] - base_b["ann_alpha"]) * 100
        d_t = best["t_alpha"] - base_b["t_alpha"]
        print(f"  {method:>8s}: best N={best['N']}, α={best['ann_alpha']:+.4f}, "
              f"t={best['t_alpha']:+.3f}, Sh={best['sharpe']:+.3f}  Δα={d_a:+.2f}pp Δt={d_t:+.2f}")

    # === Plot ===
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    methods_order = ["t_lag", "b_lag", "ratio"]
    colors = {"t_lag": "#0277bd", "b_lag": "#fb8c00", "ratio": "#d81b60"}
    Ns = N_LIST

    for method in methods_order:
        alphas = [res_df[(res_df["method"] == method) & (res_df["N"] == N) & (res_df["timed"])].iloc[0]["ann_alpha"] for N in Ns]
        ts = [res_df[(res_df["method"] == method) & (res_df["N"] == N) & (res_df["timed"])].iloc[0]["t_alpha"] for N in Ns]
        axes[0].plot(Ns, alphas, "o-", color=colors[method], lw=2, ms=9, label=f"{method} timed")
        axes[1].plot(Ns, ts, "o-", color=colors[method], lw=2, ms=9, label=f"{method} timed")
    axes[0].axhline(base_b["ann_alpha"], color="black", ls="--", alpha=0.5, label=f"Full EW timed (α={base_b['ann_alpha']:.3f})")
    axes[1].axhline(base_b["t_alpha"], color="black", ls="--", alpha=0.5, label=f"Full EW timed (t={base_b['t_alpha']:.2f})")
    axes[1].axhline(1.96, color="black", ls=":", alpha=0.4)
    axes[0].set_xlabel("Top-N"); axes[0].set_ylabel("ann_alpha")
    axes[0].set_title("(A) timed alpha 比較")
    axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)
    axes[1].set_xlabel("Top-N"); axes[1].set_ylabel("t_α")
    axes[1].set_title("(B) timed t_α 比較")
    axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)
    plt.suptitle("Phase 26: D400+D500 三種排序方法 (t / β / β_lag/β_now)", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig_phase26_compare.png"), dpi=130)
    plt.close()
    print(f"\n圖檔: fig_phase26_compare.png")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
