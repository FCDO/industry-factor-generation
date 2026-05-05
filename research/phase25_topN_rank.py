"""
Phase 25: D400/D500 中游個股 walk-forward Top-N rank (取代 Phase 24 threshold)

Phase 24 結論
- threshold t > 1.96 在 OOS 不加 alpha (Δα -0.15pp, Δt -0.65 vs full EW)
- 入選數量極不穩定 (median 10, min 1, max 25), 2018 年僅 2 檔通過
- Selection 增加 vol 但沒帶來 α

Phase 25 改進
- 永遠取 t(β_lag) 排名前 N 檔 — 規避「2018 僅 2 檔」集中風險
- 測 N = 5, 10, 15, 20 看 cardinality sensitivity
- 同時測 timed (Phase 13 EMA timing) 變體
- 與 Phase 24 baseline 並列比較

No-peek 保證: 同 Phase 24 — 5y rolling regression, threshold/rank 都基於 (t-1250, t-1] window

輸出
- phase25_tstat_matrix.csv     : 每月 × 每股 t-stat 矩陣 (re-usable for下一輪)
- phase25_topN_summary.csv     : 各 N 各變體 alpha/Sharpe/turnover
- phase25_strategy_returns.csv : 各策略日報酬
- fig_phase25_N_sweep.png      : N 對 alpha 與 Sharpe 的曲線
- fig_phase25_curves.png       : 累積資本對比
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


def load_daily_ret():
    with open(os.path.join(DB, "etl#adj_close.pickle"), "rb") as f:
        adj = pickle.load(f).set_index("date")
    adj.index = pd.to_datetime(adj.index)
    adj.columns = adj.columns.astype(str)
    keep = [c for c in adj.columns if c.isdigit() and len(c) == 4 and not c.startswith("00")]
    return adj[keep].pct_change(fill_method=None).iloc[1:]


def screen_one(r_i_win, lead_lag_win, market_lag_win):
    y_lag = r_i_win.shift(1)
    df = pd.concat({"y": r_i_win, "x": lead_lag_win, "y_lag": y_lag, "m_lag": market_lag_win}, axis=1).dropna()
    if len(df) < MIN_OBS_IN_WIN:
        return None
    try:
        res = sm.OLS(df["y"], sm.add_constant(df[["x", "y_lag", "m_lag"]])).fit(
            cov_type="HAC", cov_kwds={"maxlags": NW_LAG}
        )
        return float(res.tvalues["x"])
    except Exception:
        return None


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
    print("Phase 25: D400/D500 中游 walk-forward Top-N rank (no peek)")
    print("=" * 110)
    print(f"  Lookback: {LOOKBACK}d  |  N values: {N_LIST}  |  cost: {COST_RT*100:.1f}bp/2 one-way")

    chain = pd.read_csv(os.path.join(ROOT, "industry_chain.csv"), dtype=str)
    daily_ret = load_daily_ret()
    market = daily_ret.mean(axis=1)
    market_lag = market.shift(1)

    # Lead
    d100 = chain[(chain["industry_code"] == "D000") & (chain["sub_code"] == "D100")]["stock_id"].unique().tolist()
    dc00 = chain[(chain["industry_code"] == "D000") & (chain["sub_code"] == "DC00")]["stock_id"].unique().tolist()
    d100_in = [s for s in d100 if s in daily_ret.columns]
    dc00_in = [s for s in dc00 if s in daily_ret.columns]
    lead = 0.5 * daily_ret[d100_in].mean(axis=1) + 0.5 * daily_ret[dc00_in].mean(axis=1)
    lead_lag = lead.shift(1)

    # Universe
    d_mid = chain[(chain["industry_code"] == "D000") &
                   (chain["sub_code"].isin(["D400", "D500"]))][["stock_id", "name", "sub_code"]]
    d_mid = d_mid.drop_duplicates("stock_id", keep="first").reset_index(drop=True)
    universe = sorted([s for s in d_mid["stock_id"] if s in daily_ret.columns])
    print(f"\nUniverse: {len(universe)} 檔")

    rets_uni = daily_ret[universe]

    # Rebal dates
    first_valid = daily_ret.index[LOOKBACK]
    month_ends = pd.Series(daily_ret.index).groupby([daily_ret.index.year, daily_ret.index.month]).max()
    rebal_dates = pd.DatetimeIndex(month_ends.values)
    rebal_dates = rebal_dates[rebal_dates >= first_valid]
    print(f"Rebal dates: {len(rebal_dates)} ({rebal_dates[0].date()} → {rebal_dates[-1].date()})")

    # === Compute t-stat matrix (rebal_date × stock_id) ===
    print(f"\n[1] 計算 t-stat 矩陣 (one-time, reusable)")
    t0 = time.time()
    t_matrix = pd.DataFrame(np.nan, index=rebal_dates, columns=universe)

    for j, d in enumerate(rebal_dates):
        win_start_idx = max(0, daily_ret.index.get_loc(d) - LOOKBACK)
        win_start = daily_ret.index[win_start_idx]

        lead_lag_win = lead_lag.loc[win_start:d]
        market_lag_win = market_lag.loc[win_start:d]

        for sid in universe:
            r_i_win = rets_uni[sid].loc[win_start:d]
            t_val = screen_one(r_i_win, lead_lag_win, market_lag_win)
            if t_val is not None:
                t_matrix.loc[d, sid] = t_val

        if (j + 1) % 24 == 0 or j == len(rebal_dates) - 1:
            elapsed = time.time() - t0
            row_n = (~t_matrix.loc[d].isna()).sum()
            print(f"  [{j+1:>3d}/{len(rebal_dates)}] {d.date()}  eligible={row_n:>2d}  ({elapsed:.0f}s)")

    t_matrix.to_csv(os.path.join(OUT, "phase25_tstat_matrix.csv"), encoding="utf-8-sig")

    # === Build weight matrices for each N ===
    print(f"\n[2] 構建 weight matrices for N ∈ {N_LIST}")
    rebal_list = sorted(rebal_dates)

    weight_dict = {}  # (N, ranking_method) -> DataFrame
    for N in N_LIST:
        W = pd.DataFrame(0.0, index=daily_ret.index, columns=universe)
        for i, d_rebal in enumerate(rebal_list):
            ts = t_matrix.loc[d_rebal].dropna().sort_values(ascending=False)
            if len(ts) < N:
                # fallback: 用所有 eligible
                top = ts.index.tolist()
            else:
                top = ts.head(N).index.tolist()
            next_rebal = rebal_list[i + 1] if i + 1 < len(rebal_list) else daily_ret.index.max() + pd.Timedelta(days=1)
            hold_idx = daily_ret.index[(daily_ret.index > d_rebal) & (daily_ret.index <= next_rebal)]
            if len(top) > 0:
                W.loc[hold_idx, top] = 1.0 / len(top)
        weight_dict[N] = W
        # Diagnostics
        n_avg = (W > 0).sum(axis=1).where(lambda x: x > 0).mean()
        print(f"  N={N}: avg active stocks={n_avg:.1f}")

    # Full universe EW baseline
    W_full = pd.DataFrame(0.0, index=daily_ret.index, columns=universe)
    for d in daily_ret.index[LOOKBACK:]:
        avail = rets_uni.loc[d].dropna().index.tolist()
        if avail:
            W_full.loc[d, avail] = 1.0 / len(avail)

    # Timing signal
    timing = (lead.ewm(span=SMOOTH_N, adjust=False).mean() > 0).astype(float).shift(1).fillna(0)

    # === Compute returns for each N (always-on + timed) ===
    print(f"\n[3] 計算各 N 的 returns")
    eval_start = rebal_list[0] + pd.Timedelta(days=1)

    results = []
    daily_returns_dict = {}

    # Baseline
    g_full, n_full, at_full = perf_from_weights(W_full, rets_uni)
    W_full_t = W_full.mul(timing, axis=0).fillna(0)
    g_full_t, n_full_t, at_full_t = perf_from_weights(W_full_t, rets_uni)
    daily_returns_dict["Full_EW_net"] = n_full
    daily_returns_dict["Full_EW_timed_net"] = n_full_t

    p_full = perf_with_alpha(n_full.loc[eval_start:], market.loc[eval_start:])
    p_full_t = perf_with_alpha(n_full_t.loc[eval_start:], market.loc[eval_start:])
    results.append({"strategy": "Full EW (no select)", "N": "all", "timed": False,
                    **p_full, "ann_turn": at_full})
    results.append({"strategy": "Full EW timed (no select)", "N": "all", "timed": True,
                    **p_full_t, "ann_turn": at_full_t})

    for N in N_LIST:
        W = weight_dict[N]
        g, n, at = perf_from_weights(W, rets_uni)
        W_t = W.mul(timing, axis=0).fillna(0)
        g_t, n_t, at_t = perf_from_weights(W_t, rets_uni)
        daily_returns_dict[f"TopN{N}_net"] = n
        daily_returns_dict[f"TopN{N}_timed_net"] = n_t

        p = perf_with_alpha(n.loc[eval_start:], market.loc[eval_start:])
        p_t = perf_with_alpha(n_t.loc[eval_start:], market.loc[eval_start:])
        results.append({"strategy": f"Top-{N} always-on", "N": N, "timed": False,
                        **p, "ann_turn": at})
        results.append({"strategy": f"Top-{N} timed", "N": N, "timed": True,
                        **p_t, "ann_turn": at_t})

    res_df = pd.DataFrame(results)
    res_df.to_csv(os.path.join(OUT, "phase25_topN_summary.csv"), index=False, encoding="utf-8-sig")

    # === Print summary ===
    print(f"\n" + "=" * 110)
    print(f"策略表現 (eval start = {eval_start.date()}, 扣 {COST_RT*100:.1f}bp/2 = {COST_RT*1e4:.0f}bp RT)")
    print("=" * 110)
    print(f"\n{'Strategy':<32s} {'ann_ret':>9s} {'vol':>7s} {'Sharpe':>7s} "
          f"{'alpha':>9s} {'t_α':>7s} {'max_dd':>9s} {'ann_turn':>9s}")
    base_a = res_df[res_df["strategy"] == "Full EW (no select)"].iloc[0]
    base_b = res_df[res_df["strategy"] == "Full EW timed (no select)"].iloc[0]
    for _, r in res_df.iterrows():
        flag = "  "
        if not r["timed"]:
            d_alpha = (r["ann_alpha"] - base_a["ann_alpha"]) * 100
            d_t = r["t_alpha"] - base_a["t_alpha"]
        else:
            d_alpha = (r["ann_alpha"] - base_b["ann_alpha"]) * 100
            d_t = r["t_alpha"] - base_b["t_alpha"]
        if r["strategy"].startswith("Full"):
            flag = " ◎"
        elif d_alpha > 1.0 and d_t > 0.3:
            flag = " ✓"
        elif d_alpha > 0:
            flag = " ↑"
        else:
            flag = " ↓"
        print(f"{r['strategy']:<32s} {r['ann_ret']:>+9.4f} {r['ann_vol']:>7.4f} "
              f"{r['sharpe']:>+7.3f} {r['ann_alpha']:>+9.4f} {r['t_alpha']:>+7.3f} "
              f"{r['max_dd']:>+9.4f} {r['ann_turn']:>9.2f}{flag} "
              f"Δα={d_alpha:+.2f}pp Δt={d_t:+.2f}")

    # === Δ vs baseline ===
    print(f"\n" + "=" * 110)
    print("Selection Δ alpha (vs Full EW baseline)")
    print("=" * 110)
    print(f"\nAlways-on (vs Full EW α={base_a['ann_alpha']:+.4f}, t={base_a['t_alpha']:+.3f}):")
    for N in N_LIST:
        r = res_df[(res_df["N"] == N) & (~res_df["timed"])].iloc[0]
        d_a = (r["ann_alpha"] - base_a["ann_alpha"]) * 100
        d_t = r["t_alpha"] - base_a["t_alpha"]
        print(f"  N={N:>2d}  α={r['ann_alpha']:+.4f}  Δα={d_a:+.2f}pp  Δt={d_t:+.2f}")

    print(f"\nTimed (vs Full timed α={base_b['ann_alpha']:+.4f}, t={base_b['t_alpha']:+.3f}):")
    for N in N_LIST:
        r = res_df[(res_df["N"] == N) & (res_df["timed"])].iloc[0]
        d_a = (r["ann_alpha"] - base_b["ann_alpha"]) * 100
        d_t = r["t_alpha"] - base_b["t_alpha"]
        print(f"  N={N:>2d}  α={r['ann_alpha']:+.4f}  Δα={d_a:+.2f}pp  Δt={d_t:+.2f}  Sharpe={r['sharpe']:+.3f}")

    # === Save daily returns ===
    pd.DataFrame(daily_returns_dict).to_csv(os.path.join(OUT, "phase25_strategy_returns.csv"),
                                             encoding="utf-8-sig")

    # === Plot 1: N sweep ===
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    Ns = N_LIST
    alphas_off = [res_df[(res_df["N"] == N) & (~res_df["timed"])].iloc[0]["ann_alpha"] for N in Ns]
    alphas_on = [res_df[(res_df["N"] == N) & (res_df["timed"])].iloc[0]["ann_alpha"] for N in Ns]
    sharps_off = [res_df[(res_df["N"] == N) & (~res_df["timed"])].iloc[0]["sharpe"] for N in Ns]
    sharps_on = [res_df[(res_df["N"] == N) & (res_df["timed"])].iloc[0]["sharpe"] for N in Ns]
    ts_off = [res_df[(res_df["N"] == N) & (~res_df["timed"])].iloc[0]["t_alpha"] for N in Ns]
    ts_on = [res_df[(res_df["N"] == N) & (res_df["timed"])].iloc[0]["t_alpha"] for N in Ns]

    axes[0].plot(Ns, alphas_off, "o-", color="#0277bd", lw=2, ms=9, label="Always-on")
    axes[0].plot(Ns, alphas_on, "s-", color="#1b5e20", lw=2, ms=9, label="Timed")
    axes[0].axhline(base_a["ann_alpha"], color="#0277bd", ls="--", alpha=0.5,
                    label=f"Full EW always-on (α={base_a['ann_alpha']:.3f})")
    axes[0].axhline(base_b["ann_alpha"], color="#1b5e20", ls="--", alpha=0.5,
                    label=f"Full EW timed (α={base_b['ann_alpha']:.3f})")
    axes[0].set_xlabel("Top-N")
    axes[0].set_ylabel("ann_alpha (net 50bp)")
    axes[0].set_title("(A) Top-N alpha — selection 是否加 α?")
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)

    axes[1].plot(Ns, ts_off, "o-", color="#0277bd", lw=2, ms=9, label="Always-on")
    axes[1].plot(Ns, ts_on, "s-", color="#1b5e20", lw=2, ms=9, label="Timed")
    axes[1].axhline(base_a["t_alpha"], color="#0277bd", ls="--", alpha=0.5,
                    label=f"Full EW always-on (t={base_a['t_alpha']:.2f})")
    axes[1].axhline(base_b["t_alpha"], color="#1b5e20", ls="--", alpha=0.5,
                    label=f"Full EW timed (t={base_b['t_alpha']:.2f})")
    axes[1].axhline(1.96, color="black", ls=":", alpha=0.4)
    axes[1].set_xlabel("Top-N")
    axes[1].set_ylabel("t(α)")
    axes[1].set_title("(B) Top-N t-stat")
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)
    plt.suptitle("Phase 25: D400+D500 walk-forward Top-N selection", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig_phase25_N_sweep.png"), dpi=130)
    plt.close()
    print(f"\n圖檔: fig_phase25_N_sweep.png")

    # === Plot 2: cum curves ===
    fig, ax = plt.subplots(figsize=(11, 5.5))
    cmap = plt.cm.viridis(np.linspace(0.15, 0.85, len(N_LIST)))
    cum_full_t = (1 + n_full_t.loc[eval_start:].fillna(0)).cumprod()
    ax.plot(cum_full_t, color="black", lw=2.0, alpha=0.85, ls="--",
            label=f"Full EW timed (no select) [{cum_full_t.iloc[-1]:.2f}x] α={base_b['ann_alpha']:.3f} t={base_b['t_alpha']:.2f}")
    for i, N in enumerate(N_LIST):
        n_t_eval = daily_returns_dict[f"TopN{N}_timed_net"].loc[eval_start:]
        cum = (1 + n_t_eval.fillna(0)).cumprod()
        r = res_df[(res_df["N"] == N) & (res_df["timed"])].iloc[0]
        ax.plot(cum, color=cmap[i], lw=1.5, alpha=0.9,
                label=f"Top-{N} timed [{cum.iloc[-1]:.2f}x] α={r['ann_alpha']:.3f} t={r['t_alpha']:.2f}")
    cum_mkt = (1 + market.loc[eval_start:].fillna(0)).cumprod()
    ax.plot(cum_mkt, color="gray", lw=0.8, alpha=0.5, label=f"Market [{cum_mkt.iloc[-1]:.2f}x]")
    ax.set_yscale("log")
    ax.set_title(f"Phase 25: Top-N timed (扣 50bp) vs Full EW timed (Phase 24 baseline)")
    ax.set_xlabel("date")
    ax.set_ylabel("累積資本 (log)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig_phase25_curves.png"), dpi=130)
    plt.close()
    print(f"圖檔: fig_phase25_curves.png")

    # === 最佳 N ===
    print(f"\n" + "=" * 110)
    print("Best N (按 t_alpha)")
    print("=" * 110)
    timed_only = res_df[res_df["timed"] & (res_df["N"] != "all")]
    best = timed_only.sort_values("t_alpha", ascending=False).iloc[0]
    print(f"  Timed best: N={best['N']}, α={best['ann_alpha']:+.4f}, t={best['t_alpha']:+.3f}, "
          f"Sharpe={best['sharpe']:+.3f}, ann_turn={best['ann_turn']:.2f}")
    d_alpha = (best["ann_alpha"] - base_b["ann_alpha"]) * 100
    d_t = best["t_alpha"] - base_b["t_alpha"]
    flag = "✓ 擊敗 baseline" if d_alpha > 0 and d_t > 0 else "❌ 未擊敗 baseline"
    print(f"  vs Full EW timed: Δα={d_alpha:+.2f}pp, Δt={d_t:+.2f}  {flag}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
