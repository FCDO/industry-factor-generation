"""
Phase 20: Firm-level alpha from D100+0.5·2330 spillover (Cohen-Frazzini 風)

動機 (Phase 13/19)
- D100+0.5·2330 是台股最強 spillover 訊號 (xind alpha 15.26%, t=5.72)
- 但 industry-EW 平均所有 7 下游股票, 沒做 cross-sectional 篩選
- Cohen-Frazzini (2008) "Economic Links and Predictable Returns":
  個股對訊號的 lagged β 不同, β 高者反應延遲更大 → 抓住 β 排序可放大 alpha

Pipeline
1. Lead signal x_t = 0.5·D100(t) + 0.5·2330(t)
2. Per-stock rolling β: β_i(t) = cov_{w}(r_i_t, x_{t-1}) / var_{w}(x_{t-1}), w=250d
   (向量化計算, 全 universe 一次)
3. 月底以 β 排序成 5 個 quintile (Q1 低 → Q5 高)
4. Daily portfolio returns: 持有上月底決定的 quintile 配置
5. 變體:
   A. Q5 always-on            純 cross-sectional alpha
   B. Q5 timed by EMA(x) > 0  Phase 13 timing + firm pick
   C. Q5 - Q1 long-short      market-neutral
6. 對照 industry-EW (Phase 13 winner)

Universe: 7 跨產業下游 (G/H/I/J/F/L/5400) 中的個股, β 估計需 ≥250d 歷史

輸出
- phase20_betas_sample.csv     : 月底 β 排名 sample (前 6 個月)
- phase20_strategy_returns.csv : 各 strategy 日報酬
- phase20_summary.csv          : 各 strategy alpha/Sharpe
- fig_phase20_quintile_curves.png
- fig_phase20_strategy_compare.png
"""
from __future__ import annotations

import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
import statsmodels.api as sm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
for f in ["Microsoft JhengHei", "Microsoft YaHei", "Noto Sans CJK TC", "SimHei"]:
    if any(f in fn.name for fn in font_manager.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [f]
        break
plt.rcParams["axes.unicode_minus"] = False

ROOT = r"C:\Users\engli\OneDrive\桌面\產業因子生成"
DB = r"C:\Users\engli\finlab_db"
OUT = os.path.join(ROOT, "research", "output")

DOWNSTREAM = ["G000", "H000", "I000", "J000", "F000", "L000", "5400"]
ROLL_WIN = 250        # 1 年 rolling β
N_QUINTILES = 5
COST_RT = 0.0050      # 50 bp round-trip
SMOOTH_N = 10         # Phase 13 winner EMA span
HOLD_K = 1            # Phase 13 winner


def load_pickle_df(name):
    with open(os.path.join(DB, name), "rb") as f:
        df = pickle.load(f)
    df = df.set_index("date"); df.index = pd.to_datetime(df.index)
    df.columns = df.columns.astype(str)
    return df


def filter_common(df):
    keep = [c for c in df.columns if c.isdigit() and len(c) == 4 and not c.startswith("00")]
    return df[keep]


def perf_with_alpha(daily_ret, market, ann=252):
    df = pd.concat([daily_ret.rename("s"), market.rename("b")], axis=1).dropna()
    if len(df) < 60:
        return {}
    res = sm.OLS(df["s"], sm.add_constant(df["b"])).fit(cov_type="HAC", cov_kwds={"maxlags": 60})
    s = df["s"]; cum = (1 + s).cumprod()
    return {
        "n": len(s), "ann_ret": s.mean()*ann, "ann_vol": s.std()*np.sqrt(ann),
        "sharpe": s.mean()/s.std()*np.sqrt(ann) if s.std() > 0 else np.nan,
        "max_dd": (cum/cum.cummax() - 1).min(),
        "ann_alpha": res.params["const"]*ann, "t_alpha": res.tvalues["const"],
        "beta": res.params["b"],
    }


def main():
    print("=" * 100)
    print("Phase 20: Firm-level alpha (Cohen-Frazzini 風) — D100+0.5·2330 lagged β")
    print("=" * 100)

    chain = pd.read_csv(os.path.join(ROOT, "industry_chain.csv"), dtype=str)
    adj_close = filter_common(load_pickle_df("etl#adj_close.pickle"))
    daily_ret = adj_close.pct_change(fill_method=None).iloc[1:]
    market = daily_ret.mean(axis=1)

    # === Build lead signal ===
    d100_members = chain[(chain["industry_code"] == "D000") &
                          (chain["sub_code"] == "D100")]["stock_id"].unique().tolist()
    d100_cols = [s for s in d100_members if s in daily_ret.columns]
    d100 = daily_ret[d100_cols].mean(axis=1)
    tsmc = daily_ret["2330"]
    lead = 0.5 * d100 + 0.5 * tsmc
    lead_lag = lead.shift(1)  # x_{t-1}
    print(f"\nLead signal: 0.5·D100({len(d100_cols)} 檔) + 0.5·2330")

    # === Universe: stocks in 7 跨產業下游 ===
    uni_set = set()
    for ic in DOWNSTREAM:
        members = chain[chain["industry_code"] == ic]["stock_id"].unique()
        uni_set.update(members)
    # 排除 D100 + 2330 自己 (避免自我預測)
    uni_set -= set(d100_cols)
    uni_set -= {"2330"}
    universe = sorted([s for s in uni_set if s in daily_ret.columns])
    print(f"Universe: {len(universe)} 檔 (7 下游 ∖ {{D100, 2330}})")

    rets_uni = daily_ret[universe]

    # === 向量化 rolling β: β_i(t) = cov(r_i, lead_lag) / var(lead_lag) ===
    # 使用公式 cov = E[xy] - E[x]E[y]
    print(f"\n[1] 計算 rolling β (window={ROLL_WIN}d)...")
    t0 = time.time()
    # E[x] : Series (T,)
    E_x = lead_lag.rolling(ROLL_WIN).mean()
    E_x2 = (lead_lag ** 2).rolling(ROLL_WIN).mean()
    var_x = E_x2 - E_x ** 2  # Series

    # E[r_i] : DataFrame (T, N)
    E_r = rets_uni.rolling(ROLL_WIN).mean()
    # E[r_i * x_lag] : DataFrame
    rx = rets_uni.mul(lead_lag, axis=0)
    E_rx = rx.rolling(ROLL_WIN).mean()
    # cov = E[rx] - E[r]·E[x]
    cov_rx = E_rx.sub(E_r.mul(E_x, axis=0))
    # β = cov / var
    betas = cov_rx.div(var_x, axis=0)
    print(f"  shape: {betas.shape}, 耗時 {time.time()-t0:.1f}s")

    # === 月底 rebalance ===
    print(f"\n[2] 月底取 {N_QUINTILES} 個 quintile")
    month_ends = pd.Series(daily_ret.index).groupby([daily_ret.index.year, daily_ret.index.month]).max()
    month_ends = pd.DatetimeIndex(month_ends.values)
    valid_months = month_ends[month_ends >= betas.first_valid_index() + pd.Timedelta(days=10)]
    print(f"  rebalance 次數: {len(valid_months)}")

    # 每個月底, 切 quintile 並記錄 membership
    quintile_membership = {}  # date -> dict{q: list of stocks}
    for d in valid_months:
        b = betas.loc[d].dropna()
        if len(b) < N_QUINTILES * 5:  # need at least N*5 stocks
            continue
        # cut into N quintiles
        try:
            ranks = pd.qcut(b, N_QUINTILES, labels=False)
        except ValueError:
            ranks = pd.qcut(b.rank(method="first"), N_QUINTILES, labels=False)
        membership = {q: ranks[ranks == q].index.tolist() for q in range(N_QUINTILES)}
        quintile_membership[d] = membership

    # === Build daily weight matrices (proper for turnover & cost) ===
    print(f"[3] 構建每日 weight matrix")
    rebal_dates = sorted(quintile_membership.keys())

    # 5 個 quintile 的 weight DataFrame: shape (T, n_stocks_in_universe)
    q_weights = {q: pd.DataFrame(0.0, index=daily_ret.index, columns=universe)
                 for q in range(N_QUINTILES)}
    for i, d_rebal in enumerate(rebal_dates):
        next_rebal = rebal_dates[i + 1] if i + 1 < len(rebal_dates) else daily_ret.index.max() + pd.Timedelta(days=1)
        hold_idx = daily_ret.index[(daily_ret.index > d_rebal) & (daily_ret.index <= next_rebal)]
        for q in range(N_QUINTILES):
            stocks_q = quintile_membership[d_rebal][q]
            if len(stocks_q) == 0:
                continue
            w = 1.0 / len(stocks_q)
            q_weights[q].loc[hold_idx, stocks_q] = w

    # === Timing indicator (Phase 13 framework) ===
    sig = lead.ewm(span=SMOOTH_N, adjust=False).mean()
    timing = (sig > 0).astype(float).shift(1)  # 隔日執行

    # === Compute gross + net (with PROPER turnover from weight matrix) ===
    def perf_from_weights(W, daily_ret_uni, market, label):
        """W = weight DataFrame, daily_ret_uni = stock returns DataFrame (same columns)."""
        gross = (W * daily_ret_uni).sum(axis=1)
        # one-side daily turnover = sum |dW|
        turn = W.diff().abs().sum(axis=1)
        turn.iloc[0] = W.iloc[0].abs().sum()  # initial position
        cost = turn * (COST_RT / 2.0)
        net = gross - cost
        ann_turn = turn.mean() * 252
        return gross, net, turn, ann_turn

    print(f"\n[4] 計算各策略 gross + net (proper turnover)")

    # Always-on quintiles
    Q1_g, Q1_n, Q1_t, Q1_at = perf_from_weights(q_weights[0], rets_uni, market, "Q1")
    Q3_g, Q3_n, Q3_t, Q3_at = perf_from_weights(q_weights[2], rets_uni, market, "Q3")
    Q5_g, Q5_n, Q5_t, Q5_at = perf_from_weights(q_weights[N_QUINTILES-1], rets_uni, market, "Q5")
    print(f"  Q1 ann_turnover = {Q1_at:.3f}, Q3 = {Q3_at:.3f}, Q5 = {Q5_at:.3f}")

    # Q5 timed
    W_q5_timed = q_weights[N_QUINTILES-1].mul(timing, axis=0).fillna(0)
    Q5T_g, Q5T_n, Q5T_t, Q5T_at = perf_from_weights(W_q5_timed, rets_uni, market, "Q5_timed")
    print(f"  Q5 timed ann_turnover = {Q5T_at:.3f}")

    # Q5-Q1 long-short
    W_ls = q_weights[N_QUINTILES-1] - q_weights[0]
    LS_g, LS_n, LS_t, LS_at = perf_from_weights(W_ls, rets_uni, market, "Q5-Q1")

    # Q5-Q1 timed
    W_ls_timed = W_ls.mul(timing, axis=0).fillna(0)
    LST_g, LST_n, LST_t, LST_at = perf_from_weights(W_ls_timed, rets_uni, market, "Q5-Q1_timed")
    print(f"  Q5-Q1 ann_turn = {LS_at:.3f}, Q5-Q1 timed = {LST_at:.3f}")

    # EW universe (always-on full universe)
    W_ew = pd.DataFrame(1.0/len(universe), index=daily_ret.index, columns=universe)
    EW_g, EW_n, EW_t, EW_at = perf_from_weights(W_ew, rets_uni, market, "EW")

    # EW universe timed (Phase 13 reference)
    W_ew_timed = W_ew.mul(timing, axis=0).fillna(0)
    EWT_g, EWT_n, EWT_t, EWT_at = perf_from_weights(W_ew_timed, rets_uni, market, "EW_timed")
    print(f"  EW timed ann_turnover = {EWT_at:.3f} (應接近 Phase 13 之 0.155×252 ≈ 39)")

    # 把所有結果整理
    qA = Q5_g; qA_net = Q5_n
    qB = Q5T_g; qB_net = Q5T_n
    qC = LS_g; qC_net = LS_n
    qC_timed = LST_g; qC_timed_net = LST_n
    ew_uni = EW_g
    ew_timed = EWT_g; ew_timed_net = EWT_n
    q_df = pd.DataFrame({f"Q{q+1}": (q_weights[q] * rets_uni).sum(axis=1)
                          for q in range(N_QUINTILES)})

    # === Performance summary ===
    print(f"\n" + "=" * 100)
    print("策略表現 (扣 50bp from PROPER weight-matrix turnover)")
    print("=" * 100)

    strategies = {
        "Q1 always-on (low β)":   Q1_n,
        "Q3 always-on (mid β)":   Q3_n,
        "Q5 always-on (high β)":  qA_net,
        "Q5 timed (EMA>0)":       qB_net,
        "Q5-Q1 long-short":       qC_net,
        "Q5-Q1 long-short timed": qC_timed_net,
        "EW universe":            EW_n,
        "EW universe timed (Phase 13 ref)": ew_timed_net,
    }
    # Also save gross for sanity check
    strategies_gross = {
        "Q5 timed gross":  qB,
        "EW timed gross":  ew_timed,
    }

    summary = []
    fmt = lambda x: f"{x:+.4f}"
    print(f"\n{'Strategy':<36s} {'ann_ret':>10s} {'ann_vol':>10s} {'Sharpe':>8s} "
          f"{'alpha':>9s} {'t_α':>8s} {'max_dd':>9s}")
    for label, ret in strategies.items():
        p = perf_with_alpha(ret, market)
        if not p:
            continue
        print(f"{label:<36s} {p['ann_ret']:>+10.4f} {p['ann_vol']:>10.4f} "
              f"{p['sharpe']:>+8.3f} {p['ann_alpha']:>+9.4f} {p['t_alpha']:>+8.3f} "
              f"{p['max_dd']:>+9.4f}")
        summary.append({"strategy": label, **p})

    pd.DataFrame(summary).to_csv(os.path.join(OUT, "phase20_summary.csv"),
                                  index=False, encoding="utf-8-sig")

    # Gross sanity check
    print(f"\n>>> Gross (扣前) sanity check:")
    for label, ret in strategies_gross.items():
        p = perf_with_alpha(ret, market)
        print(f"  {label:<24s} ann_ret={p['ann_ret']:>+.4f} alpha={p['ann_alpha']:>+.4f} t_α={p['t_alpha']:>+.3f}")

    # === Q1..Q5 progression check (確認單調性) ===
    print(f"\n>>> Q1→Q5 alpha 單調性檢驗 (應該 Q5 > Q1 if Cohen-Frazzini 正確):")
    for q in range(N_QUINTILES):
        ret = q_df[f"Q{q+1}"]
        p = perf_with_alpha(ret, market)
        print(f"  Q{q+1}: ann_ret={p['ann_ret']:+.4f}, alpha={p['ann_alpha']:+.4f}, t_α={p['t_alpha']:+.3f}")

    # === Visualization 1: 累積曲線 ===
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    # Q1..Q5 always-on
    colors_q = plt.cm.viridis(np.linspace(0.1, 0.9, N_QUINTILES))
    for q in range(N_QUINTILES):
        ret = q_df[f"Q{q+1}"].fillna(0)
        cum = (1 + ret).cumprod()
        axes[0].plot(cum, color=colors_q[q], lw=1.4, alpha=0.9,
                     label=f"Q{q+1} ({'low β' if q == 0 else 'high β' if q == N_QUINTILES-1 else 'mid'}) [{cum.iloc[-1]:.1f}x]")
    cum_mkt = (1 + market.fillna(0)).cumprod()
    axes[0].plot(cum_mkt, color="gray", lw=1.0, alpha=0.6, label=f"Market EW [{cum_mkt.iloc[-1]:.1f}x]")
    axes[0].set_yscale("log")
    axes[0].set_title("(A) Q1..Q5 always-on (β-sorted quintile)")
    axes[0].set_xlabel("date"); axes[0].set_ylabel("累積資本 (log)")
    axes[0].legend(loc="upper left", fontsize=9)
    axes[0].grid(alpha=0.3)

    # Strategy comparison
    plot_strats = {
        "Q5 timed (EMA>0)":       qB_net,
        "Q5 always-on":           qA_net,
        "Q5-Q1 long-short timed": qC_timed_net,
        "EW universe timed (Phase 13 ref)": ew_timed_net,
    }
    cmap_strat = {"Q5 timed (EMA>0)": "#1b5e20",
                   "Q5 always-on": "#0277bd",
                   "Q5-Q1 long-short timed": "#6a1b9a",
                   "EW universe timed (Phase 13 ref)": "#ef6c00"}
    for label, ret in plot_strats.items():
        cum = (1 + ret.fillna(0)).cumprod()
        axes[1].plot(cum, color=cmap_strat.get(label, "black"), lw=1.5, alpha=0.9,
                     label=f"{label} [{cum.iloc[-1]:.1f}x]")
    axes[1].plot(cum_mkt, color="gray", lw=1.0, alpha=0.5, label=f"Market [{cum_mkt.iloc[-1]:.1f}x]")
    axes[1].set_yscale("log")
    axes[1].set_title("(B) firm-level vs Phase 13 industry-EW (扣 50bp est.)")
    axes[1].set_xlabel("date"); axes[1].set_ylabel("累積資本 (log)")
    axes[1].legend(loc="upper left", fontsize=8.5)
    axes[1].grid(alpha=0.3)
    plt.suptitle("Phase 20: Firm-level alpha (β-sorted) vs Industry-EW", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig_phase20_quintile_curves.png"), dpi=130)
    plt.close()
    print(f"\n圖檔: fig_phase20_quintile_curves.png")

    # === Visualization 2: Q1..Q5 alpha bar ===
    q_alphas = []
    q_ts = []
    for q in range(N_QUINTILES):
        p = perf_with_alpha(q_df[f"Q{q+1}"], market)
        q_alphas.append(p["ann_alpha"]); q_ts.append(p["t_alpha"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].bar(range(1, N_QUINTILES+1), q_alphas, color=colors_q,
                edgecolor="black", lw=0.4)
    axes[0].axhline(0, color="black", lw=0.5)
    axes[0].set_xlabel("Quintile (1=low β, 5=high β)")
    axes[0].set_ylabel("ann_alpha")
    axes[0].set_title("(A) Quintile alpha (always-on, before cost)")
    axes[0].grid(axis="y", alpha=0.3)
    axes[1].bar(range(1, N_QUINTILES+1), q_ts, color=colors_q,
                edgecolor="black", lw=0.4)
    axes[1].axhline(1.96, color="black", lw=0.5, ls="--", alpha=0.5)
    axes[1].axhline(-1.96, color="black", lw=0.5, ls="--", alpha=0.5)
    axes[1].axhline(0, color="black", lw=0.5)
    axes[1].set_xlabel("Quintile")
    axes[1].set_ylabel("t_alpha")
    axes[1].set_title("(B) Quintile t_alpha")
    axes[1].grid(axis="y", alpha=0.3)
    plt.suptitle("Phase 20: Q1→Q5 單調性檢驗", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig_phase20_quintile_alphas.png"), dpi=130)
    plt.close()
    print(f"圖檔: fig_phase20_quintile_alphas.png")

    # === Sample β snapshots ===
    sample_dates = rebal_dates[::max(1, len(rebal_dates) // 6)][:6]
    sample_rows = []
    for d in sample_dates:
        b = betas.loc[d].dropna().sort_values(ascending=False)
        for stock_id, beta_val in b.head(5).items():
            sample_rows.append({"date": d.date(), "rank": "top5", "stock_id": stock_id, "beta": beta_val})
        for stock_id, beta_val in b.tail(5).items():
            sample_rows.append({"date": d.date(), "rank": "bottom5", "stock_id": stock_id, "beta": beta_val})
    pd.DataFrame(sample_rows).to_csv(os.path.join(OUT, "phase20_betas_sample.csv"),
                                      index=False, encoding="utf-8-sig")

    # Save daily strategy returns
    out_df = pd.DataFrame({
        "Q1": q_df["Q1"], "Q3": q_df["Q3"], "Q5": q_df["Q5"],
        "Q5_timed": qB, "Q5_minus_Q1": qC, "Q5_minus_Q1_timed": qC_timed,
        "EW_universe": rets_uni.mean(axis=1),
        "EW_universe_timed": ew_uni * timing,
        "lead_signal": lead, "timing_indicator": timing,
    })
    out_df.to_csv(os.path.join(OUT, "phase20_strategy_returns.csv"),
                  encoding="utf-8-sig")

    # === 結論 ===
    print(f"\n" + "=" * 100)
    print("結論摘要")
    print("=" * 100)
    summary_df = pd.DataFrame(summary)
    base = summary_df[summary_df["strategy"] == "EW universe timed (Phase 13 ref)"].iloc[0]
    print(f"\nbaseline (Phase 13 EW timed): alpha={base['ann_alpha']:.4f}, t={base['t_alpha']:.3f}, Sh={base['sharpe']:.3f}")
    for _, r in summary_df.iterrows():
        if r["strategy"] == "EW universe timed (Phase 13 ref)":
            continue
        d_alpha = (r["ann_alpha"] - base["ann_alpha"]) * 100
        d_t = r["t_alpha"] - base["t_alpha"]
        flag = " ✓" if d_alpha > 1.0 and d_t > 0.3 else (" ↑" if d_alpha > 0 else " ↓")
        print(f"  {r['strategy']:<36s} α={r['ann_alpha']:+.4f}  t={r['t_alpha']:+.3f}  "
              f"Sh={r['sharpe']:+.3f}  Δα={d_alpha:+.2f}pp  Δt={d_t:+.2f}{flag}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
