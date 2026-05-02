"""
Phase 8: 跨產業策略 訊號平滑 + 持有期 + 扣成本回測

對 Phase 7 兩個策略做扣成本後 Pareto 掃描:
- 策略 A: 半導體 lead 7 下游 timing (long-only)
- 策略 B: 整合 23-pair LS

參數:
- smoothing N (訊號 EMA span): 1, 3, 5, 10, 20
- hold K (持有期, K日後才重評): 1, 5, 10, 20
- cost (round-trip): 0bp, 30bp, 50bp, 80bp, 100bp

台股實務 round-trip ≈ 50-80bp (手續費 0.285% + 證交稅 0.3% + 滑價 0.1-0.3%)

輸出:
- 策略 A 與 B 的 (N, K, cost) 全格點掃描
- 最佳 net Sharpe 參數組合
- Net cumulative wealth curve
"""
from __future__ import annotations

import os
import pickle
import sys

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

ROOT = r"C:\Users\user\OneDrive\桌面\產業因子生成"
DB = r"C:\Users\user\finlab_db"
OUT = os.path.join(ROOT, "research", "output")


def load_pickle_df(name):
    with open(os.path.join(DB, name), "rb") as f:
        df = pickle.load(f)
    df = df.set_index("date"); df.index = pd.to_datetime(df.index)
    df.columns = df.columns.astype(str)
    return df


def filter_common(df):
    keep = [c for c in df.columns if c.isdigit() and len(c) == 4 and not c.startswith("00")]
    return df[keep]


def k_day_holding_position(target_weight: pd.Series, hold_k: int) -> pd.Series:
    """K 日持有: 每 K 天重評一次倉位, 中間維持不變."""
    if hold_k == 1:
        return target_weight
    out = target_weight.copy()
    last = target_weight.iloc[0]
    for i, date in enumerate(target_weight.index):
        if i % hold_k == 0:
            last = target_weight.iloc[i]
        out.iloc[i] = last
    return out


def perf(daily_ret: pd.Series, ann: float = 252) -> dict:
    s = daily_ret.dropna()
    if len(s) < 60: return {}
    cum = (1 + s).cumprod()
    return {
        "n": len(s),
        "ann_ret": s.mean() * ann,
        "ann_vol": s.std() * np.sqrt(ann),
        "sharpe": s.mean() / s.std() * np.sqrt(ann) if s.std() > 0 else np.nan,
        "max_dd": (cum / cum.cummax() - 1).min(),
        "win": (s > 0).mean(),
    }


def perf_with_alpha(daily_ret: pd.Series, market: pd.Series, ann: float = 252) -> dict:
    df = pd.concat([daily_ret.rename("s"), market.rename("b")], axis=1).dropna()
    if len(df) < 60: return {}
    res = sm.OLS(df["s"], sm.add_constant(df["b"])).fit(cov_type="HAC", cov_kwds={"maxlags": 60})
    out = perf(daily_ret, ann)
    out.update({
        "ann_alpha": res.params["const"] * ann,
        "t_alpha": res.tvalues["const"],
        "beta": res.params["b"],
    })
    return out


def strategy_A(panel, downstream, smooth_n, hold_k):
    """半導體 lead 7 下游 long-only timing.
    target weight = 1 if EMA_N(D000) > 0 else 0; 然後 K 日持有.
    """
    semi = panel["D000"]
    down = panel[downstream].mean(axis=1)
    if smooth_n > 1:
        sig = semi.ewm(span=smooth_n, adjust=False).mean()
    else:
        sig = semi
    target = (sig > 0).astype(float).shift(1)  # 隔日才執行
    weight = k_day_holding_position(target, hold_k)
    gross = weight * down
    turnover_one_side = weight.diff().abs().fillna(weight.iloc[0])
    return gross, weight, turnover_one_side, down


def strategy_B(panel, sig_pairs_df, smooth_n, hold_k):
    """整合 |t|>2 cross-industry pair LS.
    對每個 industry B 計算 score_B = Σ_A sign(t_AB) × EMA_N(r_A)
    然後 cross-section rank, long top, short bottom (各 1/3).
    """
    # 構建每 industry B 的 score
    industries = list(panel.columns)
    score = pd.DataFrame(0.0, index=panel.index, columns=industries)
    n_predictors = pd.Series(0, index=industries)
    for _, row in sig_pairs_df.iterrows():
        A, B = row["A"], row["B"]
        if A not in panel.columns or B not in panel.columns: continue
        sign = float(np.sign(row["t_AB"]))
        if smooth_n > 1:
            sig_A = panel[A].ewm(span=smooth_n, adjust=False).mean()
        else:
            sig_A = panel[A]
        # standardize per-A signal so weights are comparable
        sig_norm = (sig_A - sig_A.expanding(60).mean()) / sig_A.expanding(60).std()
        score[B] = score[B].add(sign * sig_norm, fill_value=0)
        n_predictors[B] += 1

    # 只留有 predictor 的 industries
    valid_B = n_predictors[n_predictors > 0].index.tolist()
    score = score[valid_B]
    score = score.shift(1)  # 隔日執行

    # cross-section rank → tertile LS
    pos = pd.DataFrame(0.0, index=panel.index, columns=valid_B)
    for date in score.index:
        s = score.loc[date].dropna()
        if len(s) < 6: continue
        rank = s.rank(ascending=False)
        n = len(s)
        k = max(2, n // 3)
        long_set = rank[rank <= k].index
        short_set = rank[rank >= n - k + 1].index
        if len(long_set) > 0:
            pos.loc[date, long_set] = 1.0 / len(long_set)
        if len(short_set) > 0:
            pos.loc[date, short_set] = -1.0 / len(short_set)

    # K 日持有
    if hold_k > 1:
        for col in pos.columns:
            pos[col] = k_day_holding_position(pos[col], hold_k)

    gross = (pos * panel[valid_B]).sum(axis=1)
    turnover_one_side = pos.diff().abs().sum(axis=1).fillna(pos.iloc[0].abs().sum())
    return gross, pos, turnover_one_side


def main():
    print("[Phase 8] 訊號平滑 + 持有期 + 扣成本回測\n")

    chain = pd.read_csv(os.path.join(ROOT, "industry_chain.csv"), dtype=str)
    industries = chain[["industry_code","industry_name"]].drop_duplicates().set_index("industry_code")["industry_name"].to_dict()

    adj_close = filter_common(load_pickle_df("etl#adj_close.pickle"))
    daily_ret = adj_close.pct_change(fill_method=None).iloc[1:]

    # 構建 panel
    industry_members = {ind: chain[chain["industry_code"] == ind]["stock_id"].unique().tolist()
                        for ind in industries}
    panel = {}
    for ind, members in industry_members.items():
        cols = [s for s in members if s in daily_ret.columns]
        if len(cols) >= 3:
            panel[ind] = daily_ret[cols].mean(axis=1)
    panel = pd.DataFrame(panel)
    market = daily_ret.mean(axis=1)

    DOWNSTREAM = ["G000","H000","I000","J000","F000","L000","5400"]

    # 策略 B 的 pair 集合 (|t|>2, L≥3 from Phase 7c)
    spill = pd.read_csv(os.path.join(OUT, "cross_industry_spillover_D.csv"))
    sig_pairs = spill[(spill["t_AB"].abs() > 2.0) & (spill["L_loose"] >= 3)].copy()
    print(f"  Strategy B 用 |t|>2, L≥3 共 {len(sig_pairs)} pair")

    # === 策略 A 掃描 ===
    print("\n=== 策略 A: 半導體 lead 7 下游 long-only timing ===\n")
    rows_A = []
    for N in [1, 3, 5, 10, 20]:
        for K in [1, 5, 10, 20]:
            gross, weight, turn, hold_ret = strategy_A(panel, DOWNSTREAM, N, K)
            avg_turn = turn.mean()
            for cost_rt in [0.000, 0.0030, 0.0050, 0.0080, 0.0100]:
                cost_one = cost_rt / 2  # turnover one-side cost
                net = gross - turn * cost_one
                p = perf_with_alpha(net, market)
                if not p: continue
                p["smooth_N"] = N; p["hold_K"] = K; p["cost_rt"] = cost_rt
                p["avg_daily_turn"] = avg_turn
                p["ann_turn"] = avg_turn * 252
                rows_A.append(p)
    df_A = pd.DataFrame(rows_A)
    df_A.to_csv(os.path.join(OUT, "phase8_strategy_A.csv"), index=False, encoding="utf-8-sig")

    # 摘要表
    print(f"{'N':>4s} {'K':>4s} {'turn/day':>9s} {'gross_ann':>10s} {'net@50bp':>10s} {'net_sh@50bp':>11s} {'net@80bp':>10s} {'net_sh@80bp':>11s}")
    for N in [1, 3, 5, 10, 20]:
        for K in [1, 5, 10, 20]:
            sub = df_A[(df_A["smooth_N"]==N) & (df_A["hold_K"]==K)]
            r0 = sub[sub["cost_rt"]==0.0].iloc[0]
            r5 = sub[sub["cost_rt"]==0.0050].iloc[0]
            r8 = sub[sub["cost_rt"]==0.0080].iloc[0]
            print(f"{N:>4d} {K:>4d} {r0['avg_daily_turn']:>9.3f} {r0['ann_ret']:>10.4f} "
                  f"{r5['ann_ret']:>10.4f} {r5['sharpe']:>11.3f} "
                  f"{r8['ann_ret']:>10.4f} {r8['sharpe']:>11.3f}")

    print("\n=== 策略 A: net@50bp 排序 Top 10 ===")
    top_A = df_A[df_A["cost_rt"]==0.0050].sort_values("sharpe", ascending=False).head(10)
    print(top_A[["smooth_N","hold_K","ann_ret","sharpe","ann_alpha","t_alpha","avg_daily_turn","ann_turn"]]
          .to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # === 策略 B 掃描 (較慢, 用較小格點) ===
    print("\n=== 策略 B: 整合 23-pair LS ===\n")
    rows_B = []
    for N in [1, 5, 10, 20]:
        for K in [1, 5, 10, 20]:
            gross, pos, turn = strategy_B(panel, sig_pairs, N, K)
            avg_turn = turn.mean()
            for cost_rt in [0.000, 0.0050, 0.0080]:
                cost_one = cost_rt / 2
                net = gross - turn * cost_one
                p = perf_with_alpha(net, market)
                if not p: continue
                p["smooth_N"] = N; p["hold_K"] = K; p["cost_rt"] = cost_rt
                p["avg_daily_turn"] = avg_turn
                p["ann_turn"] = avg_turn * 252
                rows_B.append(p)
    df_B = pd.DataFrame(rows_B)
    df_B.to_csv(os.path.join(OUT, "phase8_strategy_B.csv"), index=False, encoding="utf-8-sig")

    print(f"{'N':>4s} {'K':>4s} {'turn/day':>9s} {'gross_ann':>10s} {'net@50bp':>10s} {'net_sh@50bp':>11s} {'net@80bp':>10s} {'net_sh@80bp':>11s}")
    for N in [1, 5, 10, 20]:
        for K in [1, 5, 10, 20]:
            sub = df_B[(df_B["smooth_N"]==N) & (df_B["hold_K"]==K)]
            if len(sub) == 0: continue
            r0 = sub[sub["cost_rt"]==0.0].iloc[0]
            r5 = sub[sub["cost_rt"]==0.0050].iloc[0]
            r8 = sub[sub["cost_rt"]==0.0080].iloc[0]
            print(f"{N:>4d} {K:>4d} {r0['avg_daily_turn']:>9.3f} {r0['ann_ret']:>10.4f} "
                  f"{r5['ann_ret']:>10.4f} {r5['sharpe']:>11.3f} "
                  f"{r8['ann_ret']:>10.4f} {r8['sharpe']:>11.3f}")

    print("\n=== 策略 B: net@50bp 排序 Top 10 ===")
    top_B = df_B[df_B["cost_rt"]==0.0050].sort_values("sharpe", ascending=False).head(10)
    print(top_B[["smooth_N","hold_K","ann_ret","sharpe","ann_alpha","t_alpha","avg_daily_turn","ann_turn"]]
          .to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # === 累積曲線: 策略 A 最佳 vs gross vs market ===
    best_A = df_A[df_A["cost_rt"]==0.0050].sort_values("sharpe", ascending=False).iloc[0]
    bestN, bestK = int(best_A["smooth_N"]), int(best_A["hold_K"])
    gross, weight, turn, hold_ret = strategy_A(panel, DOWNSTREAM, bestN, bestK)

    plt.figure(figsize=(11, 5.5))
    cum_gross = (1 + gross.dropna()).cumprod()
    cum_50 = (1 + (gross - turn * 0.0025).dropna()).cumprod()
    cum_80 = (1 + (gross - turn * 0.0040).dropna()).cumprod()
    cum_mkt = (1 + market.reindex(gross.dropna().index).fillna(0)).cumprod()
    plt.plot(cum_gross, label=f"Gross (N={bestN}, K={bestK})", lw=1.5)
    plt.plot(cum_50, label="Net @ 50bp round-trip", lw=1.5, alpha=0.85)
    plt.plot(cum_80, label="Net @ 80bp round-trip", lw=1.5, alpha=0.7)
    plt.plot(cum_mkt, label="Market EW", lw=1.3, alpha=0.6)
    plt.yscale("log")
    plt.title(f"半導體 lead 策略: 訊號平滑 EMA={bestN}, 持有 K={bestK} 日 (扣成本)")
    plt.xlabel("date"); plt.ylabel("cumulative wealth")
    plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig_strategy_A_net.png"), dpi=130)
    plt.close()
    print(f"\nA 最佳 (N={bestN}, K={bestK}): gross_ann={best_A['ann_ret']*1+turn.mean()*0.0025*252:.4f} ({best_A['ann_ret']:.4f} net)")

    # 策略 B 累積曲線
    best_B = df_B[df_B["cost_rt"]==0.0050].sort_values("sharpe", ascending=False).iloc[0]
    bestN_b, bestK_b = int(best_B["smooth_N"]), int(best_B["hold_K"])
    gross_b, pos_b, turn_b = strategy_B(panel, sig_pairs, bestN_b, bestK_b)

    plt.figure(figsize=(11, 5.5))
    cum_gross_b = (1 + gross_b.dropna()).cumprod()
    cum_50_b = (1 + (gross_b - turn_b * 0.0025).dropna()).cumprod()
    cum_80_b = (1 + (gross_b - turn_b * 0.0040).dropna()).cumprod()
    plt.plot(cum_gross_b, label=f"Gross (N={bestN_b}, K={bestK_b})", lw=1.5)
    plt.plot(cum_50_b, label="Net @ 50bp", lw=1.5, alpha=0.85)
    plt.plot(cum_80_b, label="Net @ 80bp", lw=1.5, alpha=0.7)
    plt.title(f"整合 23-pair LS: 訊號平滑 EMA={bestN_b}, 持有 K={bestK_b} 日")
    plt.xlabel("date"); plt.ylabel("cumulative wealth")
    plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig_strategy_B_net.png"), dpi=130)
    plt.close()

    # === Pareto: turnover vs net Sharpe ===
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for cost_rt, marker, color in [(0.0050, "o", "C0"), (0.0080, "s", "C1")]:
        sub = df_A[df_A["cost_rt"]==cost_rt]
        axes[0].scatter(sub["ann_turn"], sub["sharpe"], s=40, alpha=0.7,
                        marker=marker, color=color, label=f"cost {cost_rt*100:.0f}bp")
    axes[0].set_xlabel("年化 turnover (one-side)")
    axes[0].set_ylabel("Net Sharpe")
    axes[0].set_title("策略 A (半導體 lead 7 下游) Pareto")
    axes[0].axhline(0, color="gray", lw=0.5)
    axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[0].set_xscale("log")

    for cost_rt, marker, color in [(0.0050, "o", "C0"), (0.0080, "s", "C1")]:
        sub = df_B[df_B["cost_rt"]==cost_rt]
        axes[1].scatter(sub["ann_turn"], sub["sharpe"], s=40, alpha=0.7,
                        marker=marker, color=color, label=f"cost {cost_rt*100:.0f}bp")
    axes[1].set_xlabel("年化 turnover (one-side)")
    axes[1].set_ylabel("Net Sharpe")
    axes[1].set_title("策略 B (整合 23-pair LS) Pareto")
    axes[1].axhline(0, color="gray", lw=0.5)
    axes[1].legend(); axes[1].grid(alpha=0.3)
    axes[1].set_xscale("log")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig_strategy_pareto.png"), dpi=130)
    plt.close()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
