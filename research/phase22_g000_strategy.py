"""
Phase 22: G000 平面顯示器 完整策略規劃 — 4 層訊號疊加 + walk-forward

訊號源 (依 Phase 11/13/15/18 證據)
1. 鏈內純 leader: GA00 顯示器其他零組件 (Phase 11b)
2. 鏈內整體上游: G000 上游 EW (Phase 11a baseline)
3. 跨產業正向: D100+0.5·2330 (Phase 13 winner, ρ_lag(D000,G000)=4.02)
4. 跨產業負向: Q000 鋼鐵 (t_AB=-4.07, 成本傳導)

Predictor variants (10 個)
A. G000 上游 EW                         Phase 11a baseline
B. GA00                                 純 leader (Step 1: 提純)
C. 整體 G000                              鏈整體 (對照)
D. D100+0.5·2330                        Phase 13 winner (跨產業 reference)
E. 0.7·GA00 + 0.3·(D100+2330_macro)     Step 2 雙鏈整合 (GA00 主)
F. 0.5·GA00 + 0.5·(D100+2330_macro)     Step 2 平衡組合
G. 0.3·GA00 + 0.7·(D100+2330_macro)     Step 2 macro 主
H. GA00 - 0.3·Q000                      Step 3 加反向訊號
I. 0.5·GA00 + 0.3·(D100+2330) - 0.2·Q000 Step 3 三因子
J. (D100+0.5·2330) - 0.3·Q000           Step 3 跨產業 + 反向

Targets
1. G000 中下游 EW                       主 target
2. 整體 G000 EW                          對照 target

Sweep N×K×cost
Walk-forward: top-3 predictor 跑 5y/1y rolling

Output
- phase22_g000_strategy.csv  : full sweep
- phase22_walkforward.csv    : top-3 walk-forward
- fig_phase22_predictor_comparison.png
- fig_phase22_curves.png
- fig_phase22_walkforward.png
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
        plt.rcParams["font.sans-serif"] = [f]
        break
plt.rcParams["axes.unicode_minus"] = False

ROOT = r"C:\Users\engli\OneDrive\桌面\產業因子生成"
DB = r"C:\Users\engli\finlab_db"
OUT = os.path.join(ROOT, "research", "output")

PARAMS_N = [1, 3, 5, 10, 20]
PARAMS_K = [1, 5, 10, 20]
COSTS = [0.000, 0.0030, 0.0050, 0.0080, 0.0100]
WF_TRAIN_YEARS = 5


def load_pickle_df(name):
    with open(os.path.join(DB, name), "rb") as f:
        df = pickle.load(f)
    df = df.set_index("date"); df.index = pd.to_datetime(df.index)
    df.columns = df.columns.astype(str)
    return df


def filter_common(df):
    keep = [c for c in df.columns if c.isdigit() and len(c) == 4 and not c.startswith("00")]
    return df[keep]


def k_day_holding(target, hold_k):
    if hold_k == 1:
        return target
    out = target.copy(); last = target.iloc[0]
    for i in range(len(target)):
        if i % hold_k == 0:
            last = target.iloc[i]
        out.iloc[i] = last
    return out


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


def sharpe_simple(s):
    s = s.dropna()
    if len(s) < 30 or s.std() == 0:
        return np.nan
    return s.mean() / s.std() * np.sqrt(252)


def strategy(predictor, target, smooth_n, hold_k):
    sig = predictor.ewm(span=smooth_n, adjust=False).mean() if smooth_n > 1 else predictor
    target_w = (sig > 0).astype(float).shift(1)
    weight = k_day_holding(target_w, hold_k)
    gross = weight * target
    turn = weight.diff().abs().fillna(weight.iloc[0])
    return gross, weight, turn


def main():
    print("=" * 100)
    print("Phase 22: G000 平面顯示器 完整策略規劃")
    print("=" * 100)

    chain = pd.read_csv(os.path.join(ROOT, "industry_chain.csv"), dtype=str)
    adj_close = filter_common(load_pickle_df("etl#adj_close.pickle"))
    daily_ret = adj_close.pct_change(fill_method=None).iloc[1:]
    market = daily_ret.mean(axis=1)

    # === Build portfolios ===
    def build(industry, sub=None, position=None):
        d = chain[chain["industry_code"] == industry]
        if sub:
            d = d[d["sub_code"] == sub]
        if position:
            d = d[d["position"] == position]
        members = [s for s in d["stock_id"].unique() if s in daily_ret.columns]
        if len(members) < 1:
            return None, 0
        return daily_ret[members].mean(axis=1), len(members)

    # G000 部分
    g_overall, n_g_overall = build("G000")
    g_up, n_g_up = build("G000", position="上游")
    g_md_set = chain[(chain["industry_code"] == "G000") &
                      (chain["position"].isin(["中游", "下游"]))]["stock_id"].unique()
    g_md_cols = [s for s in g_md_set if s in daily_ret.columns]
    g_md = daily_ret[g_md_cols].mean(axis=1)
    ga00, n_ga00 = build("G000", sub="GA00")
    print(f"\n[G000] 整體: {n_g_overall} 檔, 上游: {n_g_up} 檔, 中下游: {len(g_md_cols)} 檔, GA00: {n_ga00} 檔")

    # D100 + 2330
    d100, n_d100 = build("D000", sub="D100")
    tsmc = daily_ret["2330"]
    d_macro = 0.5 * d100 + 0.5 * tsmc
    print(f"[D000] D100 IC設計: {n_d100} 檔, 2330 個股")

    # Q000
    q000, n_q000 = build("Q000")
    print(f"[Q000] 鋼鐵: {n_q000} 檔")

    # === Sanity: 同期相關 ===
    df_corr = pd.concat({
        "GA00": ga00, "G_up": g_up, "G_overall": g_overall,
        "D100": d100, "2330": tsmc, "D_macro": d_macro,
        "Q000": q000, "G_md": g_md,
    }, axis=1).dropna()
    print(f"\n[同期相關] 訊號源 vs G000 中下游 (G_md):")
    print(df_corr.corr()["G_md"].round(3).to_string())

    # === Predictor variants ===
    predictors = {
        "A_G_up":             g_up,
        "B_GA00":             ga00,
        "C_G_overall":        g_overall,
        "D_D_macro":          d_macro,
        "E_0.7GA+0.3D":       0.7 * ga00 + 0.3 * d_macro,
        "F_0.5GA+0.5D":       0.5 * ga00 + 0.5 * d_macro,
        "G_0.3GA+0.7D":       0.3 * ga00 + 0.7 * d_macro,
        "H_GA-0.3Q":          ga00 - 0.3 * q000,
        "I_3factor":          0.5 * ga00 + 0.3 * d_macro - 0.2 * q000,
        "J_D-0.3Q":           d_macro - 0.3 * q000,
    }
    targets = {"G_md": g_md, "G_overall": g_overall}

    # === Full sweep ===
    print(f"\n" + "=" * 100)
    print(f"[1] Strategy sweep ({len(predictors)} predictors × {len(targets)} targets × N×K×cost)")
    print("=" * 100)

    rows = []
    for tgt_label, tgt in targets.items():
        for pred_label, pred in predictors.items():
            for N in PARAMS_N:
                for K in PARAMS_K:
                    gross, weight, turn = strategy(pred, tgt, N, K)
                    avg_turn = turn.mean()
                    for cost_rt in COSTS:
                        net = gross - turn * (cost_rt / 2)
                        p = perf_with_alpha(net, market)
                        if not p:
                            continue
                        p.update({"target": tgt_label, "predictor": pred_label,
                                  "smooth_N": N, "hold_K": K, "cost_rt": cost_rt,
                                  "avg_daily_turn": avg_turn, "ann_turn": avg_turn * 252})
                        rows.append(p)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "phase22_g000_strategy.csv"),
              index=False, encoding="utf-8-sig")

    fmt = lambda x: f"{x:.4f}"
    cols_show = ["predictor", "smooth_N", "hold_K", "ann_ret", "ann_vol",
                  "sharpe", "ann_alpha", "t_alpha", "max_dd", "avg_daily_turn"]

    for tlabel in ["G_md", "G_overall"]:
        title = "G000 中下游 (主 target)" if tlabel == "G_md" else "整體 G000 (對照)"
        print(f"\n>>> Target {tlabel}: {title} — best @ 50bp 排序 (按 t_alpha)")
        sub = df[(df["target"] == tlabel) & (df["cost_rt"] == 0.0050)]
        best = (sub.sort_values("sharpe", ascending=False)
                .groupby("predictor", as_index=False).first()
                .sort_values("t_alpha", ascending=False))
        print(best[cols_show].to_string(index=False, float_format=fmt))

    # === @80bp 高成本 ===
    print(f"\n>>> Target G_md: best @ 80bp 高成本場景")
    sub80 = df[(df["target"] == "G_md") & (df["cost_rt"] == 0.0080)]
    best80 = (sub80.sort_values("sharpe", ascending=False)
              .groupby("predictor", as_index=False).first()
              .sort_values("t_alpha", ascending=False))
    print(best80[cols_show].to_string(index=False, float_format=fmt))

    # === 固定 (N=20, K=1) Phase 11a winner 同參數 ===
    print(f"\n>>> 固定 (N=20, K=1) [Phase 11a winner] @ 50bp on G_md:")
    sub_p11 = df[(df["target"] == "G_md") & (df["smooth_N"] == 20) &
                  (df["hold_K"] == 1) & (df["cost_rt"] == 0.0050)]\
              .sort_values("t_alpha", ascending=False)
    for _, r in sub_p11.iterrows():
        print(f"  {r['predictor']:<18s}  ann_ret={r['ann_ret']:>+.4f}  "
              f"sharpe={r['sharpe']:>+.3f}  alpha={r['ann_alpha']:>+.4f}  t_α={r['t_alpha']:>+.3f}")

    # === Walk-forward on top 3 predictors ===
    print(f"\n" + "=" * 100)
    print(f"[2] Walk-forward 驗證 top-3 predictors (5y train / 1y test)")
    print("=" * 100)

    # 取 G_md @ 50bp 的 top 3 predictor (按 sharpe)
    top3 = (df[(df["target"] == "G_md") & (df["cost_rt"] == 0.0050)]
            .sort_values("sharpe", ascending=False)
            .groupby("predictor", as_index=False).first()
            .sort_values("sharpe", ascending=False)
            .head(3)["predictor"].tolist())
    print(f"\nTop-3 by Sharpe @ 50bp: {top3}")

    # Pre-compute net returns for all (N, K) of top-3 predictors
    cache = {}
    for pred_label in top3:
        cache[pred_label] = {}
        pred = predictors[pred_label]
        for N in PARAMS_N:
            for K in PARAMS_K:
                gross, _, turn = strategy(pred, g_md, N, K)
                net = gross - turn * (0.0050 / 2)  # @50bp
                cache[pred_label][(N, K)] = net

    yrs = sorted(set(daily_ret.index.year))
    yr_min = min(yrs)
    test_years = [y for y in yrs if y - WF_TRAIN_YEARS >= yr_min]

    wf_rows = []
    oos_returns = {p: pd.Series(dtype=float) for p in top3}
    for test_y in test_years:
        train_start = pd.Timestamp(year=test_y - WF_TRAIN_YEARS, month=1, day=1)
        train_end = pd.Timestamp(year=test_y, month=1, day=1)
        test_end = pd.Timestamp(year=test_y + 1, month=1, day=1)
        for pred_label in top3:
            best_NK = None
            best_train_sh = -np.inf
            for (N, K), net in cache[pred_label].items():
                tr = net[(net.index >= train_start) & (net.index < train_end)]
                sh = sharpe_simple(tr)
                if not np.isnan(sh) and sh > best_train_sh:
                    best_train_sh = sh
                    best_NK = (N, K)
            te = cache[pred_label][best_NK][
                (cache[pred_label][best_NK].index >= train_end) &
                (cache[pred_label][best_NK].index < test_end)
            ]
            test_sh = sharpe_simple(te)
            test_ann = te.mean() * 252 if len(te) > 0 else np.nan
            wf_rows.append({
                "predictor": pred_label, "test_year": test_y,
                "best_N": best_NK[0], "best_K": best_NK[1],
                "train_sharpe": best_train_sh, "test_sharpe": test_sh,
                "test_ann_ret": test_ann, "test_n_days": len(te),
            })
            oos_returns[pred_label] = pd.concat([oos_returns[pred_label], te])

    wf_df = pd.DataFrame(wf_rows)
    wf_df.to_csv(os.path.join(OUT, "phase22_walkforward.csv"),
                 index=False, encoding="utf-8-sig")

    print(f"\n>>> Walk-forward 結果")
    print(f"{'predictor':<22s} {'WF_ann':>9s} {'WF_Sh':>8s} {'WF_α':>9s} {'WF_t_α':>9s} {'IS_α_OOS':>10s}")
    summary_wf = []
    for pred_label in top3:
        oos_ret = oos_returns[pred_label].sort_index()
        market_oos = market.reindex(oos_ret.index)
        p_oos = perf_with_alpha(oos_ret, market_oos)
        # In-sample winner on the OOS period for fair comparison
        is_perfs = [(sharpe_simple(net), nk, net) for nk, net in cache[pred_label].items()]
        is_perfs.sort(key=lambda x: -x[0] if not np.isnan(x[0]) else -1e9)
        is_NK = is_perfs[0][1]
        is_oos_period = cache[pred_label][is_NK][
            (cache[pred_label][is_NK].index >= oos_ret.index.min()) &
            (cache[pred_label][is_NK].index <= oos_ret.index.max())
        ]
        p_is = perf_with_alpha(is_oos_period, market.reindex(is_oos_period.index))
        print(f"{pred_label:<22s} {p_oos['ann_ret']:>+9.4f} {p_oos['sharpe']:>+8.3f} "
              f"{p_oos['ann_alpha']:>+9.4f} {p_oos['t_alpha']:>+9.3f} {p_is['ann_alpha']:>+10.4f}")
        summary_wf.append({"predictor": pred_label, **p_oos,
                           "IS_winner_NK": str(is_NK),
                           "IS_OOS_alpha": p_is.get("ann_alpha"),
                           "IS_OOS_t": p_is.get("t_alpha")})

    # === Visualization 1: best 各 predictor 比較 bar (G_md target) ===
    fig, ax = plt.subplots(figsize=(11, 6))
    sub = df[(df["target"] == "G_md") & (df["cost_rt"] == 0.0050)]
    best = (sub.sort_values("sharpe", ascending=False)
            .groupby("predictor", as_index=False).first()
            .sort_values("t_alpha"))
    colors = ["#1b5e20" if t > 5 else "#558b2f" if t > 3 else
              "#fb8c00" if t > 1.96 else "#9e9e9e" for t in best["t_alpha"]]
    ax.barh(best["predictor"], best["t_alpha"], color=colors, edgecolor="black", lw=0.4)
    for i, (t, alpha, pred) in enumerate(zip(best["t_alpha"], best["ann_alpha"], best["predictor"])):
        ax.text(t + 0.08, i, f"α={alpha:+.3f}", va="center", fontsize=8.5)
    ax.axvline(1.96, color="black", lw=0.5, ls="--", alpha=0.5)
    ax.axvline(0, color="black", lw=0.5)
    ax.set_xlabel("t_alpha (best (N,K) @ 50bp)")
    ax.set_title("Phase 22: G000 中下游 — 10 predictor 變體比較")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig_phase22_predictor_comparison.png"), dpi=130)
    plt.close()
    print(f"\n圖檔: fig_phase22_predictor_comparison.png")

    # === Visualization 2: 累積曲線 ===
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    plot_preds = ["A_G_up", "B_GA00", "F_0.5GA+0.5D", "I_3factor", "D_D_macro", "C_G_overall"]
    cmap = {"A_G_up": "#0277bd", "B_GA00": "#1b5e20",
            "F_0.5GA+0.5D": "#c62828", "I_3factor": "#6a1b9a",
            "D_D_macro": "#ef6c00", "C_G_overall": "#666666"}

    for ax, (tgt_label, tgt) in zip(axes, [("G_md", g_md), ("G_overall", g_overall)]):
        title = "G000 中下游" if tgt_label == "G_md" else "整體 G000"
        sub = df[(df["target"] == tgt_label) & (df["cost_rt"] == 0.0050)]
        best_per = (sub.sort_values("sharpe", ascending=False)
                    .groupby("predictor", as_index=False).first())
        cum_bh = (1 + tgt.fillna(0)).cumprod()
        ax.plot(cum_bh, color="#bbb", lw=1.0, alpha=0.7,
                label=f"Target B&H [{cum_bh.iloc[-1]:.1f}x]")
        for pred_label in plot_preds:
            row = best_per[best_per["predictor"] == pred_label]
            if len(row) == 0:
                continue
            r = row.iloc[0]
            N, K = int(r["smooth_N"]), int(r["hold_K"])
            gross, _, turn = strategy(predictors[pred_label], tgt, N, K)
            net = (gross - turn * 0.0025).fillna(0)
            cum = (1 + net).cumprod()
            ax.plot(cum, color=cmap.get(pred_label, "black"), lw=1.4, alpha=0.85,
                    label=f"{pred_label} (N={N},K={K}) [{cum.iloc[-1]:.1f}x, t={r['t_alpha']:.2f}]")
        ax.set_yscale("log")
        ax.set_title(f"Target: {title}")
        ax.set_xlabel("date"); ax.set_ylabel("累積資本 (log)")
        ax.legend(loc="upper left", fontsize=8.5)
        ax.grid(alpha=0.3)
    plt.suptitle("Phase 22: G000 策略 — 10 predictor 累積曲線 (扣 50bp)", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig_phase22_curves.png"), dpi=130)
    plt.close()
    print(f"圖檔: fig_phase22_curves.png")

    # === Visualization 3: Walk-forward ===
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    wf_colors = {top3[i]: c for i, c in enumerate(["#1b5e20", "#0277bd", "#6a1b9a"])}
    for pred_label in top3:
        oos_ret = oos_returns[pred_label].sort_index().fillna(0)
        cum_wf = (1 + oos_ret).cumprod()
        axes[0].plot(cum_wf, color=wf_colors[pred_label], lw=1.6,
                     label=f"{pred_label} WF [{cum_wf.iloc[-1]:.2f}x]")
    market_oos = market.reindex(oos_returns[top3[0]].sort_index().index).fillna(0)
    cum_mkt = (1 + market_oos).cumprod()
    axes[0].plot(cum_mkt, color="#666", lw=1.0, alpha=0.5,
                 label=f"Market [{cum_mkt.iloc[-1]:.2f}x]")
    axes[0].set_yscale("log")
    axes[0].set_title("(A) WF 累積資本 (扣 50bp)")
    axes[0].set_xlabel("date"); axes[0].set_ylabel("累積資本 (log)")
    axes[0].legend(loc="upper left", fontsize=9)
    axes[0].grid(alpha=0.3)

    # parameter stability
    for pred_label in top3:
        sub = wf_df[wf_df["predictor"] == pred_label].sort_values("test_year")
        axes[1].plot(sub["test_year"], sub["best_N"], "o-",
                     color=wf_colors[pred_label], label=f"{pred_label} N*", alpha=0.85)
        axes[1].plot(sub["test_year"], sub["best_K"], "s--",
                     color=wf_colors[pred_label], label=f"{pred_label} K*", alpha=0.5)
    axes[1].set_xlabel("test_year"); axes[1].set_ylabel("best (N, K)")
    axes[1].set_title("(B) WF 各年最佳 (N*, K*)")
    axes[1].set_yticks(sorted(set(PARAMS_N + PARAMS_K)))
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)
    plt.suptitle("Phase 22: G000 策略 walk-forward (5y train / 1y test)", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig_phase22_walkforward.png"), dpi=130)
    plt.close()
    print(f"圖檔: fig_phase22_walkforward.png")

    # === 結論 ===
    print(f"\n" + "=" * 100)
    print("結論摘要")
    print("=" * 100)
    base_row = (df[(df["target"] == "G_md") & (df["cost_rt"] == 0.0050) &
                    (df["predictor"] == "A_G_up")]
                .sort_values("sharpe", ascending=False).iloc[0])
    print(f"\nbaseline A_G_up (Phase 11a): "
          f"alpha={base_row['ann_alpha']:.4f}, t={base_row['t_alpha']:.3f}, Sh={base_row['sharpe']:.3f}, "
          f"(N={int(base_row['smooth_N'])}, K={int(base_row['hold_K'])})")
    sub = df[(df["target"] == "G_md") & (df["cost_rt"] == 0.0050)]
    best = (sub.sort_values("sharpe", ascending=False)
            .groupby("predictor", as_index=False).first()
            .sort_values("t_alpha", ascending=False))
    print(f"\n各 predictor vs baseline (target=G_md, 50bp):")
    for _, r in best.iterrows():
        if r["predictor"] == "A_G_up":
            continue
        d_alpha = (r["ann_alpha"] - base_row["ann_alpha"]) * 100
        d_t = r["t_alpha"] - base_row["t_alpha"]
        flag = " ✓" if d_alpha > 1.0 and d_t > 0.3 else (" ↑" if d_alpha > 0 else " ↓")
        print(f"  {r['predictor']:<18s} α={r['ann_alpha']:+.4f}  t={r['t_alpha']:+.3f}  "
              f"Sh={r['sharpe']:+.3f}  Δα={d_alpha:+.2f}pp  Δt={d_t:+.2f}{flag}")

    print(f"\nWalk-forward 衰減 (top-3):")
    for r in summary_wf:
        decay = (r["ann_alpha"] - r["IS_OOS_alpha"]) * 100
        print(f"  {r['predictor']}: WF α={r['ann_alpha']:.4f}/t={r['t_alpha']:.3f} "
              f"vs IS@{r['IS_winner_NK']} α={r['IS_OOS_alpha']:.4f}/t={r['IS_OOS_t']:.3f} "
              f"(衰減 {decay:+.2f}pp)")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
