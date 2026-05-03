"""
Phase 13: IC 設計 (D100) + 台積電 (2330) 雙因子組合 timing 策略

動機 (來自 Phase 10)
- D100 IC 設計 portfolio 是純訊息源頭 (Joint reg 8/8 主導)
- 台積電 2330 有獨立 alpha (t≈2.6, 反映全球 AI / 客戶 macro 訊號)
- Phase 12 結果: D100 ≈ D000 (corr=0.966), 單獨替換無效
- 假設: D100 + 2330 線性組合或殘差化組合, 透過捕獲 2330 獨立 alpha 提升整體訊號

雙因子方案
1. baseline:   D100 (95 檔 EW) | 2330 (個股)
2. 線性組合:    0.5/0.5 | 0.7/0.3 | 0.3/0.7
3. 殘差化組合:  D100 + λ · (2330 ⊥ D100, market) for λ ∈ {1, 2}
              捕獲 2330 idiosyncratic 訊息, 與 D100 正交, 避免共線性

Targets
A. 7 跨產業下游 EW (G/H/I/J/F/L/5400) — Phase 8/12 框架
B. D000 中下游 EW (中游 + 下游聯合) — Phase 9/10 框架

策略結構
- predictor EMA(N) > 0 → long target, 隔日執行, K 日持有
- N ∈ {1,3,5,10,20}, K ∈ {1,5,10,20}, cost ∈ {0, 30, 50, 80, 100} bp

輸出
- phase13_dual_factor_xind.csv : 7 predictors × N×K×cost (跨產業 target)
- phase13_dual_factor_inchain.csv : 7 predictors × N×K×cost (鏈內 target)
- fig_phase13_best_alpha_comparison.png : best 參數下各 predictor 比較 bar
- fig_phase13_dual_curves.png : 累積曲線
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

DOWNSTREAM = ["G000", "H000", "I000", "J000", "F000", "L000", "5400"]
PARAMS_N = [1, 3, 5, 10, 20]
PARAMS_K = [1, 5, 10, 20]
COSTS = [0.000, 0.0030, 0.0050, 0.0080, 0.0100]


def load_pickle_df(name):
    with open(os.path.join(DB, name), "rb") as f:
        df = pickle.load(f)
    df = df.set_index("date")
    df.index = pd.to_datetime(df.index)
    df.columns = df.columns.astype(str)
    return df


def filter_common(df):
    keep = [c for c in df.columns if c.isdigit() and len(c) == 4 and not c.startswith("00")]
    return df[keep]


def k_day_holding(target: pd.Series, hold_k: int) -> pd.Series:
    if hold_k == 1:
        return target
    out = target.copy()
    last = target.iloc[0]
    for i in range(len(target)):
        if i % hold_k == 0:
            last = target.iloc[i]
        out.iloc[i] = last
    return out


def perf_with_alpha(daily_ret: pd.Series, market: pd.Series, ann: float = 252) -> dict:
    df = pd.concat([daily_ret.rename("s"), market.rename("b")], axis=1).dropna()
    if len(df) < 60:
        return {}
    res = sm.OLS(df["s"], sm.add_constant(df["b"])).fit(
        cov_type="HAC", cov_kwds={"maxlags": 60}
    )
    s = df["s"]
    cum = (1 + s).cumprod()
    return {
        "n": len(s),
        "ann_ret": s.mean() * ann,
        "ann_vol": s.std() * np.sqrt(ann),
        "sharpe": s.mean() / s.std() * np.sqrt(ann) if s.std() > 0 else np.nan,
        "max_dd": (cum / cum.cummax() - 1).min(),
        "win": (s > 0).mean(),
        "ann_alpha": res.params["const"] * ann,
        "t_alpha": res.tvalues["const"],
        "beta": res.params["b"],
    }


def strategy(predictor, target, smooth_n, hold_k):
    sig = predictor.ewm(span=smooth_n, adjust=False).mean() if smooth_n > 1 else predictor
    target_w = (sig > 0).astype(float).shift(1)
    weight = k_day_holding(target_w, hold_k)
    gross = weight * target
    turnover = weight.diff().abs().fillna(weight.iloc[0])
    return gross, weight, turnover


def main():
    print("=" * 90)
    print("Phase 13: D100 IC設計 + 2330 台積電 雙因子組合 timing 策略")
    print("=" * 90)

    chain = pd.read_csv(os.path.join(ROOT, "industry_chain.csv"), dtype=str)
    industry_meta = (chain[["industry_code", "industry_name"]]
                     .drop_duplicates()
                     .set_index("industry_code")["industry_name"].to_dict())
    adj_close = filter_common(load_pickle_df("etl#adj_close.pickle"))
    daily_ret = adj_close.pct_change(fill_method=None).iloc[1:]
    market = daily_ret.mean(axis=1)

    # === industry panel ===
    panel = {}
    for ic in industry_meta:
        members = chain[chain["industry_code"] == ic]["stock_id"].unique().tolist()
        cols = [s for s in members if s in daily_ret.columns]
        if len(cols) >= 3:
            panel[ic] = daily_ret[cols].mean(axis=1)
    panel = pd.DataFrame(panel)

    # === D100 portfolio (排除 2330 避免重複 — 但 2330 不在 D100 sub_code, 是 D300) ===
    d100_members = chain[(chain["industry_code"] == "D000") &
                          (chain["sub_code"] == "D100")]["stock_id"].unique().tolist()
    d100_cols = [s for s in d100_members if s in daily_ret.columns]
    d100 = daily_ret[d100_cols].mean(axis=1)

    if "2330" not in daily_ret.columns:
        raise SystemExit("2330 not in adj_close — abort")
    tsmc = daily_ret["2330"]

    # 確認 2330 不在 D100
    assert "2330" not in d100_cols, f"2330 leaked into D100"
    print(f"\nD100 IC設計: {len(d100_cols)} 檔  (不含 2330)")
    print(f"2330 台積電: 1 檔個股")

    # === 雙因子 sanity: Phase 10 風格 ===
    df_chk = pd.concat({"D100": d100, "2330": tsmc, "mkt": market}, axis=1).dropna()
    print(f"\n[Sanity 同期相關]")
    print(f"  ρ(D100, 2330) = {df_chk['D100'].corr(df_chk['2330']):.3f}")
    print(f"  ρ(D100, mkt)  = {df_chk['D100'].corr(df_chk['mkt']):.3f}")
    print(f"  ρ(2330, mkt)  = {df_chk['2330'].corr(df_chk['mkt']):.3f}")
    print(f"  vol(D100) = {d100.std()*np.sqrt(252)*100:.2f}%  vol(2330) = {tsmc.std()*np.sqrt(252)*100:.2f}%")

    # === 殘差化: 2330 ⊥ D100, market ===
    res_orth = sm.OLS(df_chk["2330"], sm.add_constant(df_chk[["D100", "mkt"]])).fit()
    tsmc_resid = (df_chk["2330"] - res_orth.predict(sm.add_constant(df_chk[["D100", "mkt"]]))).reindex(daily_ret.index)
    var_ratio = tsmc_resid.var() / tsmc.var()
    print(f"\n[殘差化] 2330 = β·D100 + δ·mkt + ε,  β={res_orth.params['D100']:+.4f}, "
          f"δ={res_orth.params['mkt']:+.4f},  R²={res_orth.rsquared:.3f}")
    print(f"  2330_resid var ratio = {var_ratio:.3f}  (idiosyncratic 約 {var_ratio*100:.1f}% 變異)")

    # === Predictors: 7 個方案 ===
    predictors = {
        "D100":              d100,
        "2330":              tsmc,
        "0.5·D100+0.5·2330": 0.5 * d100 + 0.5 * tsmc,
        "0.7·D100+0.3·2330": 0.7 * d100 + 0.3 * tsmc,
        "0.3·D100+0.7·2330": 0.3 * d100 + 0.7 * tsmc,
        "D100+1.0·2330⊥":    d100 + 1.0 * tsmc_resid,
        "D100+2.0·2330⊥":    d100 + 2.0 * tsmc_resid,
    }

    # === Targets: 兩種 ===
    target_xind = panel[DOWNSTREAM].mean(axis=1)  # 7 跨產業下游 EW
    d_mid = chain[(chain["industry_code"] == "D000") & (chain["position"] == "中游")]["stock_id"].unique()
    d_dn = chain[(chain["industry_code"] == "D000") & (chain["position"] == "下游")]["stock_id"].unique()
    md_set = list(set(list(d_mid) + list(d_dn)) & set(daily_ret.columns))
    target_inchain = daily_ret[md_set].mean(axis=1)
    print(f"\nTargets:")
    print(f"  A. 7 跨產業下游 EW (G/H/I/J/F/L/5400)")
    print(f"  B. D000 中下游 EW ({len(md_set)} 檔)")

    # === 全 sweep ===
    targets = {"xind": target_xind, "inchain": target_inchain}
    all_rows = {"xind": [], "inchain": []}

    for tgt_label, tgt in targets.items():
        for pred_label, pred in predictors.items():
            for N in PARAMS_N:
                for K in PARAMS_K:
                    gross, weight, turn = strategy(pred, tgt, N, K)
                    avg_turn = turn.mean()
                    for cost_rt in COSTS:
                        cost_one = cost_rt / 2.0
                        net = gross - turn * cost_one
                        p = perf_with_alpha(net, market)
                        if not p:
                            continue
                        p.update({
                            "predictor": pred_label,
                            "smooth_N": N,
                            "hold_K": K,
                            "cost_rt": cost_rt,
                            "avg_daily_turn": avg_turn,
                            "ann_turn": avg_turn * 252,
                        })
                        all_rows[tgt_label].append(p)

    df_xind = pd.DataFrame(all_rows["xind"])
    df_inchain = pd.DataFrame(all_rows["inchain"])
    df_xind.to_csv(os.path.join(OUT, "phase13_dual_factor_xind.csv"),
                   index=False, encoding="utf-8-sig")
    df_inchain.to_csv(os.path.join(OUT, "phase13_dual_factor_inchain.csv"),
                      index=False, encoding="utf-8-sig")

    # === 印各 predictor 在各 target 的最佳結果 (50bp) ===
    fmt = lambda x: f"{x:.4f}"
    cols_show = ["predictor", "smooth_N", "hold_K", "ann_ret", "sharpe",
                 "ann_alpha", "t_alpha", "max_dd", "avg_daily_turn"]

    print("\n" + "=" * 90)
    print("Target A: 7 跨產業下游 EW — 各 predictor best @ 50bp (按 t_alpha 排序)")
    print("=" * 90)
    best_xind = (df_xind[df_xind["cost_rt"] == 0.0050]
                 .sort_values("sharpe", ascending=False)
                 .groupby("predictor", as_index=False).first()
                 .sort_values("t_alpha", ascending=False))
    print(best_xind[cols_show].to_string(index=False, float_format=fmt))

    print("\n" + "=" * 90)
    print("Target B: D000 中下游 EW — 各 predictor best @ 50bp (按 t_alpha 排序)")
    print("=" * 90)
    best_inchain = (df_inchain[df_inchain["cost_rt"] == 0.0050]
                    .sort_values("sharpe", ascending=False)
                    .groupby("predictor", as_index=False).first()
                    .sort_values("t_alpha", ascending=False))
    print(best_inchain[cols_show].to_string(index=False, float_format=fmt))

    # === 同 (N=20, K=1) 比較 (Phase 8 winner) on 跨產業 ===
    print("\n" + "=" * 90)
    print("Target A 跨產業 — 鎖定 Phase 8 winner (N=20, K=1) 看純參數效果")
    print("=" * 90)
    for cost_rt in [0.0, 0.0050, 0.0080]:
        print(f"\n  cost @{int(cost_rt*1e4)}bp:")
        sub = df_xind[(df_xind["smooth_N"] == 20) & (df_xind["hold_K"] == 1) &
                       (df_xind["cost_rt"] == cost_rt)].sort_values("t_alpha", ascending=False)
        for _, r in sub.iterrows():
            print(f"    {r['predictor']:<22s}  ann_ret={r['ann_ret']:>+7.4f}  "
                  f"sharpe={r['sharpe']:>+6.3f}  alpha={r['ann_alpha']:>+7.4f}  t_α={r['t_alpha']:>+6.3f}")

    # === 80bp 高成本場景 ===
    print("\n" + "=" * 90)
    print("Target A 跨產業 — Top 3 by Sharpe @ 80bp (各 predictor)")
    print("=" * 90)
    top80 = (df_xind[df_xind["cost_rt"] == 0.0080]
             .sort_values("sharpe", ascending=False)
             .groupby("predictor", as_index=False).first()
             .sort_values("t_alpha", ascending=False))
    print(top80[cols_show].to_string(index=False, float_format=fmt))

    # === Visualization 1: best alpha 比較 bar ===
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (label, best_df) in zip(axes, [("跨產業 7 下游", best_xind),
                                             ("D000 中下游", best_inchain)]):
        order = best_df.sort_values("t_alpha")
        colors = ["#1b5e20" if t > 4 else
                  "#558b2f" if t > 2 else
                  "#fb8c00" if t > 0 else "#e53935"
                  for t in order["t_alpha"]]
        ax.barh(order["predictor"], order["t_alpha"], color=colors, edgecolor="black", linewidth=0.4)
        for i, (t, alpha, pred) in enumerate(zip(order["t_alpha"], order["ann_alpha"], order["predictor"])):
            ax.text(t + 0.05, i, f"α={alpha:+.3f}", va="center", fontsize=8.5)
        ax.axvline(1.96, color="black", lw=0.5, ls="--", alpha=0.5)
        ax.axvline(0, color="black", lw=0.5)
        ax.set_xlabel("t_alpha (best (N,K) @ 50bp)")
        ax.set_title(f"Target: {label}")
        ax.grid(axis="x", alpha=0.3)
    plt.suptitle("Phase 13: 雙因子組合 — best 參數下 t_alpha 比較 (50bp)", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig_phase13_best_alpha_comparison.png"), dpi=130)
    plt.close()
    print(f"\n圖檔: fig_phase13_best_alpha_comparison.png")

    # === Visualization 2: 累積曲線 - top combo on 跨產業 ===
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, (tgt_label, tgt, best_df) in zip(
        axes,
        [("Target A: 7 跨產業下游 EW", target_xind, best_xind),
         ("Target B: D000 中下游 EW", target_inchain, best_inchain)],
    ):
        plot_preds = ["D100", "2330", "0.5·D100+0.5·2330", "D100+1.0·2330⊥", "D100+2.0·2330⊥"]
        colors_map = {"D100": "#0277bd", "2330": "#6a1b9a",
                      "0.5·D100+0.5·2330": "#1b5e20",
                      "D100+1.0·2330⊥": "#ef6c00", "D100+2.0·2330⊥": "#c62828"}
        cum_bh = (1 + tgt.fillna(0)).cumprod()
        ax.plot(cum_bh, color="#999999", lw=1.0, alpha=0.7, label=f"Target B&H ({cum_bh.iloc[-1]:.1f}x)")
        for pred_label in plot_preds:
            best_row = best_df[best_df["predictor"] == pred_label].iloc[0]
            N, K = int(best_row["smooth_N"]), int(best_row["hold_K"])
            gross, _, turn = strategy(predictors[pred_label], tgt, N, K)
            net = (gross - turn * 0.0025).fillna(0)
            cum = (1 + net).cumprod()
            ax.plot(cum, color=colors_map.get(pred_label, "black"), lw=1.5, alpha=0.85,
                    label=f"{pred_label} (N={N}, K={K}) [{cum.iloc[-1]:.1f}x]")
        ax.set_yscale("log")
        ax.set_title(tgt_label)
        ax.set_xlabel("date"); ax.set_ylabel("累積資本 (log)")
        ax.legend(loc="upper left", fontsize=8.5)
        ax.grid(alpha=0.3)
    plt.suptitle("Phase 13: 雙因子組合 timing 累積曲線 (扣 50bp)", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig_phase13_dual_curves.png"), dpi=130)
    plt.close()
    print(f"圖檔: fig_phase13_dual_curves.png")

    # === Decision matrix ===
    print("\n" + "=" * 90)
    print("結論: 雙因子是否值得?")
    print("=" * 90)
    base_xind = best_xind[best_xind["predictor"] == "D100"].iloc[0]
    print(f"\nTarget A 跨產業:")
    print(f"  baseline D100:           ann_α={base_xind['ann_alpha']:+.4f}  t_α={base_xind['t_alpha']:+.3f}  Sharpe={base_xind['sharpe']:+.3f}")
    for label, row in best_xind.iterrows():
        if row["predictor"] == "D100":
            continue
        delta_a = (row["ann_alpha"] - base_xind["ann_alpha"]) * 100
        delta_s = row["sharpe"] - base_xind["sharpe"]
        delta_t = row["t_alpha"] - base_xind["t_alpha"]
        flag = " ✓" if delta_a > 1.0 and delta_t > 0.3 else (" ↑" if delta_a > 0 else " ↓")
        print(f"  {row['predictor']:<22s} ann_α={row['ann_alpha']:+.4f}  "
              f"t_α={row['t_alpha']:+.3f}  Sharpe={row['sharpe']:+.3f}  "
              f"Δα={delta_a:+.2f}pp  Δt={delta_t:+.2f}{flag}")

    base_in = best_inchain[best_inchain["predictor"] == "D100"].iloc[0]
    print(f"\nTarget B 鏈內:")
    print(f"  baseline D100:           ann_α={base_in['ann_alpha']:+.4f}  t_α={base_in['t_alpha']:+.3f}  Sharpe={base_in['sharpe']:+.3f}")
    for label, row in best_inchain.iterrows():
        if row["predictor"] == "D100":
            continue
        delta_a = (row["ann_alpha"] - base_in["ann_alpha"]) * 100
        delta_s = row["sharpe"] - base_in["sharpe"]
        delta_t = row["t_alpha"] - base_in["t_alpha"]
        flag = " ✓" if delta_a > 1.0 and delta_t > 0.3 else (" ↑" if delta_a > 0 else " ↓")
        print(f"  {row['predictor']:<22s} ann_α={row['ann_alpha']:+.4f}  "
              f"t_α={row['t_alpha']:+.3f}  Sharpe={row['sharpe']:+.3f}  "
              f"Δα={delta_a:+.2f}pp  Δt={delta_t:+.2f}{flag}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
