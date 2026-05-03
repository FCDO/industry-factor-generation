"""
Phase 17: P000 電機機械 雙因子 — (P200+P600) + 0.5·1590 亞德客-KY

動機 (Phase 15)
- P200+P600 為 P000 鏈內純 leader (殘差化後仍 t>2 預測 P000 中下游與 xind)
- **1590 亞德客-KY 為 P000 的 macro 個股**: xind alone t=3.19, joint 雙顯著 (2.36/2.43)
- 角色類比: P200+P600 ↔ D100 IC設計; 1590 ↔ 2330 台積電

預期: 套 Phase 13 框架 (50/50 線性組合) 應為 P000 雙因子 winner.

Predictors
1. P_lead = (P200+P600) EW                  Phase 11b 鏈內 leader
2. 1590 亞德客個股                              macro (Phase 15 發現)
3. 2049 上銀個股                               對照 macro (較弱)
4. 0.5·P_lead + 0.5·1590                    Phase 13 框架 winner 候選
5. 0.7·P_lead + 0.3·1590
6. 0.3·P_lead + 0.7·1590
7. P_lead + 1.0·(1590 ⊥ P_lead, mkt)         殘差化版
8. 整體 P000                                  Phase 11a baseline
9. D100+0.5·2330                             Phase 13 跨鏈 winner (對照)

Targets
A. P000 中下游 EW (鏈內)
B. 7 跨產業下游 EW (xind)

Sweep N×K×cost, output 比較 P000 雙因子是否能達到 Phase 13 D000 雙因子的 alpha 水準
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
    df = df.set_index("date"); df.index = pd.to_datetime(df.index)
    df.columns = df.columns.astype(str)
    return df


def filter_common(df):
    keep = [c for c in df.columns if c.isdigit() and len(c) == 4 and not c.startswith("00")]
    return df[keep]


def k_day_holding(target: pd.Series, hold_k: int) -> pd.Series:
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
        "n": len(s), "ann_ret": s.mean() * ann, "ann_vol": s.std() * np.sqrt(ann),
        "sharpe": s.mean()/s.std()*np.sqrt(ann) if s.std() > 0 else np.nan,
        "max_dd": (cum/cum.cummax() - 1).min(), "win": (s > 0).mean(),
        "ann_alpha": res.params["const"]*ann, "t_alpha": res.tvalues["const"],
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
    print("=" * 100)
    print("Phase 17: P000 電機機械 雙因子 — (P200+P600) + 0.5·1590 亞德客")
    print("=" * 100)

    chain = pd.read_csv(os.path.join(ROOT, "industry_chain.csv"), dtype=str)
    adj_close = filter_common(load_pickle_df("etl#adj_close.pickle"))
    daily_ret = adj_close.pct_change(fill_method=None).iloc[1:]
    market = daily_ret.mean(axis=1)

    # === Build leaders ===
    def build_subcode(industry, sub):
        members = chain[(chain["industry_code"] == industry) &
                         (chain["sub_code"] == sub)]["stock_id"].unique().tolist()
        cols = [s for s in members if s in daily_ret.columns]
        return daily_ret[cols].mean(axis=1), len(cols)

    p200, n200 = build_subcode("P000", "P200")
    p600, n600 = build_subcode("P000", "P600")
    p_lead = pd.concat([p200, p600], axis=1).mean(axis=1)

    # P000 整體 (對照)
    p_overall_members = chain[chain["industry_code"] == "P000"]["stock_id"].unique()
    p_overall_cols = [s for s in p_overall_members if s in daily_ret.columns]
    p_overall = daily_ret[p_overall_cols].mean(axis=1)

    # P000 中下游 (target A)
    p_md_set = chain[(chain["industry_code"] == "P000") &
                      (chain["position"].isin(["中游", "下游"]))]["stock_id"].unique()
    p_md_cols = [s for s in p_md_set if s in daily_ret.columns]
    p_md = daily_ret[p_md_cols].mean(axis=1)

    # 個股
    if "1590" not in daily_ret.columns or "2049" not in daily_ret.columns:
        raise SystemExit("1590 / 2049 not in adj_close")
    s1590 = daily_ret["1590"]
    s2049 = daily_ret["2049"]

    # D100, 2330 對照
    d100_members = chain[(chain["industry_code"] == "D000") &
                          (chain["sub_code"] == "D100")]["stock_id"].unique()
    d100_cols = [s for s in d100_members if s in daily_ret.columns]
    d100 = daily_ret[d100_cols].mean(axis=1)
    s2330 = daily_ret["2330"]

    # 7 跨產業下游 (target B)
    industry_panel = {}
    for ic in chain["industry_code"].unique():
        members = chain[chain["industry_code"] == ic]["stock_id"].unique().tolist()
        cols = [s for s in members if s in daily_ret.columns]
        if len(cols) >= 3:
            industry_panel[ic] = daily_ret[cols].mean(axis=1)
    industry_panel = pd.DataFrame(industry_panel)
    target_xind = industry_panel[DOWNSTREAM].mean(axis=1)

    print(f"\nLeaders / Macro:")
    print(f"  P200 傳動: {n200} 檔  |  P600 沖壓: {n600} 檔  |  P_lead = (P200+P600)/2")
    print(f"  整體 P000: {len(p_overall_cols)} 檔  |  P000 中下游: {len(p_md_cols)} 檔")
    print(f"  1590 亞德客-KY (個股)  |  2049 上銀 (個股)")
    print(f"  D100 IC設計 (對照): {len(d100_cols)} 檔  |  2330 台積電")

    # === Sanity: corr ===
    df_corr = pd.concat({
        "P_lead": p_lead, "1590": s1590, "2049": s2049,
        "P000_overall": p_overall, "D100": d100, "2330": s2330, "mkt": market,
    }, axis=1).dropna()
    print(f"\n[Sanity] 同期相關:")
    print(df_corr.corr().round(3).to_string())

    # === 殘差化: 1590 ⊥ P_lead, mkt ===
    df_orth = pd.concat({"P_lead": p_lead, "1590": s1590, "mkt": market}, axis=1).dropna()
    res_o = sm.OLS(df_orth["1590"], sm.add_constant(df_orth[["P_lead", "mkt"]])).fit()
    s1590_resid = (df_orth["1590"] - res_o.predict(sm.add_constant(df_orth[["P_lead", "mkt"]]))).reindex(daily_ret.index)
    print(f"\n[殘差化] 1590 = β·P_lead + δ·mkt + ε,  β={res_o.params['P_lead']:+.3f}, "
          f"δ={res_o.params['mkt']:+.3f},  R²={res_o.rsquared:.3f}")
    print(f"  1590_resid var ratio = {s1590_resid.var()/s1590.var():.3f}")

    # === Predictors ===
    predictors = {
        "P_lead":              p_lead,
        "1590":                s1590,
        "2049":                s2049,
        "0.5·P_lead+0.5·1590": 0.5 * p_lead + 0.5 * s1590,
        "0.7·P_lead+0.3·1590": 0.7 * p_lead + 0.3 * s1590,
        "0.3·P_lead+0.7·1590": 0.3 * p_lead + 0.7 * s1590,
        "P_lead+1.0·1590⊥":    p_lead + 1.0 * s1590_resid,
        "0.5·P_lead+0.5·2049": 0.5 * p_lead + 0.5 * s2049,  # 對照 macro
        "整體P000":              p_overall,
        "D100+0.5·2330":        0.5 * d100 + 0.5 * s2330,    # Phase 13 跨鏈 winner
    }

    targets = {"P_md": p_md, "xind": target_xind}

    # === Sweep ===
    print("\n" + "=" * 100)
    print("Strategy sweep")
    print("=" * 100)
    rows = []
    for tgt_label, tgt in targets.items():
        for pred_label, pred in predictors.items():
            for N in PARAMS_N:
                for K in PARAMS_K:
                    gross, weight, turn = strategy(pred, tgt, N, K)
                    avg_turn = turn.mean()
                    for cost_rt in COSTS:
                        net = gross - turn * cost_rt / 2
                        p = perf_with_alpha(net, market)
                        if not p:
                            continue
                        p.update({"target": tgt_label, "predictor": pred_label,
                                  "smooth_N": N, "hold_K": K, "cost_rt": cost_rt,
                                  "avg_daily_turn": avg_turn, "ann_turn": avg_turn * 252})
                        rows.append(p)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "phase17_p000_dual_factor.csv"),
              index=False, encoding="utf-8-sig")

    # === Best per (target, predictor) @ 50bp ===
    fmt = lambda x: f"{x:.4f}"
    cols_show = ["predictor", "smooth_N", "hold_K", "ann_ret", "ann_vol", "sharpe",
                 "ann_alpha", "t_alpha", "max_dd", "avg_daily_turn"]

    for tlabel in ["P_md", "xind"]:
        title = "P000 中下游" if tlabel == "P_md" else "7 跨產業下游"
        print("\n" + "=" * 100)
        print(f"Target {tlabel}: {title} — 各 predictor best @ 50bp (按 t_alpha 排序)")
        print("=" * 100)
        sub = df[(df["target"] == tlabel) & (df["cost_rt"] == 0.0050)]
        best = (sub.sort_values("sharpe", ascending=False)
                .groupby("predictor", as_index=False).first()
                .sort_values("t_alpha", ascending=False))
        print(best[cols_show].to_string(index=False, float_format=fmt))

    # === 雙因子 vs 單因子 head-to-head, 固定 (N=10, K=1) Phase 13 winner 參數 ===
    print("\n" + "=" * 100)
    print("固定 (N=10, K=1) — Phase 13 winner 參數下各 predictor 表現 @ 50bp")
    print("=" * 100)
    for tlabel in ["P_md", "xind"]:
        title = "P000 中下游" if tlabel == "P_md" else "7 跨產業下游"
        print(f"\n  Target {tlabel}: {title}")
        sub = df[(df["target"] == tlabel) & (df["smooth_N"] == 10) &
                  (df["hold_K"] == 1) & (df["cost_rt"] == 0.0050)]\
              .sort_values("t_alpha", ascending=False)
        for _, r in sub.iterrows():
            print(f"    {r['predictor']:<22s}  ann_ret={r['ann_ret']:>+.4f}  "
                  f"sharpe={r['sharpe']:>+.3f}  alpha={r['ann_alpha']:>+.4f}  t_α={r['t_alpha']:>+.3f}")

    # === @80bp 高成本 ===
    print("\n" + "=" * 100)
    print("各 predictor best @ 80bp — Target xind (按 t_alpha)")
    print("=" * 100)
    sub80 = df[(df["target"] == "xind") & (df["cost_rt"] == 0.0080)]
    best80 = (sub80.sort_values("sharpe", ascending=False)
              .groupby("predictor", as_index=False).first()
              .sort_values("t_alpha", ascending=False))
    print(best80[cols_show].to_string(index=False, float_format=fmt))

    # === Visualization 1: methods bar chart ===
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, tlabel in zip(axes, ["P_md", "xind"]):
        title = "P000 中下游" if tlabel == "P_md" else "7 跨產業下游"
        sub = df[(df["target"] == tlabel) & (df["cost_rt"] == 0.0050)]
        best = (sub.sort_values("sharpe", ascending=False)
                .groupby("predictor", as_index=False).first()
                .sort_values("t_alpha"))
        colors = ["#1b5e20" if t > 5 else "#558b2f" if t > 3 else
                  "#fb8c00" if t > 1.96 else "#9e9e9e" if t > 0 else "#e53935"
                  for t in best["t_alpha"]]
        ax.barh(best["predictor"], best["t_alpha"], color=colors,
                edgecolor="black", lw=0.4)
        for i, (t, alpha) in enumerate(zip(best["t_alpha"], best["ann_alpha"])):
            ax.text(t + 0.08, i, f"α={alpha:+.3f}", va="center", fontsize=8.5)
        ax.axvline(1.96, color="black", lw=0.5, ls="--", alpha=0.5)
        ax.axvline(0, color="black", lw=0.5)
        ax.set_xlabel("t_alpha (best (N,K) @ 50bp)")
        ax.set_title(f"Target: {title}")
        ax.grid(axis="x", alpha=0.3)
    plt.suptitle("Phase 17: P000 雙因子組合 — 各 predictor 比較", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig_phase17_methods.png"), dpi=130)
    plt.close()
    print(f"\n圖檔: fig_phase17_methods.png")

    # === Visualization 2: 累積曲線 ===
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    plot_preds = ["P_lead", "1590", "0.5·P_lead+0.5·1590",
                  "P_lead+1.0·1590⊥", "整體P000", "D100+0.5·2330"]
    colors_map = {
        "P_lead": "#0277bd", "1590": "#6a1b9a",
        "0.5·P_lead+0.5·1590": "#1b5e20",
        "P_lead+1.0·1590⊥": "#ef6c00",
        "整體P000": "#666666", "D100+0.5·2330": "#c62828",
    }
    for ax, (tlabel, tgt) in zip(axes, [("P_md", p_md), ("xind", target_xind)]):
        title = "P000 中下游" if tlabel == "P_md" else "7 跨產業下游"
        sub = df[(df["target"] == tlabel) & (df["cost_rt"] == 0.0050)]
        best = (sub.sort_values("sharpe", ascending=False)
                .groupby("predictor", as_index=False).first())
        cum_bh = (1 + tgt.fillna(0)).cumprod()
        ax.plot(cum_bh, color="#bbb", lw=1.0, alpha=0.7,
                label=f"Target B&H [{cum_bh.iloc[-1]:.1f}x]")
        for pred_label in plot_preds:
            best_row = best[best["predictor"] == pred_label].iloc[0]
            N, K = int(best_row["smooth_N"]), int(best_row["hold_K"])
            gross, _, turn = strategy(predictors[pred_label], tgt, N, K)
            net = (gross - turn * 0.0025).fillna(0)
            cum = (1 + net).cumprod()
            ax.plot(cum, color=colors_map.get(pred_label, "black"), lw=1.4, alpha=0.85,
                    label=f"{pred_label} (N={N},K={K}) [{cum.iloc[-1]:.1f}x, t={best_row['t_alpha']:.2f}]")
        ax.set_yscale("log")
        ax.set_title(f"Target: {title}")
        ax.set_xlabel("date"); ax.set_ylabel("累積資本 (log)")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(alpha=0.3)
    plt.suptitle("Phase 17: P000 雙因子累積曲線 (扣 50bp)", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig_phase17_curves.png"), dpi=130)
    plt.close()
    print(f"圖檔: fig_phase17_curves.png")

    # === 結論 ===
    print("\n" + "=" * 100)
    print("結論: P000 雙因子 vs Phase 11a baseline + Phase 13 跨鏈對照")
    print("=" * 100)
    for tlabel in ["P_md", "xind"]:
        title = "P000 中下游" if tlabel == "P_md" else "7 跨產業下游"
        sub = df[(df["target"] == tlabel) & (df["cost_rt"] == 0.0050)]
        best = (sub.sort_values("sharpe", ascending=False)
                .groupby("predictor", as_index=False).first())
        print(f"\nTarget {tlabel} ({title}):")
        for label in ["整體P000", "P_lead", "1590", "0.5·P_lead+0.5·1590",
                       "P_lead+1.0·1590⊥", "D100+0.5·2330"]:
            row = best[best["predictor"] == label]
            if len(row) == 0:
                continue
            r = row.iloc[0]
            tag = ""
            if label == "0.5·P_lead+0.5·1590":
                tag = " ← P000 雙因子 candidate"
            elif label == "D100+0.5·2330":
                tag = " ← Phase 13 跨鏈 winner reference"
            elif label == "整體P000":
                tag = " ← Phase 11a baseline"
            print(f"  {label:<22s}  α={r['ann_alpha']:+.4f}  t_α={r['t_alpha']:+.3f}  "
                  f"Sh={r['sharpe']:+.3f}{tag}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
