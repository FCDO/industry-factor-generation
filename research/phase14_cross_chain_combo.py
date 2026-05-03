"""
Phase 14: 跨鏈 leader 組合因子

採用 Phase 9-11 找到的純 leader sub_code + Phase 13 macro 個股, 整合 6 個 leader portfolios:
- D100 IC設計     (D000)
- 2330 台積電個股                 [Phase 13 證實獨立 alpha]
- GA00 顯示器其他零組件 (G000)
- P200 傳動元件   (P000)
- P600 沖壓零組件 (P000)
- IA00 光通訊     (I000)

7 種整合方法 (predictor):
1. D100                  baseline (Phase 12)
2. D100+0.5·2330         Phase 13 winner
3. 6L-EW-raw             6 leader 原始報酬等權
4. 6L-EW-zscore          60d rolling z-score 後等權
5. 6L-PCA-PC1            full-sample PCA 第一主成分 (方向 = 共同因子)
6. 6L-Orth-sum           D100 anchor + 其他 5 個 ⊥ (D100, mkt) 殘差 sum
7. 6L-OLS-oracle         對 target_{t+1} ~ leaders_t full-sample OLS, 取係數做權重 (有 look-ahead)

2 種 target:
A. xind   = 7 跨產業下游 EW (G/H/I/J/F/L/5400)  — Phase 8/13 框架
B. broad  = 23 standard chain 中下游 super-broad (mean of per-chain EW)

策略結構: predictor EMA(N) > 0 → long target, K 日持有, 隔日執行, 扣 round-trip cost

輸出
- phase14_cross_chain.csv          : 全 sweep
- phase14_ols_oracle_coefs.csv     : OLS-oracle 係數 (記錄 in-sample loadings)
- fig_phase14_methods_comparison.png
- fig_phase14_curves.png
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

LEADER_SPEC = [
    ("D100", "D000", "D100", "IC設計"),
    ("GA00", "G000", "GA00", "顯示器其他零組件"),
    ("P200", "P000", "P200", "傳動元件"),
    ("P600", "P000", "P600", "沖壓零組件"),
    ("IA00", "I000", "IA00", "光通訊"),
]


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


def strategy(predictor: pd.Series, target: pd.Series, smooth_n: int, hold_k: int):
    sig = predictor.ewm(span=smooth_n, adjust=False).mean() if smooth_n > 1 else predictor
    target_w = (sig > 0).astype(float).shift(1)
    weight = k_day_holding(target_w, hold_k)
    gross = weight * target
    turnover = weight.diff().abs().fillna(weight.iloc[0])
    return gross, weight, turnover


def main():
    print("=" * 100)
    print("Phase 14: 跨鏈 leader 組合因子 — 6 leaders × 7 整合方法 × 2 target")
    print("=" * 100)

    chain = pd.read_csv(os.path.join(ROOT, "industry_chain.csv"), dtype=str)
    industry_meta = (chain[["industry_code", "industry_name"]].drop_duplicates()
                     .set_index("industry_code")["industry_name"].to_dict())
    adj_close = filter_common(load_pickle_df("etl#adj_close.pickle"))
    daily_ret = adj_close.pct_change(fill_method=None).iloc[1:]
    market = daily_ret.mean(axis=1)

    # === Industry panel ===
    panel = {}
    for ic in industry_meta:
        members = chain[chain["industry_code"] == ic]["stock_id"].unique().tolist()
        cols = [s for s in members if s in daily_ret.columns]
        if len(cols) >= 3:
            panel[ic] = daily_ret[cols].mean(axis=1)
    panel = pd.DataFrame(panel)

    # === Leader portfolios ===
    print("\n[Leaders]")
    leaders = {}
    for label, ic, sc, name in LEADER_SPEC:
        members = chain[(chain["industry_code"] == ic) &
                         (chain["sub_code"] == sc)]["stock_id"].unique().tolist()
        cols = [s for s in members if s in daily_ret.columns]
        leaders[label] = daily_ret[cols].mean(axis=1)
        print(f"  {label} {name:<14s} ({ic}): {len(cols)} 檔")
    if "2330" not in daily_ret.columns:
        raise SystemExit("2330 not in adj_close")
    leaders["2330"] = daily_ret["2330"]
    print(f"  2330 台積電個股        : 1 檔")

    # 同期相關
    L = pd.DataFrame(leaders)
    print(f"\n[Leaders 同期相關矩陣]")
    print(L.corr().round(3).to_string())

    # === Targets ===
    target_xind = panel[DOWNSTREAM].mean(axis=1)

    # Broad target: 23 standard chain 中下游 mean of per-chain EW
    std_chains = sorted(chain[chain["is_standard_chain"] == "True"]["industry_code"].unique())
    chain_md = {}
    for ic in std_chains:
        d = chain[chain["industry_code"] == ic]
        mid = d[d["position"] == "中游"]["stock_id"].unique()
        dn = d[d["position"] == "下游"]["stock_id"].unique()
        members = list(set(list(mid) + list(dn)) & set(daily_ret.columns))
        if len(members) >= 5:
            chain_md[ic] = daily_ret[members].mean(axis=1)
    target_broad = pd.DataFrame(chain_md).mean(axis=1)
    print(f"\n[Targets]")
    print(f"  A. xind  = 7 跨產業下游 EW (G/H/I/J/F/L/5400)")
    print(f"  B. broad = {len(chain_md)} standard chain 中下游 super-EW")

    targets = {"xind": target_xind, "broad": target_broad}

    # === Predictor 構建 ===
    pred_d100 = leaders["D100"]
    pred_p13 = 0.5 * leaders["D100"] + 0.5 * leaders["2330"]
    pred_ew_raw = L.mean(axis=1)

    # 60d rolling z-score 後 EW
    L_z = (L - L.rolling(60).mean()) / L.rolling(60).std()
    pred_ew_z = L_z.mean(axis=1)

    # PCA PC1 (full sample, in-sample)
    L_clean = L.dropna()
    L_centered = L_clean - L_clean.mean(axis=0)
    cov_L = L_centered.cov().values
    eigvals, eigvecs = np.linalg.eigh(cov_L)
    pc1_v = eigvecs[:, -1]
    if pc1_v.sum() < 0:
        pc1_v = -pc1_v  # 確保 positive 方向
    var_explained = eigvals[-1] / eigvals.sum()
    pred_pca = pd.Series(L.values @ pc1_v, index=L.index)
    print(f"\n[PCA PC1] var explained = {var_explained:.3f}")
    print(f"  loadings: {dict(zip(L.columns, np.round(pc1_v, 3)))}")

    # Orth-sum: 對 D100, mkt 殘差化其餘 5 個 leader, 然後 sum
    df_orth = pd.concat({**leaders, "mkt": market}, axis=1).dropna()
    others = [c for c in L.columns if c != "D100"]  # GA00, P200, P600, IA00, 2330
    resid_components = []
    print(f"\n[Orth-sum] 殘差化結果 (對 D100, mkt):")
    for o in others:
        res = sm.OLS(df_orth[o], sm.add_constant(df_orth[["D100", "mkt"]])).fit()
        resid = df_orth[o] - res.predict(sm.add_constant(df_orth[["D100", "mkt"]]))
        var_ratio = resid.var() / df_orth[o].var()
        print(f"  {o:<6s}: β_D100={res.params['D100']:+.3f}, β_mkt={res.params['mkt']:+.3f}, "
              f"R²={res.rsquared:.3f}, idiosync var={var_ratio:.3f}")
        resid_components.append(resid)
    resid_sum = pd.concat(resid_components, axis=1).sum(axis=1)
    pred_orth = (df_orth["D100"] + resid_sum).reindex(daily_ret.index)

    # Static predictors
    predictors_static = {
        "D100":            pred_d100,
        "D100+0.5·2330":   pred_p13,
        "6L-EW-raw":       pred_ew_raw,
        "6L-EW-zscore":    pred_ew_z,
        "6L-PCA-PC1":      pred_pca,
        "6L-Orth-sum":     pred_orth,
    }

    # === OLS-oracle predictor: per target, full-sample look-ahead ===
    def fit_ols_oracle(target_series):
        y_lead = target_series.shift(-1)
        df = pd.concat([y_lead.rename("y"), L], axis=1).dropna()
        if len(df) < 200:
            return None, None
        res = sm.OLS(df["y"], sm.add_constant(df[L.columns])).fit()
        coefs = res.params.drop("const")
        pred = (L * coefs).sum(axis=1)
        return pred, coefs

    print(f"\n[OLS-oracle] in-sample target_{{t+1}} ~ leaders_t coefficients:")
    oracle_coefs_log = []
    target_predictors = {}
    for tlabel, t in targets.items():
        pred_ols, coefs = fit_ols_oracle(t)
        target_predictors[tlabel] = {**predictors_static, "6L-OLS-oracle": pred_ols}
        coefs_dict = coefs.to_dict()
        print(f"  target={tlabel}: ", {k: round(v, 4) for k, v in coefs_dict.items()})
        oracle_coefs_log.append({"target": tlabel, **coefs_dict})
    pd.DataFrame(oracle_coefs_log).to_csv(
        os.path.join(OUT, "phase14_ols_oracle_coefs.csv"),
        index=False, encoding="utf-8-sig"
    )

    # === Strategy sweep ===
    print("\n" + "=" * 100)
    print("Strategy sweep: 2 targets × 7 predictors × 5 N × 4 K × 5 cost = 1400 cells")
    print("=" * 100)

    rows = []
    for tlabel, tgt in targets.items():
        for pred_label, pred in target_predictors[tlabel].items():
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
                            "target": tlabel,
                            "predictor": pred_label,
                            "smooth_N": N,
                            "hold_K": K,
                            "cost_rt": cost_rt,
                            "avg_daily_turn": avg_turn,
                            "ann_turn": avg_turn * 252,
                        })
                        rows.append(p)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "phase14_cross_chain.csv"),
              index=False, encoding="utf-8-sig")

    # === Print best per (target, predictor) @ 50bp ===
    fmt = lambda x: f"{x:.4f}"
    cols_show = ["predictor", "smooth_N", "hold_K", "ann_ret", "ann_vol", "sharpe",
                 "ann_alpha", "t_alpha", "max_dd", "avg_daily_turn"]

    pred_order = ["D100", "D100+0.5·2330", "6L-EW-raw", "6L-EW-zscore",
                  "6L-PCA-PC1", "6L-Orth-sum", "6L-OLS-oracle"]

    for tlabel in ["xind", "broad"]:
        title = "7 跨產業下游" if tlabel == "xind" else "23 chain 中下游 broad"
        print("\n" + "=" * 100)
        print(f"Target {tlabel.upper()}: {title} — 各 predictor best @ 50bp (按 t_alpha)")
        print("=" * 100)
        sub = df[(df["target"] == tlabel) & (df["cost_rt"] == 0.0050)]
        best = (sub.sort_values("sharpe", ascending=False)
                .groupby("predictor", as_index=False).first())
        # 按 pred_order 排序印
        best["_rank"] = best["predictor"].map({p: i for i, p in enumerate(pred_order)})
        best = best.sort_values("t_alpha", ascending=False).drop(columns=["_rank"])
        print(best[cols_show].to_string(index=False, float_format=fmt))

    # === Phase 13 winner 同 (N, K) head-to-head ===
    print("\n" + "=" * 100)
    print("固定 (N=10, K=1) — Phase 13 winner 參數下各 predictor 表現 @ 50bp")
    print("=" * 100)
    for tlabel in ["xind", "broad"]:
        title = "7 跨產業下游" if tlabel == "xind" else "23 chain 中下游 broad"
        print(f"\n  Target {tlabel.upper()}: {title}")
        sub = df[(df["target"] == tlabel) & (df["smooth_N"] == 10) &
                  (df["hold_K"] == 1) & (df["cost_rt"] == 0.0050)]\
              .sort_values("t_alpha", ascending=False)
        for _, r in sub.iterrows():
            print(f"    {r['predictor']:<22s}  ann_ret={r['ann_ret']:>+.4f}  "
                  f"sharpe={r['sharpe']:>+.3f}  alpha={r['ann_alpha']:>+.4f}  t_α={r['t_alpha']:>+.3f}")

    # === @ 80bp 高成本 ===
    print("\n" + "=" * 100)
    print("各 predictor best @ 80bp (按 t_alpha 排序, target=xind)")
    print("=" * 100)
    sub80 = df[(df["target"] == "xind") & (df["cost_rt"] == 0.0080)]
    best80 = (sub80.sort_values("sharpe", ascending=False)
              .groupby("predictor", as_index=False).first()
              .sort_values("t_alpha", ascending=False))
    print(best80[cols_show].to_string(index=False, float_format=fmt))

    # === Visualization 1: methods comparison bar ===
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, tlabel in zip(axes, ["xind", "broad"]):
        title = "7 跨產業下游" if tlabel == "xind" else "23 chain broad 中下游"
        sub = df[(df["target"] == tlabel) & (df["cost_rt"] == 0.0050)]
        best = (sub.sort_values("sharpe", ascending=False)
                .groupby("predictor", as_index=False).first()
                .sort_values("t_alpha"))
        colors = ["#1b5e20" if t > 5 else "#558b2f" if t > 3 else
                  "#fb8c00" if t > 1.96 else "#9e9e9e" if t > 0 else "#e53935"
                  for t in best["t_alpha"]]
        ax.barh(best["predictor"], best["t_alpha"], color=colors, edgecolor="black", linewidth=0.4)
        for i, (t, alpha) in enumerate(zip(best["t_alpha"], best["ann_alpha"])):
            ax.text(t + 0.08, i, f"α={alpha:+.3f}", va="center", fontsize=8.5)
        ax.axvline(1.96, color="black", lw=0.5, ls="--", alpha=0.5)
        ax.axvline(0, color="black", lw=0.5)
        ax.set_xlabel("t_alpha (best (N,K) @ 50bp)")
        ax.set_title(f"Target: {title}")
        ax.grid(axis="x", alpha=0.3)
    plt.suptitle("Phase 14: 跨鏈 leader 組合因子 — 7 整合方法比較", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig_phase14_methods_comparison.png"), dpi=130)
    plt.close()
    print(f"\n圖檔: fig_phase14_methods_comparison.png")

    # === Visualization 2: 累積曲線 ===
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, (tlabel, tgt) in zip(axes, [("xind", target_xind), ("broad", target_broad)]):
        title = "7 跨產業下游" if tlabel == "xind" else "23 chain broad 中下游"
        sub = df[(df["target"] == tlabel) & (df["cost_rt"] == 0.0050)]
        best = (sub.sort_values("sharpe", ascending=False)
                .groupby("predictor", as_index=False).first())

        plot_preds = ["D100", "D100+0.5·2330", "6L-EW-raw", "6L-PCA-PC1",
                       "6L-Orth-sum", "6L-OLS-oracle"]
        colors_map = {
            "D100": "#0277bd",
            "D100+0.5·2330": "#1b5e20",
            "6L-EW-raw": "#ef6c00",
            "6L-PCA-PC1": "#6a1b9a",
            "6L-Orth-sum": "#c62828",
            "6L-OLS-oracle": "#4527a0",
        }
        cum_bh = (1 + tgt.fillna(0)).cumprod()
        ax.plot(cum_bh, color="#999999", lw=1.0, alpha=0.7,
                label=f"Target B&H [{cum_bh.iloc[-1]:.1f}x]")
        for pred_label in plot_preds:
            best_row = best[best["predictor"] == pred_label].iloc[0]
            N, K = int(best_row["smooth_N"]), int(best_row["hold_K"])
            gross, _, turn = strategy(target_predictors[tlabel][pred_label], tgt, N, K)
            net = (gross - turn * 0.0025).fillna(0)
            cum = (1 + net).cumprod()
            ax.plot(cum, color=colors_map.get(pred_label, "black"), lw=1.4, alpha=0.85,
                    label=f"{pred_label} (N={N},K={K}) [{cum.iloc[-1]:.1f}x, t={best_row['t_alpha']:.2f}]")
        ax.set_yscale("log")
        ax.set_title(f"Target: {title}")
        ax.set_xlabel("date"); ax.set_ylabel("累積資本 (log)")
        ax.legend(loc="upper left", fontsize=8.5)
        ax.grid(alpha=0.3)
    plt.suptitle("Phase 14: 跨鏈 leader 組合因子 累積曲線 (扣 50bp)", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig_phase14_curves.png"), dpi=130)
    plt.close()
    print(f"圖檔: fig_phase14_curves.png")

    # === 結論摘要 ===
    print("\n" + "=" * 100)
    print("結論摘要")
    print("=" * 100)
    for tlabel in ["xind", "broad"]:
        title = "7 跨產業下游" if tlabel == "xind" else "23 chain broad"
        sub = df[(df["target"] == tlabel) & (df["cost_rt"] == 0.0050)]
        best_per = (sub.sort_values("sharpe", ascending=False)
                    .groupby("predictor", as_index=False).first())
        base = best_per[best_per["predictor"] == "D100+0.5·2330"].iloc[0]
        print(f"\nTarget {tlabel.upper()} ({title}) baseline = D100+0.5·2330: "
              f"alpha={base['ann_alpha']:.4f}, t={base['t_alpha']:.3f}, Sharpe={base['sharpe']:.3f}")
        for _, row in best_per.sort_values("t_alpha", ascending=False).iterrows():
            if row["predictor"] == "D100+0.5·2330":
                continue
            d_alpha = (row["ann_alpha"] - base["ann_alpha"]) * 100
            d_t = row["t_alpha"] - base["t_alpha"]
            d_sh = row["sharpe"] - base["sharpe"]
            flag = " ✓" if d_alpha > 1.0 and d_t > 0.3 else \
                   (" ↑" if d_alpha > 0 else " ↓")
            print(f"  {row['predictor']:<22s} alpha={row['ann_alpha']:.4f}  "
                  f"t={row['t_alpha']:>+.3f}  Sh={row['sharpe']:>+.3f}  "
                  f"Δα={d_alpha:+.2f}pp  Δt={d_t:+.2f}{flag}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
