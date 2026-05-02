"""
Phase 7: 跨產業 (cross-industry) link-based spillover

理念 (Cohen-Frazzini 2008 firm-level → 改建為產業層級):
    一家公司若同時出現在「產業 A 的下游」與「產業 B 的上游」,
    則代表此公司既消費 A 的產出, 又作為 B 的投入供應端,
    即 A 之輸出 → 此公司 → B 之需求, 構成 A → B 供應鏈連結.

連結強度:
    L[A→B] = |stocks in A.下游 ∩ stocks in B.上游|         (純位置式)
    M[A→B] = |stocks in A.中游 or 下游 ∩ stocks in B.上游 or 中游|  (寬鬆式)
    S[A,B] = |stocks in A ∩ stocks in B|                     (無向相似度)

檢驗步驟:
1. 構 40 個產業的月頻 EW 報酬 (各產業所有公司平均)
2. 構 L[A→B] 矩陣 (限 26 條標準鏈, 雙方都有 上游/下游)
3. 對 top-N linked pairs 做雙向預測迴歸:
       r_B(t+1) = α + β·r_A(t) + γ·r_B(t) + δ·r_market(t) + ε
4. Placebo: 隨機抽 link=0 的 pair, 比較 mean β
5. 統計每個產業 "predictor power" (出邊 mean t) 與 "predicted susceptibility"
6. 視覺化: 鄰接矩陣 + spillover 矩陣 + 網絡圖

輸出:
- cross_industry_links.csv (有向連結)
- cross_industry_spillover.csv (各 pair 雙向迴歸)
- cross_industry_node_scores.csv (預測力/被預測力)
- fig_link_heatmap.png
- fig_spillover_heatmap.png
"""
from __future__ import annotations

import os
import pickle
import sys
from itertools import product

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

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
    df = df.set_index("date")
    df.index = pd.to_datetime(df.index)
    df.columns = df.columns.astype(str)
    return df


def filter_common(df):
    keep = [c for c in df.columns if c.isdigit() and len(c) == 4 and not c.startswith("00")]
    return df[keep]


def main():
    print("[Phase 7] 跨產業 link-based spillover\n")

    # === 1. 載入資料 ===
    chain = pd.read_csv(os.path.join(ROOT, "industry_chain.csv"), dtype=str)
    industries = chain[["industry_code", "industry_name"]].drop_duplicates().set_index("industry_code")["industry_name"].to_dict()
    print(f"產業數: {len(industries)}")

    adj_close = filter_common(load_pickle_df("etl#adj_close.pickle"))
    daily_ret = adj_close.pct_change().iloc[1:]
    print(f"日報酬: {daily_ret.shape}")

    # === 2. 各產業月頻 EW 報酬 ===
    print("\n構建產業月報酬 (40 產業, EW)...")
    log_d = np.log1p(daily_ret)
    monthly_log = log_d.resample("ME").sum(min_count=10)
    monthly_simple = np.expm1(monthly_log)

    industry_monthly = {}
    industry_member_size = {}
    for ind in industries:
        members = chain[chain["industry_code"] == ind]["stock_id"].unique().tolist()
        members = [s for s in members if s in monthly_simple.columns]
        if len(members) >= 3:
            industry_monthly[ind] = monthly_simple[members].mean(axis=1)
            industry_member_size[ind] = len(members)
    panel_m = pd.DataFrame(industry_monthly)
    print(f"  納入產業: {panel_m.shape[1]} (≥3 家成員); 月數: {panel_m.shape[0]}")

    # market: 全 EW
    market_m = monthly_simple.mean(axis=1)

    # 也建 weekly
    weekly_log = log_d.resample("W-FRI").sum(min_count=3)
    weekly_simple = np.expm1(weekly_log)
    industry_weekly = {}
    for ind in industries:
        members = chain[chain["industry_code"] == ind]["stock_id"].unique().tolist()
        members = [s for s in members if s in weekly_simple.columns]
        if len(members) >= 3:
            industry_weekly[ind] = weekly_simple[members].mean(axis=1)
    panel_w = pd.DataFrame(industry_weekly)
    market_w = weekly_simple.mean(axis=1)

    # === 3. 連結矩陣 ===
    print("\n構建連結矩陣...")
    # L[A→B] = stocks in A.下游 ∩ stocks in B.上游
    A_down = {ind: set(chain[(chain["industry_code"] == ind) & (chain["position"] == "下游")]["stock_id"].unique())
              for ind in industries}
    B_up = {ind: set(chain[(chain["industry_code"] == ind) & (chain["position"] == "上游")]["stock_id"].unique())
            for ind in industries}
    A_midown = {ind: set(chain[(chain["industry_code"] == ind) & (chain["position"].isin(["中游", "下游"]))]["stock_id"].unique())
                for ind in industries}
    B_upmid = {ind: set(chain[(chain["industry_code"] == ind) & (chain["position"].isin(["上游", "中游"]))]["stock_id"].unique())
               for ind in industries}
    all_in = {ind: set(chain[chain["industry_code"] == ind]["stock_id"].unique()) for ind in industries}

    inds_have_pos = sorted([ind for ind in industries if A_down[ind] and B_up[ind]])
    print(f"  有 上游+下游 標籤的產業: {len(inds_have_pos)}")

    rows = []
    for A in industries:
        for B in industries:
            if A == B: continue
            L_strict = len(A_down[A] & B_up[B])
            L_loose = len(A_midown[A] & B_upmid[B])
            S = len(all_in[A] & all_in[B])
            rows.append({
                "A": A, "A_name": industries[A],
                "B": B, "B_name": industries[B],
                "L_strict": L_strict, "L_loose": L_loose, "S_undir": S,
            })
    link_df = pd.DataFrame(rows)
    link_df.to_csv(os.path.join(OUT, "cross_industry_links.csv"), index=False, encoding="utf-8-sig")
    print(f"  link 矩陣寫入 cross_industry_links.csv ({len(link_df)} 列)")

    print(f"  L_strict 分布: max={link_df['L_strict'].max()}, mean={link_df['L_strict'].mean():.2f}")
    print(f"  L_loose  分布: max={link_df['L_loose'].max()},  mean={link_df['L_loose'].mean():.2f}")
    print(f"  S_undir  分布: max={link_df['S_undir'].max()},  mean={link_df['S_undir'].mean():.2f}")

    print("\n  Top 20 directed links by L_loose:")
    print(link_df.nlargest(20, "L_loose")[["A_name", "B_name", "L_strict", "L_loose", "S_undir"]].to_string(index=False))

    # === 4. 雙向預測迴歸 (對所有 directed pairs, 不限 linked) ===
    print("\n執行雙向預測迴歸 (全 pair, 月頻)...")
    panel_m.index = pd.to_datetime(panel_m.index)
    market_m.index = pd.to_datetime(market_m.index)

    reg_rows = []
    pairs = [(A, B) for A in panel_m.columns for B in panel_m.columns if A != B]
    for A, B in pairs:
        df = pd.DataFrame({
            "y": panel_m[B],
            "x_A_lag": panel_m[A].shift(1),
            "x_B_lag": panel_m[B].shift(1),
            "x_M_lag": market_m.shift(1),
        }).dropna()
        if len(df) < 60:
            continue
        try:
            res = sm.OLS(df["y"], sm.add_constant(df[["x_A_lag", "x_B_lag", "x_M_lag"]])).fit(
                cov_type="HAC", cov_kwds={"maxlags": 3}
            )
            reg_rows.append({
                "A": A, "B": B,
                "n": len(df),
                "beta_AB": res.params["x_A_lag"],
                "t_AB": res.tvalues["x_A_lag"],
                "p_AB": res.pvalues["x_A_lag"],
                "r2": res.rsquared,
            })
        except Exception:
            continue
    reg_df = pd.DataFrame(reg_rows)
    print(f"  共 {len(reg_df)} 個 directed pairs 完成")

    # 合併 link 強度
    reg_df = reg_df.merge(link_df, on=["A", "B"], how="left")
    reg_df["A_name"] = reg_df["A"].map(industries)
    reg_df["B_name"] = reg_df["B"].map(industries)
    reg_df.to_csv(os.path.join(OUT, "cross_industry_spillover.csv"), index=False, encoding="utf-8-sig")
    print(f"  迴歸結果寫入 cross_industry_spillover.csv")

    # === 5. linked vs random (placebo) ===
    print("\n=== Linked vs Random Placebo ===")
    linked = reg_df[reg_df["L_loose"] >= 3].copy()
    unlinked = reg_df[reg_df["L_loose"] == 0].copy()
    print(f"  Linked (L_loose ≥ 3): n={len(linked)}, mean β = {linked['beta_AB'].mean():.4f}, "
          f"mean t = {linked['t_AB'].mean():.3f}, share sig5_pos = {((linked['p_AB']<0.05) & (linked['beta_AB']>0)).mean():.3f}")
    print(f"  Unlinked (L_loose = 0): n={len(unlinked)}, mean β = {unlinked['beta_AB'].mean():.4f}, "
          f"mean t = {unlinked['t_AB'].mean():.3f}, share sig5_pos = {((unlinked['p_AB']<0.05) & (unlinked['beta_AB']>0)).mean():.3f}")

    # 雙樣本 t 檢定: linked β > unlinked β
    t_test = stats.ttest_ind(linked["beta_AB"].dropna(), unlinked["beta_AB"].dropna(), equal_var=False)
    print(f"  H0: β(linked) = β(unlinked); two-sample t = {t_test.statistic:.4f}, p = {t_test.pvalue:.4f}")

    # 各 link bucket
    print("\n  按 L_loose 分桶:")
    reg_df["link_bucket"] = pd.cut(reg_df["L_loose"], bins=[-0.5, 0.5, 2.5, 5.5, 999],
                                    labels=["L=0", "L=1-2", "L=3-5", "L>=6"])
    print(reg_df.groupby("link_bucket", observed=True).agg(
        n=("beta_AB", "count"),
        mean_beta=("beta_AB", "mean"),
        mean_t=("t_AB", "mean"),
        share_pos=("beta_AB", lambda x: (x > 0).mean()),
        share_sig5_pos=("p_AB", lambda x: ((x < 0.05) & (reg_df.loc[x.index, "beta_AB"] > 0)).mean()),
    ).to_string(float_format=lambda x: f"{x:.4f}"))

    # === 6. Top spillover pairs ===
    print("\n=== Top 25 cross-industry directed spillovers (by t_AB) ===")
    top = reg_df.sort_values("t_AB", ascending=False).head(25)
    print(top[["A_name", "B_name", "L_strict", "L_loose", "beta_AB", "t_AB", "p_AB"]]
          .to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # === 7. Predictor / Receiver 排名 ===
    print("\n=== 各產業作為 Predictor (出邊 mean t_AB) Top 10 ===")
    sig_only = reg_df[reg_df["L_loose"] >= 1]
    pred_score = sig_only.groupby("A_name").agg(
        n_targets=("B", "count"),
        mean_t=("t_AB", "mean"),
        mean_beta=("beta_AB", "mean"),
        n_sig5_pos=("p_AB", lambda x: ((x < 0.05) & (sig_only.loc[x.index, "beta_AB"] > 0)).sum()),
    ).sort_values("mean_t", ascending=False)
    print(pred_score.head(10).to_string(float_format=lambda x: f"{x:.4f}"))

    print("\n=== 各產業作為 Receiver (入邊 mean t_AB) Top 10 ===")
    recv_score = sig_only.groupby("B_name").agg(
        n_sources=("A", "count"),
        mean_t=("t_AB", "mean"),
        mean_beta=("beta_AB", "mean"),
        n_sig5_pos=("p_AB", lambda x: ((x < 0.05) & (sig_only.loc[x.index, "beta_AB"] > 0)).sum()),
    ).sort_values("mean_t", ascending=False)
    print(recv_score.head(10).to_string(float_format=lambda x: f"{x:.4f}"))

    pred_score.to_csv(os.path.join(OUT, "cross_industry_predictor_scores.csv"), encoding="utf-8-sig")
    recv_score.to_csv(os.path.join(OUT, "cross_industry_receiver_scores.csv"), encoding="utf-8-sig")

    # === 8. 視覺化 ===
    print("\n繪圖...")
    plot_industries = [ind for ind in panel_m.columns if industry_member_size.get(ind, 0) >= 5]
    plot_industries = sorted(plot_industries, key=lambda x: industries[x])

    # link heatmap (L_loose)
    L_mat = pd.DataFrame(0, index=plot_industries, columns=plot_industries)
    for _, row in link_df.iterrows():
        if row["A"] in plot_industries and row["B"] in plot_industries:
            L_mat.loc[row["A"], row["B"]] = row["L_loose"]

    fig, ax = plt.subplots(figsize=(11, 9.5))
    im = ax.imshow(np.log1p(L_mat.values), cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(plot_industries)))
    ax.set_yticks(range(len(plot_industries)))
    ax.set_xticklabels([industries[i] for i in plot_industries], rotation=90, fontsize=8)
    ax.set_yticklabels([industries[i] for i in plot_industries], fontsize=8)
    ax.set_xlabel("B (上游)")
    ax.set_ylabel("A (下游)")
    ax.set_title("跨產業 Link 強度: L_loose[A→B] = |A 中下游 ∩ B 上中游|, log1p scale")
    fig.colorbar(im, ax=ax, fraction=0.04)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig_link_heatmap.png"), dpi=120)
    plt.close()
    print("  fig_link_heatmap.png")

    # spillover heatmap (t_AB)
    T_mat = pd.DataFrame(np.nan, index=plot_industries, columns=plot_industries)
    for _, row in reg_df.iterrows():
        if row["A"] in plot_industries and row["B"] in plot_industries:
            T_mat.loc[row["A"], row["B"]] = row["t_AB"]

    fig, ax = plt.subplots(figsize=(11, 9.5))
    im = ax.imshow(T_mat.values, cmap="RdBu_r", aspect="auto", vmin=-3, vmax=3)
    ax.set_xticks(range(len(plot_industries)))
    ax.set_yticks(range(len(plot_industries)))
    ax.set_xticklabels([industries[i] for i in plot_industries], rotation=90, fontsize=8)
    ax.set_yticklabels([industries[i] for i in plot_industries], fontsize=8)
    ax.set_xlabel("B (被預測)")
    ax.set_ylabel("A (預測者)")
    ax.set_title("跨產業 Spillover t-stat: r_B(t+1) = β·r_A(t) + ... (Newey-West)")
    fig.colorbar(im, ax=ax, fraction=0.04)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig_spillover_heatmap.png"), dpi=120)
    plt.close()
    print("  fig_spillover_heatmap.png")

    # link 強度 vs t-stat 散點
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.scatter(reg_df["L_loose"], reg_df["t_AB"], alpha=0.4, s=20)
    ax.axhline(1.96, ls="--", color="green", alpha=0.5, label="t = ±1.96")
    ax.axhline(-1.96, ls="--", color="red", alpha=0.5)
    ax.axhline(0, ls="-", color="gray", alpha=0.3)
    ax.set_xlabel("L_loose (link 強度)")
    ax.set_ylabel("t_AB (spillover 顯著性)")
    ax.set_title("Link 強度 vs Cross-industry Spillover t-stat")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig_link_vs_tstat.png"), dpi=120)
    plt.close()
    print("  fig_link_vs_tstat.png")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
