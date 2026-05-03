"""
Phase 19: Community Detection + 跨群組合因子

承接 Phase 18 的 40-node directed network, 使用 Louvain modularity 找產業 community,
從不同 community 各取 representative leader, 測試是否能突破 D100+2330 雙因子的天花板.

Phase 18 已知:
- R300 電子商務 out-degree=39 (對 39 個產業顯著正向領先) — 疑似高 beta artifact
- D000 半導體 為 #2 hub, 鋼鐵 Q000 為 #1 負向 hub
- HITS authority top: G/I/K/5200 — 多訊號接收者

本 Phase 流程:
1. 用 G_pos 對稱化為無向圖, edge weight = max(|t_AB|, |t_BA|)
2. Louvain modularity / Greedy modularity 找 community
3. 過濾 R300/5700 等可能 artifact (out_count > 30 視為 high-beta noise)
4. 各 community 取代表 leader (出邊最強, 排除 artifact)
5. 構建跨群組合 predictor:
   - 等權平均 (signal EW)
   - PCA PC1
   - OLS-oracle (look-ahead 上限)
6. 對照 Phase 13 D100+2330 baseline 與 Phase 14 ad-hoc 6-leader

輸出:
- phase19_communities.csv : 每個 industry 的 community 標籤
- phase19_community_combo.csv : 各 predictor 的 sweep 結果
- fig_phase19_community_graph.png : 上色後的 network graph
- fig_phase19_combo_comparison.png : 跨群組合 vs Phase 13 baseline
"""
from __future__ import annotations

import os
import pickle
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
import networkx as nx

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
COSTS = [0.000, 0.0050, 0.0080]
T_THRESHOLD = 2.0
ARTIFACT_OUT_DEGREE = 30  # out_count > 30 視為市場先行 artifact (R300/5700)


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
        "n": len(s), "ann_ret": s.mean() * ann, "ann_vol": s.std() * np.sqrt(ann),
        "sharpe": s.mean()/s.std()*np.sqrt(ann) if s.std() > 0 else np.nan,
        "max_dd": (cum/cum.cummax() - 1).min(),
        "ann_alpha": res.params["const"]*ann, "t_alpha": res.tvalues["const"],
        "beta": res.params["b"],
    }


def strategy(predictor, target, smooth_n, hold_k):
    sig = predictor.ewm(span=smooth_n, adjust=False).mean() if smooth_n > 1 else predictor
    target_w = (sig > 0).astype(float).shift(1)
    weight = k_day_holding(target_w, hold_k)
    gross = weight * target
    turn = weight.diff().abs().fillna(weight.iloc[0])
    return gross, weight, turn


def main():
    print("=" * 100)
    print("Phase 19: Community Detection + 跨群組合因子")
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
    industries = list(panel.columns)

    # === Load Phase 18 t-matrix ===
    df_links = pd.read_csv(os.path.join(OUT, "phase18_t_matrix.csv"), dtype={"A": str, "B": str})
    cent_df = pd.read_csv(os.path.join(OUT, "phase18_centrality.csv"), dtype={"industry_code": str})

    # === 標記 artifact 產業 (out_count 過大) ===
    artifacts = set(cent_df[cent_df["out_degree"] > ARTIFACT_OUT_DEGREE]["industry_code"])
    print(f"\n[Artifact 過濾] out_degree > {ARTIFACT_OUT_DEGREE} 視為市場先行 artifact:")
    for ic in artifacts:
        row = cent_df[cent_df["industry_code"] == ic].iloc[0]
        print(f"  {ic} {row['industry_name']:<10s}  out_count={int(row['out_degree'])}, n_stocks={int(row['n_stocks'])} → 排除")

    # === 建無向 weighted graph (對稱化) ===
    G_und = nx.Graph()
    for ic in industries:
        G_und.add_node(ic, name=industry_meta[ic])

    # 對稱化: edge weight = max(t_AB, t_BA) (only positive edges)
    edges_dict = {}
    for _, r in df_links.iterrows():
        if r["t_AB"] <= T_THRESHOLD:
            continue
        a, b = r["A"], r["B"]
        if a in artifacts or b in artifacts:
            continue
        key = tuple(sorted([a, b]))
        edges_dict.setdefault(key, []).append(r["t_AB"])
    for (a, b), ts in edges_dict.items():
        G_und.add_edge(a, b, weight=max(ts))

    print(f"\n[Undirected positive graph] {G_und.number_of_nodes()} nodes, {G_und.number_of_edges()} edges")

    # === Community Detection (Greedy Modularity) ===
    communities = list(nx.community.greedy_modularity_communities(G_und, weight="weight"))
    print(f"\n[Communities] 找到 {len(communities)} 個 community")

    # 每個 industry 的 community 標籤
    ic_to_comm = {}
    for cid, comm in enumerate(communities):
        for ic in comm:
            ic_to_comm[ic] = cid
    # isolated nodes 都歸到 community -1
    for ic in industries:
        if ic not in ic_to_comm:
            ic_to_comm[ic] = -1

    cent_df["community"] = cent_df["industry_code"].map(ic_to_comm).fillna(-1).astype(int)
    cent_df.to_csv(os.path.join(OUT, "phase19_communities.csv"),
                   index=False, encoding="utf-8-sig")

    # === 列出每個 community ===
    print(f"\n各 community 成員 (按 out_weighted 排序):")
    for cid in sorted(set(cent_df["community"])):
        sub = cent_df[cent_df["community"] == cid].sort_values("out_weighted", ascending=False)
        if cid == -1:
            print(f"\n  Community -1 (isolated, {len(sub)} 個):")
        else:
            print(f"\n  Community {cid} ({len(sub)} 個):")
        for _, r in sub.iterrows():
            artifact_flag = " [ARTIFACT]" if r["industry_code"] in artifacts else ""
            print(f"    {r['industry_code']} {r['industry_name']:<14s}  "
                  f"out_w={r['out_weighted']:.2f}, out_count={int(r['out_degree'])}, "
                  f"n_stocks={int(r['n_stocks'])}{artifact_flag}")

    # === 各 community 取 representative leader (出邊最強, 排除 artifact) ===
    leaders = {}
    for cid in sorted(set(cent_df["community"])):
        if cid == -1:
            continue
        sub = cent_df[(cent_df["community"] == cid) &
                      (~cent_df["industry_code"].isin(artifacts))].copy()
        sub = sub[sub["out_weighted"] > 0].sort_values("out_weighted", ascending=False)
        if len(sub) == 0:
            continue
        leader = sub.iloc[0]
        leaders[cid] = {
            "industry_code": leader["industry_code"],
            "industry_name": leader["industry_name"],
            "out_weighted": leader["out_weighted"],
            "n_stocks": int(leader["n_stocks"]),
        }
        print(f"  Community {cid} representative leader: "
              f"{leader['industry_code']} {leader['industry_name']:<10s} (out_w={leader['out_weighted']:.2f})")

    print(f"\n選出 {len(leaders)} 個 community leaders")

    # === 構建 leader portfolios + 2330 macro ===
    leader_panels = {}
    for cid, info in leaders.items():
        leader_panels[info["industry_code"]] = panel[info["industry_code"]]

    # 加入 2330 (Phase 13 macro)
    leader_panels["2330"] = daily_ret["2330"]

    L_df = pd.DataFrame(leader_panels)
    print(f"\n[Cross-community leader 同期相關]")
    print(L_df.corr().round(3).to_string())

    # === Predictors ===
    # D100 reference
    d100_members = chain[(chain["industry_code"] == "D000") &
                          (chain["sub_code"] == "D100")]["stock_id"].unique().tolist()
    d100_cols = [s for s in d100_members if s in daily_ret.columns]
    d100 = daily_ret[d100_cols].mean(axis=1)

    target_xind = panel[DOWNSTREAM].mean(axis=1)

    predictors = {
        "D100":           d100,
        "D100+0.5·2330":  0.5 * d100 + 0.5 * daily_ret["2330"],   # Phase 13 winner
        "comm-EW-raw":    L_df.mean(axis=1),
        "comm-PCA-PC1":   None,  # placeholder
        "comm-OLS-ora":   None,  # placeholder, 對 target 有 look-ahead
    }

    # PCA PC1 (full sample)
    L_clean = L_df.dropna()
    L_centered = L_clean - L_clean.mean(axis=0)
    cov_L = L_centered.cov().values
    eigvals, eigvecs = np.linalg.eigh(cov_L)
    pc1_v = eigvecs[:, -1]
    if pc1_v.sum() < 0:
        pc1_v = -pc1_v
    var_explained = eigvals[-1] / eigvals.sum()
    predictors["comm-PCA-PC1"] = pd.Series(L_df.values @ pc1_v, index=L_df.index)
    print(f"\n[PCA PC1] var explained = {var_explained:.3f}")
    print(f"  loadings: {dict(zip(L_df.columns, np.round(pc1_v, 3)))}")

    # OLS-oracle (look-ahead): target_{t+1} ~ leaders_t
    y_lead = target_xind.shift(-1)
    df_ols = pd.concat([y_lead.rename("y"), L_df], axis=1).dropna()
    res_ols = sm.OLS(df_ols["y"], sm.add_constant(df_ols[L_df.columns])).fit()
    coefs = res_ols.params.drop("const")
    print(f"\n[OLS-oracle] 對 xind target_{{t+1}} 係數:")
    for k, v in coefs.items():
        print(f"  {k}: {v:+.4f}")
    predictors["comm-OLS-ora"] = (L_df * coefs).sum(axis=1)

    # === Strategy sweep ===
    print(f"\n" + "=" * 100)
    print("Strategy sweep on xind target")
    print("=" * 100)

    rows = []
    for pred_label, pred in predictors.items():
        for N in PARAMS_N:
            for K in PARAMS_K:
                gross, _, turn = strategy(pred, target_xind, N, K)
                avg_turn = turn.mean()
                for cost_rt in COSTS:
                    net = gross - turn * cost_rt / 2
                    p = perf_with_alpha(net, market)
                    if not p:
                        continue
                    p.update({"predictor": pred_label, "smooth_N": N, "hold_K": K,
                              "cost_rt": cost_rt, "avg_daily_turn": avg_turn})
                    rows.append(p)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "phase19_community_combo.csv"),
              index=False, encoding="utf-8-sig")

    fmt = lambda x: f"{x:.4f}"
    cols_show = ["predictor", "smooth_N", "hold_K", "ann_ret", "ann_vol", "sharpe",
                  "ann_alpha", "t_alpha", "max_dd", "avg_daily_turn"]

    print("\n>>> 各 predictor best @ 50bp (按 t_alpha 排序):")
    sub50 = df[df["cost_rt"] == 0.0050]
    best50 = (sub50.sort_values("sharpe", ascending=False)
              .groupby("predictor", as_index=False).first()
              .sort_values("t_alpha", ascending=False))
    print(best50[cols_show].to_string(index=False, float_format=fmt))

    print("\n>>> 固定 (N=10, K=1) — Phase 13 winner 參數下 @ 50bp:")
    sub_p13 = df[(df["smooth_N"] == 10) & (df["hold_K"] == 1) & (df["cost_rt"] == 0.0050)]\
              .sort_values("t_alpha", ascending=False)
    for _, r in sub_p13.iterrows():
        print(f"  {r['predictor']:<22s}  ann_ret={r['ann_ret']:>+.4f}  "
              f"sharpe={r['sharpe']:>+.3f}  alpha={r['ann_alpha']:>+.4f}  t_α={r['t_alpha']:>+.3f}")

    # === Visualization 1: community graph ===
    fig, ax = plt.subplots(figsize=(13, 11))
    pos = nx.spring_layout(G_und, seed=42, k=1.0/np.sqrt(max(1, G_und.number_of_nodes())))
    n_comms = max(set(ic_to_comm.values())) + 1
    cmap = plt.cm.tab10
    node_colors = [cmap(ic_to_comm[n] / max(1, n_comms))
                   if ic_to_comm[n] >= 0 else "#dddddd"
                   for n in G_und.nodes]
    node_sizes = []
    for n in G_und.nodes:
        cw = cent_df[cent_df["industry_code"] == n]["out_weighted"]
        node_sizes.append(cw.iloc[0] * 30 + 200 if len(cw) > 0 else 200)
    nx.draw_networkx_nodes(G_und, pos, node_size=node_sizes, node_color=node_colors,
                           edgecolors="black", linewidths=0.6, alpha=0.85, ax=ax)
    edge_widths = [G_und[u][v]["weight"] / 3 for u, v in G_und.edges]
    nx.draw_networkx_edges(G_und, pos, width=edge_widths, alpha=0.3, ax=ax)
    labels = {n: f"{n}\n{industry_meta[n][:6]}" for n in G_und.nodes}
    nx.draw_networkx_labels(G_und, pos, labels=labels, font_size=8.5,
                            font_family="Microsoft JhengHei", ax=ax)
    # 圈出 leaders
    for cid, info in leaders.items():
        ic = info["industry_code"]
        if ic in pos:
            x, y = pos[ic]
            ax.scatter(x, y, s=node_sizes[list(G_und.nodes).index(ic)] * 1.3,
                       facecolor="none", edgecolor="red", linewidth=2.5, zorder=10)
    ax.set_title(f"Phase 19: Community detection ({n_comms} communities, leader 紅圈標記)\n"
                 f"已排除 artifact: {', '.join(sorted(artifacts))}", fontsize=11)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig_phase19_community_graph.png"), dpi=130)
    plt.close()
    print(f"\n圖檔: fig_phase19_community_graph.png")

    # === Visualization 2: combo comparison bar ===
    fig, ax = plt.subplots(figsize=(11, 5))
    plot_df = best50.sort_values("t_alpha")
    colors = ["#1b5e20" if t > 5 else "#558b2f" if t > 3 else "#fb8c00" for t in plot_df["t_alpha"]]
    ax.barh(plot_df["predictor"], plot_df["t_alpha"], color=colors, edgecolor="black", lw=0.4)
    for i, (t, alpha) in enumerate(zip(plot_df["t_alpha"], plot_df["ann_alpha"])):
        ax.text(t + 0.08, i, f"α={alpha:+.3f}", va="center", fontsize=9)
    ax.axvline(1.96, color="black", lw=0.5, ls="--", alpha=0.5)
    ax.axvline(0, color="black", lw=0.5)
    ax.set_xlabel("t_alpha (best (N,K) @ 50bp on xind target)")
    ax.set_title(f"Phase 19: 跨 community 組合 vs Phase 13 baseline")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig_phase19_combo_comparison.png"), dpi=130)
    plt.close()
    print(f"圖檔: fig_phase19_combo_comparison.png")

    # === 結論 ===
    print("\n" + "=" * 100)
    print("結論")
    print("=" * 100)
    base = best50[best50["predictor"] == "D100+0.5·2330"].iloc[0]
    print(f"\nbaseline Phase 13 D100+0.5·2330: alpha={base['ann_alpha']:.4f}, t={base['t_alpha']:.3f}, Sh={base['sharpe']:.3f}")
    for _, r in best50.iterrows():
        if r["predictor"] == "D100+0.5·2330":
            continue
        d_alpha = (r["ann_alpha"] - base["ann_alpha"]) * 100
        d_t = r["t_alpha"] - base["t_alpha"]
        flag = " ✓" if d_alpha > 1.0 and d_t > 0.3 else (" ↑" if d_alpha > 0 else " ↓")
        print(f"  {r['predictor']:<22s} alpha={r['ann_alpha']:.4f}  t={r['t_alpha']:>+.3f}  "
              f"Sh={r['sharpe']:>+.3f}  Δα={d_alpha:+.2f}pp  Δt={d_t:+.2f}{flag}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
