"""
Phase 18: 鏈間網絡分析 — 40 產業 spillover graph 建構 + 中心性

動機
- Phase 7 部分配對 spillover 已知 D000 為 hub, 但未系統化
- 將 40 個產業視為 graph node, A → B (weight=t_AB) 為 lagged regression 顯著邊
- 用 networkx 計算多種 centrality, 找系統化 hub 排序

Pipeline
1. 對 40×39=1560 個 (A, B) 跑 lagged regression: r_B(t) ~ const + β·r_A(t-1) + γ·r_B(t-1) + δ·r_mkt(t-1)
2. 建 directed weighted graph: edge A → B 存在 iff |t_AB| > 2.0, weight = t_AB
3. 計算 centrality:
   - out_degree (t-weighted)             : 領先樞紐
   - in_degree                            : Receiver
   - PageRank                             : 訊息匯聚度 (受 hub 影響者)
   - Eigenvector                          : 遞迴重要性
   - HITS hub / authority                 : 雙模式
   - Betweenness                          : 訊息橋樑

輸出
- phase18_t_matrix.csv          : 40×40 t-stat matrix (long format)
- phase18_beta_matrix.csv       : 40×40 β-matrix (long)
- phase18_centrality.csv        : 各 industry 的 6 種 centrality 排名
- fig_phase18_t_heatmap.png     : t-stat 熱圖
- fig_phase18_centrality.png    : 各 centrality top-15 比較
- fig_phase18_network.png       : graph visualization (top edges only)
"""
from __future__ import annotations

import os
import pickle
import sys
import time

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

T_THRESHOLD = 2.0  # |t| > 2 才建邊


def load_pickle_df(name):
    with open(os.path.join(DB, name), "rb") as f:
        df = pickle.load(f)
    df = df.set_index("date"); df.index = pd.to_datetime(df.index)
    df.columns = df.columns.astype(str)
    return df


def filter_common(df):
    keep = [c for c in df.columns if c.isdigit() and len(c) == 4 and not c.startswith("00")]
    return df[keep]


def main():
    print("=" * 100)
    print("Phase 18: 40 產業 spillover 網絡 + 中心性分析")
    print("=" * 100)

    chain = pd.read_csv(os.path.join(ROOT, "industry_chain.csv"), dtype=str)
    industry_meta = (chain[["industry_code", "industry_name"]].drop_duplicates()
                     .set_index("industry_code")["industry_name"].to_dict())

    adj_close = filter_common(load_pickle_df("etl#adj_close.pickle"))
    daily_ret = adj_close.pct_change(fill_method=None).iloc[1:]
    market = daily_ret.mean(axis=1)

    # === Build industry panel ===
    panel = {}
    n_stocks = {}
    for ic in industry_meta:
        members = chain[chain["industry_code"] == ic]["stock_id"].unique().tolist()
        cols = [s for s in members if s in daily_ret.columns]
        if len(cols) >= 3:
            panel[ic] = daily_ret[cols].mean(axis=1)
            n_stocks[ic] = len(cols)
    panel = pd.DataFrame(panel)
    industries = list(panel.columns)
    print(f"\n納入 {len(industries)} 個產業 (≥3 股, in adj_close)")

    # === 40×40 lagged regression matrix ===
    print(f"\n[1] 跑 {len(industries)*(len(industries)-1)} 個 lagged regression...")
    t0 = time.time()

    # Pre-compute lagged versions
    panel_lag = panel.shift(1)
    market_lag = market.shift(1)

    rows = []
    n_pairs = 0
    for A in industries:
        x_lag = panel_lag[A]
        for B in industries:
            if A == B:
                continue
            y = panel[B]
            y_lag = panel_lag[B]
            df = pd.concat({"y": y, "x": x_lag, "yl": y_lag, "ml": market_lag},
                            axis=1).dropna()
            if len(df) < 250:
                continue
            try:
                res = sm.OLS(df["y"], sm.add_constant(df[["x", "yl", "ml"]]))\
                        .fit(cov_type="HAC", cov_kwds={"maxlags": 5})
                rows.append({
                    "A": A, "A_name": industry_meta[A], "n_A": n_stocks[A],
                    "B": B, "B_name": industry_meta[B], "n_B": n_stocks[B],
                    "n": len(df),
                    "beta_AB": res.params["x"], "t_AB": res.tvalues["x"],
                    "p_AB": res.pvalues["x"],
                })
                n_pairs += 1
            except Exception:
                continue
    print(f"  完成 {n_pairs} 個有效配對, 耗時 {time.time()-t0:.1f}s")

    df_links = pd.DataFrame(rows)
    df_links.to_csv(os.path.join(OUT, "phase18_t_matrix.csv"),
                    index=False, encoding="utf-8-sig")

    t_mat = df_links.pivot(index="A", columns="B", values="t_AB")
    beta_mat = df_links.pivot(index="A", columns="B", values="beta_AB")

    # === 摘要統計 ===
    sig_pos = ((df_links["t_AB"] > T_THRESHOLD)).sum()
    sig_neg = ((df_links["t_AB"] < -T_THRESHOLD)).sum()
    print(f"\n[Spillover edges]")
    print(f"  正向 |t| > {T_THRESHOLD}: {sig_pos} 條 ({sig_pos/n_pairs*100:.1f}%)")
    print(f"  負向 |t| > {T_THRESHOLD}: {sig_neg} 條 ({sig_neg/n_pairs*100:.1f}%)")
    print(f"  無向 (|t| > {T_THRESHOLD}): {sig_pos+sig_neg} 條 ({(sig_pos+sig_neg)/n_pairs*100:.1f}%)")

    # === Build directed graph (signed edges) ===
    G_pos = nx.DiGraph()  # 正向 spillover only
    G_signed = nx.DiGraph()  # 含負向, weight = |t_AB|
    G_signed_neg = nx.DiGraph()  # 純負向

    for ic in industries:
        for G in [G_pos, G_signed, G_signed_neg]:
            G.add_node(ic, name=industry_meta[ic])

    for _, r in df_links.iterrows():
        if r["t_AB"] > T_THRESHOLD:
            G_pos.add_edge(r["A"], r["B"], weight=r["t_AB"], beta=r["beta_AB"])
        if abs(r["t_AB"]) > T_THRESHOLD:
            G_signed.add_edge(r["A"], r["B"], weight=abs(r["t_AB"]),
                              t_signed=r["t_AB"], beta=r["beta_AB"])
        if r["t_AB"] < -T_THRESHOLD:
            G_signed_neg.add_edge(r["A"], r["B"], weight=abs(r["t_AB"]),
                                   beta=r["beta_AB"])

    print(f"\n[Graph G_pos] {G_pos.number_of_nodes()} nodes, {G_pos.number_of_edges()} edges")
    print(f"[Graph G_neg] {G_signed_neg.number_of_nodes()} nodes, {G_signed_neg.number_of_edges()} edges")

    # === Centrality 計算 (用 G_pos 為主) ===
    print(f"\n[2] Centrality 計算 (基於 G_pos, 正向 spillover only)")

    # Out-degree (t-weighted)
    out_w = {n: sum(d["weight"] for _, _, d in G_pos.out_edges(n, data=True))
             for n in G_pos.nodes}
    in_w = {n: sum(d["weight"] for _, _, d in G_pos.in_edges(n, data=True))
             for n in G_pos.nodes}
    out_count = {n: G_pos.out_degree(n) for n in G_pos.nodes}
    in_count = {n: G_pos.in_degree(n) for n in G_pos.nodes}

    # PageRank
    pr = nx.pagerank(G_pos, weight="weight", alpha=0.85)
    # Eigenvector centrality (might fail if disconnected)
    try:
        eig = nx.eigenvector_centrality_numpy(G_pos, weight="weight")
    except Exception as e:
        print(f"  eigenvector_centrality_numpy failed: {e}")
        eig = {n: np.nan for n in G_pos.nodes}
    # HITS (hub & authority)
    try:
        hubs, auths = nx.hits(G_pos, max_iter=1000)
    except Exception:
        hubs = {n: np.nan for n in G_pos.nodes}
        auths = {n: np.nan for n in G_pos.nodes}
    # Betweenness
    bc = nx.betweenness_centrality(G_pos, weight="weight", normalized=True)

    # Negative centrality (from G_signed_neg)
    out_neg = {n: G_signed_neg.out_degree(n) for n in G_signed_neg.nodes}

    cent_df = pd.DataFrame({
        "industry_code": list(G_pos.nodes),
        "industry_name": [industry_meta[n] for n in G_pos.nodes],
        "n_stocks": [n_stocks[n] for n in G_pos.nodes],
        "out_degree": [out_count[n] for n in G_pos.nodes],
        "out_weighted": [out_w[n] for n in G_pos.nodes],
        "in_degree": [in_count[n] for n in G_pos.nodes],
        "in_weighted": [in_w[n] for n in G_pos.nodes],
        "pagerank": [pr[n] for n in G_pos.nodes],
        "eigenvector": [eig[n] for n in G_pos.nodes],
        "hub": [hubs[n] for n in G_pos.nodes],
        "authority": [auths[n] for n in G_pos.nodes],
        "betweenness": [bc[n] for n in G_pos.nodes],
        "neg_out_degree": [out_neg.get(n, 0) for n in G_pos.nodes],
    })
    cent_df = cent_df.sort_values("out_weighted", ascending=False)
    cent_df.to_csv(os.path.join(OUT, "phase18_centrality.csv"),
                   index=False, encoding="utf-8-sig")

    print("\n>>> Top 15 by out-weighted degree (領先樞紐 — 對最多下游有顯著正向影響):")
    cols_show = ["industry_code", "industry_name", "n_stocks",
                  "out_degree", "out_weighted", "in_degree", "pagerank", "hub"]
    print(cent_df.head(15)[cols_show].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\n>>> Top 10 by HITS authority (受領先源頭影響最深):")
    print(cent_df.sort_values("authority", ascending=False).head(10)[
        ["industry_code","industry_name","authority","in_degree","in_weighted"]
    ].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\n>>> Top 10 by betweenness (訊息橋樑):")
    print(cent_df.sort_values("betweenness", ascending=False).head(10)[
        ["industry_code","industry_name","betweenness","out_degree","in_degree"]
    ].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\n>>> Top 10 by negative-out (對下游負向領先):")
    neg_top = cent_df.sort_values("neg_out_degree", ascending=False).head(10)
    print(neg_top[["industry_code","industry_name","neg_out_degree","out_degree"]]
          .to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # === 視覺化 1: t-stat heatmap ===
    fig, ax = plt.subplots(figsize=(13, 11))
    # 排序: 按 out_weighted 降序
    order = cent_df["industry_code"].tolist()
    t_mat_sorted = t_mat.reindex(index=order, columns=order)
    labels = [f"{ic} {industry_meta[ic][:6]}" for ic in order]

    vmax = max(abs(t_mat_sorted.min().min()), abs(t_mat_sorted.max().max()))
    im = ax.imshow(t_mat_sorted.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   aspect="auto")
    ax.set_xticks(range(len(order)))
    ax.set_yticks(range(len(order)))
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("B (target)")
    ax.set_ylabel("A (predictor)")
    ax.set_title(f"Phase 18: 40×40 spillover t-stat matrix (按 out-weighted 排序)")
    plt.colorbar(im, ax=ax, label="t-stat (A → B)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig_phase18_t_heatmap.png"), dpi=130)
    plt.close()
    print(f"\n圖檔: fig_phase18_t_heatmap.png")

    # === 視覺化 2: top-15 各 centrality 比較 ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    metrics = [("out_weighted", "Out-degree (t-weighted) — 領先樞紐"),
               ("pagerank", "PageRank — 訊息匯聚度"),
               ("hub", "HITS Hub — 出邊連向 authority 多者"),
               ("authority", "HITS Authority — 接收 hub 影響大者"),
               ("betweenness", "Betweenness — 訊息橋樑"),
               ("neg_out_degree", "Negative-out — 負向領先 (成本傳導等)")]
    for ax, (metric, title) in zip(axes.flat, metrics):
        top = cent_df.sort_values(metric, ascending=True).tail(15)
        ax.barh([f"{ic} {industry_meta[ic][:6]}" for ic in top["industry_code"]],
                 top[metric], color="#1b5e20" if metric != "neg_out_degree" else "#c62828",
                 edgecolor="black", lw=0.4, alpha=0.85)
        ax.set_xlabel(metric)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="x", alpha=0.3)
    plt.suptitle("Phase 18: 各 centrality 排名 Top-15", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig_phase18_centrality.png"), dpi=130)
    plt.close()
    print(f"圖檔: fig_phase18_centrality.png")

    # === 視覺化 3: network graph ===
    # 只畫 top edges (按 |weight| top 50) + top-15 nodes
    top_nodes = set(cent_df.head(15)["industry_code"])
    G_plot = nx.DiGraph()
    edges_signed = sorted([(u, v, d) for u, v, d in G_signed.edges(data=True)],
                           key=lambda x: x[2]["weight"], reverse=True)[:60]
    for u, v, d in edges_signed:
        G_plot.add_edge(u, v, **d)

    pos = nx.spring_layout(G_plot, seed=42, k=1.0/np.sqrt(max(1, G_plot.number_of_nodes())))
    fig, ax = plt.subplots(figsize=(13, 11))
    node_sizes = [cent_df[cent_df["industry_code"] == n]["out_weighted"].iloc[0] * 60 + 200
                  for n in G_plot.nodes]
    node_colors = ["#1b5e20" if n in top_nodes else "#90caf9" for n in G_plot.nodes]
    nx.draw_networkx_nodes(G_plot, pos, node_size=node_sizes, node_color=node_colors,
                           edgecolors="black", linewidths=0.5, alpha=0.85, ax=ax)
    pos_edges = [(u, v) for u, v, d in G_plot.edges(data=True) if d["t_signed"] > 0]
    neg_edges = [(u, v) for u, v, d in G_plot.edges(data=True) if d["t_signed"] < 0]
    pos_widths = [G_plot[u][v]["weight"] / 3 for u, v in pos_edges]
    neg_widths = [G_plot[u][v]["weight"] / 3 for u, v in neg_edges]
    nx.draw_networkx_edges(G_plot, pos, edgelist=pos_edges, width=pos_widths,
                           edge_color="#1b5e20", alpha=0.5, arrows=True,
                           arrowsize=12, ax=ax)
    nx.draw_networkx_edges(G_plot, pos, edgelist=neg_edges, width=neg_widths,
                           edge_color="#c62828", alpha=0.5, arrows=True,
                           arrowsize=12, ax=ax, style="dashed")
    labels = {n: f"{n}\n{industry_meta[n][:6]}" for n in G_plot.nodes}
    nx.draw_networkx_labels(G_plot, pos, labels=labels, font_size=8.5,
                            font_family="Microsoft JhengHei", ax=ax)
    ax.set_title(f"Phase 18: 鏈間 spillover 網絡 (top 60 edges by |t|)\n"
                 f"綠色實線=正向, 紅色虛線=負向; 節點大小=out-weighted degree", fontsize=11)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig_phase18_network.png"), dpi=130)
    plt.close()
    print(f"圖檔: fig_phase18_network.png")

    # === 結論摘要 ===
    print("\n" + "=" * 100)
    print("結論摘要")
    print("=" * 100)
    top5 = cent_df.head(5)
    print(f"\nTop 5 hub (out-weighted degree):")
    for _, r in top5.iterrows():
        print(f"  {r['industry_code']} {r['industry_name']:<8s}: out_w={r['out_weighted']:.2f}, "
              f"out_count={int(r['out_degree'])}, PageRank={r['pagerank']:.4f}, hub={r['hub']:.3f}")

    # Validate Phase 7 finding: D000 is #1 hub
    d000_rank = list(cent_df["industry_code"]).index("D000") + 1
    q000_rank_neg = list(cent_df.sort_values("neg_out_degree", ascending=False)["industry_code"]).index("Q000") + 1 if "Q000" in cent_df["industry_code"].values else None
    print(f"\nPhase 7 驗證:")
    print(f"  D000 半導體 out-weighted rank: #{d000_rank}/{len(cent_df)} {'✓ 為 hub' if d000_rank <= 3 else '✗'}")
    if q000_rank_neg:
        print(f"  Q000 鋼鐵 negative-out rank: #{q000_rank_neg} {'✓ 為負向 hub' if q000_rank_neg <= 3 else '✗'}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
