"""
Phase 11: 將 Phase 9 風格的「上游 lead → 中下游 timing」策略擴展到所有 standard chain

11a: 22 鏈 strategy sweep
- 上游 EMA(20) > 0 時 long 中下游 (EW), K=1
- 報 gross 與 net@50bp 的 alpha, t_alpha, Sharpe
- 排序找出哪些鏈像半導體一樣有效

11b: top-3 鏈 sub_code 深掘
- 對 11a top-3 (按 net t_alpha) 跑 sub_code pairwise 迴歸 (Phase 9 風格)
- 找出每條鏈的 pure leader / receiver
"""
from __future__ import annotations

import itertools
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

ROOT = r"C:\Users\engli\OneDrive\桌面\產業因子生成"
DB = r"C:\Users\engli\finlab_db"
OUT = os.path.join(ROOT, "research", "output")

SMOOTH_N = 20      # 與 Phase 8/9 對齊 (D000 winner)
HOLD_K = 1
COST_RT = 0.0050   # 50bp round-trip
MIN_UP_STOCKS = 3
MIN_REST_STOCKS = 5


# ===========================================================================
# Phase 11a: 鏈內 timing 策略掃描 22 條 chain
# ===========================================================================

def perf_with_alpha(daily_ret, market):
    df = pd.concat([daily_ret.rename("s"), market.rename("b")], axis=1).dropna()
    if len(df) < 200:
        return None
    res = sm.OLS(df["s"], sm.add_constant(df["b"])).fit(
        cov_type="HAC", cov_kwds={"maxlags": 60}
    )
    s = df["s"]
    return {
        "n": len(df),
        "ann_ret": s.mean() * 252,
        "ann_vol": s.std() * np.sqrt(252),
        "sharpe": s.mean() / s.std() * np.sqrt(252) if s.std() > 0 else np.nan,
        "ann_alpha": res.params["const"] * 252,
        "t_alpha": res.tvalues["const"],
        "beta": res.params["b"],
    }


def phase_11a():
    print("=" * 80)
    print("【Phase 11a】22 鏈上→中下游 timing 策略 (EMA(20), K=1, 扣 50bp)")
    print("=" * 80)

    panel = pd.read_pickle(os.path.join(OUT, "phase_returns.pkl"))
    market = pd.read_pickle(os.path.join(OUT, "market_return.pkl"))["ret_market"]

    ew = panel.pivot_table(index="date", columns=["industry_code", "position"], values="ret_ew")
    ew.index = pd.to_datetime(ew.index)
    market.index = pd.to_datetime(market.index)

    chain_meta = panel[["industry_code", "industry_name"]].drop_duplicates()\
                    .set_index("industry_code")["industry_name"].to_dict()

    # 各 (industry, position) 中位數股數
    n_med = panel.groupby(["industry_code", "position"])["n_stocks"].median()

    rows = []
    for ic in sorted(panel["industry_code"].unique()):
        cols_for_ic = [c for c in ew.columns if c[0] == ic]
        positions_avail = [c[1] for c in cols_for_ic]
        if "上游" not in positions_avail:
            continue
        up_n = n_med.get((ic, "上游"), 0)
        if up_n < MIN_UP_STOCKS:
            continue

        rest_components, rest_n, rest_pos_label = [], 0, []
        for tgt_pos in ["中游", "下游"]:
            if tgt_pos in positions_avail:
                rest_components.append(ew[(ic, tgt_pos)])
                rest_n += n_med.get((ic, tgt_pos), 0)
                rest_pos_label.append(tgt_pos)
        if not rest_components or rest_n < MIN_REST_STOCKS:
            continue

        up = ew[(ic, "上游")]
        rest = pd.concat(rest_components, axis=1).mean(axis=1)

        # Long-only timing strategy: long rest when EMA(up) > 0
        sig = up.ewm(span=SMOOTH_N, adjust=False).mean()
        weight = (sig > 0).astype(float).shift(1)
        gross = (weight * rest)
        # 1-side cost = round-trip / 2; turnover one-side
        turnover = weight.diff().abs().fillna(weight.iloc[0])
        net = gross - turnover * (COST_RT / 2.0)

        for label, ret in [("gross", gross), ("net50", net)]:
            stat = perf_with_alpha(ret, market)
            if not stat:
                continue
            stat.update({
                "industry_code": ic,
                "industry_name": chain_meta[ic],
                "variant": label,
                "rest_positions": "+".join(rest_pos_label),
                "up_n_med": up_n,
                "rest_n_med": rest_n,
                "ann_turnover": turnover.mean() * 252,
            })
            rows.append(stat)

    df = pd.DataFrame(rows)
    df = df[["industry_code", "industry_name", "rest_positions", "variant",
             "up_n_med", "rest_n_med", "n",
             "ann_ret", "sharpe", "ann_alpha", "t_alpha", "beta",
             "ann_turnover"]]
    df.to_csv(os.path.join(OUT, "phase11a_chain_strategy_sweep.csv"),
              index=False, encoding="utf-8-sig")

    # Print
    fmt = lambda x: f"{x:.3f}"
    print(f"\n{'>>> Net@50bp 排序 (按 t_alpha 降序)':<80s}")
    net = df[df["variant"] == "net50"].sort_values("t_alpha", ascending=False).copy()
    print(net[["industry_code", "industry_name", "rest_positions",
               "up_n_med", "rest_n_med", "ann_ret", "sharpe",
               "ann_alpha", "t_alpha", "ann_turnover"]]
          .to_string(index=False, float_format=fmt))

    print(f"\n>>> Gross 排序 (參考):")
    gr = df[df["variant"] == "gross"].sort_values("t_alpha", ascending=False).copy()
    print(gr[["industry_code", "industry_name", "ann_ret", "sharpe",
              "ann_alpha", "t_alpha"]]
          .to_string(index=False, float_format=fmt))

    # 視覺化 bar chart
    fig, ax = plt.subplots(figsize=(11, 7))
    plot_df = net.sort_values("t_alpha", ascending=True)
    colors = ["#1b5e20" if t > 1.96 else
              "#9ccc65" if t > 0 else
              "#ef5350" if t < -1.96 else
              "#bdbdbd" for t in plot_df["t_alpha"]]
    labels = [f"{r.industry_code} {r.industry_name}" for _, r in plot_df.iterrows()]
    ax.barh(labels, plot_df["t_alpha"], color=colors, edgecolor="black", linewidth=0.4)
    ax.axvline(1.96, color="black", lw=0.6, ls="--", alpha=0.6)
    ax.axvline(-1.96, color="black", lw=0.6, ls="--", alpha=0.6)
    ax.axvline(0, color="black", lw=0.5)
    ax.set_xlabel("Net@50bp t_alpha (vs 市場)")
    ax.set_title("Phase 11a: 各鏈「上游 EMA(20) → long 中下游」 timing 策略 t_alpha")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig_phase11a_chain_ranking.png"), dpi=130)
    plt.close()
    print("\n  圖檔: fig_phase11a_chain_ranking.png")

    return df


# ===========================================================================
# Phase 11b: top-3 鏈 sub_code 深掘
# ===========================================================================

def reg(y, x, market, nw_lag=5):
    df = pd.concat({"y": y,
                    "x": x.shift(1),
                    "y_lag": y.shift(1),
                    "m_lag": market.shift(1)}, axis=1).dropna()
    if len(df) < 200:
        return None
    res = sm.OLS(df["y"], sm.add_constant(df[["x", "y_lag", "m_lag"]])).fit(
        cov_type="HAC", cov_kwds={"maxlags": nw_lag}
    )
    return {"n": len(df), "beta": res.params["x"],
            "t": res.tvalues["x"], "p": res.pvalues["x"]}


def phase_11b(top_chain_codes):
    print("\n" + "=" * 80)
    print(f"【Phase 11b】Top {len(top_chain_codes)} 鏈 sub_code pairwise 深掘")
    print("=" * 80)

    # Load FinLab daily returns
    with open(os.path.join(DB, "etl#adj_close.pickle"), "rb") as f:
        adj = pickle.load(f).set_index("date")
    adj.index = pd.to_datetime(adj.index)
    adj.columns = adj.columns.astype(str)
    keep = [c for c in adj.columns if c.isdigit() and len(c) == 4 and not c.startswith("00")]
    daily_ret = adj[keep].pct_change(fill_method=None).iloc[1:]
    market = daily_ret.mean(axis=1)

    chain = pd.read_csv(os.path.join(ROOT, "industry_chain.csv"), dtype=str)

    summary_rows = []
    for ic in top_chain_codes:
        ind_name = chain[chain["industry_code"] == ic]["industry_name"].iloc[0]
        print(f"\n--- {ic} {ind_name} ---")

        d = chain[chain["industry_code"] == ic]
        sub_panels, sub_meta = {}, {}
        for sc in d["sub_code"].dropna().unique():
            members = d[d["sub_code"] == sc]["stock_id"].unique().tolist()
            members = [s for s in members if s in daily_ret.columns]
            if len(members) < 3:
                continue
            sub_name = d[d["sub_code"] == sc]["sub_name"].iloc[0]
            sub_panels[sc] = daily_ret[members].mean(axis=1)
            sub_meta[sc] = (sub_name, len(members))

        if len(sub_panels) < 2:
            print(f"  ! sub_code 不足 (≥3 股) — 跳過")
            continue

        print(f"  納入 sub_codes: {len(sub_panels)} 個")
        for sc, (n, k) in sub_meta.items():
            print(f"    {sc} {n}: {k} 檔")

        # Pairwise regression
        rows = []
        for src, dst in itertools.permutations(sub_panels.keys(), 2):
            r = reg(sub_panels[dst], sub_panels[src], market)
            if r is None:
                continue
            rows.append({
                "src_code": src, "src_name": sub_meta[src][0],
                "dst_code": dst, "dst_name": sub_meta[dst][0],
                "n": r["n"], "beta": r["beta"], "t": r["t"], "p": r["p"]
            })
        res = pd.DataFrame(rows)
        if len(res) == 0:
            continue

        res.to_csv(os.path.join(OUT, f"phase11b_subcode_{ic}.csv"),
                   index=False, encoding="utf-8-sig")

        # predictor / receiver scores
        pred = res.groupby("src_name").apply(
            lambda g: pd.Series({
                "mean_t": g["t"].mean(),
                "max_t": g["t"].max(),
                "n_targets": len(g),
                "n_sig5_pos": int(((g["p"] < 0.05) & (g["beta"] > 0)).sum()),
                "n_sig5_neg": int(((g["p"] < 0.05) & (g["beta"] < 0)).sum()),
            }), include_groups=False
        ).sort_values("mean_t", ascending=False)
        recv = res.groupby("dst_name").apply(
            lambda g: pd.Series({
                "mean_t": g["t"].mean(),
                "max_t": g["t"].max(),
                "n_sources": len(g),
                "n_sig5_pos": int(((g["p"] < 0.05) & (g["beta"] > 0)).sum()),
                "n_sig5_neg": int(((g["p"] < 0.05) & (g["beta"] < 0)).sum()),
            }), include_groups=False
        ).sort_values("mean_t", ascending=False)

        fmt = lambda x: f"{x:.3f}" if isinstance(x, float) else str(x)
        print(f"\n  [PREDICTOR] 出邊 mean t (高=純 leader):")
        print(pred.to_string(float_format=lambda x: f"{x:.3f}"))
        print(f"\n  [RECEIVER] 入邊 mean t (高=純 receiver):")
        print(recv.to_string(float_format=lambda x: f"{x:.3f}"))

        # 視覺化: predictor mean_t bar
        fig, ax = plt.subplots(figsize=(9, max(3, len(pred) * 0.3)))
        plot_df = pred.sort_values("mean_t", ascending=True)
        colors = ["#1b5e20" if t > 1.96 else
                  "#9ccc65" if t > 0 else
                  "#ef5350" if t < -1.96 else
                  "#bdbdbd" for t in plot_df["mean_t"]]
        ax.barh(plot_df.index.tolist(), plot_df["mean_t"], color=colors, edgecolor="black", lw=0.4)
        ax.axvline(0, color="black", lw=0.5)
        ax.axvline(1.96, color="black", lw=0.6, ls="--", alpha=0.5)
        ax.axvline(-1.96, color="black", lw=0.6, ls="--", alpha=0.5)
        ax.set_xlabel("mean t (作為 predictor)")
        ax.set_title(f"{ic} {ind_name}: sub_code predictor 強度排行")
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT, f"fig_phase11b_{ic}_predictor.png"), dpi=130)
        plt.close()

        # 紀錄到 summary
        leaders = pred[(pred["mean_t"] > 1.0) & (pred["n_sig5_pos"] >= 2)].index.tolist()
        receivers = recv[(recv["mean_t"] > 1.0) & (recv["n_sig5_pos"] >= 2)].index.tolist()
        summary_rows.append({
            "industry_code": ic, "industry_name": ind_name,
            "n_subcodes": len(sub_panels),
            "leaders": "; ".join(leaders),
            "receivers": "; ".join(receivers),
            "max_pred_t": pred["mean_t"].max(),
            "min_pred_t": pred["mean_t"].min(),
        })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(os.path.join(OUT, "phase11b_subcode_summary.csv"),
                   index=False, encoding="utf-8-sig")
    print("\n>>> Phase 11b summary:")
    if len(summary):
        print(summary.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    return summary


# ===========================================================================
# Main
# ===========================================================================

def main():
    df_11a = phase_11a()

    # Pick top 3 chains by net t_alpha (excluding D000 since already done in Phase 9)
    net = df_11a[df_11a["variant"] == "net50"].copy()
    net = net[net["industry_code"] != "D000"]  # 排除已做過的半導體
    net = net.sort_values("t_alpha", ascending=False)
    top3 = net.head(3)["industry_code"].tolist()

    print(f"\n>>> Phase 11b 將深掘: {top3}")
    print(f"    (排除 D000 — Phase 9 已做)")

    if top3:
        phase_11b(top3)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
