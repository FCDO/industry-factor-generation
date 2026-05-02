"""半導體鏈 (D000) 內部 報酬關係深度分析
1. Position 級雙向迴歸 (從 Phase 2 結果擷取 D000)
2. Sub_code 級 pairwise 迴歸 (11 個細分類)
3. 鏈內 timing 策略: IC設計 lead → 封測/晶圓製造
"""
import os, pickle, sys, itertools
import numpy as np, pandas as pd
import statsmodels.api as sm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
for f in ["Microsoft JhengHei", "Microsoft YaHei", "Noto Sans CJK TC", "SimHei"]:
    if any(f in fn.name for fn in font_manager.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [f]; break
plt.rcParams["axes.unicode_minus"] = False

sys.stdout.reconfigure(encoding="utf-8")
DB = r"C:\Users\user\finlab_db"
ROOT = r"C:\Users\user\OneDrive\桌面\產業因子生成"
OUT = os.path.join(ROOT, "research", "output")


def load_daily_ret():
    with open(os.path.join(DB, "etl#adj_close.pickle"), "rb") as f:
        adj = pickle.load(f).set_index("date")
    adj.index = pd.to_datetime(adj.index); adj.columns = adj.columns.astype(str)
    keep = [c for c in adj.columns if c.isdigit() and len(c)==4 and not c.startswith("00")]
    return adj[keep].pct_change(fill_method=None).iloc[1:]


def reg(y, x, others, market, nw_lag=5):
    df = pd.concat({"y": y, "x": x.shift(1), "y_lag": y.shift(1), "m_lag": market.shift(1)}, axis=1).dropna()
    if len(df) < 200: return None
    res = sm.OLS(df["y"], sm.add_constant(df[["x","y_lag","m_lag"]])).fit(
        cov_type="HAC", cov_kwds={"maxlags":nw_lag}
    )
    return {"n":len(df), "beta":res.params["x"], "t":res.tvalues["x"], "p":res.pvalues["x"]}


def main():
    daily_ret = load_daily_ret()
    chain = pd.read_csv(os.path.join(ROOT, "industry_chain.csv"), dtype=str)
    d = chain[chain['industry_code']=='D000']
    market = daily_ret.mean(axis=1)

    # === 1. Position 級雙向: 上中下游 6 個方向 ===
    print("="*80)
    print("【1】半導體鏈 D000 — Position 級 (上/中/下游) 雙向迴歸 (日頻 EW)")
    print("="*80)

    pos_ret = {}
    for pos in ["上游","中游","下游"]:
        members = d[d['position']==pos]['stock_id'].unique().tolist()
        members = [s for s in members if s in daily_ret.columns]
        pos_ret[pos] = daily_ret[members].mean(axis=1)
        print(f"  {pos}: {len(members)} 檔股票")

    print(f"\n{'src → dst':<14s} {'β':>10s} {'t':>8s} {'p':>10s} {'n':>6s}")
    for src, dst in itertools.permutations(["上游","中游","下游"], 2):
        r = reg(pos_ret[dst], pos_ret[src], None, market)
        marker = " ✓" if r and abs(r["t"])>1.96 else ""
        print(f"{src} → {dst:<10s} {r['beta']:>10.4f} {r['t']:>8.3f} {r['p']:>10.4f} {r['n']:>6d}{marker}")

    # === 2. Sub_code 級 pairwise ===
    print("\n" + "="*80)
    print("【2】Sub_code 級 (11 細分類) pairwise 迴歸")
    print("="*80)

    sub_panels = {}
    sub_meta = {}
    for sc in d['sub_code'].unique():
        members = d[d['sub_code']==sc]['stock_id'].unique().tolist()
        members = [s for s in members if s in daily_ret.columns]
        if len(members) < 3: continue
        sub_name = d[d['sub_code']==sc]['sub_name'].iloc[0]
        sub_panels[sc] = daily_ret[members].mean(axis=1)
        sub_meta[sc] = (sub_name, len(members))

    print(f"  納入 sub_code: {len(sub_panels)} 個 (≥3 股)")
    for sc, (n, k) in sub_meta.items():
        print(f"    {sc} {n:<14s} : {k} 檔")

    # 逐對迴歸
    rows = []
    for src, dst in itertools.permutations(sub_panels.keys(), 2):
        r = reg(sub_panels[dst], sub_panels[src], None, market)
        if r is None: continue
        rows.append({
            "src_code": src, "src_name": sub_meta[src][0],
            "dst_code": dst, "dst_name": sub_meta[dst][0],
            "n": r["n"], "beta": r["beta"], "t": r["t"], "p": r["p"]
        })
    res = pd.DataFrame(rows)

    print("\n  Top 15 顯著 sub_code 對 (按 |t| 排序):")
    res["abs_t"] = res["t"].abs()
    top = res.sort_values("abs_t", ascending=False).head(15)
    print(top[["src_name","dst_name","beta","t","p","n"]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    res.to_csv(os.path.join(OUT, "phase9_semi_subcode_spillover.csv"), index=False, encoding="utf-8-sig")

    # 各 sub_code 作為 predictor / receiver 的整體 t 分布
    print("\n  各 sub_code 作為 PREDICTOR (出邊 mean t):")
    pred = res.groupby("src_name").agg(mean_t=("t","mean"), max_t=("t","max"), n_targets=("t","count"),
                                        n_sig5_pos=("p", lambda x: ((x<0.05) & (res.loc[x.index,"beta"]>0)).sum())
                                       ).sort_values("mean_t", ascending=False)
    print(pred.to_string(float_format=lambda x: f"{x:.3f}"))

    print("\n  各 sub_code 作為 RECEIVER (入邊 mean t):")
    recv = res.groupby("dst_name").agg(mean_t=("t","mean"), max_t=("t","max"), n_sources=("t","count"),
                                        n_sig5_pos=("p", lambda x: ((x<0.05) & (res.loc[x.index,"beta"]>0)).sum())
                                       ).sort_values("mean_t", ascending=False)
    print(recv.to_string(float_format=lambda x: f"{x:.3f}"))

    # === 3. 鏈內 timing 策略: 上游 lead 中下游 ===
    print("\n" + "="*80)
    print("【3】鏈內 timing 策略: D000 上游 EMA(N) → long 中下游 EW")
    print("="*80)

    upstream = pos_ret["上游"]
    midstream = pos_ret["中游"]
    downstream = pos_ret["下游"]
    middown = pd.concat([midstream, downstream], axis=1).mean(axis=1)

    rows_strat = []
    for N in [5, 10, 20]:
        for K in [1, 5, 10, 20]:
            sig = upstream.ewm(span=N, adjust=False).mean()
            target = (sig > 0).astype(float).shift(1)
            if K > 1:
                w = target.copy(); last = target.iloc[0]
                for i in range(len(target)):
                    if i % K == 0: last = target.iloc[i]
                    w.iloc[i] = last
            else:
                w = target
            gross = w * middown
            turn = w.diff().abs().fillna(w.iloc[0])
            for cost in [0.0, 0.0050]:
                net = gross - turn * (cost/2)
                df = pd.concat([net.rename("s"), market.rename("b")], axis=1).dropna()
                if len(df) < 60: continue
                r = sm.OLS(df["s"], sm.add_constant(df["b"])).fit(cov_type="HAC", cov_kwds={"maxlags":60})
                ann = df["s"].mean()*252
                vol = df["s"].std()*np.sqrt(252)
                sh = ann/vol if vol>0 else np.nan
                rows_strat.append({
                    "N":N, "K":K, "cost":cost,
                    "ann_ret":ann, "sharpe":sh,
                    "alpha":r.params["const"]*252, "t_alpha":r.tvalues["const"],
                    "daily_turn": turn.mean()
                })

    sdf = pd.DataFrame(rows_strat)
    print("\n  Net@50bp Top 5 (按 Sharpe):")
    top_strat = sdf[sdf["cost"]==0.0050].sort_values("sharpe", ascending=False).head(5)
    print(top_strat[["N","K","ann_ret","sharpe","alpha","t_alpha","daily_turn"]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # 最佳參數累積曲線
    best = top_strat.iloc[0]
    bestN, bestK = int(best["N"]), int(best["K"])
    sig = upstream.ewm(span=bestN, adjust=False).mean()
    target = (sig > 0).astype(float).shift(1)
    if bestK > 1:
        w = target.copy(); last = target.iloc[0]
        for i in range(len(target)):
            if i % bestK == 0: last = target.iloc[i]
            w.iloc[i] = last
    else:
        w = target
    gross = w * middown
    turn = w.diff().abs().fillna(w.iloc[0])
    net50 = (gross - turn * 0.0025).fillna(0)

    cum_strat = (1+net50).cumprod()
    cum_md = (1+middown.fillna(0)).cumprod()
    cum_up = (1+upstream.fillna(0)).cumprod()
    cum_mkt = (1+market.fillna(0)).cumprod()

    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.plot(cum_strat, color="#1b5e20", lw=1.6, label=f"D000上游 lead → 中下游 net@50bp (cum {cum_strat.iloc[-1]:.1f}x)")
    ax.plot(cum_md, color="#ef6c00", lw=1.3, alpha=0.85, label=f"D000 中下游 B&H (cum {cum_md.iloc[-1]:.1f}x)")
    ax.plot(cum_up, color="#6a1b9a", lw=1.2, alpha=0.7, label=f"D000 上游 (IC設計) B&H (cum {cum_up.iloc[-1]:.1f}x)")
    ax.plot(cum_mkt, color="#37474f", lw=1.2, alpha=0.6, label=f"Market EW (cum {cum_mkt.iloc[-1]:.1f}x)")
    ax.set_yscale("log")
    ax.set_title(f"半導體鏈內 timing 策略: 上游 EMA({bestN}) → 中下游, hold={bestK} 日 (扣 50bp)")
    ax.set_xlabel("date"); ax.set_ylabel("累積資本 (log)")
    ax.legend(loc="upper left"); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig_semi_internal_strategy.png"), dpi=130)
    plt.close()
    print(f"\n圖檔: fig_semi_internal_strategy.png")

    # === 比較 半導體鏈內 vs 跨產業 (Phase 8) 結果 ===
    print("\n" + "="*80)
    print("【對照】半導體鏈內 vs 跨產業 半導體 lead 策略")
    print("="*80)
    print("  鏈內 (本分析, 上游 → 中下游):")
    print(f"    {bestN}/{bestK}: alpha={best['alpha']:.4f}, t_α={best['t_alpha']:.3f}, Sharpe={best['sharpe']:.3f}")
    print("  跨產業 (Phase 8, 整體半導體 → 7 個下游產業):")
    print(f"    20/1: alpha=0.1242, t_α=4.198, Sharpe=1.462")


if __name__ == "__main__":
    main()
