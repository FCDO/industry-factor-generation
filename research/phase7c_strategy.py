"""
Phase 7c: 跨產業 link spillover 策略測試 + 視覺化

基於 Phase 7b 發現:
- 半導體 → 7 個下游科技產業 全部 daily t > 2.84
- 鋼鐵 → 3 個下游 汽車/電機/建材 全部 daily t < -3.6 (負向, 成本轉嫁)

策略 1: 半導體 lead long-only
  當 r_半導體(t-1) > 0 時, 等權持有 7 個下游產業; 否則持有現金 (或大盤)

策略 2: 鋼鐵 hedge
  當 r_鋼鐵(t-1) > median 時, 空 3 個下游; 否則 long 3 個下游

策略 3: 整合 cross-industry pairs LS
  用所有 |t| > 2 的有向 pair 作信號, 加權多空持倉

對照: Cohen-Frazzini (2008) 用客戶供應商連結作 firm-level LS, 1.55%/月
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


def perf_stats(strat_ret, benchmark_ret, ann=252):
    df = pd.concat([strat_ret.rename("s"), benchmark_ret.rename("b")], axis=1).dropna()
    if len(df) < 60: return {}
    res = sm.OLS(df["s"], sm.add_constant(df["b"])).fit(cov_type="HAC", cov_kwds={"maxlags": int(ann/4)})
    cum = (1 + df["s"]).cumprod()
    return {
        "n": len(df),
        "ann_ret": df["s"].mean() * ann,
        "ann_vol": df["s"].std() * np.sqrt(ann),
        "sharpe": df["s"].mean() / df["s"].std() * np.sqrt(ann) if df["s"].std() > 0 else np.nan,
        "ann_alpha": res.params["const"] * ann,
        "t_alpha": res.tvalues["const"],
        "beta": res.params["b"],
        "win": (df["s"] > 0).mean(),
        "max_dd": (cum / cum.cummax() - 1).min(),
    }


def main():
    print("[Phase 7c] 跨產業 link spillover 策略 + 視覺化\n")

    chain = pd.read_csv(os.path.join(ROOT, "industry_chain.csv"), dtype=str)
    industries = chain[["industry_code","industry_name"]].drop_duplicates().set_index("industry_code")["industry_name"].to_dict()

    adj_close = filter_common(load_pickle_df("etl#adj_close.pickle"))
    daily_ret = adj_close.pct_change().iloc[1:]

    industry_members = {ind: chain[chain["industry_code"] == ind]["stock_id"].unique().tolist()
                        for ind in industries}

    panel = {}
    for ind, members in industry_members.items():
        cols = [s for s in members if s in daily_ret.columns]
        if len(cols) >= 3:
            panel[ind] = daily_ret[cols].mean(axis=1)
    panel = pd.DataFrame(panel)
    market = daily_ret.mean(axis=1)

    # === 策略 1: 半導體 → 7 下游 long-only timing ===
    print("=== 策略 1: 半導體 lead 7 下游 (long-only) ===")
    semi = panel["D000"]
    downstream = ["G000","H000","I000","J000","F000","L000","5400"]  # 顯示/觸控/通信/被動/電腦週邊/PCB/雲端
    down_ew = panel[downstream].mean(axis=1)

    # 信號: r_semi(t-1)
    signal = semi.shift(1)
    # Long when signal > 0; cash otherwise
    strat1 = np.where(signal > 0, down_ew, 0)
    strat1 = pd.Series(strat1, index=down_ew.index).dropna()
    s1 = perf_stats(strat1, market)
    print(f"  n={s1.get('n',0)}, ann_ret={s1.get('ann_ret',0):.4f}, sharpe={s1.get('sharpe',0):.3f}, "
          f"alpha={s1.get('ann_alpha',0):.4f}, t={s1.get('t_alpha',0):.3f}")

    # 比較: 永遠持有 down_ew
    s_buy = perf_stats(down_ew, market)
    print(f"  Buy-and-hold 7下游: ann_ret={s_buy.get('ann_ret',0):.4f}, sharpe={s_buy.get('sharpe',0):.3f}, "
          f"alpha={s_buy.get('ann_alpha',0):.4f}, t={s_buy.get('t_alpha',0):.3f}")

    # 變體: long when signal>0, short when signal<0
    strat1b = np.where(signal > 0, down_ew, -down_ew)
    strat1b = pd.Series(strat1b, index=down_ew.index).dropna()
    s1b = perf_stats(strat1b, market)
    print(f"  Long/Short: ann_ret={s1b.get('ann_ret',0):.4f}, sharpe={s1b.get('sharpe',0):.3f}, "
          f"alpha={s1b.get('ann_alpha',0):.4f}, t={s1b.get('t_alpha',0):.3f}")

    # === 策略 2: 鋼鐵反向 hedge ===
    print("\n=== 策略 2: 鋼鐵 lead 3 下游 (短鏈成本轉嫁) ===")
    steel = panel["Q000"]
    steel_down = ["3000","P000","S000"]
    sd_ew = panel[steel_down].mean(axis=1)
    sig_steel = steel.shift(1)
    # 鋼鐵漲 → 下游受壓 → 短下游
    strat2 = np.where(sig_steel > sig_steel.median(), -sd_ew, sd_ew)
    strat2 = pd.Series(strat2, index=sd_ew.index).dropna()
    s2 = perf_stats(strat2, market)
    print(f"  Steel-short hedge: ann_ret={s2.get('ann_ret',0):.4f}, sharpe={s2.get('sharpe',0):.3f}, "
          f"alpha={s2.get('ann_alpha',0):.4f}, t={s2.get('t_alpha',0):.3f}")

    # === 策略 3: 多 cross-industry pair 整合 LS ===
    print("\n=== 策略 3: 整合 |t|>2 cross-industry pair 多空 ===")
    spill = pd.read_csv(os.path.join(OUT, "cross_industry_spillover_D.csv"))
    sig_pairs = spill[(spill["t_AB"].abs() > 2.0) & (spill["L_loose"] >= 3)].copy()
    print(f"  L_loose≥3 且 |t|>2 的 pair 數: {len(sig_pairs)}")
    print(sig_pairs[["A_name","B_name","L_loose","beta_AB","t_AB"]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # 對每個 pair (A→B): 訊號 = sign(t) * r_A(t-1), 持有 r_B(t)
    # 加總所有 pair 的 daily contribution
    daily_contribution = pd.Series(0.0, index=daily_ret.index)
    n_active = pd.Series(0, index=daily_ret.index)
    for _, row in sig_pairs.iterrows():
        A, B = row["A"], row["B"]
        if A not in panel.columns or B not in panel.columns: continue
        sign = np.sign(row["t_AB"])
        sig_A = panel[A].shift(1)
        # 標準化訊號為 demean (zero-cost)
        sig_norm = (sig_A - sig_A.expanding(60).mean()) / sig_A.expanding(60).std()
        contrib = sign * np.sign(sig_norm) * panel[B]
        daily_contribution = daily_contribution.add(contrib, fill_value=0)
        n_active = n_active.add(contrib.notna().astype(int), fill_value=0)

    strat3 = (daily_contribution / n_active.replace(0, np.nan)).dropna()
    s3 = perf_stats(strat3, market)
    print(f"\n  整合 LS: n={s3.get('n',0)}, ann_ret={s3.get('ann_ret',0):.4f}, sharpe={s3.get('sharpe',0):.3f}, "
          f"alpha={s3.get('ann_alpha',0):.4f}, t={s3.get('t_alpha',0):.3f}")

    # === 視覺化: 半導體 lead 累積曲線 ===
    cum_strat = (1 + strat1).cumprod()
    cum_buy = (1 + down_ew.reindex(strat1.index)).cumprod()
    cum_mkt = (1 + market.reindex(strat1.index)).cumprod()
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(cum_strat.index, cum_strat.values, label="半導體 lead 7 下游 (timing long-only)", lw=1.5)
    ax.plot(cum_buy.index, cum_buy.values, label="Buy & Hold 7 下游 EW", lw=1.4, alpha=0.7)
    ax.plot(cum_mkt.index, cum_mkt.values, label="Market EW", lw=1.4, alpha=0.6)
    ax.set_yscale("log")
    ax.set_title("半導體 lead 跨產業 timing 策略 (日頻, long-only)")
    ax.set_xlabel("date"); ax.set_ylabel("cumulative wealth")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig_semi_lead_strategy.png"), dpi=130)
    plt.close()

    # 散點: link 強度 vs 日頻 |t|
    spill_d = pd.read_csv(os.path.join(OUT, "cross_industry_spillover_D.csv"))
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sc = ax.scatter(spill_d["L_loose"], spill_d["t_AB"],
                    c=spill_d["t_AB"].abs(), cmap="viridis", alpha=0.6, s=24)
    # 標記顯著 pair
    sig_d = spill_d[(spill_d["t_AB"].abs() > 2) & (spill_d["L_loose"] >= 3)]
    for _, row in sig_d.iterrows():
        ax.annotate(f"{row['A_name'][:4]}→{row['B_name'][:4]}",
                    (row["L_loose"], row["t_AB"]), fontsize=7, alpha=0.8)
    ax.axhline(1.96, ls="--", color="green", alpha=0.5)
    ax.axhline(-1.96, ls="--", color="red", alpha=0.5)
    ax.axhline(0, ls="-", color="gray", alpha=0.3)
    ax.set_xlabel("L_loose (供應鏈 link 強度)")
    ax.set_ylabel("Daily t-stat (cross-industry spillover)")
    ax.set_title("供應鏈連結強度 vs 日頻 spillover 顯著性")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig_link_vs_tstat_daily.png"), dpi=130)
    plt.close()
    print(f"\n圖檔: fig_semi_lead_strategy.png, fig_link_vs_tstat_daily.png")

    # === 完整顯著表 ===
    print("\n=== 全部 |t|>2 且 L_loose≥1 的 cross-industry directed pairs (按 |t| 排序) ===")
    full_sig = spill_d[(spill_d["t_AB"].abs() > 2.0) & (spill_d["L_loose"] >= 1)].copy()
    full_sig["abs_t"] = full_sig["t_AB"].abs()
    full_sig = full_sig.sort_values("abs_t", ascending=False)
    print(full_sig[["A_name","B_name","L_loose","beta_AB","t_AB","p_AB"]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
