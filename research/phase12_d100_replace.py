"""
Phase 12: 用 IC 設計 (D100, 89 家) 取代整體半導體 (D000, 346 家) 重做 Phase 8 策略 A

動機
- Phase 10 證實: IC 設計 (D100) 是純訊息源頭, D000 含晶圓代工/封測等 receiver 雜訊
- Phase 9 鏈內: D000 上游 lead 中下游 alpha 14.7% (50bp)
- Phase 8 跨產業: D000 → 7 個下游產業 alpha 12.4%, t=4.20
- 假設: 用 D100 取代 D000 應提升訊噪比 → alpha 上升

策略 (與 Phase 8 strategy A 同構)
- 訊號: predictor EMA(N) > 0 → long EW(7 個下游產業)
- 隔日執行, K 日持有
- 扣 round-trip cost {0, 30, 50, 80, 100} bp
- 7 個下游產業: G000 H000 I000 J000 F000 L000 5400

輸出
- phase12_d100_vs_d000.csv  : 全 (predictor × N × K × cost) sweep
- fig_phase12_d100_vs_d000.png : 累積曲線 (D100 best vs D000 best vs B&H)
- fig_phase12_param_heatmap.png : (N, K) net alpha 熱圖
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


def k_day_holding_position(target_weight: pd.Series, hold_k: int) -> pd.Series:
    if hold_k == 1:
        return target_weight
    out = target_weight.copy()
    last = target_weight.iloc[0]
    for i in range(len(target_weight)):
        if i % hold_k == 0:
            last = target_weight.iloc[i]
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


def strategy_long_only_timing(predictor: pd.Series,
                               downstream_avg: pd.Series,
                               smooth_n: int,
                               hold_k: int):
    """predictor EMA(N) > 0 → long downstream_avg, K-day hold."""
    if smooth_n > 1:
        sig = predictor.ewm(span=smooth_n, adjust=False).mean()
    else:
        sig = predictor
    target = (sig > 0).astype(float).shift(1)
    weight = k_day_holding_position(target, hold_k)
    gross = weight * downstream_avg
    turnover = weight.diff().abs().fillna(weight.iloc[0])
    return gross, weight, turnover


def main():
    print("=" * 90)
    print("Phase 12: D100 (IC設計, 89家) 取代 D000 (整體半導體, 346家) 重做 Phase 8 策略 A")
    print("=" * 90)

    chain = pd.read_csv(os.path.join(ROOT, "industry_chain.csv"), dtype=str)
    industry_meta = (chain[["industry_code", "industry_name"]]
                     .drop_duplicates()
                     .set_index("industry_code")["industry_name"].to_dict())

    adj_close = filter_common(load_pickle_df("etl#adj_close.pickle"))
    daily_ret = adj_close.pct_change(fill_method=None).iloc[1:]
    market = daily_ret.mean(axis=1)

    # === 構建 industry panel (40 個 industry, EW) ===
    industry_members = {ic: chain[chain["industry_code"] == ic]["stock_id"].unique().tolist()
                        for ic in industry_meta}
    panel = {}
    for ic, members in industry_members.items():
        cols = [s for s in members if s in daily_ret.columns]
        if len(cols) >= 3:
            panel[ic] = daily_ret[cols].mean(axis=1)
    panel = pd.DataFrame(panel)

    # === D100 IC 設計 portfolio ===
    d100_members = chain[(chain["industry_code"] == "D000") &
                          (chain["sub_code"] == "D100")]["stock_id"].unique().tolist()
    d100_cols = [s for s in d100_members if s in daily_ret.columns]
    d100 = daily_ret[d100_cols].mean(axis=1)

    d000_n = sum(1 for s in industry_members["D000"] if s in daily_ret.columns)
    print(f"\nD100 IC 設計        : {len(d100_cols):>4d} 檔")
    print(f"D000 整體半導體 (參考): {d000_n:>4d} 檔")
    print(f"7 個下游 targets    : {DOWNSTREAM}")

    downstream_avg = panel[DOWNSTREAM].mean(axis=1)

    # === D100 vs D000 sanity checks ===
    df_chk = pd.concat({"D100": d100, "D000": panel["D000"]}, axis=1).dropna()
    print(f"\n[Sanity] D100 vs D000 同期相關 ρ = {df_chk['D100'].corr(df_chk['D000']):.3f}")
    df_lag = pd.concat({"d000": panel["D000"], "d100_lag": d100.shift(1)}, axis=1).dropna()
    res_lag = sm.OLS(df_lag["d000"], sm.add_constant(df_lag["d100_lag"]))\
              .fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    print(f"[Sanity] D100(t-1) → D000(t):  β={res_lag.params['d100_lag']:+.4f}, t={res_lag.tvalues['d100_lag']:+.3f}")
    df_lag2 = pd.concat({"down": downstream_avg, "d100_lag": d100.shift(1)}, axis=1).dropna()
    res_lag2 = sm.OLS(df_lag2["down"], sm.add_constant(df_lag2["d100_lag"]))\
               .fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    print(f"[Sanity] D100(t-1) → 7下游平均(t):  β={res_lag2.params['d100_lag']:+.4f}, t={res_lag2.tvalues['d100_lag']:+.3f}")

    # === 全 sweep ===
    print("\n" + "=" * 90)
    print("Strategy sweep: predictor × smooth_N × hold_K × cost")
    print("=" * 90)

    rows = []
    for predictor_label, predictor in [("D100", d100), ("D000", panel["D000"])]:
        for N in PARAMS_N:
            for K in PARAMS_K:
                gross, weight, turn = strategy_long_only_timing(predictor, downstream_avg, N, K)
                avg_turn = turn.mean()
                for cost_rt in COSTS:
                    cost_one = cost_rt / 2.0
                    net = gross - turn * cost_one
                    p = perf_with_alpha(net, market)
                    if not p:
                        continue
                    p.update({
                        "predictor": predictor_label,
                        "smooth_N": N,
                        "hold_K": K,
                        "cost_rt": cost_rt,
                        "avg_daily_turn": avg_turn,
                        "ann_turn": avg_turn * 252,
                    })
                    rows.append(p)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "phase12_d100_vs_d000.csv"),
              index=False, encoding="utf-8-sig")

    # === Top 排序 ===
    fmt = lambda x: f"{x:.4f}"
    cols_show = ["smooth_N", "hold_K", "ann_ret", "ann_vol", "sharpe",
                 "ann_alpha", "t_alpha", "max_dd", "avg_daily_turn"]

    print("\n>>> Top 10 by net Sharpe @ 50bp (D100 IC設計):")
    top_d100 = df[(df["predictor"] == "D100") & (df["cost_rt"] == 0.0050)]\
               .sort_values("sharpe", ascending=False).head(10)
    print(top_d100[cols_show].to_string(index=False, float_format=fmt))

    print("\n>>> Top 10 by net Sharpe @ 50bp (D000 整體半導體, 對照):")
    top_d000 = df[(df["predictor"] == "D000") & (df["cost_rt"] == 0.0050)]\
               .sort_values("sharpe", ascending=False).head(10)
    print(top_d000[cols_show].to_string(index=False, float_format=fmt))

    # === Head-to-head 同 (N, K) 比較 (50bp) ===
    print("\n" + "=" * 90)
    print("Head-to-head 同 (N, K) D100 vs D000 (net @ 50bp)")
    print("=" * 90)
    print(f"\n{'N':>3} {'K':>3} | {'D100 ann':>10} {'D100 sh':>9} {'D100 t_α':>10} | "
          f"{'D000 ann':>10} {'D000 sh':>9} {'D000 t_α':>10} | {'Δalpha':>8} {'Δsharpe':>9}")
    for N in PARAMS_N:
        for K in PARAMS_K:
            r100 = df[(df["predictor"] == "D100") & (df["smooth_N"] == N) &
                       (df["hold_K"] == K) & (df["cost_rt"] == 0.0050)]
            r000 = df[(df["predictor"] == "D000") & (df["smooth_N"] == N) &
                       (df["hold_K"] == K) & (df["cost_rt"] == 0.0050)]
            if len(r100) == 0 or len(r000) == 0:
                continue
            a, b = r100.iloc[0], r000.iloc[0]
            print(f"{N:>3d} {K:>3d} | {a['ann_ret']:>+10.4f} {a['sharpe']:>+9.3f} {a['t_alpha']:>+10.3f} | "
                  f"{b['ann_ret']:>+10.4f} {b['sharpe']:>+9.3f} {b['t_alpha']:>+10.3f} | "
                  f"{(a['ann_alpha']-b['ann_alpha'])*100:>+7.2f}% {a['sharpe']-b['sharpe']:>+9.3f}")

    # === D100 vs D000 在 80bp ===
    print("\n>>> Top 5 by net Sharpe @ 80bp (D100 vs D000):")
    print("D100:")
    print(df[(df["predictor"] == "D100") & (df["cost_rt"] == 0.0080)]
          .sort_values("sharpe", ascending=False).head(5)
          [cols_show].to_string(index=False, float_format=fmt))
    print("\nD000 (對照):")
    print(df[(df["predictor"] == "D000") & (df["cost_rt"] == 0.0080)]
          .sort_values("sharpe", ascending=False).head(5)
          [cols_show].to_string(index=False, float_format=fmt))

    # === Phase 8 對照: D000 在 (N=20, K=1) 是先前 winner ===
    print("\n" + "=" * 90)
    print("Phase 8 對照表: D000 winner (N=20, K=1) — 同參數下 D100 表現")
    print("=" * 90)
    for cost_rt in [0.0, 0.0050, 0.0080]:
        r100 = df[(df["predictor"] == "D100") & (df["smooth_N"] == 20) &
                   (df["hold_K"] == 1) & (df["cost_rt"] == cost_rt)].iloc[0]
        r000 = df[(df["predictor"] == "D000") & (df["smooth_N"] == 20) &
                   (df["hold_K"] == 1) & (df["cost_rt"] == cost_rt)].iloc[0]
        cost_label = f"@{cost_rt*1e4:.0f}bp"
        print(f"\n  cost {cost_label}:")
        print(f"    D100  ann_ret={r100['ann_ret']:+.4f}  sharpe={r100['sharpe']:+.3f}  "
              f"alpha={r100['ann_alpha']:+.4f}  t_α={r100['t_alpha']:+.3f}")
        print(f"    D000  ann_ret={r000['ann_ret']:+.4f}  sharpe={r000['sharpe']:+.3f}  "
              f"alpha={r000['ann_alpha']:+.4f}  t_α={r000['t_alpha']:+.3f}")

    # === 累積曲線 ===
    best_d100 = df[(df["predictor"] == "D100") & (df["cost_rt"] == 0.0050)]\
                .sort_values("sharpe", ascending=False).iloc[0]
    bN, bK = int(best_d100["smooth_N"]), int(best_d100["hold_K"])
    g100, _, t100 = strategy_long_only_timing(d100, downstream_avg, bN, bK)
    net100_50 = (g100 - t100 * 0.0025).fillna(0)
    net100_80 = (g100 - t100 * 0.0040).fillna(0)

    # D000 best (N=20, K=1 從 Phase 8 確認)
    g000, _, t000 = strategy_long_only_timing(panel["D000"], downstream_avg, 20, 1)
    net000_50 = (g000 - t000 * 0.0025).fillna(0)

    plt.figure(figsize=(12, 6.2))
    cum100_50 = (1 + net100_50).cumprod()
    cum100_80 = (1 + net100_80).cumprod()
    cum000_50 = (1 + net000_50).cumprod()
    cum_bh = (1 + downstream_avg.fillna(0)).cumprod()
    cum_mkt = (1 + market.fillna(0)).cumprod()
    plt.plot(cum100_50, color="#1b5e20", lw=1.8,
             label=f"D100 timing (N={bN}, K={bK}) net@50bp [{cum100_50.iloc[-1]:.1f}x]")
    plt.plot(cum100_80, color="#558b2f", lw=1.4, alpha=0.85,
             label=f"D100 net@80bp [{cum100_80.iloc[-1]:.1f}x]")
    plt.plot(cum000_50, color="#0277bd", lw=1.4, alpha=0.85,
             label=f"D000 timing (N=20, K=1) net@50bp [{cum000_50.iloc[-1]:.1f}x]")
    plt.plot(cum_bh, color="#ef6c00", lw=1.2, alpha=0.7,
             label=f"7 下游 EW B&H [{cum_bh.iloc[-1]:.1f}x]")
    plt.plot(cum_mkt, color="#37474f", lw=1.1, alpha=0.55,
             label=f"Market EW [{cum_mkt.iloc[-1]:.1f}x]")
    plt.yscale("log")
    plt.title("Phase 12: D100 IC設計 vs D000 整體半導體 — 跨產業 lead 策略 (扣成本)")
    plt.xlabel("date")
    plt.ylabel("累積資本 (log)")
    plt.legend(loc="upper left", fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig_phase12_d100_vs_d000.png"), dpi=130)
    plt.close()
    print(f"\n圖檔: fig_phase12_d100_vs_d000.png")

    # === (N, K) net alpha heatmap @ 50bp ===
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for ax, pred_label in zip(axes, ["D100", "D000"]):
        sub = df[(df["predictor"] == pred_label) & (df["cost_rt"] == 0.0050)]
        pivot = sub.pivot(index="smooth_N", columns="hold_K", values="ann_alpha")
        im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto",
                       vmin=-0.05, vmax=0.20, origin="lower")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_xlabel("hold_K")
        ax.set_ylabel("smooth_N (EMA span)")
        ax.set_title(f"{pred_label} net@50bp ann_alpha")
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                v = pivot.values[i, j]
                ax.text(j, i, f"{v:+.3f}", ha="center", va="center",
                        color="black" if abs(v) < 0.1 else "white", fontsize=8.5)
        plt.colorbar(im, ax=ax, fraction=0.04)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig_phase12_param_heatmap.png"), dpi=130)
    plt.close()
    print(f"圖檔: fig_phase12_param_heatmap.png")

    # === 結論摘要 ===
    print("\n" + "=" * 90)
    print("結論摘要")
    print("=" * 90)
    best100_50 = df[(df["predictor"] == "D100") & (df["cost_rt"] == 0.0050)]\
                 .sort_values("sharpe", ascending=False).iloc[0]
    best000_50 = df[(df["predictor"] == "D000") & (df["cost_rt"] == 0.0050)]\
                 .sort_values("sharpe", ascending=False).iloc[0]
    print(f"  D100 最佳 net@50bp: (N={int(best100_50['smooth_N'])}, K={int(best100_50['hold_K'])})  "
          f"ann_ret={best100_50['ann_ret']:.4f}  sharpe={best100_50['sharpe']:.3f}  "
          f"alpha={best100_50['ann_alpha']:.4f}  t_α={best100_50['t_alpha']:.3f}")
    print(f"  D000 最佳 net@50bp: (N={int(best000_50['smooth_N'])}, K={int(best000_50['hold_K'])})  "
          f"ann_ret={best000_50['ann_ret']:.4f}  sharpe={best000_50['sharpe']:.3f}  "
          f"alpha={best000_50['ann_alpha']:.4f}  t_α={best000_50['t_alpha']:.3f}")
    print(f"\n  Δalpha (D100 - D000): {(best100_50['ann_alpha'] - best000_50['ann_alpha'])*100:+.2f}pp")
    print(f"  Δsharpe (D100 - D000): {best100_50['sharpe'] - best000_50['sharpe']:+.3f}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
