"""
Phase 4b: 交易成本敏感性 + 累積報酬圖

對 EW 日頻 LS 策略, 估算 turnover 並做 net-of-cost alpha 敏感性分析.
台股單邊成本估計 (含手續費 0.1425% + 證交稅 0.3% + 滑價 0.1%):
- 買: 0.1425% + 0.1% = ~0.24%
- 賣: 0.1425% + 0.3% + 0.1% = ~0.54%
- 來回: ~0.78% per round trip

由於 LS 策略 多空各持有, 每換一檔需要一買一賣, 故 cost = turnover × 0.78%
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# Windows 中文字體
for font in ["Microsoft JhengHei", "Microsoft YaHei", "Noto Sans CJK TC", "SimHei"]:
    if any(font in f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [font]
        break
plt.rcParams["axes.unicode_minus"] = False

ROOT = r"C:\Users\user\OneDrive\桌面\產業因子生成"
OUT = os.path.join(ROOT, "research", "output")


def load():
    panel = pd.read_pickle(os.path.join(OUT, "phase_returns.pkl"))
    market = pd.read_pickle(os.path.join(OUT, "market_return.pkl"))
    return panel, market


def to_wide(panel, value):
    w = panel.pivot_table(index="date", columns=["industry_code", "position"], values=value)
    w.index = pd.to_datetime(w.index)
    return w


def daily_cross_chain_with_turnover(wide, n_pick=4):
    """日頻 LS 策略 + turnover 計算."""
    chains = sorted({c for c, p in wide.columns if p == "上游"} & {c for c, p in wide.columns if p == "下游"})
    up = pd.DataFrame({c: wide[(c, "上游")] for c in chains})
    down = pd.DataFrame({c: wide[(c, "下游")] for c in chains})

    signal = up.shift(1)
    common = signal.index.intersection(down.index)
    signal = signal.reindex(common)
    down = down.reindex(common)

    long_set_prev = set()
    short_set_prev = set()
    rets, turns = [], []
    for date in common:
        s = signal.loc[date].dropna()
        r = down.loc[date]
        s = s[s.index.intersection(r.dropna().index)]
        if len(s) < n_pick * 2 + 2:
            rets.append(np.nan); turns.append(np.nan); continue
        rank = s.rank(ascending=False)
        long_ch = set(rank[rank <= n_pick].index)
        short_ch = set(rank[rank >= len(s) - n_pick + 1].index)
        ls = r[list(long_ch)].mean() - r[list(short_ch)].mean()
        # turnover: 與前一日 long/short 鏈 set 不同的比例
        # 全換 turnover = 1
        long_change = len(long_ch ^ long_set_prev) / (2 * n_pick) if long_set_prev else 1
        short_change = len(short_ch ^ short_set_prev) / (2 * n_pick) if short_set_prev else 1
        turn = (long_change + short_change) / 2
        rets.append(ls); turns.append(turn)
        long_set_prev, short_set_prev = long_ch, short_ch
    return pd.Series(rets, index=common, name="ret"), pd.Series(turns, index=common, name="turnover")


def main():
    panel, market_df = load()
    market_d = market_df["ret_market"]
    market_d.index = pd.to_datetime(market_d.index)

    wide_ew = to_wide(panel, "ret_ew")

    # 日頻 LS
    rets, turns = daily_cross_chain_with_turnover(wide_ew, n_pick=4)
    df = pd.concat([rets, turns], axis=1).dropna()
    print(f"日數: {len(df)}, 平均 daily turnover: {df['turnover'].mean():.3f}")
    print(f"年化 turnover (週轉成本基準): {df['turnover'].mean() * 252:.1f} 倍")

    # 成本敏感性: round-trip cost ∈ [0%, 0.5%, 1%, 2%]
    print("\n=== 扣交易成本後年化 alpha ===")
    print(f"{'cost_per_RT':>12s} {'cost_per_day':>12s} {'gross_ann':>10s} {'net_ann':>10s} {'net_sharpe':>11s}")
    for cost_rt in [0.0, 0.005, 0.01, 0.015, 0.02]:
        net_daily = df["ret"] - df["turnover"] * cost_rt
        gross_ann = df["ret"].mean() * 252
        net_ann = net_daily.mean() * 252
        net_vol = net_daily.std() * np.sqrt(252)
        net_sharpe = net_ann / net_vol if net_vol > 0 else np.nan
        print(f"{cost_rt:>12.4f} {cost_rt * df['turnover'].mean():>12.5f} {gross_ann:>10.4f} {net_ann:>10.4f} {net_sharpe:>11.4f}")

    # 累積報酬圖
    cum_strat = (1 + df["ret"]).cumprod()
    cum_market = (1 + market_d.reindex(df.index).fillna(0)).cumprod()
    plt.figure(figsize=(11, 5))
    plt.plot(cum_strat.index, cum_strat.values, label="LS (Top4 上游 lead → long 下游)", lw=1.4)
    plt.plot(cum_market.index, cum_market.values, label="Market EW", lw=1.4, alpha=0.7)
    plt.yscale("log")
    plt.title("日頻產業鏈 spillover 多空策略 累積報酬 (log scale)")
    plt.xlabel("date"); plt.ylabel("cumulative wealth")
    plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout()
    fig_path = os.path.join(OUT, "fig_cum_returns.png")
    plt.savefig(fig_path, dpi=130)
    plt.close()
    print(f"\n累積曲線寫入 {fig_path}")

    # 顯著鏈分布圖
    sub = pd.read_csv(os.path.join(OUT, "regression_results.csv"))
    sub_ew_d = sub[(sub["weight"] == "EW") & (sub["freq"] == "D")
                   & (sub["src"] == "上游") & (sub["dst"] == "下游")].copy()
    sub_ew_d = sub_ew_d.sort_values("t_beta", ascending=True)

    plt.figure(figsize=(9, 7))
    colors = ["#cc3333" if t < -1.96 else ("#33aa33" if t > 1.96 else "#bbbbbb") for t in sub_ew_d["t_beta"]]
    plt.barh(sub_ew_d["industry_name"], sub_ew_d["t_beta"], color=colors)
    plt.axvline(1.96, ls="--", color="#33aa33", alpha=0.6, label="t = ±1.96")
    plt.axvline(-1.96, ls="--", color="#cc3333", alpha=0.6)
    plt.xlabel("t-stat of β (上游 lagged → 下游)")
    plt.title("各產業鏈 上→下 spillover β 顯著性 (EW, 日頻)")
    plt.legend(); plt.grid(alpha=0.3, axis="x")
    plt.tight_layout()
    fig2 = os.path.join(OUT, "fig_chain_tstats.png")
    plt.savefig(fig2, dpi=130)
    plt.close()
    print(f"鏈別 t-stat 寫入 {fig2}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
