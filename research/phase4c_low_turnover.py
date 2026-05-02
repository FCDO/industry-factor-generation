"""
Phase 4c: 低頻 / 低周轉策略, 評估扣成本後是否仍存活.

設計:
- 信號平滑: 過去 N 天上游 EW 累積報酬 (N=5/10/20)
- 持有期: 5 天 / 20 天 (不每天換)
- Tertile LS, 也測 quintile / 純 long
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

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


def rolling_log_ret(s: pd.Series, window: int) -> pd.Series:
    return np.log1p(s).rolling(window).sum()


def overlapping_strategy(wide, lookback=5, hold=5, n_pick=4):
    """k-day overlapping LS: 每日進場 1/hold, 共 hold 個 sub-portfolio.
    等效於每日 turnover ~ 1/hold."""
    chains = sorted({c for c, p in wide.columns if p == "上游"} & {c for c, p in wide.columns if p == "下游"})
    up = pd.DataFrame({c: wide[(c, "上游")] for c in chains})
    down = pd.DataFrame({c: wide[(c, "下游")] for c in chains})

    signal = up.apply(lambda x: rolling_log_ret(x, lookback)).shift(1)

    # 每天決定當日「新進場 sub-portfolio」, 持有 hold 天
    # 為了簡化, 直接計算每日 portfolio = 平均過去 hold 個進場日的 LS 報酬
    daily_ls = []
    common = signal.index.intersection(down.index)

    # 每天的 cross-sectional LS
    daily_individual_ls = pd.Series(index=common, dtype=float)
    for date in common:
        s = signal.loc[date].dropna()
        r = down.loc[date]
        s = s[s.index.intersection(r.dropna().index)]
        if len(s) < n_pick * 2 + 2:
            continue
        rank = s.rank(ascending=False)
        long_ch = rank[rank <= n_pick].index.tolist()
        short_ch = rank[rank >= len(s) - n_pick + 1].index.tolist()
        ls = r[long_ch].mean() - r[short_ch].mean()
        daily_individual_ls.loc[date] = ls

    # overlap: 持有 hold 天 = 取過去 hold 期 daily_ls 平均
    overlap = daily_individual_ls.rolling(hold, min_periods=hold).mean()

    # 平均每日 turnover ≈ 1/hold (直觀)
    daily_turnover = 1.0 / hold

    return overlap, daily_turnover


def perf(s, ann_factor=252, cost_per_rt=0.005, daily_turnover=None):
    s = s.dropna()
    if daily_turnover is None:
        daily_turnover = 0.5  # default
    cost_per_day = daily_turnover * cost_per_rt
    s_net = s - cost_per_day
    out = {
        "n": len(s),
        "gross_ann": s.mean() * ann_factor,
        "gross_sharpe": s.mean() / s.std() * np.sqrt(ann_factor),
        "net_ann_50bp": (s.mean() - cost_per_day) * ann_factor,
        "net_sharpe_50bp": (s.mean() - cost_per_day) / s.std() * np.sqrt(ann_factor),
        "daily_turnover": daily_turnover,
    }
    return out


def main():
    panel, _ = load()
    wide_ew = to_wide(panel, "ret_ew")

    print("=== 低周轉率策略 (扣成本敏感性: round-trip 50bp) ===\n")
    print(f"{'lookback':>10s} {'hold':>5s} {'n_pick':>7s} {'gross_ann':>10s} {'gross_sh':>10s} {'net_ann':>10s} {'net_sh':>10s} {'turn/day':>10s}")
    rows = []
    for lookback in [1, 5, 10, 20]:
        for hold in [1, 5, 10, 20]:
            for n_pick in [3, 4, 6]:
                s, turn = overlapping_strategy(wide_ew, lookback, hold, n_pick)
                if s.dropna().empty:
                    continue
                stats = perf(s, ann_factor=252, cost_per_rt=0.005, daily_turnover=turn)
                row = {"lookback": lookback, "hold": hold, "n_pick": n_pick, **stats}
                rows.append(row)
                print(f"{lookback:>10d} {hold:>5d} {n_pick:>7d} {stats['gross_ann']:>10.4f} {stats['gross_sharpe']:>10.4f} "
                      f"{stats['net_ann_50bp']:>10.4f} {stats['net_sharpe_50bp']:>10.4f} {turn:>10.4f}")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "low_turnover_search.csv"), index=False, encoding="utf-8-sig")
    print(f"\n結果寫入 low_turnover_search.csv")

    # best by net Sharpe
    print("\n=== 扣成本後 Top 5 by net_sharpe ===")
    print(df.sort_values("net_sharpe_50bp", ascending=False).head(5).to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
