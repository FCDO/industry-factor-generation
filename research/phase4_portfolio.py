"""
Phase 4: 多空投組策略 (Cohen-Frazzini 變體 + 鏈內擇時)

策略 1 (CrossChainQuintile, Cohen-Frazzini 風):
    每月底, 對每個產業鏈 c 計算 "信號 = r_up_c(過去月)"。
    截面排序後, 分 quintile (或 tertile, 鏈數有限):
        多 top 30%: long 該等鏈的下游組合
        空 bottom 30%: short 該等鏈的下游組合
    多空等權, 持有 1 個月, 月底再平衡.

策略 2 (PerChainTiming):
    對每個鏈 c, 用 過去 N 日 上游 EW 報酬 作下游擇時信號.
    若 r_up(t-1 ~ t-N) > median, 下個交易日 long 下游, 否則 long 大盤.
    報告每鏈年化 alpha 與 Sharpe.

策略 3 (DailyCrossChain):
    每日對 22 鏈做截面排序, 信號為昨日上游 EW return.
    多 top 5 / 空 bottom 5 (各 1/5), 持有 1 日.

對照: Cohen & Frazzini (2008) 月頻 long-short 約 1.55%/月; Menzly-Ozbas (2010)
策略年化 alpha ~12%.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = r"C:\Users\user\OneDrive\桌面\產業因子生成"
OUT = os.path.join(ROOT, "research", "output")


def load_panels():
    panel = pd.read_pickle(os.path.join(OUT, "phase_returns.pkl"))
    market = pd.read_pickle(os.path.join(OUT, "market_return.pkl"))
    return panel, market


def to_wide(panel: pd.DataFrame, value: str = "ret_ew") -> pd.DataFrame:
    w = panel.pivot_table(index="date", columns=["industry_code", "position"], values=value)
    w.index = pd.to_datetime(w.index)
    return w


def daily_to_monthly_log(rets_d: pd.DataFrame) -> pd.DataFrame:
    log = np.log1p(rets_d)
    m = log.resample("ME").sum(min_count=10)
    return np.expm1(m)


def perf_stats(strat_ret: pd.Series, benchmark_ret: pd.Series, ann_factor: float):
    """alpha (vs benchmark), Sharpe, 勝率, max DD."""
    df = pd.concat([strat_ret.rename("s"), benchmark_ret.rename("b")], axis=1).dropna()
    if len(df) < 12:
        return {}
    Y, X = df["s"], sm.add_constant(df["b"])
    res = sm.OLS(Y, X).fit(cov_type="HAC", cov_kwds={"maxlags": int(ann_factor / 4)})
    alpha = res.params["const"]
    t_alpha = res.tvalues["const"]
    beta = res.params["b"]
    ann_alpha = alpha * ann_factor
    ann_ret = df["s"].mean() * ann_factor
    ann_vol = df["s"].std() * np.sqrt(ann_factor)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
    win = (df["s"] > 0).mean()
    cum = (1 + df["s"]).cumprod()
    dd = (cum / cum.cummax() - 1).min()
    return {
        "n_periods": len(df),
        "ann_ret": ann_ret, "ann_vol": ann_vol, "sharpe": sharpe,
        "ann_alpha": ann_alpha, "t_alpha": t_alpha, "beta": beta,
        "win_rate": win, "max_dd": dd, "r2": res.rsquared,
    }


def strategy1_cross_chain_monthly(wide: pd.DataFrame, market_m: pd.Series) -> tuple[pd.Series, pd.DataFrame]:
    """每月: 截面排序 lagged 上游月報酬, 多空 tertile chain-down 組合."""
    monthly = daily_to_monthly_log(wide)

    # up & down per chain
    chains = sorted({c for c, p in wide.columns if p == "上游"} & {c for c, p in wide.columns if p == "下游"})

    up_m = pd.DataFrame({c: monthly[(c, "上游")] for c in chains})
    down_m = pd.DataFrame({c: monthly[(c, "下游")] for c in chains})

    # 信號: 上月上游月報酬 (跨鏈)
    signal = up_m.shift(1)

    # 持有: 當月下游報酬
    holdings_ret = down_m

    # 交集日期 & 至少 N 鏈有信號的月份
    common = signal.index.intersection(holdings_ret.index)
    signal = signal.reindex(common)
    holdings_ret = holdings_ret.reindex(common)

    # 每月: rank 信號, 取 top tertile 與 bottom tertile (動態鏈數)
    strat_ret = []
    pos_log = []
    for date in common:
        s = signal.loc[date].dropna()
        r = holdings_ret.loc[date]
        # 配合: 同月份 holdings 也要存在
        s = s[s.index.intersection(r.dropna().index)]
        if len(s) < 6:  # 至少 6 鏈才分 tertile
            strat_ret.append(np.nan)
            continue
        n_pick = max(2, len(s) // 3)
        rank = s.rank(ascending=False)
        long_ch = rank[rank <= n_pick].index.tolist()
        short_ch = rank[rank >= len(s) - n_pick + 1].index.tolist()
        long_r = r[long_ch].mean()
        short_r = r[short_ch].mean()
        ls = long_r - short_r
        strat_ret.append(ls)
        pos_log.append({"date": date, "n": len(s), "long": ",".join(long_ch),
                        "short": ",".join(short_ch),
                        "long_ret": long_r, "short_ret": short_r, "ls": ls})

    strat_ret = pd.Series(strat_ret, index=common, name="LS_strategy")
    pos_log_df = pd.DataFrame(pos_log)

    return strat_ret, pos_log_df


def strategy3_daily_cross_chain(wide: pd.DataFrame, n_pick: int = 4) -> pd.Series:
    """日頻: 每日用 t-1 上游報酬截面排序, 多空 chain-down 組合."""
    chains = sorted({c for c, p in wide.columns if p == "上游"} & {c for c, p in wide.columns if p == "下游"})
    up = pd.DataFrame({c: wide[(c, "上游")] for c in chains})
    down = pd.DataFrame({c: wide[(c, "下游")] for c in chains})

    signal = up.shift(1)
    common = signal.index.intersection(down.index)
    signal = signal.reindex(common)
    down = down.reindex(common)

    out = []
    for date in common:
        s = signal.loc[date].dropna()
        r = down.loc[date]
        s = s[s.index.intersection(r.dropna().index)]
        if len(s) < n_pick * 2 + 2:
            out.append(np.nan)
            continue
        rank = s.rank(ascending=False)
        long_ch = rank[rank <= n_pick].index.tolist()
        short_ch = rank[rank >= len(s) - n_pick + 1].index.tolist()
        ls = r[long_ch].mean() - r[short_ch].mean()
        out.append(ls)
    return pd.Series(out, index=common, name="LS_daily")


def strategy_long_only(wide: pd.DataFrame, n_pick: int = 4) -> pd.Series:
    """日頻 long-only: 多 top n_pick 鏈下游 vs 等權所有鏈下游."""
    chains = sorted({c for c, p in wide.columns if p == "上游"} & {c for c, p in wide.columns if p == "下游"})
    up = pd.DataFrame({c: wide[(c, "上游")] for c in chains})
    down = pd.DataFrame({c: wide[(c, "下游")] for c in chains})

    signal = up.shift(1)
    common = signal.index.intersection(down.index)
    signal = signal.reindex(common)
    down = down.reindex(common)

    out = []
    for date in common:
        s = signal.loc[date].dropna()
        r = down.loc[date]
        s = s[s.index.intersection(r.dropna().index)]
        if len(s) < n_pick + 2:
            out.append(np.nan)
            continue
        rank = s.rank(ascending=False)
        long_ch = rank[rank <= n_pick].index.tolist()
        ls = r[long_ch].mean() - r.dropna().mean()
        out.append(ls)
    return pd.Series(out, index=common, name="LongOnly_excessEqual")


def main():
    print("[Phase 4] 多空投組策略\n")
    panel, market_df = load_panels()
    market_d = market_df["ret_market"]
    market_d.index = pd.to_datetime(market_d.index)
    market_m = np.expm1(np.log1p(market_d).resample("ME").sum(min_count=10))

    out_summary = []

    for w_label, value_col in [("EW", "ret_ew"), ("VW", "ret_vw")]:
        print(f"\n========= 權重: {w_label} =========")
        wide = to_wide(panel, value_col)

        # 策略 1: 月頻跨鏈截面 LS
        strat_m, pos_log = strategy1_cross_chain_monthly(wide, market_m)
        stats_m = perf_stats(strat_m, market_m, ann_factor=12)
        print(f"\n[月頻] CrossChain Tertile LS, 月數 {stats_m.get('n_periods', 0)}")
        for k, v in stats_m.items():
            print(f"   {k}: {v:.4f}")
        stats_m.update({"weight": w_label, "strategy": "Monthly_CrossChain_LS"})
        out_summary.append(stats_m)

        # 策略 3: 日頻跨鏈 top4 / bot4
        strat_d = strategy3_daily_cross_chain(wide, n_pick=4)
        stats_d = perf_stats(strat_d, market_d, ann_factor=252)
        print(f"\n[日頻] CrossChain Top4-Bot4 LS, 天數 {stats_d.get('n_periods', 0)}")
        for k, v in stats_d.items():
            print(f"   {k}: {v:.4f}")
        stats_d.update({"weight": w_label, "strategy": "Daily_CrossChain_LS_4"})
        out_summary.append(stats_d)

        # 策略 long only top4 minus equal benchmark
        strat_lo = strategy_long_only(wide, n_pick=4)
        stats_lo = perf_stats(strat_lo, pd.Series(0, index=strat_lo.index), ann_factor=252)
        print(f"\n[日頻] Long Top4 - Equal All, 天數 {stats_lo.get('n_periods', 0)}")
        for k, v in stats_lo.items():
            print(f"   {k}: {v:.4f}")
        stats_lo.update({"weight": w_label, "strategy": "Daily_LongTop4_minusEqual"})
        out_summary.append(stats_lo)

        if w_label == "EW":
            # 存 EW 月頻策略月報酬
            strat_m.to_frame("ret").to_csv(os.path.join(OUT, "monthly_LS_EW.csv"), encoding="utf-8-sig")
            strat_d.to_frame("ret").to_csv(os.path.join(OUT, "daily_LS_EW.csv"), encoding="utf-8-sig")
            pos_log.to_csv(os.path.join(OUT, "monthly_LS_positions.csv"), index=False, encoding="utf-8-sig")

    sm_df = pd.DataFrame(out_summary)
    sm_df.to_csv(os.path.join(OUT, "portfolio_summary.csv"), index=False, encoding="utf-8-sig")
    print(f"\n\n=== 總摘要 (寫入 portfolio_summary.csv) ===")
    print(sm_df[["strategy", "weight", "ann_ret", "ann_vol", "sharpe", "ann_alpha", "t_alpha", "win_rate", "max_dd", "n_periods"]]
          .to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
