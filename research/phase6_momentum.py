"""
Phase 6: 跨產業動能策略 (Moskowitz-Grinblatt 1999 風)

針對 70 個 (chain × position) 組合與 26 條鏈 (合併上中下游), 跑 J/K 動能策略:
    每月底以過去 J 月報酬排序, 多 top tertile / 空 bottom tertile, 持有 K 月.

對照:
- Moskowitz-Grinblatt (1999, JF) 美國: J=6, K=6 動能 ~4.2%/季
- Chui-Titman-Wei (2010, JF) 跨國: 台灣股票動能弱甚至反轉
- Jegadeesh-Titman (1993) 經典 J/K = 6/6 框架

檢驗點:
1. 動能 vs 反轉: 台股產業層級是否與股票層級不同?
2. 上中下游差異: 哪個位置的動能訊號最強?
3. 與 Phase 4 spillover 訊號的相關性 (同一回事 or 獨立 alpha?)
"""
from __future__ import annotations

import os
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


def daily_to_monthly_log(daily: pd.DataFrame) -> pd.DataFrame:
    """日 → 月對數累積 (再轉回簡單)."""
    log = np.log1p(daily)
    m_log = log.resample("ME").sum(min_count=10)
    return np.expm1(m_log), m_log


def momentum_signal(monthly_log: pd.DataFrame, lookback: int, skip: int = 0) -> pd.DataFrame:
    """訊號 = past lookback 月 log return, 在月底 t 觀察 (排除最近 skip 月).

    例: lookback=6, skip=1 → t 月底觀察 t-7 ~ t-2 共 6 月累積報酬.
    持有: t+1 月.
    """
    return monthly_log.rolling(lookback).sum().shift(skip)


def jk_strategy(monthly_simple: pd.DataFrame, monthly_log: pd.DataFrame,
                lookback: int, hold: int, skip: int = 0,
                n_pick: int | str = "tertile",
                long_only_vs_avg: bool = False) -> pd.Series:
    """執行 J/K 動能策略, 回傳月策略報酬."""
    signal = momentum_signal(monthly_log, lookback, skip)
    # 信號要在 t 月底前可觀察, 持有 t+1 ~ t+hold 月
    sig_lag = signal.shift(1)  # 月 t 看到的是 t-1 月底訊號 → 對齊月 t 持有

    rets = []
    for date in monthly_simple.index:
        if date not in sig_lag.index:
            rets.append(np.nan); continue
        s = sig_lag.loc[date].dropna()
        r = monthly_simple.loc[date]
        s = s[s.index.intersection(r.dropna().index)]
        n = len(s)
        if n < 6:
            rets.append(np.nan); continue

        if n_pick == "tertile":
            k = max(2, n // 3)
        elif n_pick == "quintile":
            k = max(2, n // 5)
        else:
            k = int(n_pick)

        rank = s.rank(ascending=False)
        long_set = rank[rank <= k].index.tolist()
        short_set = rank[rank >= n - k + 1].index.tolist()
        if long_only_vs_avg:
            ls = r[long_set].mean() - r.dropna().mean()
        else:
            ls = r[long_set].mean() - r[short_set].mean()
        rets.append(ls)

    raw = pd.Series(rets, index=monthly_simple.index, name="ls")
    if hold > 1:
        return raw.rolling(hold, min_periods=1).mean()
    return raw


def perf_stats(strat_ret: pd.Series, benchmark_ret: pd.Series, ann: float = 12):
    df = pd.concat([strat_ret.rename("s"), benchmark_ret.rename("b")], axis=1).dropna()
    if len(df) < 12:
        return {}
    res = sm.OLS(df["s"], sm.add_constant(df["b"])).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
    ann_ret = df["s"].mean() * ann
    ann_vol = df["s"].std() * np.sqrt(ann)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
    cum = (1 + df["s"]).cumprod()
    dd = (cum / cum.cummax() - 1).min()
    return {
        "n": len(df),
        "ann_ret": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "ann_alpha": res.params["const"] * ann,
        "t_alpha": res.tvalues["const"],
        "beta": res.params["b"],
        "win_rate": (df["s"] > 0).mean(),
        "max_dd": dd,
    }


def main():
    print("[Phase 6] 跨產業動能策略\n")
    panel, market_df = load()
    market_d = market_df["ret_market"]
    market_d.index = pd.to_datetime(market_d.index)
    market_m_simple, market_m_log = daily_to_monthly_log(market_d.to_frame("m"))
    market_m_simple = market_m_simple["m"]; market_m_log = market_m_log["m"]

    wide_ew = to_wide(panel, "ret_ew")
    wide_vw = to_wide(panel, "ret_vw")

    # 月頻 panel
    m_simple_ew, m_log_ew = daily_to_monthly_log(wide_ew)
    m_simple_vw, m_log_vw = daily_to_monthly_log(wide_vw)

    # 也建構鏈層級 (合併上中下游) 月報酬
    print("構建鏈層級 (合併上中下游 EW) 月報酬...")
    chain_simple = m_simple_ew.groupby(level=0, axis=1).mean()  # 26 鏈
    chain_log = np.log1p(chain_simple)

    # 也按位置 (上游/中游/下游) 切片
    pos_panels = {}
    for pos in ["上游", "中游", "下游"]:
        cols = [c for c in m_simple_ew.columns if c[1] == pos]
        if cols:
            pos_panels[pos] = m_simple_ew[cols]

    print(f"  全 70 組合月報酬: {m_simple_ew.shape}")
    print(f"  鏈層級 (26 鏈): {chain_simple.shape}")
    for pos, p in pos_panels.items():
        print(f"  {pos} only: {p.shape}")

    # === 1. 主策略掃 J/K ===
    print("\n=== 主策略：全 70 組合 (EW), tertile LS ===")
    rows = []
    for J in [3, 6, 9, 12]:
        for K in [1, 3, 6]:
            for skip in [0, 1]:
                ls = jk_strategy(m_simple_ew, m_log_ew, J, K, skip, "tertile")
                stats = perf_stats(ls, market_m_simple)
                row = {"scope": "All70_EW", "J": J, "K": K, "skip": skip, "type": "tertile_LS", **stats}
                rows.append(row)
                print(f"  J={J}, K={K}, skip={skip}: ann_ret={stats.get('ann_ret', np.nan):.4f}, "
                      f"sharpe={stats.get('sharpe', np.nan):.4f}, "
                      f"alpha={stats.get('ann_alpha', np.nan):.4f}, t={stats.get('t_alpha', np.nan):.4f}")

    # === 2. 鏈層級 (26 鏈) ===
    print("\n=== 鏈層級 (26 鏈, 合併上中下游) tertile LS ===")
    chain_log_m = np.log1p(chain_simple)
    for J in [3, 6, 12]:
        for K in [1, 3, 6]:
            ls = jk_strategy(chain_simple, chain_log_m, J, K, skip=0, n_pick="tertile")
            stats = perf_stats(ls, market_m_simple)
            row = {"scope": "Chain26_EW", "J": J, "K": K, "skip": 0, "type": "tertile_LS", **stats}
            rows.append(row)
            print(f"  J={J}, K={K}: ann_ret={stats.get('ann_ret', np.nan):.4f}, "
                  f"sharpe={stats.get('sharpe', np.nan):.4f}, "
                  f"alpha={stats.get('ann_alpha', np.nan):.4f}, t={stats.get('t_alpha', np.nan):.4f}")

    # === 3. 按位置切片 ===
    print("\n=== 按位置切片 (各 position 分別 tertile LS, J=6, K=1) ===")
    for pos, panel_pos in pos_panels.items():
        log_pos = np.log1p(panel_pos)
        ls = jk_strategy(panel_pos, log_pos, 6, 1, 0, "tertile")
        stats = perf_stats(ls, market_m_simple)
        row = {"scope": f"WithinPosition_{pos}_EW", "J": 6, "K": 1, "skip": 0, "type": "tertile_LS", **stats}
        rows.append(row)
        print(f"  {pos}: n={stats.get('n', 0)}, ann_ret={stats.get('ann_ret', np.nan):.4f}, "
              f"sharpe={stats.get('sharpe', np.nan):.4f}, t_alpha={stats.get('t_alpha', np.nan):.4f}")

    # === 4. VW 主策略掃 ===
    print("\n=== VW 主策略 (全 70 組合 VW) ===")
    for J in [3, 6, 12]:
        for K in [1, 3]:
            ls = jk_strategy(m_simple_vw, m_log_vw, J, K, skip=0, n_pick="tertile")
            stats = perf_stats(ls, market_m_simple)
            row = {"scope": "All70_VW", "J": J, "K": K, "skip": 0, "type": "tertile_LS", **stats}
            rows.append(row)
            print(f"  J={J}, K={K}: ann_ret={stats.get('ann_ret', np.nan):.4f}, "
                  f"sharpe={stats.get('sharpe', np.nan):.4f}, t_alpha={stats.get('t_alpha', np.nan):.4f}")

    # === 5. Time-series momentum (TSMOM, Moskowitz-Ooi-Pedersen 2012) ===
    print("\n=== TSMOM (時序動能, J=6, sign of past return) ===")
    rolling6 = m_log_ew.rolling(6).sum().shift(1)
    sign_ew = np.sign(rolling6)
    tsmom_daily = (sign_ew * m_simple_ew).mean(axis=1)
    stats = perf_stats(tsmom_daily, market_m_simple)
    print(f"  TSMOM EW: ann_ret={stats.get('ann_ret', np.nan):.4f}, sharpe={stats.get('sharpe', np.nan):.4f}, "
          f"t_alpha={stats.get('t_alpha', np.nan):.4f}")
    rows.append({"scope": "All70_EW", "J": 6, "K": 1, "skip": 0, "type": "TSMOM", **stats})

    # === 整理輸出 ===
    out_df = pd.DataFrame(rows)
    out_df.to_csv(os.path.join(OUT, "momentum_results.csv"), index=False, encoding="utf-8-sig")
    print(f"\n結果寫入 momentum_results.csv ({len(out_df)} 列)")

    # === 最佳組合表現繪圖 ===
    # 選 J=6, K=1, skip=0, All70 EW (Moskowitz-Grinblatt 標準)
    ls_main = jk_strategy(m_simple_ew, m_log_ew, 6, 1, 0, "tertile")
    cum_strat = (1 + ls_main.dropna()).cumprod()
    cum_mkt = (1 + market_m_simple.reindex(ls_main.dropna().index).fillna(0)).cumprod()
    plt.figure(figsize=(11, 5))
    plt.plot(cum_strat.index, cum_strat.values, label="Industry Momentum (J=6, K=1, tertile LS)", lw=1.5)
    plt.plot(cum_mkt.index, cum_mkt.values, label="Market EW", lw=1.5, alpha=0.7)
    plt.title("跨產業動能策略 累積報酬 (月頻)")
    plt.xlabel("date"); plt.ylabel("cumulative wealth")
    plt.yscale("symlog")
    plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig_momentum_cum.png"), dpi=130)
    plt.close()
    print(f"圖檔寫入 fig_momentum_cum.png")

    # === 與 spillover 訊號的相關性 ===
    print("\n=== 與 Phase 4 月頻 spillover 訊號相關性 ===")
    try:
        spill = pd.read_csv(os.path.join(OUT, "monthly_LS_EW.csv"), index_col=0, parse_dates=True)["ret"]
        df_corr = pd.concat([ls_main.rename("mom"), spill.rename("spill")], axis=1).dropna()
        if len(df_corr) > 12:
            corr = df_corr.corr().iloc[0, 1]
            print(f"  N={len(df_corr)} 月; corr(動能, spillover) = {corr:.4f}")

            # 組合: 簡單平均
            combo = (df_corr["mom"] + df_corr["spill"]) / 2
            stats = perf_stats(combo, market_m_simple)
            print(f"  簡單平均組合 ann_ret={stats.get('ann_ret', 0):.4f}, "
                  f"sharpe={stats.get('sharpe', 0):.4f}, t_alpha={stats.get('t_alpha', 0):.4f}")
    except FileNotFoundError:
        print("  (找不到 monthly_LS_EW.csv, 跳過)")

    # === 總結最佳策略 ===
    print("\n=== Top 5 by t_alpha ===")
    top = out_df.sort_values("t_alpha", ascending=False, na_position="last").head(5)
    print(top[["scope", "J", "K", "skip", "type", "ann_ret", "sharpe", "ann_alpha", "t_alpha", "win_rate", "max_dd", "n"]]
          .to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
