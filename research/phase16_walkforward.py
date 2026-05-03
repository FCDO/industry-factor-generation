"""
Phase 16: 嚴格 walk-forward 驗證

確認 Phase 8 / 12 / 13 的最佳 (N, K) 不是 in-sample lookback bias.

Setup
- Rolling window: 5-year train → 1-year test, step 1 year
- 訓練期內計算 (N, K) sweep 的 net@50bp Sharpe, 取 best 應用到 test 期
- Concat 所有 test 期得 OOS 序列, 報 OOS Sharpe / alpha / t_α
- 對比 in-sample full-period winner 衰減幅度

Predictors
1. D000 (Phase 8 baseline, 整體半導體)
2. D100 (Phase 12 替換版)
3. D100+0.5·2330 (Phase 13 winner, 雙因子)

Target: 7 跨產業下游 EW (G/H/I/J/F/L/5400) — Phase 8/13/14 一致

Selection rule (per train window)
- For each (N, K), compute net@50bp Sharpe on train daily returns
- Pick (N*, K*) = argmax Sharpe
- Apply (N*, K*) to test window (signal continuity: 用全期 EMA 計算, 只取 test 期切片)

Outputs
- phase16_walkforward.csv : per-year (predictor, train_window, N*, K*, train_sharpe, test_sharpe, ...)
- phase16_oos_summary.csv : aggregated OOS performance per predictor
- fig_phase16_wf_curve.png : OOS cum wealth vs in-sample cum wealth
- fig_phase16_param_stability.png : (N*, K*) selection 隨時間變化
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
COST_RT = 0.0050
TRAIN_YEARS = 5


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


def k_day_holding(target: pd.Series, hold_k: int) -> pd.Series:
    if hold_k == 1:
        return target
    out = target.copy()
    last = target.iloc[0]
    for i in range(len(target)):
        if i % hold_k == 0:
            last = target.iloc[i]
        out.iloc[i] = last
    return out


def perf_with_alpha(daily_ret: pd.Series, market: pd.Series, ann: float = 252) -> dict:
    df = pd.concat([daily_ret.rename("s"), market.rename("b")], axis=1).dropna()
    if len(df) < 30:
        return {}
    res = sm.OLS(df["s"], sm.add_constant(df["b"])).fit(
        cov_type="HAC", cov_kwds={"maxlags": min(60, len(df) // 4)}
    )
    s = df["s"]
    return {
        "n": len(s),
        "ann_ret": s.mean() * 252,
        "ann_vol": s.std() * np.sqrt(252),
        "sharpe": s.mean() / s.std() * np.sqrt(252) if s.std() > 0 else np.nan,
        "ann_alpha": res.params["const"] * 252,
        "t_alpha": res.tvalues["const"],
        "beta": res.params["b"],
    }


def sharpe(s: pd.Series) -> float:
    s = s.dropna()
    if len(s) < 30 or s.std() == 0:
        return np.nan
    return s.mean() / s.std() * np.sqrt(252)


def precompute_strategy_returns(predictor, target):
    """Pre-compute net@50bp daily return series for ALL (N, K) combos.
    Returns dict { (N, K): pd.Series of net daily returns indexed by date }.
    """
    out = {}
    for N in PARAMS_N:
        sig = predictor.ewm(span=N, adjust=False).mean() if N > 1 else predictor
        target_w = (sig > 0).astype(float).shift(1)
        for K in PARAMS_K:
            weight = k_day_holding(target_w, K)
            gross = weight * target
            turn = weight.diff().abs().fillna(weight.iloc[0])
            net = gross - turn * (COST_RT / 2.0)
            out[(N, K)] = net
    return out


def main():
    print("=" * 100)
    print(f"Phase 16: Walk-Forward 驗證 (train={TRAIN_YEARS}y, test=1y, step=1y, cost=50bp)")
    print("=" * 100)

    chain = pd.read_csv(os.path.join(ROOT, "industry_chain.csv"), dtype=str)
    adj_close = filter_common(load_pickle_df("etl#adj_close.pickle"))
    daily_ret = adj_close.pct_change(fill_method=None).iloc[1:]
    market = daily_ret.mean(axis=1)

    print(f"\n樣本期間: {daily_ret.index.min().date()} → {daily_ret.index.max().date()}  "
          f"({len(daily_ret)} 個交易日)")

    # === panel & predictors ===
    panel = {}
    for ic in chain["industry_code"].unique():
        members = chain[chain["industry_code"] == ic]["stock_id"].unique().tolist()
        cols = [s for s in members if s in daily_ret.columns]
        if len(cols) >= 3:
            panel[ic] = daily_ret[cols].mean(axis=1)
    panel = pd.DataFrame(panel)

    d100_members = chain[(chain["industry_code"] == "D000") &
                          (chain["sub_code"] == "D100")]["stock_id"].unique().tolist()
    d100_cols = [s for s in d100_members if s in daily_ret.columns]
    d100 = daily_ret[d100_cols].mean(axis=1)
    tsmc = daily_ret["2330"]

    predictors = {
        "D000":           panel["D000"],
        "D100":           d100,
        "D100+0.5·2330":  0.5 * d100 + 0.5 * tsmc,
    }
    target = panel[DOWNSTREAM].mean(axis=1)
    print(f"Target: 7 跨產業下游 EW")
    print(f"Predictors: {list(predictors.keys())}")

    # === Pre-compute strategy returns (one-time, full sample) for each predictor ===
    cache = {label: precompute_strategy_returns(pred, target) for label, pred in predictors.items()}

    # === Walk-forward windows ===
    yrs = sorted(set(daily_ret.index.year))
    yr_min, yr_max = min(yrs), max(yrs)
    test_years = [y for y in yrs if y - TRAIN_YEARS >= yr_min]  # need TRAIN_YEARS prior
    print(f"\n訓練/測試切分: train [Y-{TRAIN_YEARS}, Y), test [Y, Y+1)")
    print(f"  測試年份: {test_years[0]} → {test_years[-1]}  共 {len(test_years)} 年")

    # === Walk-forward loop ===
    print("\n" + "=" * 100)
    print("Walk-Forward 結果")
    print("=" * 100)
    print(f"\n{'predictor':<18s} {'test_year':>9s}  {'N*':>3s} {'K*':>3s}  "
          f"{'train_Sh':>9s}  {'test_Sh':>9s}  {'test_ann':>10s}  {'2nd best':>10s}")

    wf_rows = []
    oos_returns = {label: pd.Series(dtype=float) for label in predictors}

    for test_y in test_years:
        train_start = pd.Timestamp(year=test_y - TRAIN_YEARS, month=1, day=1)
        train_end = pd.Timestamp(year=test_y, month=1, day=1)
        test_end = pd.Timestamp(year=test_y + 1, month=1, day=1)

        for label, ret_dict in cache.items():
            # train: pick best (N, K) by Sharpe
            best_NK, best_train_sh, second_train_sh = None, -np.inf, -np.inf
            train_perfs = []
            for (N, K), net in ret_dict.items():
                train_slice = net[(net.index >= train_start) & (net.index < train_end)]
                sh = sharpe(train_slice)
                train_perfs.append((sh, (N, K)))
                if sh > best_train_sh:
                    second_train_sh = best_train_sh
                    best_train_sh = sh
                    best_NK = (N, K)
                elif sh > second_train_sh:
                    second_train_sh = sh
            # test: apply best
            test_slice = ret_dict[best_NK][(ret_dict[best_NK].index >= train_end) &
                                            (ret_dict[best_NK].index < test_end)]
            test_sh = sharpe(test_slice)
            test_ann = test_slice.mean() * 252 if len(test_slice) > 0 else np.nan

            print(f"{label:<18s} {test_y:>9d}  {best_NK[0]:>3d} {best_NK[1]:>3d}  "
                  f"{best_train_sh:>+9.3f}  {test_sh:>+9.3f}  {test_ann:>+10.4f}  "
                  f"{second_train_sh:>+10.3f}")

            wf_rows.append({
                "predictor": label, "test_year": test_y,
                "train_start": train_start.date(), "train_end": train_end.date(),
                "best_N": best_NK[0], "best_K": best_NK[1],
                "train_sharpe": best_train_sh, "test_sharpe": test_sh,
                "test_ann_ret": test_ann, "test_n_days": len(test_slice),
                "second_best_train_sharpe": second_train_sh,
            })
            # accumulate OOS
            oos_returns[label] = pd.concat([oos_returns[label], test_slice])

    wf_df = pd.DataFrame(wf_rows)
    wf_df.to_csv(os.path.join(OUT, "phase16_walkforward.csv"),
                 index=False, encoding="utf-8-sig")

    # === OOS aggregate stats ===
    print("\n" + "=" * 100)
    print("OOS 總結 (concat 所有 test 期)")
    print("=" * 100)
    market_oos = market[market.index >= pd.Timestamp(year=test_years[0], month=1, day=1)]

    oos_summary = []
    for label, ret in oos_returns.items():
        ret = ret.sort_index()
        if len(ret) == 0:
            continue
        # full-sample baseline (in-sample winner)
        in_sample_perfs = [(sharpe(net), (N, K), net) for (N, K), net in cache[label].items()]
        in_sample_perfs.sort(key=lambda x: -x[0] if not np.isnan(x[0]) else -1e9)
        is_winner_NK = in_sample_perfs[0][1]
        is_winner_ret = cache[label][is_winner_NK]
        # Restrict in-sample winner to OOS period for fair comparison
        is_winner_oos_period = is_winner_ret[(is_winner_ret.index >= ret.index.min()) &
                                              (is_winner_ret.index <= ret.index.max())]
        p_oos = perf_with_alpha(ret, market_oos.reindex(ret.index))
        p_is = perf_with_alpha(is_winner_oos_period,
                                market_oos.reindex(is_winner_oos_period.index))
        oos_summary.append({
            "predictor": label,
            "WF_n": len(ret),
            "WF_ann_ret": p_oos.get("ann_ret"),
            "WF_sharpe": p_oos.get("sharpe"),
            "WF_alpha": p_oos.get("ann_alpha"),
            "WF_t_alpha": p_oos.get("t_alpha"),
            "IS_winner_N": is_winner_NK[0], "IS_winner_K": is_winner_NK[1],
            "IS_full_sharpe": in_sample_perfs[0][0],
            "IS_OOS_period_ann_ret": p_is.get("ann_ret"),
            "IS_OOS_period_sharpe": p_is.get("sharpe"),
            "IS_OOS_period_alpha": p_is.get("ann_alpha"),
            "IS_OOS_period_t_alpha": p_is.get("t_alpha"),
        })

    summary_df = pd.DataFrame(oos_summary)
    summary_df.to_csv(os.path.join(OUT, "phase16_oos_summary.csv"),
                       index=False, encoding="utf-8-sig")

    fmt = lambda x: f"{x:+.4f}" if isinstance(x, float) else str(x)
    print(f"\n{'predictor':<18s} {'WF_ann':>9s} {'WF_Sh':>8s} {'WF_α':>9s} {'WF_t_α':>9s}  "
          f"|| {'IS winner':>11s} {'IS_Sh_full':>11s} {'IS_α_OOS':>10s} {'IS_t_α_OOS':>11s}")
    for _, r in summary_df.iterrows():
        is_label = f"({r['IS_winner_N']},{r['IS_winner_K']})"
        print(f"{r['predictor']:<18s} {r['WF_ann_ret']:>+9.4f} {r['WF_sharpe']:>+8.3f} "
              f"{r['WF_alpha']:>+9.4f} {r['WF_t_alpha']:>+9.3f}  "
              f"|| {is_label:>11s} {r['IS_full_sharpe']:>+11.3f} "
              f"{r['IS_OOS_period_alpha']:>+10.4f} {r['IS_OOS_period_t_alpha']:>+11.3f}")

    # === Visualization 1: OOS cum wealth vs in-sample winner ===
    fig, ax = plt.subplots(figsize=(13, 6))
    colors = {"D000": "#0277bd", "D100": "#6a1b9a", "D100+0.5·2330": "#1b5e20"}
    for label, ret in oos_returns.items():
        ret = ret.sort_index().fillna(0)
        cum_wf = (1 + ret).cumprod()
        ax.plot(cum_wf, color=colors[label], lw=1.7, alpha=0.95,
                label=f"{label} WF [{cum_wf.iloc[-1]:.2f}x, Sh={sharpe(ret):.2f}]")

        # in-sample winner cum on same period
        is_NK = (summary_df[summary_df["predictor"] == label]["IS_winner_N"].iloc[0],
                 summary_df[summary_df["predictor"] == label]["IS_winner_K"].iloc[0])
        is_ret = cache[label][is_NK].reindex(ret.index).fillna(0)
        cum_is = (1 + is_ret).cumprod()
        ax.plot(cum_is, color=colors[label], lw=1.0, ls="--", alpha=0.6,
                label=f"{label} IS@(N={is_NK[0]},K={is_NK[1]}) [{cum_is.iloc[-1]:.2f}x]")

    # market on OOS period
    mkt_oos = market.reindex(oos_returns["D100"].sort_index().index).fillna(0)
    cum_mkt = (1 + mkt_oos).cumprod()
    ax.plot(cum_mkt, color="#666666", lw=1.0, alpha=0.5,
            label=f"Market EW [{cum_mkt.iloc[-1]:.2f}x]")

    ax.set_yscale("log")
    ax.set_xlabel("date (OOS)")
    ax.set_ylabel("OOS 累積資本 (log)")
    ax.set_title(f"Phase 16: Walk-Forward OOS 累積資本 (train={TRAIN_YEARS}y, test=1y)\n"
                 f"實線=WF (動態 N,K), 虛線=in-sample winner 鎖定參數")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig_phase16_wf_curve.png"), dpi=130)
    plt.close()
    print(f"\n圖檔: fig_phase16_wf_curve.png")

    # === Visualization 2: parameter stability over time ===
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for label in predictors:
        sub = wf_df[wf_df["predictor"] == label].sort_values("test_year")
        axes[0].plot(sub["test_year"], sub["best_N"], "o-", color=colors[label],
                     label=label, alpha=0.85)
        axes[1].plot(sub["test_year"], sub["best_K"], "s-", color=colors[label],
                     label=label, alpha=0.85)
    axes[0].set_xlabel("test_year"); axes[0].set_ylabel("best smooth_N (EMA span)")
    axes[0].set_title("(A) 各年訓練最佳 N*")
    axes[0].set_yticks(PARAMS_N); axes[0].grid(alpha=0.3); axes[0].legend(fontsize=9)
    axes[1].set_xlabel("test_year"); axes[1].set_ylabel("best hold_K")
    axes[1].set_title("(B) 各年訓練最佳 K*")
    axes[1].set_yticks(PARAMS_K); axes[1].grid(alpha=0.3); axes[1].legend(fontsize=9)
    plt.suptitle("Phase 16: Walk-Forward (N*, K*) 跨年穩定性", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig_phase16_param_stability.png"), dpi=130)
    plt.close()
    print(f"圖檔: fig_phase16_param_stability.png")

    # === Visualization 3: train_Sh vs test_Sh scatter (look-ahead 嚴重度) ===
    fig, ax = plt.subplots(figsize=(8, 7))
    for label in predictors:
        sub = wf_df[wf_df["predictor"] == label]
        ax.scatter(sub["train_sharpe"], sub["test_sharpe"], s=80, alpha=0.75,
                   color=colors[label], edgecolor="black", lw=0.4, label=label)
    lim = [-1, 4]
    ax.plot(lim, lim, "k--", lw=0.8, alpha=0.5, label="train = test (no overfit)")
    ax.axhline(0, color="gray", lw=0.5); ax.axvline(0, color="gray", lw=0.5)
    ax.set_xlabel("Train Sharpe (in-sample)")
    ax.set_ylabel("Test Sharpe (OOS, 1-year forward)")
    ax.set_title("Phase 16: train Sharpe vs test Sharpe — 散點離 y=x 越遠 = 過擬合越嚴重")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig_phase16_train_vs_test.png"), dpi=130)
    plt.close()
    print(f"圖檔: fig_phase16_train_vs_test.png")

    # === 結論摘要 ===
    print("\n" + "=" * 100)
    print("結論摘要 — WF vs in-sample 衰減")
    print("=" * 100)
    for _, r in summary_df.iterrows():
        decay_alpha = (r["WF_alpha"] - r["IS_OOS_period_alpha"]) * 100
        decay_sh = r["WF_sharpe"] - r["IS_OOS_period_sharpe"]
        decay_t = r["WF_t_alpha"] - r["IS_OOS_period_t_alpha"]
        print(f"\n  {r['predictor']}:")
        print(f"    in-sample winner = (N={r['IS_winner_N']}, K={r['IS_winner_K']})")
        print(f"    WF OOS:        α={r['WF_alpha']:+.4f}, t_α={r['WF_t_alpha']:+.3f}, Sh={r['WF_sharpe']:+.3f}")
        print(f"    IS 鎖定 OOS:   α={r['IS_OOS_period_alpha']:+.4f}, t_α={r['IS_OOS_period_t_alpha']:+.3f}, "
              f"Sh={r['IS_OOS_period_sharpe']:+.3f}")
        print(f"    衰減 (WF - IS):  Δα={decay_alpha:+.2f}pp, Δt={decay_t:+.3f}, ΔSh={decay_sh:+.3f}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
