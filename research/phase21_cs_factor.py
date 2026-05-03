"""
Phase 21: Cross-Sectional Alpha Factor — β × predicted lead signal

Phase 20 結論
- 純 β 排序 cross-sectional 沒有 alpha (Q5 alpha -1%, Q1 alpha 0% — 無 spread)
- 必須結合 binary timing 才產生 alpha (Q5 timed alpha 17.6%)
- 長空中性化失敗 (Q5-Q1 timed alpha -10%, β 排序 LS 帶反向 market beta)

問題: 能否轉化為 pure cross-sectional factor, 不依賴 binary timing 切換?

設計
- score_i(t) = β_i(t) × predicted_lead(t)
  其中 predicted_lead 為連續訊號 (EMA, z-score), 而非 binary sign
- 直觀: 當 lead 訊號預示未來高報酬, score 順著 β 排序; 訊號預示低報酬, score 反轉
- Long top-decile of score, short bottom-decile → pure CS factor

5 種 variant:
A. β × EMA(lead, 10)              連續, 與 Phase 13 timing span 對齊
B. β × sign(EMA(lead, 10))        離散 (作對照, 等同 Phase 20 結構)
C. β × z(lead) (60d rolling z)    standardized
D. β only (Phase 20 baseline)     已知無 alpha
E. predicted_return = β × predicted_lead 直接視為預測報酬 (continuous decile sort)

評估
- LS (decile 10 - decile 1) ann_alpha, t, Sharpe
- IC (information coef): Spearman ρ(score_i(t), r_i(t+1)) across stocks i
- IR = mean(IC) / std(IC) × sqrt(252)
- Turnover, transaction cost @ 50bp

Output:
- phase21_cs_factor_results.csv
- fig_phase21_factor_curves.png
- fig_phase21_ic_timeseries.png
"""
from __future__ import annotations

import os
import pickle
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr

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
ROLL_WIN = 250
N_DECILES = 10
COST_RT = 0.0050
SMOOTH_N = 10
ZSCORE_WIN = 60


def load_pickle_df(name):
    with open(os.path.join(DB, name), "rb") as f:
        df = pickle.load(f)
    df = df.set_index("date"); df.index = pd.to_datetime(df.index)
    df.columns = df.columns.astype(str)
    return df


def filter_common(df):
    keep = [c for c in df.columns if c.isdigit() and len(c) == 4 and not c.startswith("00")]
    return df[keep]


def perf_with_alpha(daily_ret, market, ann=252):
    df = pd.concat([daily_ret.rename("s"), market.rename("b")], axis=1).dropna()
    if len(df) < 60:
        return {}
    res = sm.OLS(df["s"], sm.add_constant(df["b"])).fit(cov_type="HAC", cov_kwds={"maxlags": 60})
    s = df["s"]; cum = (1 + s).cumprod()
    return {
        "n": len(s), "ann_ret": s.mean()*ann, "ann_vol": s.std()*np.sqrt(ann),
        "sharpe": s.mean()/s.std()*np.sqrt(ann) if s.std() > 0 else np.nan,
        "max_dd": (cum/cum.cummax() - 1).min(),
        "ann_alpha": res.params["const"]*ann, "t_alpha": res.tvalues["const"],
        "beta": res.params["b"],
    }


def perf_from_weights(W, daily_ret_uni, market):
    gross = (W * daily_ret_uni).sum(axis=1)
    turn = W.diff().abs().sum(axis=1)
    if len(turn) > 0:
        turn.iloc[0] = W.iloc[0].abs().sum()
    cost = turn * (COST_RT / 2.0)
    net = gross - cost
    return gross, net, turn


def main():
    print("=" * 100)
    print("Phase 21: Cross-Sectional Alpha Factor — β × predicted lead")
    print("=" * 100)

    chain = pd.read_csv(os.path.join(ROOT, "industry_chain.csv"), dtype=str)
    adj_close = filter_common(load_pickle_df("etl#adj_close.pickle"))
    daily_ret = adj_close.pct_change(fill_method=None).iloc[1:]
    market = daily_ret.mean(axis=1)

    # === Lead signal ===
    d100_members = chain[(chain["industry_code"] == "D000") &
                          (chain["sub_code"] == "D100")]["stock_id"].unique().tolist()
    d100_cols = [s for s in d100_members if s in daily_ret.columns]
    d100 = daily_ret[d100_cols].mean(axis=1)
    tsmc = daily_ret["2330"]
    lead = 0.5 * d100 + 0.5 * tsmc
    lead_lag = lead.shift(1)
    lead_ema = lead.ewm(span=SMOOTH_N, adjust=False).mean()
    lead_z = ((lead - lead.rolling(ZSCORE_WIN).mean()) /
              lead.rolling(ZSCORE_WIN).std())

    # === Universe ===
    uni_set = set()
    for ic in DOWNSTREAM:
        members = chain[chain["industry_code"] == ic]["stock_id"].unique()
        uni_set.update(members)
    uni_set -= set(d100_cols)
    uni_set -= {"2330"}
    universe = sorted([s for s in uni_set if s in daily_ret.columns])
    rets_uni = daily_ret[universe]
    print(f"\nUniverse: {len(universe)} 檔")

    # === Rolling β (vectorized) ===
    print(f"\n[1] 計算 rolling β (window={ROLL_WIN}d)")
    E_x = lead_lag.rolling(ROLL_WIN).mean()
    E_x2 = (lead_lag ** 2).rolling(ROLL_WIN).mean()
    var_x = E_x2 - E_x ** 2
    E_r = rets_uni.rolling(ROLL_WIN).mean()
    rx = rets_uni.mul(lead_lag, axis=0)
    E_rx = rx.rolling(ROLL_WIN).mean()
    cov_rx = E_rx.sub(E_r.mul(E_x, axis=0))
    betas = cov_rx.div(var_x, axis=0)
    print(f"  shape: {betas.shape}")

    # === 構建 5 種 score ===
    print(f"\n[2] 構建 5 種 cross-sectional score")
    # All scores aligned to daily index
    scores = {}
    # A. β × EMA(lead) — 連續
    scores["A_beta_x_EMA"] = betas.mul(lead_ema.shift(1), axis=0)
    # B. β × sign(EMA(lead)) — 離散 (對照)
    scores["B_beta_x_sign"] = betas.mul(np.sign(lead_ema.shift(1)), axis=0)
    # C. β × z-score(lead, 60d) — normalized
    scores["C_beta_x_zlead"] = betas.mul(lead_z.shift(1), axis=0)
    # D. β only — Phase 20 baseline
    scores["D_beta_only"] = betas.copy()
    # E. β × yesterday lead (no smooth)
    scores["E_beta_x_lead_lag"] = betas.mul(lead_lag, axis=0)

    # === Monthly rebal: form decile L/S portfolios ===
    print(f"\n[3] Monthly rebal, decile 10 - decile 1 long-short")
    month_ends = pd.Series(daily_ret.index).groupby(
        [daily_ret.index.year, daily_ret.index.month]
    ).max()
    month_ends = pd.DatetimeIndex(month_ends.values)

    strategy_results = {}

    for label, score in scores.items():
        # 月底 form deciles
        valid_dates = [d for d in month_ends if d in score.index and
                        score.loc[d].dropna().shape[0] >= N_DECILES * 5]

        # weight matrix
        W = pd.DataFrame(0.0, index=daily_ret.index, columns=universe)
        for i, d_rebal in enumerate(valid_dates):
            next_rebal = valid_dates[i + 1] if i + 1 < len(valid_dates) else daily_ret.index.max() + pd.Timedelta(days=1)
            hold_idx = daily_ret.index[(daily_ret.index > d_rebal) & (daily_ret.index <= next_rebal)]
            s = score.loc[d_rebal].dropna()
            if len(s) < N_DECILES * 5:
                continue
            try:
                ranks = pd.qcut(s, N_DECILES, labels=False)
            except ValueError:
                ranks = pd.qcut(s.rank(method="first"), N_DECILES, labels=False)
            top = ranks[ranks == N_DECILES - 1].index.tolist()
            bot = ranks[ranks == 0].index.tolist()
            if len(top) > 0:
                W.loc[hold_idx, top] = 1.0 / len(top)
            if len(bot) > 0:
                W.loc[hold_idx, bot] = -1.0 / len(bot)

        gross, net, turn = perf_from_weights(W, rets_uni, market)
        strategy_results[label] = {
            "gross": gross, "net": net, "turn": turn, "weights": W,
        }

    # === Compute IC (information coefficient) ===
    # IC = Spearman corr between score(t) and r(t+1) at each rebal date
    print(f"\n[4] 計算 IC (monthly Spearman 相關)")
    ic_results = {}
    for label, score in scores.items():
        ics = []
        for i, d_rebal in enumerate([d for d in month_ends if d in score.index]):
            # next month return
            try:
                next_d = month_ends[list(month_ends).index(d_rebal) + 1]
            except IndexError:
                continue
            s = score.loc[d_rebal].dropna()
            # cumulative return from d_rebal to next_d
            r_next = ((1 + rets_uni.loc[(rets_uni.index > d_rebal) & (rets_uni.index <= next_d)])
                      .prod() - 1)
            df = pd.concat([s.rename("score"), r_next.rename("r")], axis=1).dropna()
            if len(df) < 30:
                continue
            ic, _ = spearmanr(df["score"], df["r"])
            ics.append({"date": d_rebal, "ic": ic, "n": len(df)})
        ic_df = pd.DataFrame(ics)
        ic_results[label] = ic_df

    # === 印各 variant 表現 ===
    print(f"\n" + "=" * 100)
    print("各 CS factor variant 表現 (monthly rebal, decile 10 - decile 1 LS, 扣 50bp)")
    print("=" * 100)

    fmt = lambda x: f"{x:+.4f}"
    print(f"\n{'variant':<22s} {'ann_ret':>10s} {'ann_vol':>10s} {'Sharpe':>8s} "
          f"{'alpha':>9s} {'t_α':>8s} {'IC_mean':>8s} {'IR':>7s} {'ann_turn':>10s}")
    summary_rows = []
    for label, res in strategy_results.items():
        p_gross = perf_with_alpha(res["gross"], market)
        p_net = perf_with_alpha(res["net"], market)
        ic_df = ic_results[label]
        ic_mean = ic_df["ic"].mean()
        ic_std = ic_df["ic"].std()
        ir = ic_mean / ic_std * np.sqrt(12) if ic_std > 0 else np.nan
        ann_turn = res["turn"].mean() * 252
        print(f"{label:<22s} {p_net['ann_ret']:>+10.4f} {p_net['ann_vol']:>10.4f} "
              f"{p_net['sharpe']:>+8.3f} {p_net['ann_alpha']:>+9.4f} {p_net['t_alpha']:>+8.3f} "
              f"{ic_mean:>+8.4f} {ir:>+7.3f} {ann_turn:>10.3f}")
        summary_rows.append({
            "variant": label,
            "gross_ann": p_gross["ann_ret"], "gross_alpha": p_gross["ann_alpha"], "gross_t": p_gross["t_alpha"],
            "net_ann": p_net["ann_ret"], "net_vol": p_net["ann_vol"],
            "net_sharpe": p_net["sharpe"], "net_alpha": p_net["ann_alpha"],
            "net_t_alpha": p_net["t_alpha"],
            "ic_mean": ic_mean, "ic_std": ic_std, "ir": ir,
            "ann_turn": ann_turn,
        })
    pd.DataFrame(summary_rows).to_csv(os.path.join(OUT, "phase21_cs_factor_results.csv"),
                                       index=False, encoding="utf-8-sig")

    # === Phase 13/20 baseline 對照 ===
    print(f"\n>>> Reference baselines:")
    timing = (lead_ema > 0).astype(float).shift(1)
    W_q5_timed = pd.DataFrame(0.0, index=daily_ret.index, columns=universe)
    # Phase 20 Q5 timed
    score_d = scores["D_beta_only"]
    valid_dates = [d for d in month_ends if d in score_d.index and
                    score_d.loc[d].dropna().shape[0] >= 25]
    for i, d_rebal in enumerate(valid_dates):
        next_rebal = valid_dates[i + 1] if i + 1 < len(valid_dates) else daily_ret.index.max() + pd.Timedelta(days=1)
        hold_idx = daily_ret.index[(daily_ret.index > d_rebal) & (daily_ret.index <= next_rebal)]
        s = score_d.loc[d_rebal].dropna()
        try:
            ranks = pd.qcut(s, 5, labels=False)
        except ValueError:
            ranks = pd.qcut(s.rank(method="first"), 5, labels=False)
        top = ranks[ranks == 4].index.tolist()
        if len(top) > 0:
            W_q5_timed.loc[hold_idx, top] = 1.0 / len(top)
    W_q5_timed = W_q5_timed.mul(timing, axis=0).fillna(0)
    g_q5t, n_q5t, t_q5t = perf_from_weights(W_q5_timed, rets_uni, market)
    p_q5t = perf_with_alpha(n_q5t, market)
    print(f"  Phase 20 Q5 timed (long-only timed):  alpha={p_q5t['ann_alpha']:+.4f}, t={p_q5t['t_alpha']:+.3f}, Sh={p_q5t['sharpe']:+.3f}, ann_turn={t_q5t.mean()*252:.2f}")

    W_ew_timed = pd.DataFrame(1.0/len(universe), index=daily_ret.index, columns=universe).mul(timing, axis=0).fillna(0)
    g_ewt, n_ewt, _ = perf_from_weights(W_ew_timed, rets_uni, market)
    p_ewt = perf_with_alpha(n_ewt, market)
    print(f"  Phase 13 EW timed (long-only timed):  alpha={p_ewt['ann_alpha']:+.4f}, t={p_ewt['t_alpha']:+.3f}, Sh={p_ewt['sharpe']:+.3f}")

    # === Visualization 1: cumulative wealth curves ===
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    colors = {"A_beta_x_EMA": "#1b5e20", "B_beta_x_sign": "#0277bd",
              "C_beta_x_zlead": "#6a1b9a", "D_beta_only": "#9e9e9e",
              "E_beta_x_lead_lag": "#ef6c00"}
    for label, res in strategy_results.items():
        cum = (1 + res["net"].fillna(0)).cumprod()
        axes[0].plot(cum, color=colors[label], lw=1.5, alpha=0.9,
                     label=f"{label} [{cum.iloc[-1]:.2f}x]")
    cum_q5t = (1 + n_q5t.fillna(0)).cumprod()
    cum_ewt = (1 + n_ewt.fillna(0)).cumprod()
    axes[0].plot(cum_q5t, color="black", lw=1.0, ls="--", alpha=0.7,
                 label=f"Phase 20 Q5 timed [{cum_q5t.iloc[-1]:.2f}x]")
    axes[0].plot(cum_ewt, color="gray", lw=1.0, ls=":", alpha=0.7,
                 label=f"Phase 13 EW timed [{cum_ewt.iloc[-1]:.2f}x]")
    axes[0].set_yscale("log")
    axes[0].set_title("(A) CS factor LS 累積資本 (扣 50bp)")
    axes[0].set_xlabel("date"); axes[0].set_ylabel("累積資本 (log)")
    axes[0].legend(loc="upper left", fontsize=9)
    axes[0].grid(alpha=0.3)

    # IC time-series
    for label, ic_df in ic_results.items():
        if len(ic_df) == 0:
            continue
        # rolling 12m mean
        ic_roll = ic_df.set_index("date")["ic"].rolling(12).mean()
        axes[1].plot(ic_roll, color=colors[label], lw=1.4, alpha=0.85,
                     label=f"{label} (mean IC={ic_df['ic'].mean():+.3f})")
    axes[1].axhline(0, color="black", lw=0.5)
    axes[1].set_title("(B) IC time-series (12m rolling mean)")
    axes[1].set_xlabel("date"); axes[1].set_ylabel("Spearman ρ(score, next-period r)")
    axes[1].legend(loc="lower left", fontsize=9)
    axes[1].grid(alpha=0.3)
    plt.suptitle("Phase 21: CS factor — β × predicted lead 變體比較", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig_phase21_factor_curves.png"), dpi=130)
    plt.close()
    print(f"\n圖檔: fig_phase21_factor_curves.png")

    # === 結論 ===
    print(f"\n" + "=" * 100)
    print("結論 — CS factor 是否成功?")
    print("=" * 100)
    print(f"\n判定標準: net t_α > 2.0 且 IC > 0.02 視為「真 CS alpha」")
    summary_df = pd.DataFrame(summary_rows)
    for _, r in summary_df.iterrows():
        flag = " ✓" if (r["net_t_alpha"] > 2.0 and r["ic_mean"] > 0.02) else \
               (" ↑" if r["net_t_alpha"] > 0 else " ↓")
        print(f"  {r['variant']:<22s} net_α={r['net_alpha']:>+.4f}, t={r['net_t_alpha']:>+.3f}, "
              f"Sh={r['net_sharpe']:>+.3f}, IC={r['ic_mean']:>+.4f}, IR={r['ir']:>+.3f}{flag}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
