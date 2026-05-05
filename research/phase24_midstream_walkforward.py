"""
Phase 24: D400/D500 中游個股 walk-forward β-screening 組合
（Phase 23 是 in-sample look-ahead, 此版本嚴格 no-peek）

設計
- Universe: D400 + D500 (~60 檔, 動態過濾 FinLab 有歷史的)
- 每月底 rebalance:
  使用過去 5y (1250d) 跑 r_i = α + β·lead(t-1) + γ·r_i(t-1) + δ·market(t-1)
  HAC SE (lag=5), 取 t(β_lag) > 1.96 入選
- 持倉: EW within selected, 持有到下個月底
- 第一次選股日: 2007-01 + 5y = 2012 年初
- 同步比較:
  A. D400+D500 EW always-on (無篩選 baseline)
  B. D400+D500 EW timed (無篩選 + Phase 13 timing)
  C. WF-screen EW always-on (本研究核心)
  D. WF-screen EW timed (本研究核心 + timing)
  E. Phase 23 in-sample top-15 (look-ahead 對照, 應比 WF 好 = 證明 in-sample bias 確實存在)

No-peek 保證
- 5y window 嚴格以 (t-1250d, t-1d] 為界
- timing signal .shift(1)
- threshold ex-ante 設 1.96, 不掃
- universe membership 來自 chain (現況 snapshot, 實務上需要 point-in-time 但本研究先接受)

輸出
- phase24_selection_timeline.csv : 每月入選名單 + 數量
- phase24_strategy_returns.csv   : 各 strategy 日報酬
- phase24_summary.csv            : 各 strategy alpha/Sharpe/turnover
- fig_phase24_curves.png         : 累積資本對比
- fig_phase24_selection_count.png: 入選數量時序
"""
from __future__ import annotations

import os
import pickle
import sys
import time
import warnings

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

warnings.filterwarnings("ignore")

ROOT = r"C:\Users\engli\OneDrive\桌面\產業因子生成"
DB = r"C:\Users\engli\finlab_db"
OUT = os.path.join(ROOT, "research", "output")

LOOKBACK = 1250    # 5y trading days
T_THRESHOLD = 1.96  # ex-ante
NW_LAG = 5
COST_RT = 0.0050   # 50bp round-trip
SMOOTH_N = 10      # Phase 13 timing EMA span
MIN_OBS_IN_WIN = 800  # 必要觀測數 (5y 裡至少 ~3.2y 有資料)


def load_daily_ret():
    with open(os.path.join(DB, "etl#adj_close.pickle"), "rb") as f:
        adj = pickle.load(f).set_index("date")
    adj.index = pd.to_datetime(adj.index)
    adj.columns = adj.columns.astype(str)
    keep = [c for c in adj.columns if c.isdigit() and len(c) == 4 and not c.startswith("00")]
    return adj[keep].pct_change(fill_method=None).iloc[1:]


def screen_one(r_i_win, lead_lag_win, market_lag_win):
    """Run regression on a 5y window, return t(β_lag)."""
    y_lag = r_i_win.shift(1)
    df = pd.concat({"y": r_i_win, "x": lead_lag_win, "y_lag": y_lag, "m_lag": market_lag_win}, axis=1).dropna()
    if len(df) < MIN_OBS_IN_WIN:
        return None
    try:
        res = sm.OLS(df["y"], sm.add_constant(df[["x", "y_lag", "m_lag"]])).fit(
            cov_type="HAC", cov_kwds={"maxlags": NW_LAG}
        )
        return float(res.tvalues["x"])
    except Exception:
        return None


def perf_with_alpha(daily_ret, market, ann=252):
    df = pd.concat([daily_ret.rename("s"), market.rename("b")], axis=1).dropna()
    if len(df) < 60:
        return {}
    res = sm.OLS(df["s"], sm.add_constant(df["b"])).fit(cov_type="HAC", cov_kwds={"maxlags": 60})
    s = df["s"]; cum = (1 + s).cumprod()
    return {
        "n": len(s),
        "ann_ret": s.mean() * ann,
        "ann_vol": s.std() * np.sqrt(ann),
        "sharpe": s.mean() / s.std() * np.sqrt(ann) if s.std() > 0 else np.nan,
        "max_dd": (cum / cum.cummax() - 1).min(),
        "ann_alpha": res.params["const"] * ann,
        "t_alpha": res.tvalues["const"],
        "beta": res.params["b"],
    }


def perf_from_weights(W, daily_ret_uni, cost_rt=COST_RT):
    gross = (W * daily_ret_uni).sum(axis=1)
    turn = W.diff().abs().sum(axis=1)
    turn.iloc[0] = W.iloc[0].abs().sum()
    cost = turn * (cost_rt / 2.0)
    net = gross - cost
    ann_turn = turn.mean() * 252
    return gross, net, ann_turn


def main():
    print("=" * 110)
    print("Phase 24: D400/D500 中游個股 walk-forward β-screening (no peek)")
    print("=" * 110)
    print(f"  Lookback: {LOOKBACK}d (~5y)  |  threshold t > {T_THRESHOLD}  |  cost {COST_RT*100:.0f}bp RT")

    chain = pd.read_csv(os.path.join(ROOT, "industry_chain.csv"), dtype=str)
    daily_ret = load_daily_ret()
    market = daily_ret.mean(axis=1)
    market_lag = market.shift(1)
    print(f"\nDaily return matrix: {daily_ret.shape} ({daily_ret.index.min().date()} → {daily_ret.index.max().date()})")

    # === Lead signal ===
    d100 = chain[(chain["industry_code"] == "D000") & (chain["sub_code"] == "D100")]["stock_id"].unique().tolist()
    dc00 = chain[(chain["industry_code"] == "D000") & (chain["sub_code"] == "DC00")]["stock_id"].unique().tolist()
    d100_in = [s for s in d100 if s in daily_ret.columns]
    dc00_in = [s for s in dc00 if s in daily_ret.columns]
    lead = 0.5 * daily_ret[d100_in].mean(axis=1) + 0.5 * daily_ret[dc00_in].mean(axis=1)
    lead_lag = lead.shift(1)
    print(f"Lead: 0.5·D100({len(d100_in)} 檔) + 0.5·DC00({len(dc00_in)} 檔)")

    # === Universe (dedupe stock_id, 取 D400 ∪ D500) ===
    d_mid = chain[(chain["industry_code"] == "D000") &
                   (chain["sub_code"].isin(["D400", "D500"]))][["stock_id", "name", "sub_code"]]
    # 跨 D400/D500 的: 取 D400 標籤 (反正 EW 不影響)
    d_mid = d_mid.drop_duplicates("stock_id", keep="first").reset_index(drop=True)
    universe = sorted([s for s in d_mid["stock_id"] if s in daily_ret.columns])
    print(f"\nUniverse: {len(universe)} 檔 unique (D400 ∪ D500, FinLab 有資料)")

    rets_uni = daily_ret[universe]

    # === 月底 rebalance dates (warm-up 5y) ===
    first_valid = daily_ret.index[LOOKBACK]
    month_ends = pd.Series(daily_ret.index).groupby([daily_ret.index.year, daily_ret.index.month]).max()
    month_ends = pd.DatetimeIndex(month_ends.values)
    rebal_dates = month_ends[month_ends >= first_valid]
    print(f"\n月底 rebalance dates: {len(rebal_dates)} (first = {rebal_dates[0].date()}, last = {rebal_dates[-1].date()})")

    # === Per-month walk-forward screening ===
    print(f"\n[1] Walk-forward screening: {len(rebal_dates)} months × {len(universe)} stocks regressions")
    t0 = time.time()
    selection_rows = []          # date, n_selected, mean_t, list
    selected_dict = {}            # date -> list of stocks

    for j, d in enumerate(rebal_dates):
        win_end = d  # inclusive
        win_start_idx = max(0, daily_ret.index.get_loc(d) - LOOKBACK)
        win_start = daily_ret.index[win_start_idx]

        lead_lag_win = lead_lag.loc[win_start:win_end]
        market_lag_win = market_lag.loc[win_start:win_end]

        sels = []
        ts = []
        for sid in universe:
            r_i_win = rets_uni[sid].loc[win_start:win_end]
            t_val = screen_one(r_i_win, lead_lag_win, market_lag_win)
            if t_val is None:
                continue
            ts.append((sid, t_val))
            if t_val > T_THRESHOLD:
                sels.append(sid)

        selected_dict[d] = sels
        mean_t = np.nan if not ts else np.mean([t for _, t in ts])
        max_t = np.nan if not ts else max(t for _, t in ts)
        selection_rows.append({
            "date": d.date(),
            "n_eligible": len(ts),
            "n_selected": len(sels),
            "mean_t": mean_t,
            "max_t": max_t,
            "selected_ids": ";".join(sels[:30]),  # 避免欄位太大
        })
        if (j + 1) % 24 == 0 or j == len(rebal_dates) - 1:
            elapsed = time.time() - t0
            print(f"  [{j+1:>3d}/{len(rebal_dates)}] {d.date()}  selected={len(sels):>2d}/{len(ts):>2d}  "
                  f"mean_t={mean_t:>+.2f}  max_t={max_t:>+.2f}  ({elapsed:.0f}s)")

    sel_df = pd.DataFrame(selection_rows)
    sel_df.to_csv(os.path.join(OUT, "phase24_selection_timeline.csv"), index=False, encoding="utf-8-sig")
    print(f"\n  入選統計: median n={sel_df['n_selected'].median():.0f}, "
          f"min={sel_df['n_selected'].min()}, max={sel_df['n_selected'].max()}")

    # === Build daily weight matrices ===
    print(f"\n[2] Build weight matrices")

    # WF-screened EW
    W_wf = pd.DataFrame(0.0, index=daily_ret.index, columns=universe)
    rebal_list = sorted(selected_dict.keys())
    for i, d_rebal in enumerate(rebal_list):
        next_rebal = rebal_list[i + 1] if i + 1 < len(rebal_list) else daily_ret.index.max() + pd.Timedelta(days=1)
        hold_idx = daily_ret.index[(daily_ret.index > d_rebal) & (daily_ret.index <= next_rebal)]
        sels = selected_dict[d_rebal]
        if len(sels) > 0:
            W_wf.loc[hold_idx, sels] = 1.0 / len(sels)

    # Universe EW always-on (持有所有有資料個股, 等權)
    W_full_ew = pd.DataFrame(0.0, index=daily_ret.index, columns=universe)
    for d in daily_ret.index:
        avail = rets_uni.loc[d].dropna().index.tolist()
        if avail:
            W_full_ew.loc[d, avail] = 1.0 / len(avail)

    # Phase 23 in-sample top-15 (look-ahead bias)
    p23 = pd.read_csv(os.path.join(OUT, "phase23_top_picks.csv"), dtype={"stock_id": str})
    p23_ids = [s for s in p23["stock_id"].unique() if s in universe][:15]
    print(f"  Phase 23 top picks (look-ahead): {len(p23_ids)} 檔 — {p23_ids}")
    W_p23 = pd.DataFrame(0.0, index=daily_ret.index, columns=universe)
    W_p23.loc[:, p23_ids] = 1.0 / len(p23_ids)
    # mask 無資料的日期
    W_p23 = W_p23.where(rets_uni.notna(), 0.0)
    # re-normalize daily
    row_sum = W_p23.sum(axis=1)
    W_p23 = W_p23.div(row_sum.replace(0, np.nan), axis=0).fillna(0)

    # Timing signal (Phase 13)
    timing = (lead.ewm(span=SMOOTH_N, adjust=False).mean() > 0).astype(float).shift(1).fillna(0)

    # === Compute returns ===
    print(f"\n[3] 計算 daily returns + cost")
    g_wf, n_wf, at_wf = perf_from_weights(W_wf, rets_uni)
    g_full, n_full, at_full = perf_from_weights(W_full_ew, rets_uni)
    g_p23, n_p23, at_p23 = perf_from_weights(W_p23, rets_uni)

    # Timed variants
    W_wf_timed = W_wf.mul(timing, axis=0).fillna(0)
    g_wf_t, n_wf_t, at_wf_t = perf_from_weights(W_wf_timed, rets_uni)

    W_full_timed = W_full_ew.mul(timing, axis=0).fillna(0)
    g_full_t, n_full_t, at_full_t = perf_from_weights(W_full_timed, rets_uni)

    W_p23_timed = W_p23.mul(timing, axis=0).fillna(0)
    g_p23_t, n_p23_t, at_p23_t = perf_from_weights(W_p23_timed, rets_uni)

    print(f"  ann turnover:")
    print(f"    A. Full EW always-on:  {at_full:.2f}")
    print(f"    B. Full EW timed:      {at_full_t:.2f}")
    print(f"    C. WF screen always:   {at_wf:.2f}")
    print(f"    D. WF screen timed:    {at_wf_t:.2f}")
    print(f"    E. P23 top15 always:   {at_p23:.2f} (look-ahead)")
    print(f"    F. P23 top15 timed:    {at_p23_t:.2f}")

    # 限制到 first rebalance 之後 (warm-up 結束)
    first_rebal = rebal_list[0]
    eval_start = first_rebal + pd.Timedelta(days=1)

    strategies = {
        "A. Full EW always-on (no select, no time)":  n_full,
        "B. Full EW timed (no select)":                n_full_t,
        "C. WF-screen always-on":                       n_wf,
        "D. WF-screen timed":                           n_wf_t,
        "E. P23 top15 always-on (LOOK-AHEAD)":         n_p23,
        "F. P23 top15 timed (LOOK-AHEAD)":             n_p23_t,
    }
    strategies_gross = {
        "A_gross": g_full, "B_gross": g_full_t,
        "C_gross": g_wf,   "D_gross": g_wf_t,
        "E_gross": g_p23,  "F_gross": g_p23_t,
    }

    # === Performance summary ===
    print(f"\n" + "=" * 110)
    print(f"策略表現 (eval start = {eval_start.date()}, 扣 {COST_RT*100:.0f}bp from proper turnover)")
    print("=" * 110)
    print(f"\n{'Strategy':<46s} {'ann_ret':>9s} {'vol':>7s} {'Sharpe':>7s} "
          f"{'alpha':>9s} {'t_α':>7s} {'max_dd':>9s} {'ann_turn':>9s}")
    summary = []
    turn_map = {"A": at_full, "B": at_full_t, "C": at_wf, "D": at_wf_t, "E": at_p23, "F": at_p23_t}
    for label, ret in strategies.items():
        ret_eval = ret.loc[eval_start:]
        m_eval = market.loc[eval_start:]
        p = perf_with_alpha(ret_eval, m_eval)
        if not p:
            continue
        key = label[0]
        print(f"{label:<46s} {p['ann_ret']:>+9.4f} {p['ann_vol']:>7.4f} "
              f"{p['sharpe']:>+7.3f} {p['ann_alpha']:>+9.4f} {p['t_alpha']:>+7.3f} "
              f"{p['max_dd']:>+9.4f} {turn_map.get(key, 0):>9.2f}")
        summary.append({"strategy": label, **p, "ann_turnover": turn_map.get(key, 0)})

    # Gross sanity
    print(f"\nGross (扣前) sanity:")
    for label, ret in strategies_gross.items():
        ret_eval = ret.loc[eval_start:]
        m_eval = market.loc[eval_start:]
        p = perf_with_alpha(ret_eval, m_eval)
        if p:
            print(f"  {label:<10s} ann_ret={p['ann_ret']:+.4f} alpha={p['ann_alpha']:+.4f} t_α={p['t_alpha']:+.3f}")

    pd.DataFrame(summary).to_csv(os.path.join(OUT, "phase24_summary.csv"), index=False, encoding="utf-8-sig")

    # === Plot ===
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    cmap = {
        "A. Full EW always-on (no select, no time)":  "#90a4ae",
        "B. Full EW timed (no select)":                "#0277bd",
        "C. WF-screen always-on":                       "#388e3c",
        "D. WF-screen timed":                           "#1b5e20",
        "E. P23 top15 always-on (LOOK-AHEAD)":         "#fb8c00",
        "F. P23 top15 timed (LOOK-AHEAD)":             "#bf360c",
    }
    cum_mkt = (1 + market.loc[eval_start:].fillna(0)).cumprod()
    for label, ret in strategies.items():
        ret_eval = ret.loc[eval_start:]
        cum = (1 + ret_eval.fillna(0)).cumprod()
        ls = "--" if "LOOK-AHEAD" in label else "-"
        axes[0].plot(cum, color=cmap[label], lw=1.4, alpha=0.92, ls=ls,
                     label=f"{label[:28]} [{cum.iloc[-1]:.1f}x]")
    axes[0].plot(cum_mkt, color="black", lw=0.8, alpha=0.5, label=f"Market [{cum_mkt.iloc[-1]:.1f}x]")
    axes[0].set_yscale("log")
    axes[0].set_title(f"(A) WF-screen vs benchmarks (扣 {COST_RT*100:.0f}bp)")
    axes[0].set_ylabel("累積資本 (log)")
    axes[0].legend(loc="upper left", fontsize=8.5)
    axes[0].grid(alpha=0.3)

    # Selection count timeline
    axes[1].plot(pd.to_datetime(sel_df["date"]), sel_df["n_selected"], color="#1b5e20", lw=1.6,
                 label="入選個股數")
    axes[1].plot(pd.to_datetime(sel_df["date"]), sel_df["n_eligible"], color="#90a4ae", lw=0.9, ls=":",
                 label="可選個股數 (有 5y 歷史)")
    axes[1].axhline(0, color="black", lw=0.5)
    axes[1].set_title("(B) 每月入選個股數時序 (t > 1.96 過去 5y)")
    axes[1].set_ylabel("家數")
    axes[1].legend(loc="upper left", fontsize=10)
    axes[1].grid(alpha=0.3)
    plt.suptitle("Phase 24: D400+D500 中游 walk-forward β-screening", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig_phase24_curves.png"), dpi=130)
    plt.close()
    print(f"\n圖檔: fig_phase24_curves.png")

    # Save daily returns
    daily_out = pd.DataFrame({
        "WF_screen_gross": g_wf, "WF_screen_net": n_wf,
        "WF_screen_timed_gross": g_wf_t, "WF_screen_timed_net": n_wf_t,
        "Full_EW_net": n_full, "Full_EW_timed_net": n_full_t,
        "P23_top15_net": n_p23, "P23_top15_timed_net": n_p23_t,
        "lead": lead, "timing": timing,
    })
    daily_out.to_csv(os.path.join(OUT, "phase24_strategy_returns.csv"), encoding="utf-8-sig")

    # === Δ vs benchmark ===
    print(f"\n" + "=" * 110)
    print("Δ alpha vs full-universe baseline (driven by selection alone)")
    print("=" * 110)
    sm_df = pd.DataFrame(summary)
    base = sm_df[sm_df["strategy"] == "A. Full EW always-on (no select, no time)"].iloc[0]
    base_t = sm_df[sm_df["strategy"] == "B. Full EW timed (no select)"].iloc[0]
    for _, r in sm_df.iterrows():
        if "always-on" in r["strategy"]:
            comp = base
        else:
            comp = base_t
        d_alpha = (r["ann_alpha"] - comp["ann_alpha"]) * 100
        d_t = r["t_alpha"] - comp["t_alpha"]
        flag = " ✓" if d_alpha > 1.0 and d_t > 0.3 else (" ↑" if d_alpha > 0 else " ↓")
        print(f"  {r['strategy']:<46s} α={r['ann_alpha']:+.4f}  t={r['t_alpha']:+.3f}  "
              f"Δα={d_alpha:+.2f}pp  Δt={d_t:+.2f}{flag}")

    print(f"\n關鍵問題:")
    print("  Q1. WF-screen always-on (C) 是否擊敗 Full EW always-on (A)?")
    print("       若是 → selection 有 cross-sectional 真 alpha (Phase 21 在 D400+D500 sub-universe 內救援成功)")
    print("  Q2. P23 top15 (look-ahead, E) 是否大幅擊敗 WF-screen (C)?")
    print("       若是 → 確認 Phase 23 是 in-sample bias, WF 是真實 OOS estimate")
    print("  Q3. WF-screen timed (D) 是否擊敗 Full timed (B)?")
    print("       若是 → 選股 + timing 雙層整合有效")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
