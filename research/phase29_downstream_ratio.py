"""
Phase 29: D000 下游 6 個 sub_code 個股 walk-forward ratio rank

Phase 26 成功經驗 (D400/D500 中游 60 檔)
- ratio N=10 always-on α=10.41%, t=2.58, Sharpe 1.33 (Δα +4.94pp vs Full EW)
- ratio (β_lag/β_now) > t-stat > β_lag

Phase 29 擴展
- Universe: D000 下游 6 sub_code 合併 (D600/D700/D800/D900/DA00/DB00, 142 檔 unique)
  - D600 生產製程及檢測設備 52
  - D700 基板 10
  - D800 導線架 9
  - D900 IC封裝測試 33
  - DA00 IC模組 22
  - DB00 IC通路 36
- 同 Phase 26 設計: 5y rolling, ratio = β_lag/β_now, top-N (5/10/15/20)
- 額外診斷: top picks 的 sub_code 分布 + 各 sub_code 內最強個股

Phase 9 已知背景
- 6 sub_code 全為 receiver: D100 → 各下游 t=3.4-4.0, DC00 → 各下游 t=2.9-3.9
- 但 D900 IC封測 反向預測中上游 (t=2.5-3.1) — 異常 (Phase 11b I500 同類)

輸出
- phase29_downstream_summary.csv     : 各 N × timed × ratio summary
- phase29_subcode_breakdown.csv      : top picks 的 sub_code 分布
- phase29_factor_matrices.csv        : β_lag, β_now matrices
- fig_phase29_curves.png             : 累積資本對比 (ratio vs Full EW)
- fig_phase29_subcode_distribution.png : 哪些 sub_code 最常被 ratio 挑中
"""
from __future__ import annotations

import os
import pickle
import sys
import time
import warnings
from collections import Counter

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

LOOKBACK = 1250
NW_LAG = 5
COST_RT = 0.0050
SMOOTH_N = 10
MIN_OBS_IN_WIN = 800

DOWNSTREAM_SUBS = ["D600", "D700", "D800", "D900", "DA00", "DB00"]
N_LIST = [5, 10, 15, 20]


def load_daily_ret():
    with open(os.path.join(DB, "etl#adj_close.pickle"), "rb") as f:
        adj = pickle.load(f).set_index("date")
    adj.index = pd.to_datetime(adj.index)
    adj.columns = adj.columns.astype(str)
    keep = [c for c in adj.columns if c.isdigit() and len(c) == 4 and not c.startswith("00")]
    return adj[keep].pct_change(fill_method=None).iloc[1:]


def screen_joint(r_i_win, lead_win, lead_lag_win, market_lag_win):
    y_lag = r_i_win.shift(1)
    df = pd.concat({"y": r_i_win, "x_now": lead_win, "x_lag": lead_lag_win,
                     "y_lag": y_lag, "m_lag": market_lag_win}, axis=1).dropna()
    if len(df) < MIN_OBS_IN_WIN:
        return None, None, None
    try:
        res = sm.OLS(df["y"],
                     sm.add_constant(df[["x_now", "x_lag", "y_lag", "m_lag"]])).fit(
            cov_type="HAC", cov_kwds={"maxlags": NW_LAG}
        )
        return float(res.tvalues["x_lag"]), float(res.params["x_lag"]), float(res.params["x_now"])
    except Exception:
        return None, None, None


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
    print("Phase 29: D000 下游 6 sub_code (D600/D700/D800/D900/DA00/DB00) ratio rank")
    print("=" * 110)

    chain = pd.read_csv(os.path.join(ROOT, "industry_chain.csv"), dtype=str)
    daily_ret = load_daily_ret()
    market = daily_ret.mean(axis=1)
    market_lag = market.shift(1)

    # === Lead signal: 0.5·D100 + 0.5·DC00 (純上游) ===
    d100 = chain[(chain["industry_code"] == "D000") & (chain["sub_code"] == "D100")]["stock_id"].unique().tolist()
    dc00 = chain[(chain["industry_code"] == "D000") & (chain["sub_code"] == "DC00")]["stock_id"].unique().tolist()
    d100_in = [s for s in d100 if s in daily_ret.columns]
    dc00_in = [s for s in dc00 if s in daily_ret.columns]
    lead = 0.5 * daily_ret[d100_in].mean(axis=1) + 0.5 * daily_ret[dc00_in].mean(axis=1)
    lead_lag = lead.shift(1)
    print(f"\nLead: 0.5·D100({len(d100_in)} 檔) + 0.5·DC00({len(dc00_in)} 檔)")

    # === Universe: D000 下游 6 sub_code unique ===
    d_down = chain[(chain["industry_code"] == "D000") &
                    (chain["sub_code"].isin(DOWNSTREAM_SUBS))][["stock_id", "name", "sub_code"]]
    # 一檔股可能跨多 sub_code, 取第一個
    d_down_unique = d_down.drop_duplicates("stock_id", keep="first").reset_index(drop=True)
    universe = sorted([s for s in d_down_unique["stock_id"] if s in daily_ret.columns])
    rets_uni = daily_ret[universe]
    name_map = dict(zip(d_down_unique["stock_id"], d_down_unique["name"]))
    sub_map = dict(zip(d_down_unique["stock_id"], d_down_unique["sub_code"]))

    print(f"\nUniverse: {len(universe)} 檔 unique (D000 下游 6 sub_code)")
    print(f"  各 sub_code 在 universe 的分布:")
    sub_counts = Counter([sub_map[s] for s in universe])
    for sc in DOWNSTREAM_SUBS:
        sub_name = chain[(chain["sub_code"] == sc)]["sub_name"].iloc[0]
        print(f"    {sc} {sub_name}: {sub_counts.get(sc, 0)} 檔")

    first_valid = daily_ret.index[LOOKBACK]
    month_ends = pd.Series(daily_ret.index).groupby([daily_ret.index.year, daily_ret.index.month]).max()
    rebal_dates = pd.DatetimeIndex(month_ends.values)
    rebal_dates = rebal_dates[rebal_dates >= first_valid]
    rebal_list = sorted(rebal_dates)
    print(f"\nRebal dates: {len(rebal_list)} ({rebal_list[0].date()} → {rebal_list[-1].date()})")

    # === 計算 β_lag, β_now 矩陣 ===
    print(f"\n[1] 計算 ratio = β_lag/β_now matrix ({len(rebal_list)} months × {len(universe)} stocks)")
    t0 = time.time()
    t_lag_m = pd.DataFrame(np.nan, index=rebal_dates, columns=universe)
    b_lag_m = pd.DataFrame(np.nan, index=rebal_dates, columns=universe)
    b_now_m = pd.DataFrame(np.nan, index=rebal_dates, columns=universe)

    for j, d in enumerate(rebal_dates):
        win_start_idx = max(0, daily_ret.index.get_loc(d) - LOOKBACK)
        win_start = daily_ret.index[win_start_idx]
        lead_win = lead.loc[win_start:d]
        lead_lag_win = lead_lag.loc[win_start:d]
        market_lag_win = market_lag.loc[win_start:d]

        for sid in universe:
            r_i_win = rets_uni[sid].loc[win_start:d]
            t_lag, b_lag, b_now = screen_joint(r_i_win, lead_win, lead_lag_win, market_lag_win)
            if t_lag is not None:
                t_lag_m.loc[d, sid] = t_lag
                b_lag_m.loc[d, sid] = b_lag
                b_now_m.loc[d, sid] = b_now
        if (j + 1) % 24 == 0 or j == len(rebal_dates) - 1:
            print(f"  [{j+1:>3d}/{len(rebal_list)}] {d.date()}  ({time.time()-t0:.0f}s)")

    ratio_m = b_lag_m / b_now_m.where(b_now_m > 0.05)

    # 儲存
    pd.concat({"t_lag": t_lag_m, "b_lag": b_lag_m, "b_now": b_now_m, "ratio": ratio_m},
               axis=1).to_csv(os.path.join(OUT, "phase29_factor_matrices.csv"), encoding="utf-8-sig")

    # === Diagnostic 1: 最後一期 top 15 + sub_code ===
    last_d = rebal_list[-1]
    print(f"\n[Diagnostic 1] {last_d.date()} ratio top 15 + sub_code 分布:")
    s = ratio_m.loc[last_d].dropna()
    s = s[s > 0].sort_values(ascending=False).head(15)
    print(f"{'rank':>4s} {'stock':>5s} {'name':<10s} {'sub':>5s}  {'ratio':>7s}  {'β_lag':>7s}  {'β_now':>7s}")
    for rank_, (sid, score) in enumerate(s.items()):
        print(f"{rank_+1:>4d} {sid:>5s} {name_map.get(sid, ''):<10s} {sub_map.get(sid,''):>5s}  "
              f"{score:>+7.3f}  {b_lag_m.loc[last_d, sid]:>+7.3f}  {b_now_m.loc[last_d, sid]:>+7.3f}")

    # === Diagnostic 2: 全期間 ratio top-10 出現次數的 sub_code 分布 ===
    print(f"\n[Diagnostic 2] 全期間 ({len(rebal_list)} months) ratio top-10 sub_code 出現分布:")
    sub_appearances = Counter()
    sub_eligible = Counter()
    for d in rebal_list:
        scores = ratio_m.loc[d].dropna()
        scores = scores[scores > 0]
        if len(scores) == 0:
            continue
        # 統計 eligible 樣本中 sub_code 比例
        for sid in scores.index:
            sub_eligible[sub_map.get(sid, "?")] += 1
        # 統計 top 10 中 sub_code 比例
        top_10 = scores.sort_values(ascending=False).head(10).index
        for sid in top_10:
            sub_appearances[sub_map.get(sid, "?")] += 1

    print(f"  {'sub':>5s}  {'top10 出現':>11s}  {'eligible':>10s}  {'top10/eligible':>15s}  {'over-rep':>9s}")
    for sc in DOWNSTREAM_SUBS:
        sub_name = chain[(chain["sub_code"] == sc)]["sub_name"].iloc[0][:15]
        appear = sub_appearances.get(sc, 0)
        eligi = sub_eligible.get(sc, 0)
        ratio_rep = appear / eligi if eligi > 0 else 0
        # over-rep: 該 sub 在 top 中 vs 在 universe 中的比例
        n_in_uni = sub_counts.get(sc, 0)
        baseline_rep = n_in_uni / len(universe)
        over = ratio_rep / baseline_rep if baseline_rep > 0 else 0
        print(f"  {sc:>5s} {sub_name:<10s} {appear:>4d}  {eligi:>8d}  {ratio_rep:>11.3f}  "
              f"{over:>+9.2f}× {'✓ over' if over > 1.3 else ('↓ under' if over < 0.7 else '~')}")

    # === Build portfolios ===
    print(f"\n[2] 構 portfolios for N ∈ {N_LIST}")
    weight_dict = {}
    selection_log = {N: [] for N in N_LIST}

    for N in N_LIST:
        W = pd.DataFrame(0.0, index=daily_ret.index, columns=universe)
        for i, d in enumerate(rebal_list):
            scores = ratio_m.loc[d].dropna()
            scores = scores[scores > 0]
            if len(scores) == 0:
                continue
            top = scores.sort_values(ascending=False).head(N).index.tolist()
            next_rebal = rebal_list[i + 1] if i + 1 < len(rebal_list) else daily_ret.index.max() + pd.Timedelta(days=1)
            hold_idx = daily_ret.index[(daily_ret.index > d) & (daily_ret.index <= next_rebal)]
            if len(top) > 0:
                W.loc[hold_idx, top] = 1.0 / len(top)
            for r_, sid in enumerate(top):
                selection_log[N].append({"date": d.date(), "rank": r_+1, "stock_id": sid,
                                          "name": name_map.get(sid, ""), "sub": sub_map.get(sid, ""),
                                          "ratio": ratio_m.loc[d, sid]})
        weight_dict[N] = W
        n_avg = (W > 0).sum(axis=1).where(lambda x: x > 0).mean()
        print(f"  N={N}: avg active = {n_avg:.1f}")

    # 儲存 selection log for N=10
    pd.DataFrame(selection_log[10]).to_csv(os.path.join(OUT, "phase29_subcode_breakdown.csv"),
                                              index=False, encoding="utf-8-sig")

    # === Full EW baseline ===
    W_full = pd.DataFrame(0.0, index=daily_ret.index, columns=universe)
    for d in daily_ret.index[LOOKBACK:]:
        avail = rets_uni.loc[d].dropna().index.tolist()
        if avail:
            W_full.loc[d, avail] = 1.0 / len(avail)

    # Timing
    timing = (lead.ewm(span=SMOOTH_N, adjust=False).mean() > 0).astype(float).shift(1).fillna(0)

    # === Returns ===
    eval_start = rebal_list[0] + pd.Timedelta(days=1)
    print(f"\n[3] 計算 returns (eval start = {eval_start.date()})")

    g_full, n_full, at_full = perf_from_weights(W_full, rets_uni)
    W_full_t = W_full.mul(timing, axis=0).fillna(0)
    g_full_t, n_full_t, at_full_t = perf_from_weights(W_full_t, rets_uni)
    p_full = perf_with_alpha(n_full.loc[eval_start:], market.loc[eval_start:])
    p_full_t = perf_with_alpha(n_full_t.loc[eval_start:], market.loc[eval_start:])

    rows = [
        {"variant": "Full_EW", "N": "all", "timed": False, **p_full, "ann_turn": at_full},
        {"variant": "Full_EW", "N": "all", "timed": True, **p_full_t, "ann_turn": at_full_t},
    ]

    daily_returns_dict = {"Full_EW_net": n_full, "Full_EW_timed_net": n_full_t}

    for N in N_LIST:
        W = weight_dict[N]
        g, n, at = perf_from_weights(W, rets_uni)
        W_t = W.mul(timing, axis=0).fillna(0)
        g_t, n_t, at_t = perf_from_weights(W_t, rets_uni)
        daily_returns_dict[f"ratio{N}_net"] = n
        daily_returns_dict[f"ratio{N}_timed_net"] = n_t
        p = perf_with_alpha(n.loc[eval_start:], market.loc[eval_start:])
        p_t = perf_with_alpha(n_t.loc[eval_start:], market.loc[eval_start:])
        rows.append({"variant": "ratio", "N": N, "timed": False, **p, "ann_turn": at})
        rows.append({"variant": "ratio", "N": N, "timed": True, **p_t, "ann_turn": at_t})

    res_df = pd.DataFrame(rows)
    res_df.to_csv(os.path.join(OUT, "phase29_downstream_summary.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(daily_returns_dict).to_csv(os.path.join(OUT, "phase29_strategy_returns.csv"),
                                              encoding="utf-8-sig")

    # === Print ===
    print(f"\n" + "=" * 110)
    print(f"策略表現 (扣 50bp RT, eval start = {eval_start.date()})")
    print("=" * 110)
    base_a = res_df[(res_df["variant"] == "Full_EW") & (~res_df["timed"])].iloc[0]
    base_b = res_df[(res_df["variant"] == "Full_EW") & (res_df["timed"])].iloc[0]
    print(f"\nFull EW (D000 下游 {len(universe)} 檔):")
    print(f"  always-on: α={base_a['ann_alpha']:+.4f}, t={base_a['t_alpha']:+.3f}, "
          f"Sh={base_a['sharpe']:+.3f}, ann_ret={base_a['ann_ret']:.4f}")
    print(f"  timed:     α={base_b['ann_alpha']:+.4f}, t={base_b['t_alpha']:+.3f}, "
          f"Sh={base_b['sharpe']:+.3f}, ann_ret={base_b['ann_ret']:.4f}")

    print(f"\n[Always-on] (vs Full EW α={base_a['ann_alpha']:+.4f}, t={base_a['t_alpha']:+.3f}, Sh={base_a['sharpe']:+.3f})")
    print(f"{'N':>3s}  {'α':>9s}  {'t_α':>8s}  {'Sh':>7s}  {'turn':>6s}  Δα/Δt vs Full EW")
    for _, r in res_df[(res_df["variant"] == "ratio") & (~res_df["timed"])].iterrows():
        d_a = (r["ann_alpha"] - base_a["ann_alpha"]) * 100
        d_t = r["t_alpha"] - base_a["t_alpha"]
        d_sh = r["sharpe"] - base_a["sharpe"]
        flag = "✓✓" if d_a > 3 and d_t > 0.5 else ("✓" if d_a > 1 and d_t > 0.2 else ("↑" if d_a > 0 else "↓"))
        print(f"{r['N']:>3d}  {r['ann_alpha']:>+9.4f}  {r['t_alpha']:>+8.3f}  "
              f"{r['sharpe']:>+7.3f}  {r['ann_turn']:>6.2f}  {d_a:+.2f}pp / {d_t:+.2f} (ΔSh={d_sh:+.3f}) {flag}")

    print(f"\n[Timed] (vs Full timed α={base_b['ann_alpha']:+.4f}, t={base_b['t_alpha']:+.3f}, Sh={base_b['sharpe']:+.3f})")
    print(f"{'N':>3s}  {'α':>9s}  {'t_α':>8s}  {'Sh':>7s}  {'turn':>6s}  Δα/Δt vs Full timed")
    for _, r in res_df[(res_df["variant"] == "ratio") & (res_df["timed"])].iterrows():
        d_a = (r["ann_alpha"] - base_b["ann_alpha"]) * 100
        d_t = r["t_alpha"] - base_b["t_alpha"]
        d_sh = r["sharpe"] - base_b["sharpe"]
        flag = "✓✓" if d_a > 3 and d_t > 0.5 else ("✓" if d_a > 1 and d_t > 0.2 else ("↑" if d_a > 0 else "↓"))
        print(f"{r['N']:>3d}  {r['ann_alpha']:>+9.4f}  {r['t_alpha']:>+8.3f}  "
              f"{r['sharpe']:>+7.3f}  {r['ann_turn']:>6.2f}  {d_a:+.2f}pp / {d_t:+.2f} (ΔSh={d_sh:+.3f}) {flag}")

    # === 對照 Phase 26 中游 ===
    print(f"\n" + "=" * 110)
    print("對照 Phase 26 中游 (D400+D500, 60 檔)")
    print("=" * 110)
    print(f"  Phase 26 ratio N=10 always-on: α=10.41%, t=2.58, Sh=1.33, Δα=+4.94pp ✓✓")
    p29_n10_a = res_df[(res_df["variant"] == "ratio") & (res_df["N"] == 10) & (~res_df["timed"])].iloc[0]
    print(f"  Phase 29 ratio N=10 always-on: α={p29_n10_a['ann_alpha']*100:.2f}%, "
          f"t={p29_n10_a['t_alpha']:.2f}, Sh={p29_n10_a['sharpe']:.2f}, "
          f"Δα={(p29_n10_a['ann_alpha']-base_a['ann_alpha'])*100:+.2f}pp")

    # === Plot 1: cum curves ===
    fig, ax = plt.subplots(figsize=(11, 5.5))
    cmap = plt.cm.viridis(np.linspace(0.15, 0.85, len(N_LIST)))
    cum_full = (1 + n_full.loc[eval_start:].fillna(0)).cumprod()
    ax.plot(cum_full, color="black", lw=2.0, alpha=0.7, ls="--",
            label=f"Full EW always-on [{cum_full.iloc[-1]:.2f}x] α={base_a['ann_alpha']:.3f}")
    cum_full_t = (1 + n_full_t.loc[eval_start:].fillna(0)).cumprod()
    ax.plot(cum_full_t, color="gray", lw=2.0, alpha=0.7, ls="--",
            label=f"Full EW timed [{cum_full_t.iloc[-1]:.2f}x] α={base_b['ann_alpha']:.3f}")
    for i, N in enumerate(N_LIST):
        n_eval = daily_returns_dict[f"ratio{N}_net"].loc[eval_start:]
        cum = (1 + n_eval.fillna(0)).cumprod()
        r = res_df[(res_df["variant"] == "ratio") & (res_df["N"] == N) & (~res_df["timed"])].iloc[0]
        ax.plot(cum, color=cmap[i], lw=1.5, alpha=0.92,
                label=f"ratio N={N} always-on [{cum.iloc[-1]:.2f}x] α={r['ann_alpha']:.3f}")
    ax.set_yscale("log")
    ax.set_title(f"Phase 29: D000 下游 ratio rank (5y WF, 扣 50bp)")
    ax.set_xlabel("date"); ax.set_ylabel("累積資本 (log)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig_phase29_curves.png"), dpi=130)
    plt.close()
    print(f"\n圖檔: fig_phase29_curves.png")

    # === Plot 2: sub_code distribution ===
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    sub_names_short = {
        "D600": "D600 製程設備", "D700": "D700 基板", "D800": "D800 導線架",
        "D900": "D900 IC封測", "DA00": "DA00 IC模組", "DB00": "DB00 IC通路",
    }
    cats = DOWNSTREAM_SUBS
    over_rates = []
    appear_rates = []
    for sc in cats:
        appear = sub_appearances.get(sc, 0)
        eligi = sub_eligible.get(sc, 0)
        n_in_uni = sub_counts.get(sc, 0)
        baseline_rep = n_in_uni / len(universe)
        ratio_rep = appear / eligi if eligi > 0 else 0
        over = ratio_rep / baseline_rep if baseline_rep > 0 else 0
        over_rates.append(over)
        appear_rates.append(appear)

    bars = axes[0].bar(cats, appear_rates, color=["#0277bd", "#2e7d32", "#f57c00", "#c62828", "#6a1b9a", "#00838f"])
    axes[0].set_xticklabels([sub_names_short[c] for c in cats], rotation=20, ha="right", fontsize=9)
    axes[0].set_title("(A) 全期間 top-10 出現次數")
    axes[0].set_ylabel("出現次數 (across rebalances)")
    axes[0].grid(axis="y", alpha=0.3)
    for b, v in zip(bars, appear_rates):
        axes[0].text(b.get_x() + b.get_width()/2, v + 5, f"{v}", ha="center", fontsize=9)

    bars = axes[1].bar(cats, over_rates,
                        color=["#0277bd" if v >= 1 else "#c62828" for v in over_rates])
    axes[1].axhline(1.0, color="black", lw=0.8, ls="--", label="平均代表性 (1.0×)")
    axes[1].set_xticklabels([sub_names_short[c] for c in cats], rotation=20, ha="right", fontsize=9)
    axes[1].set_title("(B) Top-10 中該 sub 比例 / 該 sub 在 universe 比例")
    axes[1].set_ylabel("over-representation (×)")
    axes[1].legend(fontsize=9)
    axes[1].grid(axis="y", alpha=0.3)
    for b, v in zip(bars, over_rates):
        axes[1].text(b.get_x() + b.get_width()/2, v + 0.05, f"{v:.2f}×", ha="center", fontsize=9)

    plt.suptitle("Phase 29: D000 下游 ratio rank top-10 sub_code 分布", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig_phase29_subcode_distribution.png"), dpi=130)
    plt.close()
    print(f"圖檔: fig_phase29_subcode_distribution.png")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
