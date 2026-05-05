"""
Phase 32: D000 下游 純 3M-avg 成交金額 top-N (無 ratio)

動機 (Phase 31 對照組已初測, 此處全面 sweep)
- Phase 31 對照組: Pure TV top-10/15/20 timed Sharpe 0.69-0.82 (不及 Full EW timed 1.62)
- 但只試了 3 個 N, 且未顯示 always-on / 個股組成
- 完整檢驗「純買流動性大的個股」是否有任何 alpha 來源

設計
- Universe: D000 下游 126 檔 (同 Phase 29-31)
- N ∈ {5, 10, 15, 20, 25, 30} (更細的 sweep)
- always-on + timed
- 對照: Full EW (no select), Phase 29 ratio N=20 (有選股)

Diagnostic
- 最後一期 top-N 個股名單 + sub_code 分布
- 全期間 top-N 出現頻率最高的個股 (持股穩定度)

輸出
- phase32_pure_tv_summary.csv
- phase32_strategy_returns.csv
- phase32_top_holdings.csv : 最常持有的個股
- fig_phase32_N_sweep.png
- fig_phase32_curves.png
"""
from __future__ import annotations

import os
import pickle
import sys
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
COST_RT = 0.0050
SMOOTH_N = 10
TV_ROLL = 60

DOWNSTREAM_SUBS = ["D600", "D700", "D800", "D900", "DA00", "DB00"]
N_LIST = [5, 10, 15, 20, 25, 30]


def load_daily_ret():
    with open(os.path.join(DB, "etl#adj_close.pickle"), "rb") as f:
        adj = pickle.load(f).set_index("date")
    adj.index = pd.to_datetime(adj.index)
    adj.columns = adj.columns.astype(str)
    keep = [c for c in adj.columns if c.isdigit() and len(c) == 4 and not c.startswith("00")]
    return adj[keep].pct_change(fill_method=None).iloc[1:]


def load_trading_value(daily_index):
    with open(os.path.join(DB, "price#成交金額.pickle"), "rb") as f:
        tv = pickle.load(f).set_index("date")
    tv.index = pd.to_datetime(tv.index)
    tv.columns = tv.columns.astype(str)
    tv_avg = tv.rolling(TV_ROLL, min_periods=20).mean()
    tv_avg_daily = tv_avg.reindex(daily_index, method="ffill").shift(1)
    return tv_avg_daily


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
    print("Phase 32: D000 下游 純 3M-avg 成交金額 top-N (無 ratio)")
    print("=" * 110)

    chain = pd.read_csv(os.path.join(ROOT, "industry_chain.csv"), dtype=str)
    daily_ret = load_daily_ret()
    market = daily_ret.mean(axis=1)
    tv_avg_daily = load_trading_value(daily_ret.index)

    # Lead (僅供 timing)
    d100 = chain[(chain["industry_code"] == "D000") & (chain["sub_code"] == "D100")]["stock_id"].unique().tolist()
    dc00 = chain[(chain["industry_code"] == "D000") & (chain["sub_code"] == "DC00")]["stock_id"].unique().tolist()
    d100_in = [s for s in d100 if s in daily_ret.columns]
    dc00_in = [s for s in dc00 if s in daily_ret.columns]
    lead = 0.5 * daily_ret[d100_in].mean(axis=1) + 0.5 * daily_ret[dc00_in].mean(axis=1)

    d_down = chain[(chain["industry_code"] == "D000") &
                    (chain["sub_code"].isin(DOWNSTREAM_SUBS))][["stock_id", "name", "sub_code"]]
    d_down = d_down.drop_duplicates("stock_id", keep="first").reset_index(drop=True)
    universe = sorted([s for s in d_down["stock_id"] if s in daily_ret.columns])
    rets_uni = daily_ret[universe]
    name_map = dict(zip(d_down["stock_id"], d_down["name"]))
    sub_map = dict(zip(d_down["stock_id"], d_down["sub_code"]))
    print(f"\nUniverse: {len(universe)} 檔 (D000 下游)")

    first_valid = daily_ret.index[LOOKBACK]
    month_ends = pd.Series(daily_ret.index).groupby([daily_ret.index.year, daily_ret.index.month]).max()
    rebal_dates = pd.DatetimeIndex(month_ends.values)
    rebal_dates = rebal_dates[rebal_dates >= first_valid]
    rebal_list = sorted(rebal_dates)
    print(f"Rebal dates: {len(rebal_list)}")

    # === Diagnostic 1: 最後一期 TV top 30 ===
    last_d = rebal_list[-1]
    tv_today = tv_avg_daily.loc[last_d]
    tv_uni_today = tv_today.loc[[c for c in universe if c in tv_today.index]].dropna()
    tv_sorted = tv_uni_today.sort_values(ascending=False)
    print(f"\n[Diagnostic 1] {last_d.date()} TV top 20 (3M-avg, 億/日):")
    print(f"{'rank':>4s} {'stock':>5s} {'name':<10s} {'sub':>5s}  {'TV(億/日)':>12s}")
    for rank_, (sid, tv_v) in enumerate(tv_sorted.head(20).items()):
        print(f"{rank_+1:>4d} {sid:>5s} {name_map.get(sid, ''):<10s} {sub_map.get(sid,''):>5s}  "
              f"{tv_v/1e8:>12.3f}")

    # === Build portfolios for each N ===
    print(f"\n[1] 構 portfolios: pure TV top-N for N ∈ {N_LIST}")
    weight_dict = {}
    holding_freq = {}  # N -> {stock_id: count}
    for N in N_LIST:
        W = pd.DataFrame(0.0, index=daily_ret.index, columns=universe)
        freq = Counter()
        for i, d in enumerate(rebal_list):
            tv_today = tv_avg_daily.loc[d] if d in tv_avg_daily.index else None
            if tv_today is None or tv_today.dropna().empty:
                continue
            tv_uni = tv_today.loc[[c for c in universe if c in tv_today.index]].dropna()
            top = tv_uni.sort_values(ascending=False).head(N).index.tolist()
            for sid in top:
                freq[sid] += 1
            next_rebal = rebal_list[i + 1] if i + 1 < len(rebal_list) else daily_ret.index.max() + pd.Timedelta(days=1)
            hold_idx = daily_ret.index[(daily_ret.index > d) & (daily_ret.index <= next_rebal)]
            if len(top) > 0:
                W.loc[hold_idx, top] = 1.0 / len(top)
        weight_dict[N] = W
        holding_freq[N] = freq
        n_avg = (W > 0).sum(axis=1).where(lambda x: x > 0).mean()
        print(f"  N={N}: avg active = {n_avg:.1f}")

    # === Diagnostic 2: 最常被持有的個股 (N=10) ===
    print(f"\n[Diagnostic 2] 全期間 ({len(rebal_list)} months) N=10 最常入選的個股:")
    top_held = sorted(holding_freq[10].items(), key=lambda x: -x[1])[:15]
    print(f"  {'stock':>5s} {'name':<10s} {'sub':>5s} {'入選次數':>10s} {'比例':>8s}")
    for sid, cnt in top_held:
        ratio = cnt / len(rebal_list)
        print(f"  {sid:>5s} {name_map.get(sid, ''):<10s} {sub_map.get(sid,''):>5s} "
              f"{cnt:>8d}  {ratio:>7.1%}")

    holdings_df = pd.DataFrame([{"N": N, "stock_id": sid, "name": name_map.get(sid,""),
                                  "sub": sub_map.get(sid,""), "frequency": cnt}
                                  for N, freq in holding_freq.items()
                                  for sid, cnt in sorted(freq.items(), key=lambda x: -x[1])[:30]])
    holdings_df.to_csv(os.path.join(OUT, "phase32_top_holdings.csv"), index=False, encoding="utf-8-sig")

    # === Full EW baseline ===
    W_full = pd.DataFrame(0.0, index=daily_ret.index, columns=universe)
    for d in daily_ret.index[LOOKBACK:]:
        avail = rets_uni.loc[d].dropna().index.tolist()
        if avail:
            W_full.loc[d, avail] = 1.0 / len(avail)

    timing = (lead.ewm(span=SMOOTH_N, adjust=False).mean() > 0).astype(float).shift(1).fillna(0)

    eval_start = rebal_list[0] + pd.Timedelta(days=1)
    print(f"\n[2] 計算 returns (eval start = {eval_start.date()})")

    g_full, n_full, at_full = perf_from_weights(W_full, rets_uni)
    W_full_t = W_full.mul(timing, axis=0).fillna(0)
    g_full_t, n_full_t, at_full_t = perf_from_weights(W_full_t, rets_uni)
    p_full = perf_with_alpha(n_full.loc[eval_start:], market.loc[eval_start:])
    p_full_t = perf_with_alpha(n_full_t.loc[eval_start:], market.loc[eval_start:])

    rows = [
        {"strategy": "Full EW always-on", "N": "all", "timed": False, **p_full, "ann_turn": at_full},
        {"strategy": "Full EW timed", "N": "all", "timed": True, **p_full_t, "ann_turn": at_full_t},
    ]
    daily_returns_dict = {"Full_EW": n_full, "Full_EW_timed": n_full_t}

    for N in N_LIST:
        W = weight_dict[N]
        g, n, at = perf_from_weights(W, rets_uni)
        W_t = W.mul(timing, axis=0).fillna(0)
        g_t, n_t, at_t = perf_from_weights(W_t, rets_uni)
        daily_returns_dict[f"TV_top{N}"] = n
        daily_returns_dict[f"TV_top{N}_timed"] = n_t
        p = perf_with_alpha(n.loc[eval_start:], market.loc[eval_start:])
        p_t = perf_with_alpha(n_t.loc[eval_start:], market.loc[eval_start:])
        rows.append({"strategy": f"TV top-{N} always-on", "N": N, "timed": False, **p, "ann_turn": at})
        rows.append({"strategy": f"TV top-{N} timed", "N": N, "timed": True, **p_t, "ann_turn": at_t})

    res_df = pd.DataFrame(rows)
    res_df.to_csv(os.path.join(OUT, "phase32_pure_tv_summary.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(daily_returns_dict).to_csv(os.path.join(OUT, "phase32_strategy_returns.csv"),
                                              encoding="utf-8-sig")

    # === Print ===
    print(f"\n" + "=" * 110)
    print(f"策略表現 (扣 50bp RT)")
    print("=" * 110)
    base_a = res_df.iloc[0]; base_b = res_df.iloc[1]
    print(f"\nbaselines:")
    print(f"  Full EW always-on:  α={base_a['ann_alpha']:+.4f}, t={base_a['t_alpha']:+.3f}, Sh={base_a['sharpe']:+.3f}")
    print(f"  Full EW timed:      α={base_b['ann_alpha']:+.4f}, t={base_b['t_alpha']:+.3f}, Sh={base_b['sharpe']:+.3f}")
    print(f"  Phase 29 ratio N=20 timed (winner): α=+0.1336, t=+4.408, Sh=+1.633")

    print(f"\n[Always-on] (vs Full EW α={base_a['ann_alpha']:+.4f}, Sh={base_a['sharpe']:+.3f})")
    print(f"  {'N':>3s}  {'α':>9s}  {'t_α':>8s}  {'Sh':>7s}  {'turn':>6s}  Δα/ΔSh vs Full")
    for _, r in res_df[(res_df["strategy"].str.startswith("TV")) & (~res_df["timed"])].iterrows():
        d_a = (r["ann_alpha"] - base_a["ann_alpha"]) * 100
        d_sh = r["sharpe"] - base_a["sharpe"]
        flag = "✓" if d_a > 0.5 and d_sh > 0 else ("↑" if d_a > 0 else "↓")
        print(f"  {r['N']:>3d}  {r['ann_alpha']:>+9.4f}  {r['t_alpha']:>+8.3f}  "
              f"{r['sharpe']:>+7.3f}  {r['ann_turn']:>6.2f}  {d_a:+.2f}pp / {d_sh:+.3f} {flag}")

    print(f"\n[Timed] (vs Full timed α={base_b['ann_alpha']:+.4f}, Sh={base_b['sharpe']:+.3f})")
    print(f"  {'N':>3s}  {'α':>9s}  {'t_α':>8s}  {'Sh':>7s}  {'turn':>6s}  Δα/ΔSh vs Full timed")
    for _, r in res_df[(res_df["strategy"].str.startswith("TV")) & (res_df["timed"])].iterrows():
        d_a = (r["ann_alpha"] - base_b["ann_alpha"]) * 100
        d_sh = r["sharpe"] - base_b["sharpe"]
        flag = "✓" if d_a > 0.5 and d_sh > 0 else ("↑" if d_a > 0 else "↓")
        print(f"  {r['N']:>3d}  {r['ann_alpha']:>+9.4f}  {r['t_alpha']:>+8.3f}  "
              f"{r['sharpe']:>+7.3f}  {r['ann_turn']:>6.2f}  {d_a:+.2f}pp / {d_sh:+.3f} {flag}")

    # === Best N ===
    print(f"\n" + "=" * 110)
    print("Best-N (按 Sharpe 排序, timed)")
    print("=" * 110)
    timed_only = res_df[(res_df["strategy"].str.startswith("TV")) & (res_df["timed"])].sort_values("sharpe", ascending=False)
    for _, r in timed_only.iterrows():
        d_sh = r["sharpe"] - base_b["sharpe"]
        flag = "✓" if d_sh > 0 else "↓"
        print(f"  {r['strategy']}: α={r['ann_alpha']:+.4f}, t={r['t_alpha']:+.3f}, "
              f"Sh={r['sharpe']:+.3f}, turn={r['ann_turn']:.2f}  ΔSh={d_sh:+.3f} {flag}")

    # === Plot 1: N sweep ===
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    Ns = N_LIST
    a_off = [res_df[(res_df["N"] == N) & (~res_df["timed"]) & res_df["strategy"].str.startswith("TV")].iloc[0]["ann_alpha"] for N in Ns]
    a_on = [res_df[(res_df["N"] == N) & (res_df["timed"]) & res_df["strategy"].str.startswith("TV")].iloc[0]["ann_alpha"] for N in Ns]
    sh_off = [res_df[(res_df["N"] == N) & (~res_df["timed"]) & res_df["strategy"].str.startswith("TV")].iloc[0]["sharpe"] for N in Ns]
    sh_on = [res_df[(res_df["N"] == N) & (res_df["timed"]) & res_df["strategy"].str.startswith("TV")].iloc[0]["sharpe"] for N in Ns]

    axes[0].plot(Ns, a_off, "o-", color="#0277bd", lw=2, ms=8, label="always-on")
    axes[0].plot(Ns, a_on, "s-", color="#1b5e20", lw=2, ms=8, label="timed")
    axes[0].axhline(base_a["ann_alpha"], color="#0277bd", ls="--", alpha=0.5, label=f"Full EW always (α={base_a['ann_alpha']:.3f})")
    axes[0].axhline(base_b["ann_alpha"], color="#1b5e20", ls="--", alpha=0.5, label=f"Full EW timed (α={base_b['ann_alpha']:.3f})")
    axes[0].axhline(0.1336, color="#d81b60", ls=":", alpha=0.7, label="Phase 29 ratio N=20 timed (α=0.134)")
    axes[0].set_xlabel("Top-N"); axes[0].set_ylabel("ann_alpha")
    axes[0].set_title("(A) Pure TV top-N alpha")
    axes[0].legend(fontsize=8.5)
    axes[0].grid(alpha=0.3)

    axes[1].plot(Ns, sh_off, "o-", color="#0277bd", lw=2, ms=8, label="always-on")
    axes[1].plot(Ns, sh_on, "s-", color="#1b5e20", lw=2, ms=8, label="timed")
    axes[1].axhline(base_a["sharpe"], color="#0277bd", ls="--", alpha=0.5, label=f"Full EW always (Sh={base_a['sharpe']:.2f})")
    axes[1].axhline(base_b["sharpe"], color="#1b5e20", ls="--", alpha=0.5, label=f"Full EW timed (Sh={base_b['sharpe']:.2f})")
    axes[1].axhline(1.633, color="#d81b60", ls=":", alpha=0.7, label="Phase 29 ratio N=20 timed (Sh=1.63)")
    axes[1].set_xlabel("Top-N"); axes[1].set_ylabel("Sharpe")
    axes[1].set_title("(B) Pure TV top-N Sharpe")
    axes[1].legend(fontsize=8.5)
    axes[1].grid(alpha=0.3)
    plt.suptitle("Phase 32: D000 下游 純成交金額 top-N (無 ratio)", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig_phase32_N_sweep.png"), dpi=130)
    plt.close()
    print(f"\n圖檔: fig_phase32_N_sweep.png")

    # === Plot 2: cum curves ===
    fig, ax = plt.subplots(figsize=(11, 5.5))
    cmap = plt.cm.viridis(np.linspace(0.15, 0.85, len(N_LIST)))
    cum_full_t = (1 + n_full_t.loc[eval_start:].fillna(0)).cumprod()
    ax.plot(cum_full_t, color="black", lw=2.0, alpha=0.75, ls="--",
            label=f"Full EW timed [{cum_full_t.iloc[-1]:.2f}x] α={base_b['ann_alpha']:.3f}")
    for i, N in enumerate(N_LIST):
        n_t_eval = daily_returns_dict[f"TV_top{N}_timed"].loc[eval_start:]
        cum = (1 + n_t_eval.fillna(0)).cumprod()
        r = res_df[(res_df["N"] == N) & (res_df["timed"])].iloc[0]
        ax.plot(cum, color=cmap[i], lw=1.4, alpha=0.92,
                label=f"TV top-{N} timed [{cum.iloc[-1]:.2f}x] α={r['ann_alpha']:.3f}, Sh={r['sharpe']:.2f}")
    cum_mkt = (1 + market.loc[eval_start:].fillna(0)).cumprod()
    ax.plot(cum_mkt, color="gray", lw=0.8, alpha=0.5, label=f"Market [{cum_mkt.iloc[-1]:.2f}x]")
    ax.set_yscale("log")
    ax.set_title("Phase 32: 純成交金額 top-N timed (扣 50bp)")
    ax.set_xlabel("date"); ax.set_ylabel("累積資本 (log)")
    ax.legend(loc="upper left", fontsize=8.5)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig_phase32_curves.png"), dpi=130)
    plt.close()
    print(f"圖檔: fig_phase32_curves.png")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
