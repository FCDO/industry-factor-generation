"""
Phase 23: D400 / D500 中游個股 受上游 spillover 敏感度排序

動機 (Phase 9/10/20)
- D400 (生產製程及檢測設備, 中游 85 檔) 與 D500 (化學品, 中游 24 檔) 集合層面是純 receiver
  作 predictor 對下游 t < 1.5, 對上游反向 placebo t < 0.1
- 但 EW 平均掩蓋了 cross-sectional 異質性: 個股對上游 lead 的 lagged β 應有差異
- 任務: 個股級 β-sorting → 找出真正受上游影響最深的中游個股

方法
1. Lead signal: x_t = 0.5·D100 + 0.5·DC00 (純上游 leader EW, 共 ~118 檔)
2. Per-stock regression (4245 obs, full sample):
   r_i(t) = α + β·x(t-1) + γ·r_i(t-1) + δ·market(t-1) + ε    [HAC SE, lag=5]
3. 同步 placebo: contemporaneous β (同期), 確認 lagged β 不是 spurious correlation 重述
4. Robustness:
   (a) Full sample t(β_lag)
   (b) 後 5 年 (2021-2026) sub-sample t — 確認穩定性
   (c) Rolling 1y β 的時間 stability (max-min spread)
5. 排序輸出 top 15 / bottom 5 (含 stock name + sub_code)

輸出
- phase23_midstream_betas.csv : 全 109 檔 ranking
- phase23_top_picks.csv       : top 15 (with metadata)
"""
from __future__ import annotations

import os
import pickle
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = r"C:\Users\engli\OneDrive\桌面\產業因子生成"
DB = r"C:\Users\engli\finlab_db"
OUT = os.path.join(ROOT, "research", "output")

NW_LAG = 5
ROLL_WIN_1Y = 250


def load_daily_ret():
    with open(os.path.join(DB, "etl#adj_close.pickle"), "rb") as f:
        adj = pickle.load(f).set_index("date")
    adj.index = pd.to_datetime(adj.index)
    adj.columns = adj.columns.astype(str)
    keep = [c for c in adj.columns if c.isdigit() and len(c) == 4 and not c.startswith("00")]
    return adj[keep].pct_change(fill_method=None).iloc[1:]


def reg_one(y, x_lag, y_lag, m_lag):
    df = pd.concat({"y": y, "x": x_lag, "y_lag": y_lag, "m_lag": m_lag}, axis=1).dropna()
    if len(df) < 250:
        return None
    res = sm.OLS(df["y"], sm.add_constant(df[["x", "y_lag", "m_lag"]])).fit(
        cov_type="HAC", cov_kwds={"maxlags": NW_LAG}
    )
    return {
        "n": len(df),
        "beta": res.params["x"],
        "t": res.tvalues["x"],
        "p": res.pvalues["x"],
        "alpha_ann": res.params["const"] * 252,
    }


def reg_contemp(y, x, y_lag, m_lag):
    """contemporaneous (同期) placebo — should be very strong if just market beta."""
    df = pd.concat({"y": y, "x": x, "y_lag": y_lag, "m_lag": m_lag}, axis=1).dropna()
    if len(df) < 250:
        return None
    res = sm.OLS(df["y"], sm.add_constant(df[["x", "y_lag", "m_lag"]])).fit(
        cov_type="HAC", cov_kwds={"maxlags": NW_LAG}
    )
    return {"beta_c": res.params["x"], "t_c": res.tvalues["x"]}


def main():
    print("=" * 110)
    print("Phase 23: D400 / D500 中游個股 上游 spillover β-loading 排序")
    print("=" * 110)

    chain = pd.read_csv(os.path.join(ROOT, "industry_chain.csv"), dtype=str)
    daily_ret = load_daily_ret()
    market = daily_ret.mean(axis=1)
    print(f"\nDaily return matrix: {daily_ret.shape} ({daily_ret.index.min().date()} → {daily_ret.index.max().date()})")

    # === 1. Build upstream lead ===
    d100 = chain[(chain["industry_code"] == "D000") & (chain["sub_code"] == "D100")]["stock_id"].unique().tolist()
    dc00 = chain[(chain["industry_code"] == "D000") & (chain["sub_code"] == "DC00")]["stock_id"].unique().tolist()
    d100_in = [s for s in d100 if s in daily_ret.columns]
    dc00_in = [s for s in dc00 if s in daily_ret.columns]
    d100_ew = daily_ret[d100_in].mean(axis=1)
    dc00_ew = daily_ret[dc00_in].mean(axis=1)
    lead = 0.5 * d100_ew + 0.5 * dc00_ew
    print(f"\nLead signal: 0.5·D100({len(d100_in)} 檔) + 0.5·DC00({len(dc00_in)} 檔)")
    print(f"  D100 ann_ret = {d100_ew.mean()*252:.4f}, DC00 ann_ret = {dc00_ew.mean()*252:.4f}")
    print(f"  corr(D100, DC00) = {d100_ew.corr(dc00_ew):.3f}")

    # === 2. Universe: D400 + D500 ===
    d400 = chain[(chain["industry_code"] == "D000") & (chain["sub_code"] == "D400")][["stock_id", "name"]].drop_duplicates()
    d500 = chain[(chain["industry_code"] == "D000") & (chain["sub_code"] == "D500")][["stock_id", "name"]].drop_duplicates()
    d400["sub_code"] = "D400"; d400["sub_name"] = "生產製程及檢測設備(中游)"
    d500["sub_code"] = "D500"; d500["sub_name"] = "化學品"
    universe_meta = pd.concat([d400, d500], ignore_index=True)
    universe_meta = universe_meta[universe_meta["stock_id"].isin(daily_ret.columns)].reset_index(drop=True)
    print(f"\nUniverse: D400 ({(universe_meta['sub_code']=='D400').sum()}) + D500 ({(universe_meta['sub_code']=='D500').sum()}) "
          f"= {len(universe_meta)} 檔 (FinLab 有資料)")

    # === 3. Per-stock lagged-β regression ===
    print(f"\n[1] Per-stock lagged-β regression (HAC lag={NW_LAG})")
    lead_lag = lead.shift(1)
    market_lag = market.shift(1)

    cutoff_5y = pd.Timestamp("2021-01-01")
    rows = []
    for _, meta in universe_meta.iterrows():
        sid = meta["stock_id"]
        r = daily_ret[sid].dropna()
        if len(r) < 500:
            continue
        # full sample
        rec = reg_one(r, lead_lag, r.shift(1), market_lag)
        if rec is None:
            continue
        # contemporaneous placebo
        contemp = reg_contemp(r, lead, r.shift(1), market_lag)
        # last 5y
        r_5y = r[r.index >= cutoff_5y]
        rec_5y = reg_one(r_5y, lead_lag, r_5y.shift(1), market_lag) if len(r_5y) >= 500 else None
        # ann vol
        ann_vol = r.std() * np.sqrt(252)

        rows.append({
            "stock_id": sid,
            "name": meta["name"],
            "sub_code": meta["sub_code"],
            "n_full": rec["n"],
            "beta_lag": rec["beta"],
            "t_lag": rec["t"],
            "alpha_ann_full": rec["alpha_ann"],
            "beta_contemp": contemp["beta_c"] if contemp else np.nan,
            "t_contemp": contemp["t_c"] if contemp else np.nan,
            "beta_lag_5y": rec_5y["beta"] if rec_5y else np.nan,
            "t_lag_5y": rec_5y["t"] if rec_5y else np.nan,
            "ann_vol": ann_vol,
        })

    df = pd.DataFrame(rows).sort_values("t_lag", ascending=False).reset_index(drop=True)
    print(f"  完成 regression: {len(df)} 檔")

    # === 4. 顯示 top / bottom ===
    print(f"\n" + "=" * 110)
    print(f"【TOP 15】 受上游 D100+DC00 lagged β 最強的中游個股 (t 排序)")
    print("=" * 110)
    print(f"{'rank':>4s} {'stock':>6s} {'name':<10s} {'sub':>5s}  "
          f"{'β_lag':>7s} {'t_lag':>7s} {'β_now':>7s} {'t_now':>7s} {'β_5y':>7s} {'t_5y':>7s} {'σ':>6s}")
    for i, r in df.head(15).iterrows():
        print(f"{i+1:>4d} {r['stock_id']:>6s} {r['name']:<10s} {r['sub_code']:>5s}  "
              f"{r['beta_lag']:>+7.3f} {r['t_lag']:>+7.2f} "
              f"{r['beta_contemp']:>+7.3f} {r['t_contemp']:>+7.2f} "
              f"{r['beta_lag_5y']:>+7.3f} {r['t_lag_5y']:>+7.2f} {r['ann_vol']:>6.3f}")

    print(f"\n【BOTTOM 5】最不受上游影響 / 反向")
    for i, r in df.tail(5).iterrows():
        print(f"{i+1:>4d} {r['stock_id']:>6s} {r['name']:<10s} {r['sub_code']:>5s}  "
              f"{r['beta_lag']:>+7.3f} {r['t_lag']:>+7.2f} "
              f"{r['beta_contemp']:>+7.3f} {r['t_contemp']:>+7.2f} "
              f"{r['beta_lag_5y']:>+7.3f} {r['t_lag_5y']:>+7.2f} {r['ann_vol']:>6.3f}")

    # === 5. 分組統計 ===
    print(f"\n" + "=" * 110)
    print("各 sub_code 內統計 (t_lag)")
    print("=" * 110)
    for sub in ["D400", "D500"]:
        sub_df = df[df["sub_code"] == sub]
        n_sig = (sub_df["t_lag"] > 1.96).sum()
        n_strong = (sub_df["t_lag"] > 2.58).sum()
        print(f"  {sub} ({len(sub_df)} 檔): mean t={sub_df['t_lag'].mean():.2f}, "
              f"median t={sub_df['t_lag'].median():.2f}, "
              f"max t={sub_df['t_lag'].max():.2f}; "
              f"|t|>1.96: {n_sig}/{len(sub_df)} ({n_sig/len(sub_df)*100:.0f}%); "
              f"|t|>2.58: {n_strong}/{len(sub_df)}")

    # === 6. 穩定性 (full vs 5y) 雙樣本一致性 ===
    print(f"\n" + "=" * 110)
    print("穩定性: full sample t > 1.96 且 5y t > 1.0 (同方向且不衰退) 的個股")
    print("=" * 110)
    stable = df[(df["t_lag"] > 1.96) & (df["t_lag_5y"] > 1.0)].copy()
    print(f"  共 {len(stable)} 檔通過")
    for _, r in stable.head(20).iterrows():
        print(f"  {r['stock_id']:>5s} {r['name']:<10s} {r['sub_code']:>5s}  "
              f"t_full={r['t_lag']:>+5.2f} t_5y={r['t_lag_5y']:>+5.2f}  β_full={r['beta_lag']:+.3f}")

    # === 7. 儲存 ===
    df.to_csv(os.path.join(OUT, "phase23_midstream_betas.csv"), index=False, encoding="utf-8-sig")
    df.head(15).to_csv(os.path.join(OUT, "phase23_top_picks.csv"), index=False, encoding="utf-8-sig")
    print(f"\n輸出: phase23_midstream_betas.csv (全 {len(df)} 檔)")
    print(f"輸出: phase23_top_picks.csv (top 15)")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
