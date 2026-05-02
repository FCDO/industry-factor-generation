"""
Phase 7b: 跨產業 link spillover - 日/週頻 robustness

呼應 Phase 2 發現 (within-chain spillover 在日頻最強, 月頻衰退),
檢驗 cross-industry spillover 是否在日/週頻才顯著.
"""
from __future__ import annotations

import os
import pickle
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

ROOT = r"C:\Users\user\OneDrive\桌面\產業因子生成"
DB = r"C:\Users\user\finlab_db"
OUT = os.path.join(ROOT, "research", "output")


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


def run_pairwise_regression(panel: pd.DataFrame, market: pd.Series, nw_lag: int, min_obs: int):
    """對 panel 的所有 directed pair 跑 r_B(t+1) = α + β·r_A(t) + γ·r_B(t) + δ·r_M(t) + ε"""
    rows = []
    for A in panel.columns:
        for B in panel.columns:
            if A == B: continue
            df = pd.DataFrame({
                "y": panel[B],
                "xA": panel[A].shift(1),
                "xB": panel[B].shift(1),
                "xM": market.shift(1),
            }).dropna()
            if len(df) < min_obs: continue
            try:
                res = sm.OLS(df["y"], sm.add_constant(df[["xA","xB","xM"]])).fit(
                    cov_type="HAC", cov_kwds={"maxlags": nw_lag}
                )
                rows.append({
                    "A": A, "B": B, "n": len(df),
                    "beta_AB": res.params["xA"],
                    "t_AB": res.tvalues["xA"],
                    "p_AB": res.pvalues["xA"],
                })
            except Exception:
                continue
    return pd.DataFrame(rows)


def main():
    print("[Phase 7b] 跨產業 link spillover (日/週頻 robustness)\n")

    chain = pd.read_csv(os.path.join(ROOT, "industry_chain.csv"), dtype=str)
    industries = chain[["industry_code","industry_name"]].drop_duplicates().set_index("industry_code")["industry_name"].to_dict()

    adj_close = filter_common(load_pickle_df("etl#adj_close.pickle"))
    daily_ret = adj_close.pct_change().iloc[1:]

    log_d = np.log1p(daily_ret)

    # 構建各頻率 panel
    print("構建各頻率產業報酬...")
    industry_members = {ind: chain[chain["industry_code"] == ind]["stock_id"].unique().tolist()
                        for ind in industries}

    def build_panel(rets):
        out = {}
        for ind, members in industry_members.items():
            cols = [s for s in members if s in rets.columns]
            if len(cols) >= 3:
                out[ind] = rets[cols].mean(axis=1)
        return pd.DataFrame(out)

    panel_d = build_panel(daily_ret)
    weekly_simple = np.expm1(log_d.resample("W-FRI").sum(min_count=3))
    panel_w = build_panel(weekly_simple)
    monthly_simple = np.expm1(log_d.resample("ME").sum(min_count=10))
    panel_m = build_panel(monthly_simple)

    market_d = daily_ret.mean(axis=1)
    market_w = weekly_simple.mean(axis=1)
    market_m = monthly_simple.mean(axis=1)

    print(f"  Panel D: {panel_d.shape}, W: {panel_w.shape}, M: {panel_m.shape}")

    # 載入 link 資料
    link_df = pd.read_csv(os.path.join(OUT, "cross_industry_links.csv"))

    # 排除 醫療器材 (代碼 C200) 與 其他 (X000) 作主測試
    EXCLUDED = ["C200", "X000"]

    results_by_freq = {}
    for freq, panel, market, nw_lag, min_obs in [
        ("D", panel_d, market_d, 5, 250),
        ("W", panel_w, market_w, 4, 100),
        ("M", panel_m, market_m, 3, 60),
    ]:
        print(f"\n=== 頻率: {freq} ===")
        reg = run_pairwise_regression(panel, market, nw_lag, min_obs)
        reg = reg.merge(link_df[["A","B","L_strict","L_loose","S_undir"]], on=["A","B"], how="left")
        reg["A_name"] = reg["A"].map(industries)
        reg["B_name"] = reg["B"].map(industries)
        reg.to_csv(os.path.join(OUT, f"cross_industry_spillover_{freq}.csv"), index=False, encoding="utf-8-sig")
        results_by_freq[freq] = reg

        # 報告
        print(f"  pairs: {len(reg)}")
        for label, sub in [("All", reg),
                           ("excl. 醫療器材/其他", reg[~reg["A"].isin(EXCLUDED) & ~reg["B"].isin(EXCLUDED)])]:
            print(f"  --- {label} ---")
            for blo, bhi, name in [(0,0,"L=0"),(1,2,"L=1-2"),(3,5,"L=3-5"),(6,9999,"L>=6")]:
                d = sub[(sub["L_loose"]>=blo) & (sub["L_loose"]<=bhi)]
                if len(d) > 0:
                    sig5_pos = ((d["p_AB"]<0.05) & (d["beta_AB"]>0)).mean()
                    print(f"    {name:<6s} n={len(d):4d} mean_β={d['beta_AB'].mean():+.4f} mean_t={d['t_AB'].mean():+.3f} share_pos={(d['beta_AB']>0).mean():.3f} sig5+={sig5_pos:.3f}")
            # 雙樣本 t
            linked = sub[sub["L_loose"]>=3]
            unlinked = sub[sub["L_loose"]==0]
            if len(linked) > 5 and len(unlinked) > 5:
                t = stats.ttest_ind(linked["beta_AB"], unlinked["beta_AB"], equal_var=False)
                print(f"    H0: β(linked) = β(unlinked); t = {t.statistic:+.3f}, p = {t.pvalue:.4f}")

    # === 對核心連結特別關注 ===
    KEY_PAIRS = [
        ("D000", "G000"),  # 半導體 → 平面顯示器
        ("D000", "F000"),  # 半導體 → 電腦週邊
        ("D000", "L000"),  # 半導體 → 印刷電路板
        ("D000", "I000"),  # 半導體 → 通信網路
        ("D000", "H000"),  # 半導體 → 觸控面板
        ("D000", "J000"),  # 半導體 → 被動元件
        ("D000", "5400"),  # 半導體 → 雲端運算
        ("K000", "F000"),  # 連接器 → 電腦週邊
        ("P000", "3000"),  # 電機機械 → 汽車
        ("I000", "F000"),  # 通信網路 → 電腦週邊
        ("L000", "F000"),  # 印刷電路板 → 電腦週邊
        ("J000", "F000"),  # 被動元件 → 電腦週邊
        ("N000", "O000"),  # 石化 → 紡織
        ("Q000", "3000"),  # 鋼鐵 → 汽車
        ("Q000", "P000"),  # 鋼鐵 → 電機機械
        ("Q000", "S000"),  # 鋼鐵 → 建材營造
        ("C100", "C200"),  # 製藥 → 醫療器材
        ("C100", "C300"),  # 製藥 → 食品生技
        ("C100", "C400"),  # 製藥 → 再生醫療
    ]

    print("\n=== 核心供應鏈 pair 各頻率表現 ===")
    print(f"{'A→B':<28s} {'L_loose':>8s} {'D: β / t':>14s} {'W: β / t':>14s} {'M: β / t':>14s}")
    for A, B in KEY_PAIRS:
        if A not in industries or B not in industries: continue
        link_v = link_df.loc[(link_df["A"]==A) & (link_df["B"]==B), "L_loose"].iloc[0] if len(link_df.loc[(link_df["A"]==A) & (link_df["B"]==B)])>0 else 0
        line = f"{industries[A]:<8s} -> {industries[B]:<10s} {link_v:>8d}"
        for freq in ["D","W","M"]:
            r = results_by_freq[freq]
            row = r[(r["A"]==A) & (r["B"]==B)]
            if len(row) > 0:
                b = row["beta_AB"].iloc[0]
                t = row["t_AB"].iloc[0]
                marker = "*" if abs(t) > 1.96 else ""
                line += f" {b:+.3f}/{t:+.2f}{marker:<2s}"
            else:
                line += "       n/a   "
        print(line)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
