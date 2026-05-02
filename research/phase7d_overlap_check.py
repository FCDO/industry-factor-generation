"""
Phase 7d: 重疊股票 robustness check + in-sample bias 檢驗

確認 Phase 7c 半導體 → 下游 spillover 真實性, 排除以下污染:
1. 重疊股票: 半導體 ∩ 下游 中的股票會在 r_A 與 r_B 都出現, 自相關可能造成假陽性
2. In-sample bias: 7 下游是看到 Phase 7b 結果後選的, 應做 split-sample (前半挑選, 後半驗證)

設計:
1. 對每個 pair (A→B), 重新計算 r_A_excl_B = A 中**不在 B**的股票 EW;
   r_B_excl_A = B 中**不在 A**的股票 EW. 重跑迴歸.
2. Split-sample 1: 用 2007-2015 樣本挑選顯著 pairs, 在 2016-2024 驗證
"""
from __future__ import annotations

import os
import pickle
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = r"C:\Users\user\OneDrive\桌面\產業因子生成"
DB = r"C:\Users\user\finlab_db"
OUT = os.path.join(ROOT, "research", "output")


def load_pickle_df(name):
    with open(os.path.join(DB, name), "rb") as f:
        df = pickle.load(f)
    df = df.set_index("date"); df.index = pd.to_datetime(df.index)
    df.columns = df.columns.astype(str)
    return df


def filter_common(df):
    keep = [c for c in df.columns if c.isdigit() and len(c) == 4 and not c.startswith("00")]
    return df[keep]


def reg_AB(rA, rB, rM, nw_lag):
    df = pd.DataFrame({
        "y": rB, "xA": rA.shift(1), "xB": rB.shift(1), "xM": rM.shift(1)
    }).dropna()
    if len(df) < 200: return None
    res = sm.OLS(df["y"], sm.add_constant(df[["xA","xB","xM"]])).fit(
        cov_type="HAC", cov_kwds={"maxlags": nw_lag}
    )
    return {"n": len(df), "beta": res.params["xA"], "t": res.tvalues["xA"], "p": res.pvalues["xA"]}


def main():
    print("[Phase 7d] 重疊股票 + Split-sample robustness\n")

    chain = pd.read_csv(os.path.join(ROOT, "industry_chain.csv"), dtype=str)
    industries = chain[["industry_code","industry_name"]].drop_duplicates().set_index("industry_code")["industry_name"].to_dict()

    adj_close = filter_common(load_pickle_df("etl#adj_close.pickle"))
    daily_ret = adj_close.pct_change().iloc[1:]

    # 各產業全部成員 set
    ind_stocks = {ind: set(chain[chain["industry_code"] == ind]["stock_id"].unique()) for ind in industries}

    # 全市場 EW
    market = daily_ret.mean(axis=1)

    # 待驗證的核心 pair (Phase 7c top)
    KEY_PAIRS = [
        ("D000", "G000"), ("D000", "F000"), ("D000", "L000"),
        ("D000", "I000"), ("D000", "H000"), ("D000", "J000"),
        ("D000", "5400"), ("D000", "K000"),
        ("Q000", "3000"), ("Q000", "P000"), ("Q000", "S000"),
        ("K000", "F000"), ("P000", "3000"), ("I000", "F000"),
    ]

    # === (1) 重疊股票排除測試 ===
    print("=== 重疊股票 robustness check ===")
    print(f"{'A→B':<28s} {'A_only':>8s} {'B_only':>8s} {'overlap':>8s} {'orig β/t':>14s} {'excl β/t':>14s}")
    rows = []
    for A, B in KEY_PAIRS:
        if A not in ind_stocks or B not in ind_stocks: continue
        A_set, B_set = ind_stocks[A], ind_stocks[B]
        overlap = A_set & B_set
        A_only = A_set - B_set
        B_only = B_set - A_set

        A_only_cols = [s for s in A_only if s in daily_ret.columns]
        B_only_cols = [s for s in B_only if s in daily_ret.columns]
        A_full_cols = [s for s in A_set if s in daily_ret.columns]
        B_full_cols = [s for s in B_set if s in daily_ret.columns]

        if len(A_only_cols) < 3 or len(B_only_cols) < 3:
            continue

        rA_full = daily_ret[A_full_cols].mean(axis=1)
        rB_full = daily_ret[B_full_cols].mean(axis=1)
        rA_excl = daily_ret[A_only_cols].mean(axis=1)
        rB_excl = daily_ret[B_only_cols].mean(axis=1)

        orig = reg_AB(rA_full, rB_full, market, 5)
        excl = reg_AB(rA_excl, rB_excl, market, 5)
        if orig is None or excl is None:
            continue

        marker_o = "*" if abs(orig["t"]) > 1.96 else ""
        marker_e = "*" if abs(excl["t"]) > 1.96 else ""
        print(f"{industries[A]:<8s} -> {industries[B]:<10s} {len(A_only_cols):>8d} {len(B_only_cols):>8d} {len(overlap):>8d} "
              f"{orig['beta']:+.3f}/{orig['t']:+.2f}{marker_o:<2s} {excl['beta']:+.3f}/{excl['t']:+.2f}{marker_e:<2s}")
        rows.append({
            "A_name": industries[A], "B_name": industries[B],
            "A_only": len(A_only_cols), "B_only": len(B_only_cols), "overlap": len(overlap),
            "orig_beta": orig["beta"], "orig_t": orig["t"],
            "excl_beta": excl["beta"], "excl_t": excl["t"],
        })

    pd.DataFrame(rows).to_csv(os.path.join(OUT, "overlap_check.csv"), index=False, encoding="utf-8-sig")

    # === (2) Split-sample test ===
    print("\n=== Split-sample 驗證 (前半挑選, 後半測試) ===")
    panel = {}
    for ind in industries:
        members = list(ind_stocks[ind])
        cols = [s for s in members if s in daily_ret.columns]
        if len(cols) >= 3:
            panel[ind] = daily_ret[cols].mean(axis=1)
    panel = pd.DataFrame(panel)

    split_date = "2016-01-01"
    panel_train = panel[panel.index < split_date]
    panel_test = panel[panel.index >= split_date]
    market_train = market[market.index < split_date]
    market_test = market[market.index >= split_date]

    print(f"  Train: {panel_train.index[0].date()} ~ {panel_train.index[-1].date()} ({len(panel_train)} 天)")
    print(f"  Test:  {panel_test.index[0].date()} ~ {panel_test.index[-1].date()} ({len(panel_test)} 天)")

    # Train: 找出 |t|>2 且 β>0 的 directed pairs
    train_rows = []
    for A in panel.columns:
        for B in panel.columns:
            if A == B: continue
            r = reg_AB(panel_train[A], panel_train[B], market_train, 5)
            if r is None: continue
            train_rows.append({"A": A, "B": B, **r})
    train_df = pd.DataFrame(train_rows)
    train_sig = train_df[(train_df["t"] > 2.0) & (train_df["beta"] > 0)]
    print(f"  Train 顯著正向 pair 數: {len(train_sig)}")

    # Test: 對 train_sig 中的 pair 重跑 OOS 迴歸
    test_rows = []
    for _, row in train_sig.iterrows():
        A, B = row["A"], row["B"]
        r = reg_AB(panel_test[A], panel_test[B], market_test, 5)
        if r is None: continue
        test_rows.append({
            "A_name": industries[A], "B_name": industries[B],
            "train_beta": row["beta"], "train_t": row["t"],
            "test_beta": r["beta"], "test_t": r["t"], "test_p": r["p"],
        })
    test_df = pd.DataFrame(test_rows)
    if len(test_df) > 0:
        test_df = test_df.sort_values("test_t", ascending=False)
        test_df.to_csv(os.path.join(OUT, "split_sample_oos.csv"), index=False, encoding="utf-8-sig")

        share_oos_pos = (test_df["test_beta"] > 0).mean()
        share_oos_sig = ((test_df["test_p"] < 0.05) & (test_df["test_beta"] > 0)).mean()
        print(f"\n  OOS β > 0 比例: {share_oos_pos:.3f} (基準 0.5 = no-info)")
        print(f"  OOS p<0.05 且 β>0 比例: {share_oos_sig:.3f}")
        print(f"  OOS mean β: {test_df['test_beta'].mean():.4f}, mean t: {test_df['test_t'].mean():.3f}")

        from scipy import stats as st
        # 雙尾 binomial: H0: P(test_beta > 0) = 0.5
        n_pos = (test_df["test_beta"] > 0).sum()
        n_total = len(test_df)
        binom_p = st.binomtest(n_pos, n_total, 0.5, alternative="greater").pvalue
        print(f"  Binomial test (H0: P(β>0)=0.5): {n_pos}/{n_total}, p = {binom_p:.4f}")

        print(f"\n  OOS 仍顯著的 top 15 pairs:")
        oos_sig = test_df[(test_df["test_p"] < 0.05) & (test_df["test_beta"] > 0)].head(15)
        print(oos_sig.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
