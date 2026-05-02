"""
Phase 1: 構建產業鏈組合日報酬

依 industry_chain.csv 將股票分組為 (industry_code, position)，
對每組計算 等權 (EW) 與 市值加權 (VW) 日報酬，
輸出長表 panel: (date, industry_code, position) -> ret_ew, ret_vw, n_stocks

只處理 is_standard_chain == True 的鏈，position ∈ {上游, 中游, 下游}。
"""
from __future__ import annotations

import os
import pickle
import sys

import numpy as np
import pandas as pd

ROOT = r"C:\Users\user\OneDrive\桌面\產業因子生成"
DB = r"C:\Users\user\finlab_db"
OUT = os.path.join(ROOT, "research", "output")
os.makedirs(OUT, exist_ok=True)


def load_pickle_df(name: str) -> pd.DataFrame:
    """載入 finlab_db pickle 並標準化為 DatetimeIndex × ticker columns。"""
    path = os.path.join(DB, name)
    with open(path, "rb") as f:
        df = pickle.load(f)
    df = df.set_index("date")
    df.index = pd.to_datetime(df.index)
    df.columns = df.columns.astype(str)
    return df


def filter_common_stocks(df: pd.DataFrame) -> pd.DataFrame:
    """只保留 4 碼數字、不是 00 開頭的普通股 ticker。"""
    keep = [c for c in df.columns if c.isdigit() and len(c) == 4 and not c.startswith("00")]
    return df[keep]


def build_portfolio_returns(
    rets: pd.DataFrame,
    mvs: pd.DataFrame,
    members: list[str],
) -> pd.DataFrame:
    """對 members 中的股票，計算每日 EW、VW 報酬與有效股票數。

    EW: 簡單平均（忽略 NaN）
    VW: 用前一日市值加權（避免 lookahead）
    """
    members = [s for s in members if s in rets.columns]
    if not members:
        return pd.DataFrame(columns=["ret_ew", "ret_vw", "n_stocks"], index=rets.index)

    sub_r = rets[members]
    n = sub_r.notna().sum(axis=1)

    ew = sub_r.mean(axis=1, skipna=True)

    mv_members = [s for s in members if s in mvs.columns]
    if mv_members:
        sub_mv = mvs[mv_members].reindex(index=rets.index).shift(1)
        sub_mv = sub_mv.where(sub_r[mv_members].notna())
        weights = sub_mv.div(sub_mv.sum(axis=1), axis=0)
        vw = (sub_r[mv_members] * weights).sum(axis=1, min_count=1)
    else:
        vw = pd.Series(np.nan, index=rets.index)

    out = pd.concat({"ret_ew": ew, "ret_vw": vw, "n_stocks": n}, axis=1)
    return out


def main():
    print(f"[Phase 1] 構建產業鏈組合日報酬")
    print(f"  ROOT: {ROOT}")

    # 1) 載入價格與市值
    print("\n[1/4] 載入 FinLab 價格、市值資料...")
    adj_close = load_pickle_df("etl#adj_close.pickle")
    market_value = load_pickle_df("etl#market_value.pickle")
    print(f"  adj_close: {adj_close.shape}  {adj_close.index.min().date()} ~ {adj_close.index.max().date()}")
    print(f"  market_value: {market_value.shape}  {market_value.index.min().date()} ~ {market_value.index.max().date()}")

    adj_close = filter_common_stocks(adj_close)
    market_value = filter_common_stocks(market_value)
    print(f"  filtered普通股: close={adj_close.shape[1]}, mv={market_value.shape[1]}")

    # 2) 計算日簡單報酬
    print("\n[2/4] 計算日報酬...")
    rets = adj_close.pct_change()
    rets = rets.iloc[1:]
    print(f"  rets shape: {rets.shape}")
    print(f"  ret 中位數 std: {rets.std().median():.4f}")

    # 3) 載入產業鏈分類，過濾標準上中下游
    print("\n[3/4] 載入產業鏈分類...")
    chain = pd.read_csv(os.path.join(ROOT, "industry_chain.csv"), dtype=str)
    std = chain[chain["is_standard_chain"] == "True"].copy()
    std = std[std["position"].isin(["上游", "中游", "下游"])]
    print(f"  standard 鏈樣本: {len(std)} 列")
    print(f"  涵蓋產業: {std['industry_name'].nunique()} 個")

    # 一檔股票若同時在某 (industry, position) 出現多次（不同 sub_code），去重一次
    std = std.drop_duplicates(["stock_id", "industry_code", "position"])
    print(f"  去重後 (stock × industry × position): {len(std)} 列")

    # 4) 對每個 (industry_code, position) 構建組合報酬
    print("\n[4/4] 構建組合日報酬...")
    out_rows = []
    groups = std.groupby(["industry_code", "industry_name", "position"])
    for (ind_code, ind_name, pos), g in groups:
        members = g["stock_id"].unique().tolist()
        port = build_portfolio_returns(rets, market_value, members)
        port = port.reset_index().rename(columns={"index": "date", "date": "date"})
        port["industry_code"] = ind_code
        port["industry_name"] = ind_name
        port["position"] = pos
        port["n_listed"] = len(members)
        out_rows.append(port)
        print(f"  {ind_code} {ind_name:8s} {pos}: {len(members):3d} listed, n_stocks 中位數={port['n_stocks'].median():.0f}")

    panel = pd.concat(out_rows, ignore_index=True)
    panel = panel[["date", "industry_code", "industry_name", "position", "ret_ew", "ret_vw", "n_stocks", "n_listed"]]

    # 計算市場報酬：所有共同股票 EW
    market_ret = rets.mean(axis=1, skipna=True).rename("ret_market")

    # 輸出
    panel_path = os.path.join(OUT, "phase_returns.pkl")
    panel.to_pickle(panel_path)
    print(f"\n  panel 寫入 {panel_path}")
    print(f"  panel shape: {panel.shape}")

    market_path = os.path.join(OUT, "market_return.pkl")
    market_ret.to_frame().to_pickle(market_path)
    print(f"  market_ret 寫入 {market_path}")

    # 印簡要統計
    print("\n=== 簡要統計（EW 日報酬）===")
    pivot = panel.pivot_table(index="date", columns=["industry_code", "position"], values="ret_ew")
    print(f"  pivot shape: {pivot.shape}")
    print(f"  覆蓋天數: {pivot.shape[0]}")
    print(f"  欄位數 (industry × position): {pivot.shape[1]}")

    # 各鏈各位置的覆蓋與年化波動度
    summary = panel.groupby(["industry_code", "industry_name", "position"]).agg(
        n_days=("date", "count"),
        n_stocks_med=("n_stocks", "median"),
        ret_mean=("ret_ew", "mean"),
        ret_std=("ret_ew", "std"),
    ).reset_index()
    summary["ann_ret"] = summary["ret_mean"] * 252
    summary["ann_vol"] = summary["ret_std"] * np.sqrt(252)
    summary_path = os.path.join(OUT, "phase_returns_summary.csv")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"  summary 寫入 {summary_path}")
    print(summary[["industry_name", "position", "n_stocks_med", "ann_ret", "ann_vol"]].to_string(index=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
