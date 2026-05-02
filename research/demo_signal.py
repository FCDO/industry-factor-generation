"""示範: 訊號是怎麼一行一行組成出來的"""
import os, pickle, sys
import numpy as np, pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
DB = r"C:\Users\user\finlab_db"
ROOT = r"C:\Users\user\OneDrive\桌面\產業因子生成"

# === Step 1: 原始材料 — adj_close.pickle ===
with open(os.path.join(DB, "etl#adj_close.pickle"), "rb") as f:
    adj_close = pickle.load(f).set_index("date")
adj_close.index = pd.to_datetime(adj_close.index)
adj_close.columns = adj_close.columns.astype(str)
print("Step 1: 原始 adj_close (調整收盤價)")
print(adj_close[["2330","2454","3034"]].tail(3))

# === Step 2: 個股日報酬 ===
daily_ret = adj_close.pct_change(fill_method=None).iloc[1:]
print("\nStep 2: 個股日報酬 (price.pct_change)")
print(daily_ret[["2330","2454","3034"]].tail(3))

# === Step 3: 從 industry_chain.csv 取出「半導體」成員 ===
chain = pd.read_csv(os.path.join(ROOT, "industry_chain.csv"), dtype=str)
semi_stocks = chain[chain["industry_code"]=="D000"]["stock_id"].unique().tolist()
semi_stocks = [s for s in semi_stocks if s in daily_ret.columns]
print(f"\nStep 3: D000 半導體成員: 共 {len(semi_stocks)} 檔, 前 10 = {semi_stocks[:10]}")

# === Step 4: 半導體產業日報酬 = 該日所有半導體股報酬等權平均 ===
r_semi = daily_ret[semi_stocks].mean(axis=1)
print("\nStep 4: r_半導體 = 每日等權平均")
print(r_semi.tail(5))

# === Step 5: EMA(20) 平滑 ===
N = 20
ema = r_semi.ewm(span=N, adjust=False).mean()
print(f"\nStep 5: EMA({N})  α = 2/(N+1) = {2/(N+1):.4f}")
print("  公式: EMA_t = α·r_t + (1-α)·EMA_{t-1}")
print("  最後 5 日:")
print(pd.DataFrame({"r_半導體": r_semi, "EMA20": ema}).tail(5))

# === Step 6: lag 1 + sign rule ===
signal = (ema > 0).astype(int).shift(1)
print(f"\nStep 6: 隔日訊號 = 1 if EMA20(t-1) > 0 else 0")
print(pd.DataFrame({
    "r_半導體": r_semi,
    "EMA20": ema,
    "EMA20_lag1": ema.shift(1),
    "position(0/1)": signal
}).tail(10))

# === Step 7: 用部位 × 7 下游 EW 報酬 ===
DOWN = ["G000","H000","I000","J000","F000","L000","5400"]
down_ret = {}
for ind in DOWN:
    members = chain[chain["industry_code"]==ind]["stock_id"].unique().tolist()
    members = [s for s in members if s in daily_ret.columns]
    down_ret[ind] = daily_ret[members].mean(axis=1)
down_ret = pd.DataFrame(down_ret)
down_ew = down_ret.mean(axis=1)
print("\nStep 7: 7 下游組合 (各自 EW 後再 7 個等權)")
print("  最後 5 日 7 條鏈報酬:")
print(down_ret.tail(5))
print("\n  最後 5 日 7 鏈等權 down_ew:")
print(down_ew.tail(5))

# === 完整 timeline 示範一個月 ===
print("\n=== 一個月 timeline (2024-06) ===")
demo = pd.DataFrame({
    "r_半導體": r_semi,
    "EMA20": ema,
    "signal_lag1(0/1)": signal,
    "down_ew_today": down_ew,
    "strat_ret": signal * down_ew,
}).loc["2024-06-01":"2024-06-30"]
print(demo.round(5).to_string())
