"""畫 EMA(20) 訊號 17 年圖, 綠=部位 on (long 7下游), 紅=部位 off (現金)"""
import os, pickle, sys
import numpy as np, pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
for f in ["Microsoft JhengHei", "Microsoft YaHei", "Noto Sans CJK TC", "SimHei"]:
    if any(f in fn.name for fn in font_manager.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [f]; break
plt.rcParams["axes.unicode_minus"] = False

sys.stdout.reconfigure(encoding="utf-8")
DB = r"C:\Users\user\finlab_db"
ROOT = r"C:\Users\user\OneDrive\桌面\產業因子生成"
OUT = os.path.join(ROOT, "research", "output")

with open(os.path.join(DB, "etl#adj_close.pickle"), "rb") as f:
    adj_close = pickle.load(f).set_index("date")
adj_close.index = pd.to_datetime(adj_close.index)
adj_close.columns = adj_close.columns.astype(str)
keep = [c for c in adj_close.columns if c.isdigit() and len(c)==4 and not c.startswith("00")]
adj_close = adj_close[keep]
daily_ret = adj_close.pct_change(fill_method=None).iloc[1:]

chain = pd.read_csv(os.path.join(ROOT, "industry_chain.csv"), dtype=str)
semi = chain[chain["industry_code"]=="D000"]["stock_id"].unique().tolist()
semi = [s for s in semi if s in daily_ret.columns]
r_semi = daily_ret[semi].mean(axis=1)

ema = r_semi.ewm(span=20, adjust=False).mean()
position = (ema.shift(1) > 0).astype(int)

DOWN = ["G000","H000","I000","J000","F000","L000","5400"]
down_panels = []
for ind in DOWN:
    members = chain[chain["industry_code"]==ind]["stock_id"].unique().tolist()
    members = [s for s in members if s in daily_ret.columns]
    down_panels.append(daily_ret[members].mean(axis=1).rename(ind))
down_ew = pd.concat(down_panels, axis=1).mean(axis=1)
market = daily_ret.mean(axis=1)

# Strategy A net@50bp
gross = position * down_ew
turnover = position.diff().abs().fillna(position.iloc[0])
net_50 = gross - turnover * 0.0025  # one-side cost

# 累積曲線
cum_strat = (1 + net_50.fillna(0)).cumprod()
cum_down = (1 + down_ew.fillna(0)).cumprod()
cum_mkt = (1 + market.fillna(0)).cumprod()
cum_semi = (1 + r_semi.fillna(0)).cumprod()

# === 圖 ===
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8.5), sharex=True,
                               gridspec_kw={"height_ratios": [1, 1.4]})

# Panel 1: EMA(20) + 訊號背景
on_mask = position == 1
off_mask = position == 0
# 在 ax1 用 axvspan 標出 on/off 段
# 找出 position 變動的點
pos_diff = position.diff().fillna(0)
flips = pos_diff[pos_diff != 0].index
segs = []
prev = position.index[0]
prev_state = position.iloc[0]
for f in flips:
    segs.append((prev, f, prev_state))
    prev = f
    prev_state = position.loc[f]
segs.append((prev, position.index[-1], prev_state))

for s, e, st in segs:
    color = "#a5d6a7" if st == 1 else "#ef9a9a"
    ax1.axvspan(s, e, alpha=0.30, color=color, zorder=0)

ax1.plot(ema.index, ema.values, color="#0d47a1", lw=1.0, label="EMA(20) 訊號", zorder=2)
ax1.axhline(0, color="black", lw=0.6, ls="-", alpha=0.7)
ax1.set_ylabel("EMA(20) of r_半導體")
ax1.set_title("EMA(20) 訊號 17 年時序圖（綠 = 部位 ON，紅 = 部位 OFF）")
ax1.legend(loc="upper left")
ax1.grid(alpha=0.3, axis="y")

# 統計訊號翻面次數與在場比例
n_flips = (position.diff().abs() == 1).sum()
on_share = on_mask.mean()
ax1.text(0.99, 0.05, f"在場比例 {on_share*100:.1f}% | 訊號翻面次數 {n_flips}",
         transform=ax1.transAxes, ha="right", va="bottom",
         bbox=dict(facecolor="white", alpha=0.85, edgecolor="gray"))

# Panel 2: 累積曲線
for s, e, st in segs:
    color = "#a5d6a7" if st == 1 else "#ef9a9a"
    ax2.axvspan(s, e, alpha=0.20, color=color, zorder=0)

ax2.plot(cum_strat, color="#1b5e20", lw=1.7, label=f"策略 A net@50bp (cum {cum_strat.iloc[-1]:.1f}x)", zorder=3)
ax2.plot(cum_down, color="#ff6f00", lw=1.3, alpha=0.85, label=f"7 下游 Buy & Hold (cum {cum_down.iloc[-1]:.1f}x)", zorder=2)
ax2.plot(cum_semi, color="#6a1b9a", lw=1.2, alpha=0.7, label=f"半導體 Buy & Hold (cum {cum_semi.iloc[-1]:.1f}x)", zorder=2)
ax2.plot(cum_mkt, color="#37474f", lw=1.2, alpha=0.6, label=f"Market EW (cum {cum_mkt.iloc[-1]:.1f}x)", zorder=2)
ax2.set_yscale("log")
ax2.set_ylabel("累積資本 (log scale, 起點 = 1)")
ax2.set_xlabel("date")
ax2.set_title("累積報酬：策略 A vs Buy & Hold")
ax2.legend(loc="upper left")
ax2.grid(alpha=0.3)

plt.tight_layout()
out_path = os.path.join(OUT, "fig_signal_timeline.png")
plt.savefig(out_path, dpi=130)
plt.close()
print(f"圖檔: {out_path}")
print(f"在場天數 / 總天數 = {on_mask.sum()}/{len(position)} = {on_share*100:.1f}%")
print(f"訊號翻面次數 = {n_flips}, 平均 {n_flips/len(position)*252:.1f} 次/年")
print(f"策略 A net@50bp 累積 = {cum_strat.iloc[-1]:.2f}x; 7 下游 BH = {cum_down.iloc[-1]:.2f}x; Market = {cum_mkt.iloc[-1]:.2f}x")
