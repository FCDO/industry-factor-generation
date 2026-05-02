"""
Phase 2: Menzly-Ozbas (2010) 風格預測迴歸

對每個產業鏈 c, 每對位置 (src, dst), 在頻率 freq:
    r_dst_c(t+1) = α + β·r_src_c(t) + γ·r_dst_c(t) + δ·r_market(t) + ε

H0: β = 0 (無 spillover); H1: β > 0 (上游報酬領先下游)
標準誤: Newey-West (HAC), lag = 規則自動

對照論文: Menzly & Ozbas (2010), JF, "Market Segmentation and
Cross-predictability of Returns"。他們用 BEA I-O 表，我們改用 TPEx 鏈內 上中下游。

頻率: 日 (lag 5)、週 (lag 4)、月 (lag 3)。
報酬版本: EW (主) 與 VW (穩健性).

輸出:
- regression_results.csv: 每個 (chain, freq, src→dst, weight) 的 β, t-stat, R², N
- regression_summary.txt: 各方向、各頻率的顯著比例與 mean coef
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = r"C:\Users\user\OneDrive\桌面\產業因子生成"
OUT = os.path.join(ROOT, "research", "output")

POSITIONS = ["上游", "中游", "下游"]
PAIRS = [
    ("上游", "下游"),
    ("下游", "上游"),
    ("上游", "中游"),
    ("中游", "上游"),
    ("中游", "下游"),
    ("下游", "中游"),
]

FREQ_PARAMS = {
    # freq_label: (resample_rule, NW lag, min_obs)
    "D": (None, 5, 200),
    "W": ("W-FRI", 4, 60),
    "M": ("ME", 3, 36),
}


def load_panels():
    panel = pd.read_pickle(os.path.join(OUT, "phase_returns.pkl"))
    market = pd.read_pickle(os.path.join(OUT, "market_return.pkl"))
    return panel, market


def to_wide(panel: pd.DataFrame, value: str = "ret_ew") -> pd.DataFrame:
    """轉為 wide (date × (industry_code, position))"""
    w = panel.pivot_table(index="date", columns=["industry_code", "position"], values=value)
    return w


def resample_log(returns: pd.DataFrame, rule: str | None) -> pd.DataFrame:
    """以對數累積後再轉回簡單報酬，避免簡單平均偏誤。"""
    if rule is None:
        return returns
    log = np.log1p(returns)
    agg = log.resample(rule).sum(min_count=1)
    out = np.expm1(agg)
    return out


def newey_west_reg(y: pd.Series, X: pd.DataFrame, nw_lag: int):
    """OLS + Newey-West HAC. 回傳 result。"""
    df = pd.concat([y, X], axis=1).dropna()
    if len(df) < max(nw_lag * 3, 30):
        return None, len(df)
    Y = df.iloc[:, 0]
    Xv = sm.add_constant(df.iloc[:, 1:])
    res = sm.OLS(Y, Xv).fit(cov_type="HAC", cov_kwds={"maxlags": nw_lag})
    return res, len(df)


def run_one(
    wide: pd.DataFrame,
    market: pd.Series,
    industry_code: str,
    src: str,
    dst: str,
    freq_label: str,
):
    """單一 (chain, src→dst, freq) 的預測迴歸。回傳 dict 或 None。"""
    rule, nw_lag, min_obs = FREQ_PARAMS[freq_label]

    if (industry_code, src) not in wide.columns or (industry_code, dst) not in wide.columns:
        return None

    r_src = wide[(industry_code, src)]
    r_dst = wide[(industry_code, dst)]
    r_mkt = market

    if rule is not None:
        r_src = resample_log(r_src.to_frame("x"), rule)["x"]
        r_dst = resample_log(r_dst.to_frame("x"), rule)["x"]
        r_mkt = resample_log(r_mkt.to_frame("x"), rule)["x"]

    # 預測式: r_dst(t) = α + β·r_src(t-1) + γ·r_dst(t-1) + δ·r_mkt(t-1) + ε
    df = pd.DataFrame({
        "y": r_dst,
        "x_src_lag1": r_src.shift(1),
        "x_dst_lag1": r_dst.shift(1),
        "x_mkt_lag1": r_mkt.shift(1),
    }).dropna()

    if len(df) < min_obs:
        return None

    res, n = newey_west_reg(df["y"], df[["x_src_lag1", "x_dst_lag1", "x_mkt_lag1"]], nw_lag)
    if res is None:
        return None

    return {
        "industry_code": industry_code,
        "src": src,
        "dst": dst,
        "freq": freq_label,
        "n": n,
        "beta": res.params["x_src_lag1"],
        "t_beta": res.tvalues["x_src_lag1"],
        "p_beta": res.pvalues["x_src_lag1"],
        "gamma_dst_ar1": res.params["x_dst_lag1"],
        "t_gamma": res.tvalues["x_dst_lag1"],
        "delta_mkt": res.params["x_mkt_lag1"],
        "t_delta": res.tvalues["x_mkt_lag1"],
        "alpha": res.params["const"],
        "r2": res.rsquared,
        "r2_adj": res.rsquared_adj,
    }


def main():
    print("[Phase 2] Menzly-Ozbas 風格預測迴歸\n")
    panel, market_df = load_panels()
    market_ret = market_df["ret_market"]
    market_ret.index = pd.to_datetime(market_ret.index)
    print(f"  panel: {panel.shape},  market: {market_ret.shape}")

    industries = panel[["industry_code", "industry_name"]].drop_duplicates().set_index("industry_code")["industry_name"].to_dict()
    print(f"  涵蓋 {len(industries)} 個產業鏈")

    all_rows = []
    for weight_label, value_col in [("EW", "ret_ew"), ("VW", "ret_vw")]:
        wide = to_wide(panel, value_col)
        # 確保 datetime index
        wide.index = pd.to_datetime(wide.index)
        for ind_code, ind_name in industries.items():
            for src, dst in PAIRS:
                for freq in FREQ_PARAMS:
                    r = run_one(wide, market_ret, ind_code, src, dst, freq)
                    if r is None:
                        continue
                    r["industry_name"] = ind_name
                    r["weight"] = weight_label
                    all_rows.append(r)

    res_df = pd.DataFrame(all_rows)
    res_df = res_df[[
        "industry_code", "industry_name", "weight", "freq",
        "src", "dst", "n",
        "beta", "t_beta", "p_beta",
        "gamma_dst_ar1", "t_gamma",
        "delta_mkt", "t_delta",
        "alpha", "r2", "r2_adj",
    ]]
    out_csv = os.path.join(OUT, "regression_results.csv")
    res_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\n  迴歸結果寫入 {out_csv}  ({len(res_df)} 列)")

    # === 摘要 ===
    print("\n=== 摘要：β (src→dst spillover) 顯著比例與平均值 ===")
    summary_lines = []

    def grp_summary(d, label):
        if len(d) == 0:
            return None
        sig5 = (d["p_beta"] < 0.05).sum()
        sig10 = (d["p_beta"] < 0.10).sum()
        pos = (d["beta"] > 0).sum()
        return {
            "scope": label,
            "n_chains": len(d),
            "mean_beta": d["beta"].mean(),
            "median_beta": d["beta"].median(),
            "mean_t": d["t_beta"].mean(),
            "share_pos": pos / len(d),
            "share_sig5_2sided": sig5 / len(d),
            "share_sig10_2sided": sig10 / len(d),
            "share_sig5_pos": ((d["p_beta"] < 0.05) & (d["beta"] > 0)).sum() / len(d),
            "share_sig5_neg": ((d["p_beta"] < 0.05) & (d["beta"] < 0)).sum() / len(d),
        }

    rows = []
    for w in ["EW", "VW"]:
        for freq in ["D", "W", "M"]:
            for src, dst in PAIRS:
                d = res_df[(res_df["weight"] == w) & (res_df["freq"] == freq)
                           & (res_df["src"] == src) & (res_df["dst"] == dst)]
                s = grp_summary(d, f"{w}-{freq} {src}->{dst}")
                if s:
                    rows.append(s)

    sm_df = pd.DataFrame(rows)
    print(sm_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    sm_df.to_csv(os.path.join(OUT, "regression_summary.csv"), index=False, encoding="utf-8-sig")

    # === Fama-MacBeth 風格：每個 (freq, weight, pair) 的截面 t-stat of mean beta ===
    print("\n=== 截面 mean β 的 t-stat (Fama-MacBeth 風) ===")
    print("    H0: 跨鏈 mean β = 0")
    fm_rows = []
    for w in ["EW", "VW"]:
        for freq in ["D", "W", "M"]:
            for src, dst in PAIRS:
                d = res_df[(res_df["weight"] == w) & (res_df["freq"] == freq)
                           & (res_df["src"] == src) & (res_df["dst"] == dst)]
                if len(d) < 5:
                    continue
                betas = d["beta"].values
                mean_b = betas.mean()
                se = betas.std(ddof=1) / np.sqrt(len(betas))
                t = mean_b / se if se > 0 else np.nan
                from scipy import stats
                p = 2 * (1 - stats.t.cdf(abs(t), df=len(betas)-1)) if not np.isnan(t) else np.nan
                fm_rows.append({
                    "weight": w, "freq": freq, "pair": f"{src}->{dst}",
                    "n_chains": len(d),
                    "mean_beta": mean_b, "t_cs": t, "p_cs": p,
                })
    fm_df = pd.DataFrame(fm_rows)
    print(fm_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    fm_df.to_csv(os.path.join(OUT, "regression_fm_summary.csv"), index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
