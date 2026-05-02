"""
Phase 3: Granger 雙向因果檢定 (Hong-Torous-Valkanov 風格)

對每個產業鏈 c, 對 (src, dst) ∈ {(上,下), (上,中), (中,下) ...}:
    H0: r_src 對 r_dst 無 Granger 因果

統計量: F 檢定 (受限 vs 非受限)
頻率: 日 (lag 5), 週 (lag 4), 月 (lag 3)

對照: Hong, Torous & Valkanov (2007), JFE。他們發現 14/34 美股產業
領先大盤 1-2 月。我們改檢驗鏈內 上↔中↔下 的領先關係。
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests

warnings.filterwarnings("ignore")

ROOT = r"C:\Users\user\OneDrive\桌面\產業因子生成"
OUT = os.path.join(ROOT, "research", "output")

PAIRS = [
    ("上游", "下游"),
    ("下游", "上游"),
    ("上游", "中游"),
    ("中游", "上游"),
    ("中游", "下游"),
    ("下游", "中游"),
]

FREQ_PARAMS = {
    "D": (None, [1, 2, 3, 5], 200),
    "W": ("W-FRI", [1, 2, 4], 60),
    "M": ("ME", [1, 2, 3], 36),
}


def load_panels():
    panel = pd.read_pickle(os.path.join(OUT, "phase_returns.pkl"))
    return panel


def to_wide(panel: pd.DataFrame, value: str = "ret_ew") -> pd.DataFrame:
    w = panel.pivot_table(index="date", columns=["industry_code", "position"], values=value)
    w.index = pd.to_datetime(w.index)
    return w


def resample_log(s: pd.Series, rule: str | None) -> pd.Series:
    if rule is None:
        return s
    log = np.log1p(s)
    return np.expm1(log.resample(rule).sum(min_count=1))


def granger_one(r_src: pd.Series, r_dst: pd.Series, lags: list[int], min_obs: int):
    """檢驗 r_src 是否 Granger-cause r_dst。

    statsmodels grangercausalitytests 第二欄 → 第一欄: x2 → x1
    我們要檢驗 src → dst, 故 data = (dst, src)
    """
    df = pd.concat({"dst": r_dst, "src": r_src}, axis=1).dropna()
    if len(df) < min_obs:
        return None

    arr = df[["dst", "src"]].values
    out = {"n": len(df)}
    for L in lags:
        try:
            res = grangercausalitytests(arr, maxlag=L, verbose=False)
            f_stat = res[L][0]["ssr_ftest"][0]
            f_p = res[L][0]["ssr_ftest"][1]
            out[f"F_lag{L}"] = f_stat
            out[f"p_lag{L}"] = f_p
        except Exception:
            out[f"F_lag{L}"] = np.nan
            out[f"p_lag{L}"] = np.nan
    return out


def main():
    print("[Phase 3] Granger 雙向因果檢定\n")
    panel = load_panels()

    industries = panel[["industry_code", "industry_name"]].drop_duplicates().set_index("industry_code")["industry_name"].to_dict()

    rows = []
    for weight_label, value_col in [("EW", "ret_ew"), ("VW", "ret_vw")]:
        wide = to_wide(panel, value_col)
        for ind_code, ind_name in industries.items():
            for src, dst in PAIRS:
                if (ind_code, src) not in wide.columns or (ind_code, dst) not in wide.columns:
                    continue
                r_src_d = wide[(ind_code, src)]
                r_dst_d = wide[(ind_code, dst)]
                for freq, (rule, lags, min_obs) in FREQ_PARAMS.items():
                    r_src = resample_log(r_src_d, rule)
                    r_dst = resample_log(r_dst_d, rule)
                    out = granger_one(r_src, r_dst, lags, min_obs)
                    if out is None:
                        continue
                    out.update({
                        "weight": weight_label, "industry_code": ind_code,
                        "industry_name": ind_name, "src": src, "dst": dst, "freq": freq,
                        "lags": ",".join(str(l) for l in lags),
                    })
                    rows.append(out)

    res_df = pd.DataFrame(rows)
    out_csv = os.path.join(OUT, "granger_results.csv")
    res_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"  Granger 結果寫入 {out_csv}  ({len(res_df)} 列)")

    # 摘要：每個 (freq, weight, pair) 的顯著比例
    print("\n=== Granger 摘要 (各 lag 的顯著鏈比例 @ p<0.05) ===")
    sig_rows = []
    for w in ["EW", "VW"]:
        for freq in ["D", "W", "M"]:
            lags_list = [int(l) for l in FREQ_PARAMS[freq][1]]
            for src, dst in PAIRS:
                d = res_df[(res_df["weight"] == w) & (res_df["freq"] == freq)
                           & (res_df["src"] == src) & (res_df["dst"] == dst)]
                if len(d) == 0:
                    continue
                row = {"weight": w, "freq": freq, "pair": f"{src}->{dst}", "n_chains": len(d)}
                for L in lags_list:
                    pcol = f"p_lag{L}"
                    if pcol in d.columns:
                        row[f"sig5_lag{L}"] = (d[pcol] < 0.05).sum()
                        row[f"share_lag{L}"] = (d[pcol] < 0.05).mean()
                sig_rows.append(row)
    sig_df = pd.DataFrame(sig_rows)
    sig_df.to_csv(os.path.join(OUT, "granger_summary.csv"), index=False, encoding="utf-8-sig")
    print(sig_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # 強顯著鏈：日頻 EW 上→下、上→中
    print("\n=== EW 日頻 上→下 (Granger F, lag5) — 各鏈 ===")
    sub = res_df[(res_df["weight"] == "EW") & (res_df["freq"] == "D")
                 & (res_df["src"] == "上游") & (res_df["dst"] == "下游")].copy()
    if "p_lag5" in sub.columns:
        sub = sub.sort_values("p_lag5")
        print(sub[["industry_code", "industry_name", "F_lag1", "p_lag1", "F_lag5", "p_lag5", "n"]]
              .to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n=== EW 月頻 上→下 (Granger F, lag1/2/3) — 各鏈 ===")
    sub = res_df[(res_df["weight"] == "EW") & (res_df["freq"] == "M")
                 & (res_df["src"] == "上游") & (res_df["dst"] == "下游")].copy()
    sub = sub.sort_values("p_lag1")
    print(sub[["industry_code", "industry_name", "F_lag1", "p_lag1", "F_lag2", "p_lag2", "F_lag3", "p_lag3", "n"]]
          .to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
