"""
Phase 15: G000 平面顯示器 / P000 電機機械 源頭辨認 (Phase 10 風格)

動機 (Phase 11a)
- G000 平面顯示器 t=4.96 (alpha 14.5%)
- P000 電機機械 t=4.74 (alpha 12.3%)
皆超越 D000 半導體 t=4.31 (alpha 12.4%); 但 11a 僅用「整體鏈上游」做訊號, 未確認真實源頭.

Phase 11b 鏈內 sub_code 結果:
- G000 → GA00 (其他零組件) 為純 leader
- P000 → P200 (傳動元件) + P600 (沖壓零組件) 為純 leader

本 Phase (Phase 10 風格) 進一步問:
1. Pairwise 預測力: leader sub_code 是否 dominate 整體鏈?
2. Joint reg: target_{t+1} ~ leader_sub + 整體鏈 + mkt — 誰存活?
3. Orthogonalization: leader ⊥ 整體鏈, 殘差是否仍有預測力?
4. 個股 macro signal: 各鏈是否有「自己的 2330」? 候選:
   - G000: 友達 2409 / 群創 3481 (面板雙雄)
   - P000: 上銀 2049 / 亞德客-KY 1590 (傳動 / 氣動龍頭)

Targets (各鏈分開測):
- 鏈內: 各鏈自己的中下游 EW
- 跨產業: 7 跨產業下游 EW (Phase 8/13/14 框架)

輸出
- phase15_joint_reg.csv : 各 (chain, target) 的 pairwise + joint + orth 結果
- phase15_macro_stocks.csv : 個股 macro signal 結果
- fig_phase15_chain_sources.png : 各鏈 leader vs 整體鏈 t-stat 比較
"""
from __future__ import annotations

import os
import pickle
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
for f in ["Microsoft JhengHei", "Microsoft YaHei", "Noto Sans CJK TC", "SimHei"]:
    if any(f in fn.name for fn in font_manager.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [f]
        break
plt.rcParams["axes.unicode_minus"] = False

ROOT = r"C:\Users\engli\OneDrive\桌面\產業因子生成"
DB = r"C:\Users\engli\finlab_db"
OUT = os.path.join(ROOT, "research", "output")

DOWNSTREAM = ["G000", "H000", "I000", "J000", "F000", "L000", "5400"]


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


def reg_simple(y, x, market, nw=5):
    df = pd.concat({"y": y, "x": x.shift(1), "y_lag": y.shift(1),
                    "m_lag": market.shift(1)}, axis=1).dropna()
    if len(df) < 200:
        return None
    res = sm.OLS(df["y"], sm.add_constant(df[["x", "y_lag", "m_lag"]]))\
            .fit(cov_type="HAC", cov_kwds={"maxlags": nw})
    return {"n": len(df), "beta": res.params["x"],
            "t": res.tvalues["x"], "p": res.pvalues["x"], "r2": res.rsquared}


def reg_joint(y, x1, x2, market, nw=5):
    """y_{t+1} ~ x1_t + x2_t + y_t + mkt_t."""
    df = pd.concat({"y": y, "x1": x1.shift(1), "x2": x2.shift(1),
                    "y_lag": y.shift(1), "m_lag": market.shift(1)}, axis=1).dropna()
    if len(df) < 200:
        return None
    res = sm.OLS(df["y"], sm.add_constant(df[["x1", "x2", "y_lag", "m_lag"]]))\
            .fit(cov_type="HAC", cov_kwds={"maxlags": nw})
    return {
        "n": len(df),
        "b1": res.params["x1"], "t1": res.tvalues["x1"], "p1": res.pvalues["x1"],
        "b2": res.params["x2"], "t2": res.tvalues["x2"], "p2": res.pvalues["x2"],
        "r2": res.rsquared,
    }


def verdict_label(t1, t2):
    if t1 > 1.96 and t2 < 1.96 and t2 > -1.96:
        return "✓ leader 主導"
    if t2 > 1.96 and t1 < 1.96 and t1 > -1.96:
        return "整體鏈主導"
    if t1 > 1.96 and t2 > 1.96:
        return "雙顯著"
    if t1 < -1.96 or t2 < -1.96:
        return "(負向)"
    return "皆不顯著"


def main():
    print("=" * 100)
    print("Phase 15: G000 / P000 源頭辨認 (Phase 10 風格 joint regression + 個股 macro)")
    print("=" * 100)

    chain = pd.read_csv(os.path.join(ROOT, "industry_chain.csv"), dtype=str)
    adj_close = filter_common(load_pickle_df("etl#adj_close.pickle"))
    daily_ret = adj_close.pct_change(fill_method=None).iloc[1:]
    market = daily_ret.mean(axis=1)

    # === Build chain panels ===
    def build_subcode(industry, sub):
        members = chain[(chain["industry_code"] == industry) &
                         (chain["sub_code"] == sub)]["stock_id"].unique().tolist()
        cols = [s for s in members if s in daily_ret.columns]
        return daily_ret[cols].mean(axis=1), len(cols)

    def build_chain_overall(industry):
        members = chain[chain["industry_code"] == industry]["stock_id"].unique().tolist()
        cols = [s for s in members if s in daily_ret.columns]
        return daily_ret[cols].mean(axis=1), len(cols)

    def build_chain_middown(industry):
        d = chain[chain["industry_code"] == industry]
        mid = d[d["position"] == "中游"]["stock_id"].unique()
        dn = d[d["position"] == "下游"]["stock_id"].unique()
        members = list(set(list(mid) + list(dn)) & set(daily_ret.columns))
        return daily_ret[members].mean(axis=1), len(members)

    # === G000 setup ===
    print("\n" + "─" * 100)
    print("G000 平面顯示器")
    print("─" * 100)
    g_overall, g_n = build_chain_overall("G000")
    g_md, g_md_n = build_chain_middown("G000")
    g_a, g_a_n = build_subcode("G000", "GA00")
    print(f"  整體 G000: {g_n} 檔  |  G000 中下游: {g_md_n} 檔  |  GA00 其他零組件: {g_a_n} 檔")

    # 候選個股
    g_macro_stocks = {"2409 友達": "2409", "3481 群創": "3481"}
    g_macro = {label: daily_ret[sid] for label, sid in g_macro_stocks.items()
               if sid in daily_ret.columns}
    print(f"  G000 個股 macro 候選: {list(g_macro.keys())}")

    # === P000 setup ===
    print("\n" + "─" * 100)
    print("P000 電機機械")
    print("─" * 100)
    p_overall, p_n = build_chain_overall("P000")
    p_md, p_md_n = build_chain_middown("P000")
    p_200, p_200_n = build_subcode("P000", "P200")
    p_600, p_600_n = build_subcode("P000", "P600")
    p_lead = pd.concat([p_200, p_600], axis=1).mean(axis=1)
    print(f"  整體 P000: {p_n} 檔  |  P000 中下游: {p_md_n} 檔  |  P200 傳動: {p_200_n} 檔  |  P600 沖壓: {p_600_n} 檔")

    p_macro_stocks = {"2049 上銀": "2049", "1590 亞德客-KY": "1590"}
    p_macro = {label: daily_ret[sid] for label, sid in p_macro_stocks.items()
               if sid in daily_ret.columns}
    print(f"  P000 個股 macro 候選: {list(p_macro.keys())}")

    # === Targets ===
    industry_panel = {}
    for ic in chain["industry_code"].unique():
        members = chain[chain["industry_code"] == ic]["stock_id"].unique().tolist()
        cols = [s for s in members if s in daily_ret.columns]
        if len(cols) >= 3:
            industry_panel[ic] = daily_ret[cols].mean(axis=1)
    industry_panel = pd.DataFrame(industry_panel)
    target_xind = industry_panel[DOWNSTREAM].mean(axis=1)

    # ==========================================================================
    # 部分 1: 鏈內 — Pairwise + Joint + Orth
    # ==========================================================================
    print("\n" + "=" * 100)
    print("【1】Pairwise 預測力 (target = 各鏈中下游 / xind)")
    print("=" * 100)

    cases = [
        # (chain, leader_label, leader_series, overall_label, overall_series, target_label, target)
        ("G000", "GA00",      g_a,     "整體G000", g_overall, "G000中下游", g_md),
        ("G000", "GA00",      g_a,     "整體G000", g_overall, "xind",      target_xind),
        ("P000", "P200+P600", p_lead,  "整體P000", p_overall, "P000中下游", p_md),
        ("P000", "P200",      p_200,   "整體P000", p_overall, "P000中下游", p_md),
        ("P000", "P600",      p_600,   "整體P000", p_overall, "P000中下游", p_md),
        ("P000", "P200+P600", p_lead,  "整體P000", p_overall, "xind",      target_xind),
    ]
    print(f"\n{'Chain':<6s} {'Target':<14s} {'Leader':<14s} {'Overall':<10s} "
          f"{'Lead β/t':>14s} {'Over β/t':>14s} {'Joint t1/t2':>16s} {'Verdict':>14s}")
    rows = []
    for chain_code, lead_label, lead_ser, ov_label, ov_ser, tgt_label, tgt in cases:
        r1 = reg_simple(tgt, lead_ser, market)
        r2 = reg_simple(tgt, ov_ser, market)
        rj = reg_joint(tgt, lead_ser, ov_ser, market)
        if r1 is None or r2 is None or rj is None:
            continue
        v = verdict_label(rj["t1"], rj["t2"])
        s1 = "*" if abs(r1["t"]) > 1.96 else " "
        s2 = "*" if abs(r2["t"]) > 1.96 else " "
        print(f"{chain_code:<6s} {tgt_label:<14s} {lead_label:<14s} {ov_label:<10s}  "
              f"{r1['beta']:>+.3f}/{r1['t']:>+.2f}{s1} "
              f"{r2['beta']:>+.3f}/{r2['t']:>+.2f}{s2} "
              f"{rj['t1']:>+.2f}/{rj['t2']:>+.2f}  {v}")
        rows.append({"chain": chain_code, "target": tgt_label,
                     "leader": lead_label, "overall": ov_label,
                     "lead_beta": r1["beta"], "lead_t": r1["t"], "lead_p": r1["p"],
                     "over_beta": r2["beta"], "over_t": r2["t"], "over_p": r2["p"],
                     "joint_b1": rj["b1"], "joint_t1": rj["t1"], "joint_p1": rj["p1"],
                     "joint_b2": rj["b2"], "joint_t2": rj["t2"], "joint_p2": rj["p2"],
                     "verdict": v})
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "phase15_joint_reg.csv"),
                              index=False, encoding="utf-8-sig")

    # ==========================================================================
    # 部分 2: Orthogonalization
    # ==========================================================================
    print("\n" + "=" * 100)
    print("【2】Orthogonalization — leader ⊥ 整體鏈, 殘差是否仍預測 target?")
    print("=" * 100)

    def orth_test(leader, overall, tgt, label):
        df = pd.concat({"L": leader, "O": overall, "m": market}, axis=1).dropna()
        res = sm.OLS(df["L"], sm.add_constant(df[["O", "m"]])).fit()
        L_resid = (df["L"] - res.predict(sm.add_constant(df[["O", "m"]]))).reindex(daily_ret.index)
        var_ratio = L_resid.var() / leader.var()
        r = reg_simple(tgt, L_resid, market)
        if r is None:
            return None
        print(f"  {label}: idiosync_var={var_ratio:.3f}  resid β/t = {r['beta']:+.4f}/{r['t']:+.3f}  "
              f"(* if |t|>1.96)  R²(L on O,mkt)={res.rsquared:.3f}")
        return {"label": label, "var_ratio": var_ratio, "resid_beta": r["beta"],
                "resid_t": r["t"], "resid_p": r["p"]}

    print(f"\n  G000 — predicting G000 中下游:")
    orth_test(g_a, g_overall, g_md, "GA00 ⊥ 整體G000")
    print(f"\n  G000 — predicting xind:")
    orth_test(g_a, g_overall, target_xind, "GA00 ⊥ 整體G000")
    print(f"\n  P000 — predicting P000 中下游:")
    orth_test(p_lead, p_overall, p_md, "P200+P600 ⊥ 整體P000")
    orth_test(p_200, p_overall, p_md, "P200 ⊥ 整體P000")
    orth_test(p_600, p_overall, p_md, "P600 ⊥ 整體P000")
    print(f"\n  P000 — predicting xind:")
    orth_test(p_lead, p_overall, target_xind, "P200+P600 ⊥ 整體P000")

    # ==========================================================================
    # 部分 3: 個股 macro signal — 類似 2330 之於 D100
    # ==========================================================================
    print("\n" + "=" * 100)
    print("【3】個股 macro signal — 是否各鏈有自己的「2330」?")
    print("=" * 100)

    macro_rows = []

    def stock_macro_test(label, stock_ret, leader_ser, target_label, tgt):
        # Pairwise 預測 target
        r_alone = reg_simple(tgt, stock_ret, market)
        # Joint with leader
        rj = reg_joint(tgt, stock_ret, leader_ser, market)
        if r_alone is None or rj is None:
            return
        # ρ with leader
        rho = pd.concat([stock_ret, leader_ser], axis=1).dropna().corr().iloc[0, 1]
        v = verdict_label(rj["t1"], rj["t2"])
        print(f"  {label:<22s} → {target_label:<14s}  "
              f"alone β/t={r_alone['beta']:+.4f}/{r_alone['t']:+.2f}  "
              f"ρ(stock, leader)={rho:.3f}  "
              f"joint t1(stock)={rj['t1']:+.2f}, t2(leader)={rj['t2']:+.2f}  {v}")
        macro_rows.append({"stock": label, "target": target_label,
                           "alone_beta": r_alone["beta"], "alone_t": r_alone["t"],
                           "rho_with_leader": rho,
                           "joint_t_stock": rj["t1"], "joint_t_leader": rj["t2"],
                           "verdict": v})

    print(f"\n  G000 (vs GA00):")
    for label, stock in g_macro.items():
        stock_macro_test(label, stock, g_a, "G000中下游", g_md)
        stock_macro_test(label, stock, g_a, "xind", target_xind)
    print(f"\n  P000 (vs P200+P600):")
    for label, stock in p_macro.items():
        stock_macro_test(label, stock, p_lead, "P000中下游", p_md)
        stock_macro_test(label, stock, p_lead, "xind", target_xind)

    # 對照: 2330 vs D100 (Phase 10 結論的 reference)
    print(f"\n  [REFERENCE] D100 vs 2330:")
    d100_members = chain[(chain["industry_code"] == "D000") &
                          (chain["sub_code"] == "D100")]["stock_id"].unique().tolist()
    d100_cols = [s for s in d100_members if s in daily_ret.columns]
    d100 = daily_ret[d100_cols].mean(axis=1)
    d_md = chain[(chain["industry_code"] == "D000") &
                  (chain["position"].isin(["中游", "下游"]))]["stock_id"].unique()
    d_md_cols = [s for s in d_md if s in daily_ret.columns]
    d_md_ret = daily_ret[d_md_cols].mean(axis=1)
    if "2330" in daily_ret.columns:
        stock_macro_test("2330 台積電", daily_ret["2330"], d100, "D000中下游", d_md_ret)
        stock_macro_test("2330 台積電", daily_ret["2330"], d100, "xind", target_xind)

    pd.DataFrame(macro_rows).to_csv(os.path.join(OUT, "phase15_macro_stocks.csv"),
                                     index=False, encoding="utf-8-sig")

    # ==========================================================================
    # Visualization: chain leader vs overall t-stat
    # ==========================================================================
    plot_data = pd.DataFrame(rows)
    plot_data = plot_data[plot_data["target"].isin(["G000中下游", "P000中下游"])]
    plot_data["pair"] = plot_data["chain"] + " | " + plot_data["leader"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    # subplot 1: pairwise t (leader vs overall)
    axes[0].barh(plot_data["pair"], plot_data["lead_t"], color="#1b5e20",
                 alpha=0.85, edgecolor="black", lw=0.4, label="leader t")
    axes[0].barh(plot_data["pair"], plot_data["over_t"], color="#0277bd",
                 alpha=0.5, edgecolor="black", lw=0.4, label="整體鏈 t")
    axes[0].axvline(1.96, color="black", lw=0.5, ls="--", alpha=0.5)
    axes[0].set_xlabel("Pairwise t-stat → 各自鏈中下游")
    axes[0].set_title("(A) Pairwise: leader vs 整體鏈")
    axes[0].legend(); axes[0].grid(axis="x", alpha=0.3)

    # subplot 2: joint t1 vs t2
    axes[1].scatter(plot_data["joint_t1"], plot_data["joint_t2"],
                    s=120, c="#c62828", alpha=0.8, edgecolor="black")
    for _, r in plot_data.iterrows():
        axes[1].annotate(r["pair"], (r["joint_t1"], r["joint_t2"]),
                         xytext=(5, 5), textcoords="offset points", fontsize=8)
    axes[1].axvline(1.96, color="gray", lw=0.5, ls="--", alpha=0.5)
    axes[1].axhline(1.96, color="gray", lw=0.5, ls="--", alpha=0.5)
    axes[1].axvline(0, color="black", lw=0.5)
    axes[1].axhline(0, color="black", lw=0.5)
    axes[1].set_xlabel("Joint t (leader)")
    axes[1].set_ylabel("Joint t (整體鏈)")
    axes[1].set_title("(B) Joint reg: leader 主導區 = 右下")
    axes[1].grid(alpha=0.3)
    plt.suptitle("Phase 15: G000 / P000 源頭辨認", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig_phase15_chain_sources.png"), dpi=130)
    plt.close()
    print(f"\n圖檔: fig_phase15_chain_sources.png")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
