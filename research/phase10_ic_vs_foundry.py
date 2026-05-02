"""IC 設計 vs 晶圓代工: 誰是真正的領先源頭

三層測試:
1. Pairwise: IC設計 ↔ 晶圓代工 雙向預測迴歸
2. Joint regression (核心): r_target(t+1) = α + β1·r_IC設計(t) + β2·r_晶圓代工(t) + γ·r_target(t) + δ·r_market(t)
   → β1, β2 誰存活, 誰死掉, 揭示真正的源頭
3. Orthogonalization: 把 IC設計 對 晶圓代工 殘差化, 反之亦然, 看殘差訊號是否仍有預測力
4. 額外: 台積電 (2330) 單獨作為 predictor, 與 IC設計 portfolio 比較
"""
import os, pickle, sys
import numpy as np, pandas as pd
import statsmodels.api as sm

sys.stdout.reconfigure(encoding="utf-8")
DB = r"C:\Users\user\finlab_db"
ROOT = r"C:\Users\user\OneDrive\桌面\產業因子生成"
OUT = os.path.join(ROOT, "research", "output")


def load():
    with open(os.path.join(DB, "etl#adj_close.pickle"), "rb") as f:
        adj = pickle.load(f).set_index("date")
    adj.index = pd.to_datetime(adj.index); adj.columns = adj.columns.astype(str)
    keep = [c for c in adj.columns if c.isdigit() and len(c)==4 and not c.startswith("00")]
    return adj[keep].pct_change(fill_method=None).iloc[1:]


def build_panel(daily_ret, chain, code_map):
    panel = {}
    for label, (industry_code, sub_code) in code_map.items():
        d = chain[chain["industry_code"]==industry_code]
        if sub_code:
            d = d[d["sub_code"]==sub_code]
        members = d["stock_id"].unique().tolist()
        members = [s for s in members if s in daily_ret.columns]
        if len(members) >= 3:
            panel[label] = daily_ret[members].mean(axis=1)
            print(f"  {label}: {len(members)} 檔")
    return pd.DataFrame(panel)


def reg_simple(y, x, others, market, nw=5):
    df = pd.concat({"y":y, "x":x.shift(1), "y_lag":y.shift(1), "m":market.shift(1)}, axis=1).dropna()
    if len(df) < 200: return None
    res = sm.OLS(df["y"], sm.add_constant(df[["x","y_lag","m"]])).fit(cov_type="HAC", cov_kwds={"maxlags":nw})
    return {"n":len(df), "beta":res.params["x"], "t":res.tvalues["x"], "p":res.pvalues["x"], "r2":res.rsquared}


def reg_joint(y, x1, x2, market, nw=5):
    """Joint: y(t+1) = α + β1·x1(t) + β2·x2(t) + γ·y(t) + δ·m(t)"""
    df = pd.concat({
        "y":y, "x1":x1.shift(1), "x2":x2.shift(1),
        "y_lag":y.shift(1), "m":market.shift(1)
    }, axis=1).dropna()
    if len(df) < 200: return None
    res = sm.OLS(df["y"], sm.add_constant(df[["x1","x2","y_lag","m"]])).fit(cov_type="HAC", cov_kwds={"maxlags":nw})
    return {
        "n":len(df),
        "b1":res.params["x1"], "t1":res.tvalues["x1"], "p1":res.pvalues["x1"],
        "b2":res.params["x2"], "t2":res.tvalues["x2"], "p2":res.pvalues["x2"],
        "r2":res.rsquared
    }


def main():
    daily_ret = load()
    chain = pd.read_csv(os.path.join(ROOT, "industry_chain.csv"), dtype=str)
    market = daily_ret.mean(axis=1)

    print("="*90)
    print("【Setup】建構 predictor portfolios")
    print("="*90)
    pred_panel = build_panel(daily_ret, chain, {
        "IC設計":      ("D000", "D100"),
        "IP設計":      ("D000", "DC00"),
        "晶圓代工":    ("D000", "D300"),
        "封測":        ("D000", "D900"),
    })
    # 加入台積電單獨
    if "2330" in daily_ret.columns:
        pred_panel["台積電2330"] = daily_ret["2330"]
        print("  台積電2330: 1 檔 (個股)")
    if "2454" in daily_ret.columns:
        pred_panel["聯發科2454"] = daily_ret["2454"]
        print("  聯發科2454: 1 檔 (個股, IC設計龍頭)")

    # 計算 predictor 之間的同期相關
    print("\n=== Predictor 之間日同期相關 (corr) ===")
    print(pred_panel.corr().round(3))

    # === Targets ===
    print("\n=== Targets: D000 中下游 + Phase 8 7 個跨產業下游 ===")
    target_codes = {
        "D000中下游": [c for c in ["D000_中游","D000_下游"]],  # 用 position aggregation
        "G000平面顯示器":("G000",None),"H000觸控面板":("H000",None),
        "I000通信網路":("I000",None),"J000被動元件":("J000",None),
        "F000電腦週邊":("F000",None),"L000PCB":("L000",None),
        "5400雲端運算":("5400",None),
    }
    target_panel = {}
    # D000 中下游 = 中游 + 下游 EW 聯合
    d_mid = chain[(chain["industry_code"]=="D000") & (chain["position"]=="中游")]["stock_id"].unique()
    d_dn = chain[(chain["industry_code"]=="D000") & (chain["position"]=="下游")]["stock_id"].unique()
    md_set = list(set(list(d_mid)+list(d_dn)) & set(daily_ret.columns))
    target_panel["D000中下游"] = daily_ret[md_set].mean(axis=1)
    for name, (ic, sc) in [(k, v) for k, v in target_codes.items() if k != "D000中下游"]:
        d = chain[chain["industry_code"]==ic]
        members = [s for s in d["stock_id"].unique() if s in daily_ret.columns]
        target_panel[name] = daily_ret[members].mean(axis=1)
    target_panel = pd.DataFrame(target_panel)
    print(f"  共 {target_panel.shape[1]} 個 target")

    # === 1. Pairwise: IC設計 vs 晶圓代工 ===
    print("\n" + "="*90)
    print("【1】Pairwise (單一 predictor) — 預測各 target")
    print("="*90)
    print(f"\n{'Target':<18s} {'IC設計 β/t':>16s} {'晶圓代工 β/t':>16s} {'IP設計 β/t':>16s} {'2330 β/t':>14s} {'2454 β/t':>14s}")
    for tname in target_panel.columns:
        y = target_panel[tname]
        line = f"{tname:<18s}"
        for pname in ["IC設計","晶圓代工","IP設計","台積電2330","聯發科2454"]:
            r = reg_simple(y, pred_panel[pname], None, market)
            star = "*" if abs(r["t"])>1.96 else " "
            line += f" {r['beta']:>+.3f}/{r['t']:>+.2f}{star} "
        print(line)

    # === 2. Joint regression: IC設計 + 晶圓代工 一起放進去 ===
    print("\n" + "="*90)
    print("【2】Joint Regression — 同時放 IC設計 與 晶圓代工 入迴歸, 看誰存活")
    print("="*90)
    print(f"\n{'Target':<18s} {'IC設計 β1':>14s} {'IC設計 t1':>10s} {'晶圓代工 β2':>14s} {'晶圓代工 t2':>10s} {'verdict':>20s}")
    rows = []
    for tname in target_panel.columns:
        r = reg_joint(target_panel[tname], pred_panel["IC設計"], pred_panel["晶圓代工"], market)
        if r is None: continue
        verdict = ""
        if r["t1"]>1.96 and r["t2"]<1.96:
            verdict = "✓ IC設計 主導"
        elif r["t2"]>1.96 and r["t1"]<1.96:
            verdict = "✓ 晶圓代工 主導"
        elif r["t1"]>1.96 and r["t2"]>1.96:
            verdict = "兩者皆 sig"
        elif r["t1"]<-1.96 or r["t2"]<-1.96:
            verdict = "(負向) 反轉"
        else:
            verdict = "皆不顯著"
        print(f"{tname:<18s} {r['b1']:>+14.4f} {r['t1']:>+10.3f} {r['b2']:>+14.4f} {r['t2']:>+10.3f} {verdict:>20s}")
        rows.append({"target":tname, **r, "verdict":verdict})
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "phase10_joint_reg.csv"), index=False, encoding="utf-8-sig")

    # === 3. 額外: 台積電 vs IC設計 ===
    print("\n" + "="*90)
    print("【2b】Joint Regression — 台積電2330 vs IC設計 portfolio")
    print("="*90)
    print(f"\n{'Target':<18s} {'2330 β1':>14s} {'2330 t1':>10s} {'IC設計 β2':>14s} {'IC設計 t2':>10s} {'verdict':>20s}")
    for tname in target_panel.columns:
        r = reg_joint(target_panel[tname], pred_panel["台積電2330"], pred_panel["IC設計"], market)
        if r is None: continue
        verdict = ""
        if r["t1"]>1.96 and r["t2"]<1.96: verdict = "✓ 台積電 主導"
        elif r["t2"]>1.96 and r["t1"]<1.96: verdict = "✓ IC設計 主導"
        elif r["t1"]>1.96 and r["t2"]>1.96: verdict = "兩者 sig"
        elif r["t1"]<-1.96 or r["t2"]<-1.96: verdict = "(負向) 反轉"
        else: verdict = "皆不顯著"
        print(f"{tname:<18s} {r['b1']:>+14.4f} {r['t1']:>+10.3f} {r['b2']:>+14.4f} {r['t2']:>+10.3f} {verdict:>20s}")

    # === 4. Orthogonalization 殘差訊號 ===
    print("\n" + "="*90)
    print("【3】Orthogonalization — 將 predictor 對另一 predictor 殘差化")
    print("="*90)
    # IC設計 對 晶圓代工 殘差化 → IC設計獨有訊息
    # 晶圓代工 對 IC設計 殘差化 → 晶圓代工獨有訊息
    ic = pred_panel["IC設計"]
    fd = pred_panel["晶圓代工"]
    df_orth = pd.concat({"ic":ic, "fd":fd, "m":market}, axis=1).dropna()

    # IC殘差 = IC - β·foundry - δ·market
    res_ic = sm.OLS(df_orth["ic"], sm.add_constant(df_orth[["fd","m"]])).fit()
    ic_resid = df_orth["ic"] - res_ic.predict(sm.add_constant(df_orth[["fd","m"]]))

    # 晶圓殘差 = 晶圓 - β·IC - δ·market
    res_fd = sm.OLS(df_orth["fd"], sm.add_constant(df_orth[["ic","m"]])).fit()
    fd_resid = df_orth["fd"] - res_fd.predict(sm.add_constant(df_orth[["ic","m"]]))

    print(f"\nResidual signals:")
    print(f"  IC設計⊥晶圓代工:   var ratio = {ic_resid.var()/ic.var():.3f}")
    print(f"  晶圓代工⊥IC設計:   var ratio = {fd_resid.var()/fd.var():.3f}")
    print(f"  原 IC 與 晶圓代工 corr: {df_orth['ic'].corr(df_orth['fd']):.3f}")

    print(f"\n{'Target':<18s} {'IC殘差 β/t':>18s} {'晶圓殘差 β/t':>18s}")
    for tname in target_panel.columns:
        y = target_panel[tname]
        r1 = reg_simple(y, ic_resid, None, market)
        r2 = reg_simple(y, fd_resid, None, market)
        if r1 is None or r2 is None: continue
        s1 = "*" if abs(r1["t"])>1.96 else " "
        s2 = "*" if abs(r2["t"])>1.96 else " "
        print(f"{tname:<18s} {r1['beta']:>+10.4f}/{r1['t']:>+5.2f}{s1}   {r2['beta']:>+10.4f}/{r2['t']:>+5.2f}{s2}")


if __name__ == "__main__":
    main()
