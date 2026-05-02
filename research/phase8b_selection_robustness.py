"""驗證 7 下游選擇是否 in-sample 偏誤
比較三種選法:
A. 我原本的 7 個 (in-sample, 主觀)
B. 純結構: L_loose ≥ 9 的全部下游 (no return data)
C. Train/Test 切分: 用 2007-2015 的迴歸結果挑選, 在 2016-2024 測試
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


def industry_panel(daily_ret, chain):
    panel = {}
    for ind in chain["industry_code"].unique():
        members = chain[chain["industry_code"]==ind]["stock_id"].unique().tolist()
        cols = [s for s in members if s in daily_ret.columns]
        if len(cols) >= 3:
            panel[ind] = daily_ret[cols].mean(axis=1)
    return pd.DataFrame(panel)


def strategy_A(panel, semi_code, downstream, smooth_n, hold_k, cost_one=0.0025):
    semi = panel[semi_code]
    cols = [c for c in downstream if c in panel.columns]
    if len(cols) == 0: return None
    down = panel[cols].mean(axis=1)
    sig = semi.ewm(span=smooth_n, adjust=False).mean()
    target = (sig > 0).astype(float).shift(1)
    if hold_k > 1:
        out = target.copy(); last = target.iloc[0]
        for i in range(len(target)):
            if i % hold_k == 0: last = target.iloc[i]
            out.iloc[i] = last
        weight = out
    else:
        weight = target
    gross = weight * down
    turn = weight.diff().abs().fillna(weight.iloc[0])
    net = gross - turn * cost_one
    return gross, net, turn, weight, down


def perf(net, market, ann=252):
    df = pd.concat([net.rename("s"), market.rename("b")], axis=1).dropna()
    if len(df) < 60: return {}
    res = sm.OLS(df["s"], sm.add_constant(df["b"])).fit(cov_type="HAC", cov_kwds={"maxlags":60})
    return {
        "n": len(df),
        "ann_ret": df["s"].mean()*ann,
        "ann_vol": df["s"].std()*np.sqrt(ann),
        "sharpe": df["s"].mean()/df["s"].std()*np.sqrt(ann) if df["s"].std()>0 else np.nan,
        "ann_alpha": res.params["const"]*ann,
        "t_alpha": res.tvalues["const"],
    }


def main():
    daily_ret = load()
    chain = pd.read_csv(os.path.join(ROOT, "industry_chain.csv"), dtype=str)
    industries = chain[["industry_code","industry_name"]].drop_duplicates().set_index("industry_code")["industry_name"].to_dict()
    panel = industry_panel(daily_ret, chain)
    market = daily_ret.mean(axis=1)

    SEMI = "D000"

    # === 選法 A: 我原本的 7 個 ===
    SET_A = ["G000","H000","I000","J000","F000","L000","5400"]
    SET_A_names = [industries.get(c, c) for c in SET_A]

    # === 選法 B: L_loose ≥ 9 (純結構, 無報酬資料) ===
    link = pd.read_csv(os.path.join(OUT, "cross_industry_links.csv"))
    semi_links = link[(link["A"]==SEMI) & (link["L_loose"]>=9)].sort_values("L_loose", ascending=False)
    SET_B = semi_links["B"].tolist()
    SET_B_names = [industries.get(c, c) for c in SET_B]

    # === 選法 C: Train/Test 切分 ===
    split_date = "2016-01-01"
    panel_train = panel[panel.index < split_date]
    market_train = market[market.index < split_date]
    panel_test = panel[panel.index >= split_date]
    market_test = market[market.index >= split_date]

    # 只用 train 跑半導體 → X 迴歸, 取 t > 2 的 X
    SEMI_train = panel_train[SEMI]
    sel_C = []
    for B in panel_train.columns:
        if B == SEMI: continue
        df = pd.DataFrame({
            "y": panel_train[B],
            "xA": SEMI_train.shift(1),
            "xB": panel_train[B].shift(1),
            "xM": market_train.shift(1)
        }).dropna()
        if len(df) < 200: continue
        try:
            res = sm.OLS(df["y"], sm.add_constant(df[["xA","xB","xM"]])).fit(
                cov_type="HAC", cov_kwds={"maxlags":5}
            )
            if res.tvalues["xA"] > 2.0 and res.params["xA"] > 0:
                sel_C.append((B, res.tvalues["xA"]))
        except Exception:
            continue
    sel_C.sort(key=lambda x: -x[1])
    SET_C = [b for b, t in sel_C]
    SET_C_names = [industries.get(c, c) for c in SET_C]

    print("="*100)
    print("【選法 A】我原本選的 7 個 (in-sample, 主觀)")
    print("  ", " / ".join(SET_A_names))
    print(f"\n【選法 B】純結構 L_loose ≥ 9 (沒看任何報酬資料): {len(SET_B)} 個")
    print("  ", " / ".join([f"{n}(L={l})" for n, l in zip(SET_B_names, semi_links['L_loose'].tolist())]))
    print(f"\n【選法 C】Train (2007-2015) 內挑 t>2 且 β>0 的: {len(SET_C)} 個")
    print("  ", " / ".join([f"{n}(t={t:.2f})" for n, t in zip(SET_C_names, [t for _, t in sel_C])]))

    # 比較三種策略的全期表現 + OOS 表現
    print("\n" + "="*100)
    print(f"\n{'選法':<8s} {'全期 net@50bp':>40s}    {'OOS only (2016-2024) net@50bp':>40s}")
    print(f"{'':<8s} {'ann_ret':>10s} {'sharpe':>9s} {'alpha':>8s} {'t_α':>7s}    {'ann_ret':>10s} {'sharpe':>9s} {'alpha':>8s} {'t_α':>7s}")
    for label, sset in [("A (主觀)", SET_A), ("B (結構)", SET_B), ("C (OOS)", SET_C)]:
        for N, K in [(20, 1), (10, 20)]:
            r_full = strategy_A(panel, SEMI, sset, N, K)
            if r_full is None: continue
            _, net_full, _, _, _ = r_full
            p_full = perf(net_full, market)

            # OOS test
            r_oos = strategy_A(panel.loc[split_date:], SEMI, sset, N, K)
            _, net_oos, _, _, _ = r_oos
            p_oos = perf(net_oos, market.loc[split_date:])

            print(f"{label} N={N},K={K} "
                  f"{p_full.get('ann_ret', np.nan):>10.4f} {p_full.get('sharpe', np.nan):>9.3f} "
                  f"{p_full.get('ann_alpha', np.nan):>8.4f} {p_full.get('t_alpha', np.nan):>7.3f}    "
                  f"{p_oos.get('ann_ret', np.nan):>10.4f} {p_oos.get('sharpe', np.nan):>9.3f} "
                  f"{p_oos.get('ann_alpha', np.nan):>8.4f} {p_oos.get('t_alpha', np.nan):>7.3f}")


if __name__ == "__main__":
    main()
