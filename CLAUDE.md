# 產業因子生成

研究台股產業鏈中上中下游股價報酬關係，建構以「產業位置」為基礎的因子。

## 資料來源

| 類型 | 來源 | 用途 |
|---|---|---|
| 產業鏈關係 | [TPEx 產業價值鏈資訊平台](https://ic.tpex.org.tw/) | 公司 → 產業 → 上中下游位置對照 |
| 股價／報酬 | FinLab API | 日報酬，後續因子建構 |

## 目錄結構

```
產業因子生成/
├── CLAUDE.md                       # 本檔
├── .gitignore
├── industry_chain.csv              # 爬蟲產出，主要對照表
├── scraper/
│   └── scrape_industry_chain.py    # TPEx 爬蟲
└── research/                       # Phase 1~22 實證分析
    ├── research_report.md          # 完整研究報告
    ├── phase1_build_returns.py     # 構建產業鏈組合日報酬
    ├── phase2_predictive_reg.py    # Menzly-Ozbas 預測迴歸
    ├── phase3_granger.py           # Granger 因果檢定
    ├── phase4_portfolio.py         # 內部多空投組
    ├── phase4b_cost_sensitivity.py # 成本敏感性
    ├── phase4c_low_turnover.py     # 低周轉版
    ├── phase6_momentum.py          # 跨產業動能
    ├── phase7_cross_industry.py    # 跨產業 link spillover
    ├── phase7b_freq_robustness.py  # 日週月頻 robustness
    ├── phase7c_strategy.py         # 半導體 lead 策略
    ├── phase7d_overlap_check.py    # 重疊股票 + split-sample
    ├── phase8_net_strategy.py      # 訊號平滑+扣成本
    ├── phase8b_selection_robustness.py  # 下游選擇 robustness
    ├── phase9_semi_internal.py     # 半導體鏈內部 (sub_code 級)
    ├── phase10_ic_vs_foundry.py    # IC設計 vs 晶圓代工 源頭辨認
    ├── phase11_chain_extension.py  # 23 鏈 timing 掃描 + top-3 sub_code 深掘
    ├── phase12_d100_replace.py     # D100 替換 D000 重做 Phase 8 (null 結果)
    ├── phase13_dual_factor.py      # IC設計 + 台積電 雙因子組合 (核心 winner)
    ├── phase14_cross_chain_combo.py # 6-leader 跨鏈整合 (negative)
    ├── phase15_g000_p000_sources.py # G000/P000 源頭辨認 + 個股 macro 檢驗
    ├── phase16_walkforward.py      # 嚴格 walk-forward 驗證
    ├── phase17_p000_dual_factor.py # P000 雙因子嘗試 (null)
    ├── phase18_network_centrality.py # 40 產業 network + 多種 centrality
    ├── phase19_community_combo.py  # Community detection + 跨群組合 (null)
    ├── phase20_firm_level_alpha.py # Cohen-Frazzini β-sorted long-only timed
    ├── phase21_cs_factor.py        # Cross-sectional factor 變體 (全部 negative)
    ├── phase22_g000_strategy.py    # G000 完整策略規劃 + walk-forward
    ├── phase23_midstream_pickup.py # D400/D500 中游個股 in-sample β-loading (look-ahead)
    ├── phase24_midstream_walkforward.py # WF threshold t>1.96 篩選 (失敗)
    ├── phase25_topN_rank.py        # WF Top-N rank (改進但未擊敗 baseline)
    ├── phase26_delay_ratio.py      # WF β_lag/β_now 延遲份額排序 (selection α 救援成功)
    ├── phase27_revenue_filter.py   # ratio + YoY>0 過濾 rank→filter (失敗 -4pp)
    ├── phase28_universe_prefilter.py # ratio + 3M-avg-YoY 預過濾 filter→rank (仍輸 no-filter)
    ├── phase29_downstream_ratio.py # D000 下游 6 sub_code ratio rank (timed 三項全勝)
    ├── phase30_market_cap.py       # ratio + 市值預過濾 (全失敗) + 純 MC 對照
    ├── phase31_trading_value.py    # ratio + 3M 成交金額預過濾 (比 MC 更傷)
    ├── phase32_pure_tv_rank.py     # 純成交金額 top-N (無 ratio, 全失敗)
    ├── phase33_pure_mc_rank.py     # 純市值 top-N (比 TV 略好但仍失敗)
    └── output/                     # 110+ CSV/PNG 結果檔
```

## industry_chain.csv 欄位

| 欄位 | 型別 | 說明 |
|---|---|---|
| stock_id | str | 股票代號（含 -KY 等後綴） |
| name | str | 公司簡稱 |
| market_type | str | 本國上市 / 外國上市 / 本國上櫃 / 本國興櫃 / 外國知名企業 / 創櫃 |
| industry_code | str | 產業鏈代碼（如 D000） |
| industry_name | str | 產業鏈名稱（如 半導體） |
| position | str | 上游 / 中游 / 下游，或自訂主題（如「核心技術」） |
| is_standard_chain | bool | True 表示 position 為標準上中下游 |
| sub_code | str | 細分類代碼（如 D300） |
| sub_name | str | 細分類名稱（如 IC/晶圓製造） |

長表結構：一家公司可能跨多個產業／位置（例：台達電 17 個產業位置）。

## 爬蟲使用

```bash
python scraper/scrape_industry_chain.py
```

流程：
1. 抓首頁解析 40 個產業代碼
2. 對每個產業頁解析 chain-panel → sub_code 對應上中下游
3. 解析 companyList_<sub_code> 區塊取得各市場別公司
4. 去重 (stock_id × industry_code × sub_code) 後輸出 CSV

禮貌間隔 0.8 秒，全程約 40 秒。

## 進度

### 已完成
- [x] 確認資料來源（TPEx 產業價值鏈資訊平台，免費、不需登入）
- [x] 撰寫爬蟲，產出 6,339 筆 / 2,341 家公司 / 40 個產業鏈
- [x] **Phase 1-4: 鏈內 (上中下游) spillover 驗證**
  - 預測迴歸 + Granger 因果, 確認台股存在「上→下」spillover (日頻 EW t_FM=3.19, p=0.004)
  - 22 鏈中 10 鏈 (45.5%) 上游顯著 Granger-cause 下游, 與 HTV (2007) 美股 41% 相當
  - 內部 LS 策略 gross alpha 16.4%, t=5.78
  - 低周轉版 net@50bp 約 8-9%
- [x] **Phase 6: 跨產業動能** — 弱於美股 (J=9/K=6 t=2.55), 與 spillover 訊號正交 (corr 0.16)
- [x] **Phase 7: 跨產業 link spillover** — 用公司多重歸屬構建有向連結矩陣
  - **半導體 D000 是台股領先樞紐**: 日頻領先 12+ 下游產業 |t|>2.6
  - **鋼鐵 Q000 為負向領先**: 成本轉嫁壓力使汽車/電機/建材跟跌
  - 重疊股票排除後仍顯著, OOS 91.8% pair 維持 β>0 (binomial p<0.0001)
- [x] **Phase 8: 訊號平滑+持有期+扣成本** — 半導體 lead 策略可實作
  - EMA(20)+K=1, **net@50bp 17.2%, Sharpe 1.46, alpha t=4.20** (80bp 成本下仍存活)
  - 集中投注強 hub 顯著優於分散整合 23-pair LS (後者扣成本後失效)
- [x] **Phase 9: 半導體鏈內部 (sub_code 級)** — 揭示真正領先者
  - **IC 設計 (D100, 89家)** 與 **IP 設計 (DC00, 14家)** 為純 leader (mean t > 3.4, 10/10 outputs sig)
  - 化學品 / 設備 / 基板 / 導線架 為純 receiver (mean t < 0)
  - 鏈內 timing 策略 net@50bp **20.6%, t=4.63** (略勝跨產業策略)
- [x] **Phase 10: IC 設計 vs 晶圓代工 源頭辨認** — 訊息源就是 IC 設計
  - Joint regression: 8/8 target 全 IC 設計主導, 晶圓代工 t 從 1-2 掉到 ±0.5
  - Orthogonalization: IC 設計⊥晶圓代工殘差 (僅 17% 變異) 仍 8/8 顯著; 晶圓代工殘差 0/8
  - 台積電 2330 有獨立 alpha (t≈2.6), 反映全球 AI/客戶 macro 訊號
  - **訊息傳導鏈: 客戶 → IC 設計 → 晶圓代工 → 後段製程**
- [x] **Phase 11: 鏈內 timing 策略擴展到 23 條 standard chain + top-3 sub_code 深掘**
  - 11a 上→中下游 timing (EMA(20), K=1, net@50bp): 13/23 鏈 t>1.96, 6/23 t>4
  - **半導體不是最強**: G000 平面顯示器 t=4.96 (alpha 14.5%), P000 電機機械 t=4.74 (alpha 12.3%) 均勝過 D000 t=4.31
  - 唯一失敗: 5800 運動科技 (樣本太薄, t=-0.41); Q000 鋼鐵鏈內無效 (跨產業才是 leader)
  - 11b sub_code 純 leader: G000 → 其他零組件 GA00 (t=1.86, 7/12 sig); P000 → 沖壓零組件 P600 + 傳動元件 P200; I000 → 光通訊 IA00 + 網路IC + 記憶體 + 網路設備 (多源頭結構, 不像 D000 一枝獨秀)
  - 異常: I000 鏈內印刷電路板 I500 為**反向訊號** (作 predictor mean t=-2.04, 6/11 negative sig), 可能反映 PCB 庫存週期與終端對沖
- [x] **Phase 12: D100 取代 D000 重做 Phase 8 跨產業策略 — null 結果**
  - 假設: D100 純訊號源應提升 alpha; 實測: D100 ≈ D000 (corr=0.966)
  - Best (N=20, K=1) @ 50bp: D100 alpha 12.56% / t=4.20 vs D000 alpha 12.17% / t=4.36 (Δ 僅 +0.39pp)
  - **解釋**: D100 ⊂ D000 且半導體股同質性高, EW 平均後幾乎是同一 portfolio
  - 次要發現: D100 在弱平滑+長持有 (N=3, K=20) 下 Δalpha 達 +4.05pp, 訊號穩定度較好但實用 winner 參數差距小
- [x] **Phase 13: D100 + 2330 雙因子組合 — 核心 winner**
  - 為何有效: ρ(D100, 2330) = **0.460** (vs D000 為 0.966), 2330 帶 76% idiosyncratic 訊息
  - **0.5·D100+0.5·2330 (N=10, K=1) net@50bp: alpha 15.26%, t=5.72, Sharpe 1.65** (跨產業 7 下游)
  - 鏈內 D000 中下游 target: alpha **18.25%, t=6.41, Sharpe 1.81** — 全研究最強訊號
  - 殘差化版 (D100+1.0·2330⊥) t=5.09 略遜 50/50, 因丟棄 2330 market-correlated alpha
  - @80bp 高成本: 0.7·D100+0.3·2330 (N=20, K=1) Sharpe 1.26 與純 D100 低周轉版打平
  - **驗證 Phase 10 結論**: 2330 確實有 D100 沒有的獨立 alpha, 線性組合是最簡也最佳整合
- [x] **Phase 14: 跨鏈 leader 組合 — negative result, 訊號集中於 D100+2330**
  - 整合 6 leader (D100/GA00/P200/P600/IA00/2330) × 7 整合方法 (EW raw/zscore/PCA/Orth/OLS-oracle)
  - **OLS-oracle (look-ahead) alpha 14.89% / t=5.68 仍輸 Phase 13 baseline (15.26%/5.72)**
  - 主因: leader 共線性過高 (D100 與 GA00/IA00 corr 0.84-0.88), 只有 2330 真正獨立 (~0.43)
  - OLS oracle 係數: D100=0.087, 2330=0.045, 其他 4 個 sub_code 都 ≈0 (IA00 還是負的)
  - **實務含意**: 台股可交易 spillover 訊號高度集中, D100+2330 已吸收絕大多數
- [x] **Phase 15: G000 / P000 源頭辨認 (Phase 10 風格)**
  - G000 GA00 ⊥ 整體 G000 殘差仍預測 xind t=2.31, joint reg 確認 leader 主導
  - P000 P200+P600 ⊥ 整體 P000 殘差預測 P000 中下游 t=2.68, xind t=2.47, leader 主導
  - **個股 macro 訊號**: G000 候選 (友達 2409 / 群創 3481) 全部失敗 (alone t<1, joint 不顯著)
  - **P000 找到了「P000 的 2330」: 1590 亞德客-KY** → xind alone t=3.19, joint reg 雙顯著 (t1=2.36, t2=2.43)
  - 上銀 2049 較弱 (alone t=2.15, joint 整體鏈主導)
  - 啟示: 1590 為 P000 鏈的 macro 個股, 可套 Phase 13 雙因子框架構 (P200+P600) + 0.5·1590
- [x] **Phase 16: 嚴格 walk-forward 驗證 (5y train → 1y test, step 1y)**
  - 樣本: 2007-2026 (15 個 OOS 年), target = 7 跨產業下游 EW
  - **D100+0.5·2330 WF OOS: α=9.78%, t=3.33, Sharpe 1.46** (vs IS-locked α=11.85%, t=4.25)
  - D000 WF: α=7.06%, t=2.20; D100 WF: α=7.25%, t=2.23 — Phase 13 雙因子在 WF 仍領先
  - 衰減 ~2pp / Δt~-0.9 在 reasonable 範圍, 不是 in-sample lookback artifact
  - (N*, K*) 跨年穩定性: 早期 (2012-17) 偏好 (20,1), 後期 (2018-24) 偏好 (?,20) longer holding, 2025+ 又回 (?,1)
  - 2018 (中美貿易戰) / 2022 (升息+科技調整) 為三 predictor 共同 OOS 失敗年, 為市場 regime risk
- [x] **Phase 17: P000 雙因子 (P200+P600) + 0.5·1590 — null 結果, Phase 13 框架不可移植**
  - ρ(P_lead, 1590)=0.514 與 ρ(D100, 2330)=0.460 相近, 1590 殘差變異 73.4% ≈ 2330 的 76.4% — 機構相同
  - 但結果相反: P_lead+0.5·1590 best (N=3, K=20) net@50bp xind t=2.68 vs **整體 P000 t=4.47 (alpha 11.97%, Sharpe 1.37)**
  - P000 鏈內 target 同樣失敗: 雙因子 t=3.21 vs 整體 P000 t=4.94
  - **失敗原因**:
    1. 訊號強度不對稱: P_lead t=3.77 < D100 t=4.20, 1590 t=2.19 < 2330 t=2.73 — 弱訊號組合雜訊累積大於訊號
    2. 整體 P000 (131 檔) ⊃ P_lead (27 檔) corr 0.90 — P000 不像 D000 ≡ D100 (corr 0.97), 整體已含中下游 macro/diversification
    3. best (N, K) = (3, 20) 偏離正常區間 — 短平滑+長持有是雜訊累積特徵
  - **重要結論**: D100+2330 雙因子 winner status 特殊於 D000 鏈結構 (sub_code leader ≈ 整體鏈), 不可機械套用到其他鏈
  - **P000 最佳實作仍是整體 P000 (N=20, K=1) net@50bp Sharpe 1.56**
- [x] **Phase 18: 40 產業 spillover network + centrality**
  - 跑 1560 個 lagged regression (3.1s), 建 directed weighted graph
  - 正向邊 |t|>2: 131 條 (8.4%); 負向邊: 71 條 (4.6%)
  - **Out-weighted top hub**: R300 電子商務 (artifact, 39 邊), **D000 半導體 #2 真 hub (out_w 76, 26 邊)**, 5700 體驗科技 (artifact)
  - **Negative-out hub #1**: Q000 鋼鐵 (35 條負邊), 對 B000/G000/K000/I000 全 t<-3.5 (成本傳導完美驗證)
  - **HITS authority** (純 receiver): G000/I000/K000/J000/F000 → Phase 8 7 跨產業下游名單高度重合
  - **Betweenness #1**: D000 (0.053) 為訊息傳遞核心橋樑
  - 5xxx 主題分類 (R300/5700/5800/5200) 因含小型高 beta 股有 high-out-degree artifact
- [x] **Phase 19: Community detection + 跨群組合 (null)**
  - Greedy modularity 找到 9 個 community, 3 個有實質: Community 0 (科技電子 16 個), 1 (傳統民生 10 個), 2 (生技通信 5 個)
  - 從 5 個 community + 2330 取代表 leader 構 cross-community combo
  - **跨 community leader corr 仍 0.77-0.90** (除 2330 ~0.43-0.52), community detection 沒能找到正交訊號
  - comm-OLS-oracle (look-ahead) alpha 15.31%/t=5.18 仍輸 D100+0.5·2330 (15.26%/5.72)
  - **副產品**: OLS oracle 揭示 N000 石化係數 -0.107 (反向訊號), 與 I500 PCB 異常呼應
  - **核心結論**: D100+2330 接近台股 spillover 訊號天花板, 共同因子支配所有產業 leader
- [x] **Phase 20: Firm-level alpha (Cohen-Frazzini) — long-only timed 改善有限**
  - Universe: 7 跨產業下游 620 檔, 排除 D100+2330 自我預測
  - β_i(t) = rolling 250d cov(r_i, lead_lag) / var(lead_lag), monthly rebal 5 分位數
  - **Q1→Q5 always-on alpha 全部接近 0**: Q5 alpha -0.96%, Q1 +0.10%, **無 cross-sectional 單調性**
  - **Q5 timed (long-only)**: net@50bp alpha 17.69%, t=5.00, Sharpe 1.55 (vs EW timed alpha 10.14%, t=4.84)
  - Δalpha +7.5pp 但 t_α 僅 +0.16 — alpha 增益被高 vol (14.65%) 稀釋
  - **Q5-Q1 long-short timed FAIL**: alpha -10.24%, t=-4.56 (台股無 cross-sectional β anomaly, 反而負向)
- [x] **Phase 21: Cross-sectional factor 全部 negative — 訊號不可 CS 化**
  - 5 種 score variant (β×EMA, β×sign, β×z, β only, β×lag): **全部 net alpha < 0**
  - 最佳 C_β×z(lead): alpha -6.67%, t=-2.12, IC=+0.004
  - 純 β only: alpha -3.81%, t=-1.30 (low-beta anomaly: 高 β 反而 underperform)
  - **核心結論**: D100+2330 是 time-series timing 訊號, 不是 stock-selection 訊號
  - 多因子整合應視為 regime indicator (動態 beta tilt), 不可與 size/value/momentum 並列
- [x] **Phase 22: G000 平面顯示器 完整策略規劃 (10 predictor + walk-forward)**
  - 10 個 predictor 變體: 鏈內 (G_up/GA00/G_overall) + 跨產業 (D_macro) + 反向 (Q000) 線性組合
  - **驚人發現: D_macro alone (Phase 13 訊號) 在 G000 中下游 alpha 15.79%, t=5.11, 比 G000 內部訊號 (A_G_up alpha 12.54%) 還強**
  - 加 GA00 鏈內訊號邊際提升小, Q000 反向訊號無實質貢獻 (H_GA-0.3Q 反而變差)
  - **Walk-forward winner**: G_0.3GA+0.7D (30% GA00 + 70% D_macro), WF α=11.76%, t=3.68, Sharpe 1.55, 衰減僅 -0.42pp
  - 三因子 I_3factor IS 看似最強 (t=5.30) 但 OOS 衰減 -4.03pp (overfit Q000 無實質訊號)
  - **G000 實盤建議: G_0.3GA+0.7D (N=10, K=1)** 或更簡單的 D_macro alone (WF α=11.57%, t=3.61)
- [x] **Phase 23: D400/D500 中游個股 in-sample β-loading 排序 (look-ahead bias 範例)**
  - 動機: Phase 9 顯示 D400 設備、D500 化學品集合層面是純 receiver, 但 cross-sectional 異質性可能被 EW 平均掩蓋
  - Universe: 60 檔 unique (D400 設備 44 + D500 化學品 16, FinLab 4 位數 stock_id)
  - Per-stock lagged-β regression on full sample (4245d), 控 r_i_lag + market_lag + HAC SE
  - **24/60 (40%) 全期 |t|>1.96, 22 檔 5y 仍穩定**
  - **Top picks**: 6208 日揚 (t=4.65 最強), 3535 晶彩科 (β=0.21 最大), 5434 崇越/3010 華立/1785 光洋科 (D500)
  - β_contemp 才是 market beta (t>10), β_lag/β_contemp 比率 (~24-36%) 才是真延遲份額
  - **重要警告**: 全期跑 = look-ahead, 不可實盤; 為 Phase 24-26 的 baseline / upper bound
- [x] **Phase 24: D400/D500 walk-forward β-screening (threshold t > 1.96, 5y rolling) — 失敗**
  - 月底 rebalance, 過去 5y window 跑 lagged-β regression, t > 1.96 入選, EW within selected
  - 169 個月 × 76 檔 (D400∪D500 unique) regression, no-peek 嚴格保證
  - **Δα vs Full EW**: -0.15pp (always-on) / +1.12pp 但 Δt -0.56 (timed)
  - 入選數量極不穩定 (median 10, min 1, max 25), 2018 僅 2 檔通過 → 集中風險
  - α 上升被 vol 同步上升吃光, t 反降; 確認 Phase 23 in-sample bias ≈ 1-2pp α
- [x] **Phase 25: D400/D500 walk-forward Top-N rank — 比 threshold 改進但仍輸 baseline**
  - 改取 t-stat 排名前 N 檔 (避免「2018 僅 2 檔」極端集中), N ∈ {5, 10, 15, 20}
  - vs Phase 24: Sharpe 1.27 → **1.54**, t 3.09 → **3.59** (Top-15 timed)
  - **Top-15 timed**: α=13.78% (+1.03pp vs Full EW timed), Sharpe 1.54 ≈ baseline 1.59
  - **Top-5 always-on**: α=8.74% (+3.27pp), 純 cross-sectional alpha 出現 (但 vol 大 Sh 反降)
  - timed 軌道仍未擊敗 Full EW timed, selection α 與 timing α 重疊
- [x] **Phase 26: D400/D500 walk-forward β_lag/β_now「延遲份額」排序 — selection α 救援成功**
  - 動機: t-stat 沒區分「真延遲 vs 同步反應 + 大樣本」, 改算 β_lag/β_now ratio
  - Joint regression: r_i = α + β_now·lead(t) + β_lag·lead(t-1) + γ·r_i_lag + δ·m_lag
  - **ratio N=10 always-on**: α=**10.41%**, t=**2.58**, Sharpe **1.33** (vs Full EW α=5.48% / t=1.77 / Sh 1.18)
  - **Δα = +4.94pp ✓✓** — 確認 cross-sectional 真實 alpha 存在於中游 sub-universe
  - 比 b_lag rank (+3.10pp) 和 t_lag rank (+2.79pp) 更強, 證明「延遲份額」排除「同期反應+大樣本」雜音
  - Timed 軌道三種排序持平 baseline (timing 已賺光時序 α, selection 邊際小)
  - **發現新個股**: 1742 台蠟、4764 雙鍵 (D500 化學品) 在 ratio rank 進 top, t_lag rank 看不出
  - **重要結論**: Phase 21 「CS 因子全 negative」結論在中游 sub-universe + ratio 排序下被 partially 推翻
- [x] **Phase 27: D400/D500 ratio rank + 月營收 YoY > 0 過濾 (rank→filter) — 失敗**
  - 假設: ratio 找技術延遲反應, YoY 過濾基本面已轉弱的個股
  - 結果: filter 把 ratio 最強個股大量排除 (43% 跳過, 含 #1 6208 日揚 YoY -18%)
  - **N=10 always-on Δα = -4.00pp** vs no filter (10.41% → 6.41%), ΔSh -0.273 ↓
  - 由 ratio 弱的個股遞補, 訊號被稀釋, turnover 增加 4 倍 (1.88 → 6.92)
  - 唯一 N=5 例外救援 (filter 強迫遞補等於分散集中風險)
- [x] **Phase 28: D400/D500 ratio rank + 月營收 3M-avg YoY 預過濾 (filter→rank) — 仍輸 no-filter**
  - 改 universe-level pre-filter (剔除 3M-avg YoY < threshold 個股), 然後 ratio rank top-N
  - **Δα vs Phase 27 ≥0%**: +2.26pp (filter-then-rank 比 rank-then-filter 合理, 保留 ratio 最強個股)
  - 但**全部 threshold 仍未擊敗無過濾基準** (≥-30% 微升 Sharpe +0.017, 其他都傷)
  - 失敗原因: 半導體中游是強週期股, ratio 強訊號常出現在週期觸底 (YoY 為負時), 過濾把 buy opportunity 殺掉
  - **6208 日揚 案例**: ratio #1 但 3M-avg YoY -9.58%, ≥-10% / ≥0% 門檻會把最強訊號剔除
  - **Phase 23-28 最終結論**:
    - 中游個股 selection 最佳實作 = **Phase 26 ratio N=10 always-on (no YoY filter)** — α=10.41%, Sharpe 1.33
    - 月營收 YoY 在週期股難以整合, 唯一安全用法是「剔除 3M-avg < -30% 真正爛公司」(避免下市/陷阱)
    - timed 軌道 selection 與 timing 重疊, Full EW timed 仍是 Sharpe 王 (1.59)
- [x] **Phase 29: D000 下游 6 sub_code (D600/D700/D800/D900/DA00/DB00) ratio rank — timed 軌道首度三項全勝**
  - Universe: 126 檔 unique (D600 設備 43 + D700 基板 7 + D800 導線架 7 + D900 IC封測 27 + DA00 IC模組 13 + DB00 IC通路 29)
  - 同 Phase 26 設計 (ratio = β_lag/β_now, 5y rolling, top-N), N ∈ {5, 10, 15, 20}
  - **best ratio N=20 timed**: α=**13.36%**, t=**4.41**, Sharpe **1.63** (vs Full EW timed 11.81/3.84/1.62)
  - **Δα +1.55pp ✓ Δt +0.57 ✓ ΔSh +0.014 ✓ — Phase 23-29 系列首個 timed 三項全勝設定**
  - **best N 從 10 (中游) → 20 (下游)**: universe 翻倍 (60→126), 維持 ~16% 集中度
  - **Sub_code over-representation 揭示真正 receiver**:
    - **DB00 IC 通路 1.40×** (29 檔, top-10 出現 615 次) ✓ over
    - **D700 基板 2.13×** (7 檔但 over-rep 最高) ✓✓
    - D600 設備 0.67× ❌ under-rep (43 檔最大 sub 反而 ratio 偏低 — β_now 太大被分母稀釋)
  - 2026-04 top 15: 蔚華科/美達科技/博大/威健/三顧/志旭/增你強/豐藝/意德士/佳穎/易華電/全科/創見/長科/敦吉
  - **重要**: top picks 含 8067 志旭 (市值 NaN, 流動性陷阱) → Phase 30/31 揭露實盤 trade-off
- [x] **Phase 30: 市值預過濾 + ratio rank — 全面失敗**
  - A. ratio + MC top X% filter (X = 30/50/70%): **全部 -2 ~ -5pp Δα ↓**
  - B. Pure MC top-N (對照組, 無 ratio) timed Sh 0.78-0.92 (大幅輸 ratio 1.63)
  - 失敗原因: ratio top 個股集中中小型股 (20-200 億區間), MC top X% 把強訊號 strip 掉
  - **大型股 β_now 大** (機構覆蓋率高, 同步性強) → ratio 自然偏低 → MC top 與 ratio top **結構性 mismatch**
- [x] **Phase 31: 3M 成交金額預過濾 + ratio rank — 比 MC 還傷**
  - A. ratio + TV top X% filter: top70/50/30% Δα = -3.86 / -5.51 / **-6.57pp** (timed N=20)
  - B. Pure TV top-N timed Sh 0.69-0.82 (比 Pure MC 還差)
  - **核心啟示**: 延遲反應與低流動性正相關 — Hou (2007) 經典 information slowness
  - 8067 志旭 (ratio top 6) 日均成交金額 0.001 億 = 真流動性陷阱
  - **paper α vs executable α trade-off**: 13.36% (no filter) → 9.50% (TV top70%, 排除流動性最差 30%)
- [x] **Phase 32: 純成交金額 top-N (無 ratio) — 完全失敗**
  - N ∈ {5, 10, 15, 20, 25, 30} 全部 timed Sharpe < 0.86, α < 6.45%
  - 全部 t_α < 1.5 (統計上無顯著 alpha)
  - **2308 台達電 169/169 月 (100%) 入選 top 10**, 本質是「半導體下游大型股 ETF」
  - 確認大型股 alpha 早被 priced in
- [x] **Phase 33: 純市值 top-N (無 ratio) — 失敗但比 TV 略好**
  - N ∈ {5, 10, 15, 20, 25, 30}, eval 從 2013-05 開始 (MC 資料起點)
  - **N=10 always-on Δα +2.26pp** (Phase 30-33 純 size 排序首次正向 Δα), 但 Sharpe 仍 -0.11 ↓
  - timed best N=30: α=8.83%, t=2.31, Sharpe 1.14 (仍輸 baseline 1.63 / ratio winner 1.63)
  - **MC > TV** (所有 N): timed Sh +0.10~+0.29 — size 比 liquidity 穩定, 但都不是 alpha 來源
  - 99% 入選: 台達電/和碩/群聯/大聯大/聯強/力成 — 穩定大型權值股組合
  - **Phase 29-33 總結**:
    - alpha 來源 = **中小型股的 information slowness**, 不是大型股的 market efficiency
    - 任何品質/流動性類過濾 (YoY/MC/TV) 都會 strip ratio top picks → -2~-7pp α
    - 排序方法 timed Sh 排序: **ratio (1.63) > Full EW (1.63) > MC top-N (1.14) > TV top-N (0.86)**
    - 實盤折衷: ratio + TV top70% (剔除底部 30% 流動性) → α=9.50%, Sh=1.19, 仍勝 Full EW always-on

### 待辦
- [ ] **I500 印刷電路板反向訊號異常研究**
  - Phase 11b 發現 I000 鏈內 I500 作 predictor 為 mean t=-2.04 (6/11 negative sig)
  - Phase 19 OLS-oracle 發現 N000 石化係數 -0.107 也是反向訊號, 同類型異常
  - 假設: 庫存週期反向領先, 或代工 vs 終端對沖效應
  - 驗證: 個股拆解 + 跨產業檢驗 + 整合多反向訊號為 contrarian factor
- [ ] **產業內中性 β CS 因子 (Phase 21 救援) — Phase 26/29 已驗證**
  - Phase 26 ratio N=10 always-on 在 D400+D500 中游證實 selection α 存在 (+4.94pp)
  - Phase 29 ratio N=20 timed 在 D000 下游 timed 軌道首度三項全勝 (Sh 1.63, +0.014)
  - 待擴展: 套到其他鏈中游 sub_code (P200/P600 電機機械、GA00 平面顯示器、IA00 通信網路)
  - 待擴展: 對 Phase 8 跨產業 7 下游 universe 套用 ratio rank (更大舞台)
  - 對照: BAB (Frazzini-Pedersen) 台股可行性
- [ ] **基本面/品質整合替代方案 (Phase 27-33 全失敗後)**
  - 已試失敗: YoY (Phase 27/28)、市值 (Phase 30/33)、成交金額 (Phase 31/32) — 全 strip 強訊號
  - 核心矛盾: alpha 集中中小型 information slowness, 任何「大盤股偏好」過濾都反向
  - 試: YoY 動能轉折 (由下轉上 = 加分) 取代絕對水準
  - 試: 毛利率/ROE/經營現金流等更穩定的基本面變數
  - 試: YoY 整合進 timing (組合平均 YoY < threshold 時降低部位) 而非 selection
- [ ] **paper α vs executable α 流動性 trade-off 研究**
  - Phase 31 揭露: ratio rank 純訊號 α=13.36% 含 8067 志旭等流動性陷阱不可實盤
  - 折衷 ratio + TV top70% α=9.50% (失 3.86pp 換取流動性)
  - 待: Amihud 流動性指標 (return / dollar volume) 而非 raw TV
  - 待: 動態 N (流動性大時擴大 N, 反之集中) 取代固定 N
  - 待: 計算實盤可行 α 上界 (排除日均成交 < 1 億 / 個股部位 < 5% 該股 ADV)
- [ ] **其他鏈套用 Phase 22 風策略規劃**
  - P000 已試過 (Phase 17 fail), G000 已試過 (Phase 22 success)
  - 可對 Phase 11a 中 t>4 鏈一一套用: I000/F000/L000/J000/H000/K000/S000
  - 預期: 純鏈內 vs D_macro 對比, 確認 D_macro 是否普適
- [ ] 多產業歸屬規則：一家公司若跨多產業，主產業如何決定
  - 候選：(a) FinLab 主產業 join，(b) 取營收最大產業，(c) 全部保留做加權

## 注意事項

- 5xxx 系列（區塊鏈、AI、雲端、資安、大數據、體驗、運動）為「新興主題分類」，position 用自訂維度（如「核心技術」「應用與服務」），非傳統上中下游 → 用 `is_standard_chain` 欄位區分
- 5 個產業（軟體服務 R000、其他 X000、文化創意 Y000、金融 U000、交通運輸 T000）網站本身無上中下游分類，position 為 NaN
- 「其他」(X000) 涵蓋 346 家但無分類，因子研究時通常排除
