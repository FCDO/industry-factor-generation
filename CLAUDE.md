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
└── research/                       # Phase 1~13 實證分析
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
    └── output/                     # 50+ CSV/PNG 結果檔
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

### 待辦
- [ ] **跨鏈 leader 組合因子** (下一步)
  - 把 D100、GA00、P600+P200、IA00 等多源頭整合為 pure-leader 因子
  - 測試是否優於單鏈訊號, 並用 PCA / orthogonalization 處理共線性
  - 對照 Phase 13 雙因子框架, 是否能進一步加入 2330 構成多源頭 + macro 個股組合
- [ ] **Phase 14: 平面顯示器/電機機械源頭辨認 (Phase 10 風格)**
  - Phase 11a 顯示 G000 t=4.96, P000 t=4.74 都超越半導體
  - 對 G000 (GA00 vs 整體 G000) 與 P000 (P600+P200 vs 整體 P000) 做 joint reg, 確認真實源頭
  - 預期類似 Phase 10 結論: 純 leader sub_code 主導, 整體鏈為 noise dilution
  - 找出 G000 / P000 是否各自有「macro 個股」(類似 2330 之於 D100), 可套 Phase 13 雙因子框架
- [ ] **I500 印刷電路板反向訊號異常研究**
  - Phase 11b 發現 I000 鏈內 I500 作 predictor 為 mean t=-2.04 (6/11 negative sig)
  - 假設: PCB 庫存週期反向領先, 或代工 vs 終端對沖效應
  - 驗證: 個股拆解 + 跨產業檢驗
- [ ] 多產業歸屬規則：一家公司若跨多產業，主產業如何決定
  - 候選：(a) FinLab 主產業 join，(b) 取營收最大產業，(c) 全部保留做加權
- [ ] 嚴格 walk-forward 驗證：rolling 訓練/測試, 確認 Phase 8/11 的最佳 (N, K) 不是 in-sample lookback
- [ ] 個股層級因子化：把 IC 設計訊號傳導模型轉為 firm-level alpha factor (Cohen-Frazzini 風)
- [ ] 鏈間網絡分析：以 link 矩陣計算 centrality / community detection, 系統化定義產業 hub

## 注意事項

- 5xxx 系列（區塊鏈、AI、雲端、資安、大數據、體驗、運動）為「新興主題分類」，position 用自訂維度（如「核心技術」「應用與服務」），非傳統上中下游 → 用 `is_standard_chain` 欄位區分
- 5 個產業（軟體服務 R000、其他 X000、文化創意 Y000、金融 U000、交通運輸 T000）網站本身無上中下游分類，position 為 NaN
- 「其他」(X000) 涵蓋 346 家但無分類，因子研究時通常排除
