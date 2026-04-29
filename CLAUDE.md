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
└── scraper/
    └── scrape_industry_chain.py    # TPEx 爬蟲
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

### 待辦
- [ ] 多產業歸屬規則：一家公司若跨多產業，主產業如何決定
  - 候選：(a) FinLab 主產業 join，(b) 取營收最大產業，(c) 全部保留做加權
- [ ] 串接 FinLab 報酬資料，產出 (stock_id, date, return) × industry_chain 的合併表
- [ ] 產業 × 產業共現公司數 → 鄰接矩陣
- [ ] 因子建構：上游 vs 下游報酬差、領先落後（Granger）、產業內位置 spread

## 注意事項

- 5xxx 系列（區塊鏈、AI、雲端、資安、大數據、體驗、運動）為「新興主題分類」，position 用自訂維度（如「核心技術」「應用與服務」），非傳統上中下游 → 用 `is_standard_chain` 欄位區分
- 5 個產業（軟體服務 R000、其他 X000、文化創意 Y000、金融 U000、交通運輸 T000）網站本身無上中下游分類，position 為 NaN
- 「其他」(X000) 涵蓋 346 家但無分類，因子研究時通常排除
