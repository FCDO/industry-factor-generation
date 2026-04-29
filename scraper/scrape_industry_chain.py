"""
爬取 TPEx 產業價值鏈資訊平台 (https://ic.tpex.org.tw/)

輸出 industry_chain.csv，欄位：
    stock_id        股票代號
    name            公司簡稱
    market_type     本國上市 / 外國上市 / 本國上櫃 / 本國興櫃 / 外國知名企業 / 創櫃
    industry_code   產業鏈代碼 (如 D000)
    industry_name   產業鏈名稱 (如 半導體)
    position        上游 / 中游 / 下游
    sub_code        子分類代碼 (如 D100)
    sub_name        子分類名稱 (如 IC設計)
"""
from __future__ import annotations

import re
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE = "https://ic.tpex.org.tw"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}
SLEEP = 0.8  # 禮貌性間隔


def fetch(url: str) -> str:
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    r.encoding = "utf-8"
    return r.text


def parse_industry_index(html: str) -> dict[str, str]:
    """從首頁抓 (industry_code -> industry_name)。"""
    soup = BeautifulSoup(html, "html.parser")
    out: dict[str, str] = {}
    for a in soup.select('a[href*="introduce.php?ic="]'):
        href = a.get("href", "")
        m = re.search(r"ic=([A-Z0-9]{4})", href)
        if not m:
            continue
        code = m.group(1)
        # 名稱在最後一個 <span> 中
        spans = a.find_all("span")
        name = ""
        for sp in spans:
            txt = sp.get_text(strip=True)
            cls = " ".join(sp.get("class") or [])
            if txt and "ic-sprite" not in cls:
                name = txt
        if code and name and code not in out:
            out[code] = name
    return out


STANDARD_POSITIONS = {"上游", "中游", "下游"}


def parse_chain_positions(soup: BeautifulSoup) -> dict[str, tuple[str, str]]:
    """
    從產業頁 chain-panel 解析 sub_code -> (position, sub_name).
    position 多數為 上游/中游/下游；新興主題產業可能為自訂分類
    （如「核心技術」「應用與服務」），原樣保留。
    """
    result: dict[str, tuple[str, str]] = {}
    chain_panel = soup.find("div", class_="chain-panel")
    if not chain_panel:
        return result

    for chain in chain_panel.find_all("div", class_="chain", recursive=True):
        title_tag = chain.find("div", class_="chain-title-panel")
        if not title_tag:
            continue
        position = title_tag.get_text(strip=True)
        for sub in chain.find_all("div", class_="company-chain-panel"):
            sub_id = sub.get("id", "")
            m = re.match(r"ic_link_(\w+)", sub_id)
            if not m:
                continue
            sub_code = m.group(1)
            sub_name = sub.get_text(separator="", strip=True)
            result[sub_code] = (position, sub_name)
    return result


# 子分類內公司列表的市場別標籤
MARKET_PATTERN = re.compile(
    r"^(本國上市公司|外國上市公司|本國上櫃公司|本國興櫃公司|外國知名企業|創櫃公司)"
)


def parse_company_lists(soup: BeautifulSoup) -> dict[str, list[dict]]:
    """
    解析所有 companyList_<sub_code> 區塊，回傳 sub_code -> [公司 dict].
    每個公司 dict: {stock_id, name, market_type}.
    """
    out: dict[str, list[dict]] = {}
    for div in soup.find_all("div", id=re.compile(r"^companyList_")):
        m = re.match(r"companyList_(\w+)", div.get("id", ""))
        if not m:
            continue
        sub_code = m.group(1)
        companies: list[dict] = []
        current_market = ""
        # 走訪所有 td.company
        for td in div.find_all("td", class_="company"):
            b = td.find("b")
            if b:
                txt = b.get_text(strip=True)
                m2 = MARKET_PATTERN.match(txt)
                if m2:
                    current_market = m2.group(1)
                continue
            a = td.find("a", href=re.compile(r"stk_code="))
            if not a:
                continue
            href = a.get("href", "")
            mc = re.search(r"stk_code=([A-Za-z0-9.\-]+)", href)
            if not mc:
                continue
            stock_id = mc.group(1)
            name = a.get("title") or a.get_text(strip=True)
            companies.append(
                {
                    "stock_id": stock_id,
                    "name": name,
                    "market_type": current_market,
                }
            )
        if companies:
            out[sub_code] = companies
    return out


def scrape_industry(industry_code: str, industry_name: str) -> list[dict]:
    """爬單一產業頁，回傳扁平化的公司紀錄。"""
    url = f"{BASE}/introduce.php?ic={industry_code}"
    html = fetch(url)
    soup = BeautifulSoup(html, "html.parser")

    chain_pos = parse_chain_positions(soup)
    company_lists = parse_company_lists(soup)

    rows: list[dict] = []
    for sub_code, companies in company_lists.items():
        position, sub_name = chain_pos.get(sub_code, ("", ""))
        for c in companies:
            rows.append(
                {
                    **c,
                    "industry_code": industry_code,
                    "industry_name": industry_name,
                    "position": position,
                    "is_standard_chain": position in STANDARD_POSITIONS,
                    "sub_code": sub_code,
                    "sub_name": sub_name,
                }
            )
    return rows


def main() -> None:
    out_dir = Path(__file__).parent.parent
    out_csv = out_dir / "industry_chain.csv"

    print("[1/2] 抓取產業列表...")
    home_html = fetch(f"{BASE}/")
    industries = parse_industry_index(home_html)
    print(f"  找到 {len(industries)} 個產業")

    print("[2/2] 逐一抓取產業頁...")
    all_rows: list[dict] = []
    for i, (code, name) in enumerate(sorted(industries.items()), 1):
        try:
            rows = scrape_industry(code, name)
            print(f"  [{i:>2}/{len(industries)}] {code} {name}: {len(rows)} 家")
            all_rows.extend(rows)
        except Exception as e:
            print(f"  [{i:>2}/{len(industries)}] {code} {name}: 失敗 - {e}")
        time.sleep(SLEEP)

    df = pd.DataFrame(all_rows)
    df = df[
        [
            "stock_id",
            "name",
            "market_type",
            "industry_code",
            "industry_name",
            "position",
            "is_standard_chain",
            "sub_code",
            "sub_name",
        ]
    ]
    # 同一公司在同一 sub_code 下因更細產品次分類可能重複，去重
    before = len(df)
    df = df.drop_duplicates(
        subset=["stock_id", "industry_code", "sub_code"]
    ).reset_index(drop=True)
    print(f"  去重 (stock_id × industry × sub_code): {before} -> {len(df)}")

    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\n完成：{len(df)} 筆紀錄 -> {out_csv}")
    print(f"  涵蓋公司數（去重）: {df['stock_id'].nunique()}")
    print(f"  涵蓋產業數: {df['industry_code'].nunique()}")
    print("\n位置分佈：")
    print(df["position"].value_counts())


if __name__ == "__main__":
    main()
