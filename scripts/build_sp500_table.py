import io
import time
import requests
import pandas as pd
import yfinance as yf

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
OUT_PATH = "ninja/sp500_universe.csv"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; research-bot/1.0)"}

def fetch_sp500_list():
    html = requests.get(WIKI_URL, headers=HEADERS, timeout=30).text
    df = pd.read_html(io.StringIO(html))[0]
    return df[["Symbol", "Security", "GICS Sector", "GICS Sub-Industry"]].rename(columns={
        "Symbol": "ticker",
        "Security": "company_name",
        "GICS Sector": "gics_sector",
        "GICS Sub-Industry": "gics_sub_industry",
    })

def fetch_market_caps(tickers):
    caps = {}
    total = len(tickers)
    for i, sym in enumerate(tickers, 1):
        try:
            caps[sym] = yf.Ticker(sym).fast_info.market_cap
        except Exception:
            caps[sym] = None
        if i % 50 == 0:
            print(f"  {i}/{total} fetched...")
            time.sleep(1)
    return caps

def main():
    print("Fetching S&P 500 list from Wikipedia...")
    df = fetch_sp500_list()
    print(f"  {len(df)} companies loaded")

    print("Fetching market caps from yfinance...")
    caps = fetch_market_caps(df["ticker"].tolist())

    df["market_cap_usd"] = df["ticker"].map(caps)
    df = df.sort_values("market_cap_usd", ascending=False).reset_index(drop=True)

    df.to_csv(OUT_PATH, index=False)
    print(f"Saved {len(df)} rows to {OUT_PATH}")

    # Quick sanity check
    top3 = df.head(3)[["ticker", "market_cap_usd"]].to_string(index=False)
    print(f"Top 3 by market cap:\n{top3}")

if __name__ == "__main__":
    main()
