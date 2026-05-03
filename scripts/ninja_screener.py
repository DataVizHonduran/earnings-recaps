"""
Daily SP500 earnings screener via SEC EDGAR.
Selects 30 random tickers, finds Item 2.02 8-K filings (earnings only) within
last 45 days, fetches the exhibit, and saves to ninja/.
Git commit/push handled by the workflow.
"""

import json
import os
import random
import re
import time
from datetime import date, timedelta
from pathlib import Path

import requests

# ── Config ────────────────────────────────────────────────────────────────────
HEADERS    = {"User-Agent": "ResearchBot jeannealbertoreading@gmail.com"}
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
BASE_ARCHIVE = "https://www.sec.gov/Archives/edgar/data"

REPO_ROOT  = Path(__file__).resolve().parent.parent
NINJA_DIR  = REPO_ROOT / "ninja"
SP500_JSON = REPO_ROOT / "sp500.json"

SAMPLE_SIZE = 30
WINDOW_DAYS = 45
SLEEP_SEC   = 0.15   # SEC rate limit: 10 req/sec

TRANSCRIPT_MARKERS = [
    "Operator:", "OPERATOR:", "Question-and-Answer", "Q&A SESSION",
    "Good morning,", "Good afternoon,", "Thank you for standing by",
    "Thank you for joining", "conference call", "CONFERENCE CALL",
]

# ── EDGAR helpers ─────────────────────────────────────────────────────────────
def get_cik(ticker: str) -> tuple[str, str]:
    r = requests.get(TICKERS_URL, headers=HEADERS, timeout=15)
    r.raise_for_status()
    for entry in r.json().values():
        if entry["ticker"] == ticker.upper():
            return str(entry["cik_str"]).zfill(10), entry["title"]
    raise ValueError(f"{ticker} not found in EDGAR")


def get_earnings_8ks(cik_padded: str) -> list[dict]:
    """Return Item 2.02 8-K filings filed within WINDOW_DAYS."""
    cutoff = (date.today() - timedelta(days=WINDOW_DAYS)).isoformat()
    url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"
    r = requests.get(url, headers=HEADERS, timeout=15)
    r.raise_for_status()
    recent = r.json()["filings"]["recent"]
    result = []
    for form, fdate, acc, primary, items in zip(
        recent["form"], recent["filingDate"],
        recent["accessionNumber"], recent["primaryDocument"],
        recent["items"],
    ):
        if fdate < cutoff:
            break  # filings are newest-first; stop once past window
        if form == "8-K" and "2.02" in str(items):
            result.append({"date": fdate, "accession": acc, "primary": primary})
    return result


def get_exhibit_urls(cik_raw: str, accession: str, primary_doc: str) -> list[tuple[int, str]]:
    acc_clean = accession.replace("-", "")
    base = f"{BASE_ARCHIVE}/{cik_raw}/{acc_clean}"
    r = requests.get(f"{base}/{primary_doc}", headers=HEADERS, timeout=15)
    r.raise_for_status()
    hrefs = re.findall(r'(?<![:\w])href=["\']([^"\'#?]+)["\']', r.text, re.IGNORECASE)
    seen: dict[str, int] = {}
    for h in hrefs:
        name = h.split("/")[-1].lower()
        if not name or name == primary_doc.lower():
            continue
        ext = name.rsplit(".", 1)[-1] if "." in name else ""
        if ext not in ("htm", "html", "txt", "pdf"):
            continue
        score = 10 if "transcript" in name else (5 if re.search(r"ex.?99|exhibit.?99", name) else 1)
        url = f"{base}/{name}"
        if url not in seen or seen[url] < score:
            seen[url] = score
    return sorted([(s, u) for u, s in seen.items()], key=lambda x: -x[0])


def fetch_text(url: str) -> str:
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    text = r.text
    if "<html" in text[:300].lower():
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"&nbsp;|&#160;", " ", text)
        text = re.sub(r"&amp;", "&", text)
        text = re.sub(r"&#8212;", "—", text)
        text = re.sub(r"&#[0-9]+;", " ", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def is_transcript(text: str) -> bool:
    return sum(1 for m in TRANSCRIPT_MARKERS if m in text) >= 2


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    NINJA_DIR.mkdir(exist_ok=True)

    tickers: list[str] = json.loads(SP500_JSON.read_text())
    batch = random.sample(tickers, SAMPLE_SIZE)
    today = date.today().isoformat()
    saved = []

    print(f"\n{today} — screening {SAMPLE_SIZE} random SP500 tickers via EDGAR (Item 2.02)\n")

    for ticker in batch:
        time.sleep(SLEEP_SEC)
        try:
            cik_padded, company = get_cik(ticker)
        except ValueError:
            print(f"  {ticker:6s}  not in EDGAR")
            continue
        except Exception as e:
            print(f"  {ticker:6s}  CIK error: {e}")
            continue

        cik_raw = str(int(cik_padded))

        time.sleep(SLEEP_SEC)
        try:
            filings = get_earnings_8ks(cik_padded)
        except Exception as e:
            print(f"  {ticker:6s}  submissions error: {e}")
            continue

        if not filings:
            print(f"  {ticker:6s}  no earnings 8-K in window")
            continue

        filing = filings[0]  # most recent qualifying filing
        fname = f"{ticker}_{filing['date']}.txt"
        fpath = NINJA_DIR / fname

        if fpath.exists():
            print(f"  {ticker:6s}  {filing['date']}  already saved")
            continue

        time.sleep(SLEEP_SEC)
        try:
            candidates = get_exhibit_urls(cik_raw, filing["accession"], filing["primary"])
        except Exception as e:
            print(f"  {ticker:6s}  exhibit list error: {e}")
            continue

        text = None
        source_url = ""
        for _, url in candidates:
            time.sleep(SLEEP_SEC)
            try:
                t = fetch_text(url)
                if len(t) >= 300:
                    text = t
                    source_url = url
                    break
            except Exception:
                continue

        if not text:
            print(f"  {ticker:6s}  {filing['date']}  no usable exhibit")
            continue

        label = "TRANSCRIPT" if is_transcript(text) else "PRESS_RELEASE"
        if label != "TRANSCRIPT":
            print(f"  {ticker:6s}  {filing['date']}  {label}  skipped")
            continue

        print(f"  {ticker:6s}  {filing['date']}  TRANSCRIPT  {len(text):,} chars  → ninja/{fname}")

        fpath.write_text(
            f"Ticker: {ticker}\nCompany: {company}\nFiled: {filing['date']}\n"
            f"Type: {label}\nSource: {source_url}\n\n" + text,
            encoding="utf-8",
        )
        saved.append(fname)

    print(f"\n{len(saved)} transcript(s) saved.")
    if not saved:
        print("Nothing new — workflow will skip commit.")


if __name__ == "__main__":
    main()
