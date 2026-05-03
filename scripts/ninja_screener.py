"""
Daily SP500 earnings screener via api-ninjas.com.
Selects 30 random tickers, fetches transcripts, keeps those within last 45 days,
saves to ninja/. Git commit/push handled by the workflow.
"""

import json
import os
import random
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import requests

# ── Config ────────────────────────────────────────────────────────────────────
API_KEY    = os.environ.get("API_NINJAS_KEY", "")
REPO_ROOT  = Path(__file__).resolve().parent.parent
NINJA_DIR  = REPO_ROOT / "ninja"
SP500_JSON = REPO_ROOT / "sp500.json"

SAMPLE_SIZE = 30
WINDOW_DAYS = 45
SLEEP_SEC   = 0.35   # ~50 req/min free-tier headroom

# ── Quarter → estimated report mid-date ───────────────────────────────────────
# (year_offset, month, day) — Q4 reports land in Feb of following year
QUARTER_MAP = {1: (0, 5, 1), 2: (0, 8, 1), 3: (0, 11, 1), 4: (1, 2, 1)}


def estimated_report_date(year: int, quarter: int) -> date:
    yr_off, mo, day = QUARTER_MAP.get(quarter, (0, 5, 1))
    return date(year + yr_off, mo, day)


def within_window(year: int, quarter: int) -> bool:
    cutoff = date.today() - timedelta(days=WINDOW_DAYS)
    return estimated_report_date(year, quarter) >= cutoff


# ── API ───────────────────────────────────────────────────────────────────────
def fetch_transcript(ticker: str) -> dict | None:
    try:
        r = requests.get(
            "https://api.api-ninjas.com/v1/earningstranscript",
            params={"ticker": ticker},
            headers={"X-Api-Key": API_KEY},
            timeout=15,
        )
        if r.status_code == 404:
            return None
        r.raise_for_status()
        data = r.json()
        if isinstance(data, list):
            return data[0] if data else None
        return data or None
    except Exception as e:
        print(f"  [{ticker}] error: {e}")
        return None


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    if not API_KEY:
        sys.exit("API_NINJAS_KEY env var not set")

    NINJA_DIR.mkdir(exist_ok=True)

    tickers: list[str] = json.loads(SP500_JSON.read_text())
    batch = random.sample(tickers, SAMPLE_SIZE)
    today = date.today().isoformat()
    saved = []

    print(f"\n{today} — screening {SAMPLE_SIZE} random SP500 tickers via api-ninjas\n")

    for ticker in batch:
        time.sleep(SLEEP_SEC)
        data = fetch_transcript(ticker)

        if not data:
            print(f"  {ticker:6s}  no data")
            continue

        year    = data.get("year") or data.get("fiscal_year")
        quarter = data.get("quarter") or data.get("fiscal_quarter")
        company = data.get("company", ticker)
        transcript = data.get("transcript", "")

        if not year or not quarter or not transcript:
            print(f"  {ticker:6s}  incomplete (year={year} q={quarter} len={len(transcript)})")
            continue

        year, quarter = int(year), int(quarter)
        est = estimated_report_date(year, quarter)
        ok  = within_window(year, quarter)

        print(f"  {ticker:6s}  {year} Q{quarter}  est={est}  {'IN WINDOW' if ok else 'too old'}")

        if not ok:
            continue

        fname = f"{ticker}_{year}_Q{quarter}.txt"
        fpath = NINJA_DIR / fname

        if fpath.exists():
            print(f"         already saved")
            continue

        fpath.write_text(
            f"Ticker: {ticker}\nCompany: {company}\nYear: {year}\n"
            f"Quarter: {quarter}\nEstimated Report Date: {est}\n\n"
            + transcript,
            encoding="utf-8",
        )
        print(f"         saved → ninja/{fname}")
        saved.append(fname)

    print(f"\n{len(saved)} transcript(s) saved.")
    if not saved:
        print("Nothing new — workflow will skip commit.")


if __name__ == "__main__":
    main()
