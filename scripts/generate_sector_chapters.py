#!/usr/bin/env python3
"""
Generate GICS Level 1 Sector Chapters via Gemma 4.

Reads synthesized company summaries from ninja/synthesized/GICS Level 1/{Sector}/,
compresses them into a compact per-ticker table, calls Gemma 4 once per sector,
and writes Beige Book-style sector chapters to industry_reports/{Sector}_05-03-26.md.

Usage:
    HF_TOKEN=hf_xxx python3 scripts/generate_sector_chapters.py
"""

import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from huggingface_hub import InferenceClient

MODEL_ID  = "google/gemma-4-31B-it"
HF_TOKEN  = os.environ.get("HF_TOKEN", "")
REPO_ROOT = Path(__file__).resolve().parent.parent
L1_BASE   = REPO_ROOT / "ninja" / "synthesized" / "GICS Level 1"
UNIVERSE  = REPO_ROOT / "ninja" / "sp500_universe.csv"
OUT_DIR   = REPO_ROOT / "industry_reports"
DATE_TAG  = "05-03-26"
MAX_CHARS = 90000
FORCE     = os.environ.get("FORCE", "").lower() in ("1", "true", "yes")

SUB_INDUSTRY_TO_GROUP = {
    "Oil & Gas Equipment & Services": "Energy Equipment & Services",
    "Integrated Oil & Gas": "Oil, Gas & Consumable Fuels",
    "Oil & Gas Exploration & Production": "Oil, Gas & Consumable Fuels",
    "Oil & Gas Refining & Marketing": "Oil, Gas & Consumable Fuels",
    "Oil & Gas Storage & Transportation": "Oil, Gas & Consumable Fuels",
    "Coal & Consumable Fuels": "Oil, Gas & Consumable Fuels",
    "Commodity Chemicals": "Chemicals",
    "Diversified Chemicals": "Chemicals",
    "Fertilizers & Agricultural Chemicals": "Chemicals",
    "Industrial Gases": "Chemicals",
    "Specialty Chemicals": "Chemicals",
    "Construction Materials": "Construction Materials",
    "Metal, Glass & Plastic Containers": "Containers & Packaging",
    "Paper & Plastic Packaging Products & Materials": "Containers & Packaging",
    "Aluminum": "Metals & Mining",
    "Copper": "Metals & Mining",
    "Diversified Metals & Mining": "Metals & Mining",
    "Gold": "Metals & Mining",
    "Precious Metals & Minerals": "Metals & Mining",
    "Silver": "Metals & Mining",
    "Steel": "Metals & Mining",
    "Forest Products": "Paper & Forest Products",
    "Paper Products": "Paper & Forest Products",
    "Aerospace & Defense": "Capital Goods",
    "Agricultural & Farm Machinery": "Capital Goods",
    "Building Products": "Capital Goods",
    "Construction & Engineering": "Capital Goods",
    "Construction Machinery & Heavy Transportation Equipment": "Capital Goods",
    "Electrical Components & Equipment": "Capital Goods",
    "Heavy Electrical Equipment": "Capital Goods",
    "Industrial Conglomerates": "Capital Goods",
    "Industrial Machinery & Supplies & Components": "Capital Goods",
    "Trading Companies & Distributors": "Capital Goods",
    "Commercial Printing": "Commercial & Professional Services",
    "Diversified Support Services": "Commercial & Professional Services",
    "Environmental & Facilities Services": "Commercial & Professional Services",
    "Human Resource & Employment Services": "Commercial & Professional Services",
    "Office Services & Supplies": "Commercial & Professional Services",
    "Research & Consulting Services": "Commercial & Professional Services",
    "Security & Alarm Services": "Commercial & Professional Services",
    "Air Freight & Logistics": "Transportation",
    "Cargo Ground Transportation": "Transportation",
    "Marine Transportation": "Transportation",
    "Passenger Airlines": "Transportation",
    "Passenger Ground Transportation": "Transportation",
    "Rail Transportation": "Transportation",
    "Automobile Manufacturers": "Automobiles & Components",
    "Automotive Parts & Equipment": "Automobiles & Components",
    "Tires & Rubber": "Automobiles & Components",
    "Apparel, Accessories & Luxury Goods": "Consumer Durables & Apparel",
    "Consumer Electronics": "Consumer Durables & Apparel",
    "Footwear": "Consumer Durables & Apparel",
    "Home Furnishings": "Consumer Durables & Apparel",
    "Homebuilding": "Consumer Durables & Apparel",
    "Household Appliances": "Consumer Durables & Apparel",
    "Housewares & Specialties": "Consumer Durables & Apparel",
    "Leisure Products": "Consumer Durables & Apparel",
    "Apparel Retail": "Consumer Discretionary Distribution & Retail",
    "Automotive Retail": "Consumer Discretionary Distribution & Retail",
    "Broadline Retail": "Consumer Discretionary Distribution & Retail",
    "Computer & Electronics Retail": "Consumer Discretionary Distribution & Retail",
    "Distributors": "Consumer Discretionary Distribution & Retail",
    "Home Improvement Retail": "Consumer Discretionary Distribution & Retail",
    "Homefurnishing Retail": "Consumer Discretionary Distribution & Retail",
    "Other Specialty Retail": "Consumer Discretionary Distribution & Retail",
    "Casinos & Gaming": "Consumer Services",
    "Hotels, Resorts & Cruise Lines": "Consumer Services",
    "Leisure Facilities": "Consumer Services",
    "Restaurants": "Consumer Services",
    "Specialized Consumer Services": "Consumer Services",
    "Consumer Staples Merchandise Retail": "Consumer Staples Distribution & Retail",
    "Food Distributors": "Consumer Staples Distribution & Retail",
    "Food Retail": "Consumer Staples Distribution & Retail",
    "Agricultural Products & Services": "Food, Beverage & Tobacco",
    "Brewers": "Food, Beverage & Tobacco",
    "Distillers & Vintners": "Food, Beverage & Tobacco",
    "Packaged Foods & Meats": "Food, Beverage & Tobacco",
    "Soft Drinks & Non-alcoholic Beverages": "Food, Beverage & Tobacco",
    "Tobacco": "Food, Beverage & Tobacco",
    "Household Products": "Household & Personal Products",
    "Personal Care Products": "Household & Personal Products",
    "Health Care Distributors": "Health Care Equipment & Services",
    "Health Care Equipment": "Health Care Equipment & Services",
    "Health Care Facilities": "Health Care Equipment & Services",
    "Health Care Services": "Health Care Equipment & Services",
    "Health Care Supplies": "Health Care Equipment & Services",
    "Health Care Technology": "Health Care Equipment & Services",
    "Managed Health Care": "Health Care Equipment & Services",
    "Biotechnology": "Pharmaceuticals, Biotechnology & Life Sciences",
    "Life Sciences Tools & Services": "Pharmaceuticals, Biotechnology & Life Sciences",
    "Pharmaceuticals": "Pharmaceuticals, Biotechnology & Life Sciences",
    "Diversified Banks": "Banks",
    "Regional Banks": "Banks",
    "Asset Management & Custody Banks": "Financial Services",
    "Consumer Finance": "Financial Services",
    "Financial Exchanges & Data": "Financial Services",
    "Investment Banking & Brokerage": "Financial Services",
    "Multi-Sector Holdings": "Financial Services",
    "Other Diversified Financial Services": "Financial Services",
    "Transaction & Payment Processing Services": "Financial Services",
    "Insurance Brokers": "Insurance",
    "Life & Health Insurance": "Insurance",
    "Multi-line Insurance": "Insurance",
    "Property & Casualty Insurance": "Insurance",
    "Reinsurance": "Insurance",
    "Application Software": "Software & Services",
    "Data Processing & Outsourced Services": "Software & Services",
    "IT Consulting & Other Services": "Software & Services",
    "Internet Services & Infrastructure": "Software & Services",
    "Systems Software": "Software & Services",
    "Communications Equipment": "Technology Hardware & Equipment",
    "Electronic Components": "Technology Hardware & Equipment",
    "Electronic Equipment & Instruments": "Technology Hardware & Equipment",
    "Electronic Manufacturing Services": "Technology Hardware & Equipment",
    "Technology Distributors": "Technology Hardware & Equipment",
    "Technology Hardware, Storage & Peripherals": "Technology Hardware & Equipment",
    "Semiconductor Materials & Equipment": "Semiconductors & Semiconductor Equipment",
    "Semiconductors": "Semiconductors & Semiconductor Equipment",
    "Integrated Telecommunication Services": "Telecommunication Services",
    "Wireless Telecommunication Services": "Telecommunication Services",
    "Advertising": "Media & Entertainment",
    "Broadcasting": "Media & Entertainment",
    "Cable & Satellite": "Media & Entertainment",
    "Interactive Home Entertainment": "Media & Entertainment",
    "Interactive Media & Services": "Media & Entertainment",
    "Movies & Entertainment": "Media & Entertainment",
    "Publishing": "Media & Entertainment",
    "Electric Utilities": "Utilities",
    "Gas Utilities": "Utilities",
    "Independent Power Producers & Energy Traders": "Utilities",
    "Multi-Utilities": "Utilities",
    "Water Utilities": "Utilities",
    "Data Center REITs": "Equity REITs",
    "Diversified REITs": "Equity REITs",
    "Health Care REITs": "Equity REITs",
    "Hotel & Resort REITs": "Equity REITs",
    "Industrial REITs": "Equity REITs",
    "Multi-Family Residential REITs": "Equity REITs",
    "Office REITs": "Equity REITs",
    "Other Specialized REITs": "Equity REITs",
    "Retail REITs": "Equity REITs",
    "Self-Storage REITs": "Equity REITs",
    "Single-Family Residential REITs": "Equity REITs",
    "Telecom Tower REITs": "Equity REITs",
    "Timber REITs": "Equity REITs",
    "Real Estate Services": "Real Estate Management & Development",
}


PROMPT_TEMPLATE = """\
[ROLE]: Federal Reserve economic analyst writing a Beige Book-style sector chapter.
[TASK]: Synthesize the per-company economic signals below into a Sector Chapter for {sector}. \
Produce exactly 3 sections:

1. Sector Consensus — identify common themes across the majority of firms. Use Beige Book qualifiers \
(stable, robust, slight, moderate, modest). Name at least 4–5 specific companies as examples. \
Call out product lines, geographies, or end-markets where signals diverge within the consensus.

2. Divergences & Outliers — name specific companies and their industry groups (shown in brackets) \
whose signals contradict the broader sector trend. Explain the mechanism, not just the direction \
(e.g., tariff exposure, end-market mix, customer concentration).

3. Key Thematic Headings — provide 3–4 sentences each for the following four headings. \
Cite named companies for each heading. Avoid vague generalities — ground every sentence in a \
specific signal from the data.
   - Consumer/Business Demand
   - Labor Markets & Wages
   - Price & Cost Pressures
   - Inventory & Logistics

Constraints: Do NOT mention stock prices, EPS, PE ratios, or any financial metric. \
Neutral, observational, professional tone. Reference specific companies and industry groups by name throughout.
[FORMAT]: Markdown with ## section headers. ~700 words total.
[DATA]:
{data}"""


def read_body(path: Path) -> str:
    body = path.read_text(encoding="utf-8", errors="replace")
    sep = body.find("=" * 10)
    return body[sep + 60:].strip() if sep != -1 else body.strip()


def build_data_table(sector_dir: Path, universe: dict) -> str:
    files = sorted(sector_dir.glob("*_synthesized.txt"))
    rows = []
    for f in files:
        ticker = f.name.split("_")[0]
        sub = universe.get(ticker, {}).get("sub_industry", "")
        group = SUB_INDUSTRY_TO_GROUP.get(sub, sub or "Other")
        body = read_body(f)
        rows.append(f"--- {ticker} [{group}] ---\n{body}")
    table = "\n\n".join(rows)
    return table[:MAX_CHARS]


def call_gemma(sector: str, data: str) -> str:
    client = InferenceClient(model=MODEL_ID, token=HF_TOKEN, timeout=300)
    messages = [{"role": "user", "content": PROMPT_TEMPLATE.format(sector=sector, data=data)}]
    for attempt in range(5):
        try:
            stream = client.chat.completions.create(
                messages=messages,
                temperature=0.2,
                max_tokens=2200,
                stream=True,
            )
            parts = []
            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta.content
                if delta:
                    parts.append(delta)
                    print(delta, end="", flush=True)
            print()
            return "".join(parts)
        except Exception as e:
            is_rate_limit = any(x in str(e) for x in (
                "429", "503", "Too Many Requests", "Service Temporarily Unavailable"
            ))
            if is_rate_limit and attempt < 4:
                wait = 60 * (attempt + 1)
                print(f"\n  rate-limited — waiting {wait}s (attempt {attempt+1}/5)", flush=True)
                time.sleep(wait)
            else:
                raise


def main():
    if not HF_TOKEN:
        sys.exit("ERROR: HF_TOKEN env var required")

    df = pd.read_csv(UNIVERSE)
    universe = {
        row.ticker: {"sub_industry": row.gics_sub_industry}
        for row in df.itertuples()
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sectors = sorted(d.name for d in L1_BASE.iterdir() if d.is_dir())
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    for i, sector in enumerate(sectors, 1):
        out_path = OUT_DIR / f"{sector}_{DATE_TAG}.md"
        if out_path.exists() and not FORCE:
            print(f"[{i}/{len(sectors)}] {sector} — skipped (exists)")
            continue

        print(f"\n[{i}/{len(sectors)}] {sector} — building data table ...", flush=True)
        sector_dir = L1_BASE / sector
        data = build_data_table(sector_dir, universe)
        n_files = len(list(sector_dir.glob("*_synthesized.txt")))
        print(f"  {n_files} companies, {len(data)} chars → calling Gemma ...", flush=True)

        try:
            chapter = call_gemma(sector, data)
        except Exception as e:
            print(f"  FAILED: {e}")
            continue

        header = (
            f"# {sector} — Sector Chapter | Q1 2026\n"
            f"*Aggregate Economic Report | Generated: {ts} | Model: {MODEL_ID}*\n\n"
        )
        out_path.write_text(header + chapter, encoding="utf-8")
        print(f"  → wrote {out_path.name}")

        try:
            subprocess.run(["git", "-C", str(REPO_ROOT), "add", str(out_path)], check=True)
            subprocess.run(
                ["git", "-C", str(REPO_ROOT), "commit", "-m",
                 f"Sector Chapter: {sector} — {DATE_TAG} 🤖"],
                check=True,
            )
            subprocess.run(["git", "-C", str(REPO_ROOT), "pull", "--rebase", "--autostash"], check=True)
            subprocess.run(["git", "-C", str(REPO_ROOT), "push"], check=True)
            print(f"  → committed + pushed")
        except subprocess.CalledProcessError as e:
            print(f"  git error: {e}")

        if i < len(sectors):
            time.sleep(5)

    print("\nAll sectors processed.")


if __name__ == "__main__":
    main()
