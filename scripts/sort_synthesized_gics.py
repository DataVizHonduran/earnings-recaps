import glob
import os
import shutil
import sys

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UNIVERSE_CSV = os.path.join(REPO, "ninja", "sp500_universe.csv")
SYNTHESIZED = os.path.join(REPO, "ninja", "synthesized")
L1_BASE = os.path.join(SYNTHESIZED, "GICS Level 1")
L2_BASE = os.path.join(SYNTHESIZED, "GICS Level 2")

SUB_INDUSTRY_TO_GROUP = {
    # Energy
    "Oil & Gas Equipment & Services": "Energy Equipment & Services",
    "Integrated Oil & Gas": "Oil, Gas & Consumable Fuels",
    "Oil & Gas Exploration & Production": "Oil, Gas & Consumable Fuels",
    "Oil & Gas Refining & Marketing": "Oil, Gas & Consumable Fuels",
    "Oil & Gas Storage & Transportation": "Oil, Gas & Consumable Fuels",
    "Coal & Consumable Fuels": "Oil, Gas & Consumable Fuels",
    # Materials
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
    # Industrials
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
    # Consumer Discretionary
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
    # Consumer Staples
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
    # Health Care
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
    # Financials
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
    # Information Technology
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
    # Communication Services
    "Integrated Telecommunication Services": "Telecommunication Services",
    "Wireless Telecommunication Services": "Telecommunication Services",
    "Advertising": "Media & Entertainment",
    "Broadcasting": "Media & Entertainment",
    "Cable & Satellite": "Media & Entertainment",
    "Interactive Home Entertainment": "Media & Entertainment",
    "Interactive Media & Services": "Media & Entertainment",
    "Movies & Entertainment": "Media & Entertainment",
    "Publishing": "Media & Entertainment",
    # Utilities
    "Electric Utilities": "Utilities",
    "Gas Utilities": "Utilities",
    "Independent Power Producers & Energy Traders": "Utilities",
    "Multi-Utilities": "Utilities",
    "Water Utilities": "Utilities",
    # Real Estate
    "Data Center REITs": "Equity Real Estate Investment Trusts (REITs)",
    "Diversified REITs": "Equity Real Estate Investment Trusts (REITs)",
    "Health Care REITs": "Equity Real Estate Investment Trusts (REITs)",
    "Hotel & Resort REITs": "Equity Real Estate Investment Trusts (REITs)",
    "Industrial REITs": "Equity Real Estate Investment Trusts (REITs)",
    "Multi-Family Residential REITs": "Equity Real Estate Investment Trusts (REITs)",
    "Office REITs": "Equity Real Estate Investment Trusts (REITs)",
    "Other Specialized REITs": "Equity Real Estate Investment Trusts (REITs)",
    "Retail REITs": "Equity Real Estate Investment Trusts (REITs)",
    "Self-Storage REITs": "Equity Real Estate Investment Trusts (REITs)",
    "Single-Family Residential REITs": "Equity Real Estate Investment Trusts (REITs)",
    "Telecom Tower REITs": "Equity Real Estate Investment Trusts (REITs)",
    "Timber REITs": "Equity Real Estate Investment Trusts (REITs)",
    "Real Estate Services": "Real Estate Management & Development",
}


def load_universe():
    df = pd.read_csv(UNIVERSE_CSV)
    return {
        row.ticker: (row.gics_sector, row.gics_sub_industry)
        for row in df.itertuples()
    }


def main():
    universe = load_universe()

    files = [f for f in glob.glob(os.path.join(SYNTHESIZED, "*.txt"))]
    sorted_count = skipped = unknown = 0

    for src in files:
        fname = os.path.basename(src)
        ticker = fname.split("_")[0]

        if ticker not in universe:
            print(f"UNKNOWN ticker: {ticker} ({fname})", file=sys.stderr)
            unknown += 1
            continue

        sector, sub_industry = universe[ticker]
        group = SUB_INDUSTRY_TO_GROUP.get(sub_industry)
        if group is None:
            print(f"UNMAPPED sub-industry: {sub_industry!r} ({ticker})", file=sys.stderr)
            unknown += 1
            continue

        l1_dest = os.path.join(L1_BASE, sector, fname)
        l2_dest = os.path.join(L2_BASE, group, fname)

        already = os.path.exists(l1_dest) and os.path.exists(l2_dest)

        os.makedirs(os.path.dirname(l1_dest), exist_ok=True)
        os.makedirs(os.path.dirname(l2_dest), exist_ok=True)
        shutil.copy2(src, l1_dest)
        shutil.copy2(src, l2_dest)

        if already:
            skipped += 1
        else:
            sorted_count += 1

    total = len(files)
    print(f"Done. {total} files total: {sorted_count} newly sorted, {skipped} refreshed, {unknown} unknown.")


if __name__ == "__main__":
    main()
