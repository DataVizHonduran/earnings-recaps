#!/usr/bin/env python3
"""
Pricing Power Report — extracts the "Pricing Power & Inflation" section from each
synthesized company file, synthesizes per-sector narratives via Gemma, and writes
a self-contained HTML report to industry_reports/pricing_power_{DATE_TAG}.html.

Usage:
    HF_TOKEN=hf_xxx python3 scripts/generate_pricing_power_report.py
"""

import os
import re
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

_now      = datetime.now(timezone.utc)
DATE_TAG  = _now.strftime("%m-%d-%y")
_quarter  = f"Q{(_now.month - 1) // 3 + 1} {_now.year}"

SECTOR_ORDER = [
    "Consumer Discretionary",
    "Consumer Staples",
    "Energy",
    "Financials",
    "Health Care",
    "Industrials",
    "Information Technology",
    "Communication Services",
    "Materials",
    "Real Estate",
    "Utilities",
]

# Regex: capture text between "**Pricing Power & Inflation:**" and next "**" section header
_PRICING_RE = re.compile(
    r"\*\*Pricing Power & Inflation:\*\*\s*(.*?)(?=\n\*\*|\Z)",
    re.DOTALL | re.IGNORECASE,
)


def extract_pricing_section(text: str) -> str:
    """Pull the Pricing Power & Inflation block from a synthesized file."""
    m = _PRICING_RE.search(text)
    if not m:
        return ""
    return m.group(1).strip()


def load_sector_pricing(sector: str) -> str:
    """Gather pricing snippets for all companies in a sector."""
    sector_dir = L1_BASE / sector
    if not sector_dir.exists():
        return ""

    rows = []
    for path in sorted(sector_dir.glob("*_synthesized.txt")):
        ticker = path.name.split("_")[0]
        raw = path.read_text(encoding="utf-8", errors="replace")
        snippet = extract_pricing_section(raw)
        if snippet and "not discussed" not in snippet.lower():
            rows.append(f"--- {ticker} ---\n{snippet}")

    table = "\n\n".join(rows)
    return table[:80000]


SECTOR_PROMPT = """\
[ROLE]: Federal Reserve price-stability analyst writing a Beige Book-style sector chapter.
[TASK]: Based on the pricing signals below for the {sector} sector, write a 300-400 word
Pricing Power Chapter covering exactly 4 sections:

1. **Pricing Environment** — Are firms raising, holding, or cutting prices? Describe
   cadence (frequent/quarterly/one-off) and magnitude where evidence exists.

2. **Cost Pass-Through** — Are input cost increases (commodities, labor, freight, energy)
   being passed to customers, absorbed into margins, or offset by efficiency gains?
   Name at least 2-3 specific companies as examples.

3. **Leaders & Laggards** — Name 3-4 companies with notably strong pricing power
   vs. those conceding price or offering discounts. Explain the mechanism
   (brand strength, sole-source contracts, demand inelasticity, competitive pressure, etc.).

4. **Margin Outlook** — Net pricing impact on gross or operating margins:
   expanding, stable, or compressing? Tie to specific company signals.

Constraints: No EPS, stock prices, or financial ratios. Fed-speak qualifiers only
(modest, moderate, robust, slight, stable, softening). Cite specific company names throughout.
[FORMAT]: Markdown with ## section headers.
[DATA]:
{data}"""

EXEC_SUMMARY_PROMPT = """\
[ROLE]: Chief Economist at a major central bank assessing corporate pricing power.
[TASK]: Read the 11 GICS sector chapters below and write a 350-400 word Pricing Power
Executive Summary for {quarter}. Structure as:

**Aggregate Pricing Trend:** One paragraph (3-4 sentences) on the S&P 500-wide direction
of pricing — are companies broadly retaining, gaining, or losing pricing power vs. prior
quarter? Use Beige Book qualifiers.

**Strongest Sectors:** 2-3 sentences naming the sectors with the most durable pricing
power and the mechanism behind it (brand, inelastic demand, supply constraint, etc.).

**Weakest Sectors:** 2-3 sentences naming sectors facing price compression or discounting,
and the competitive or macro driver.

**Inflation Signal:** 2-3 sentences on what aggregate corporate pricing behavior implies
for consumer and producer price trajectories. Use measured language
("contacts noted", "several firms indicated", "reports suggest").

Constraints: No stock prices, EPS, or financial metrics. Neutral, central-bank tone.
[SECTOR CHAPTERS]:
{chapters}"""


def call_gemma(prompt: str) -> str:
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN not set")
    client = InferenceClient(model=MODEL_ID, token=HF_TOKEN, timeout=300)
    for attempt in range(5):
        try:
            stream = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=1200,
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


def md_to_html(text: str) -> str:
    lines = text.split("\n")
    out = []
    in_ul = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if in_ul:
                out.append("</ul>")
                in_ul = False
            out.append("")
            continue
        if stripped.startswith("## "):
            if in_ul:
                out.append("</ul>")
                in_ul = False
            out.append(f"<h2>{_inline(stripped[3:])}</h2>")
        elif stripped.startswith("### "):
            if in_ul:
                out.append("</ul>")
                in_ul = False
            out.append(f"<h3>{_inline(stripped[4:])}</h3>")
        elif stripped.startswith("- ") or stripped.startswith("* "):
            if not in_ul:
                out.append("<ul>")
                in_ul = True
            out.append(f"<li>{_inline(stripped[2:])}</li>")
        else:
            if in_ul:
                out.append("</ul>")
                in_ul = False
            out.append(f"<p>{_inline(stripped)}</p>")
    if in_ul:
        out.append("</ul>")
    return "\n".join(out)


def _inline(text: str) -> str:
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"\*(.+?)\*", r"<em>\1</em>", text)
    return text


def build_html(exec_html: str, sector_chapters: dict[str, str], generated_at: str) -> str:
    toc_items = "".join(
        f'<li><a href="#{s.lower().replace(" ", "-")}">{s}</a></li>'
        for s in sector_chapters
    )

    sector_html = ""
    for sector, body in sector_chapters.items():
        anchor = sector.lower().replace(" ", "-")
        sector_html += f"""
        <section class="sector" id="{anchor}">
            <div class="sector-header">
                <h2>{sector}</h2>
            </div>
            <div class="sector-body">
                {md_to_html(body)}
            </div>
        </section>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Pricing Power Monitor — {_quarter}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Georgia', serif;
         background: #f4f1eb; color: #2c2c2c; line-height: 1.7; }}

  .masthead {{ background: #1a1a2e; color: #f0e6d3; padding: 40px 48px; }}
  .masthead h1 {{ font-size: 2.2em; letter-spacing: .02em; margin-bottom: 6px; }}
  .masthead .sub {{ color: #a89880; font-size: .9em; }}

  .wrapper {{ max-width: 1100px; margin: 0 auto; padding: 40px 24px 80px; display: grid;
             grid-template-columns: 220px 1fr; gap: 40px; }}

  .sidebar {{ position: sticky; top: 24px; align-self: start; }}
  .sidebar h3 {{ font-size: .75em; text-transform: uppercase; letter-spacing: .1em;
               color: #888; margin-bottom: 12px; }}
  .sidebar ul {{ list-style: none; }}
  .sidebar li {{ margin: 6px 0; }}
  .sidebar a {{ color: #444; text-decoration: none; font-size: .88em; }}
  .sidebar a:hover {{ color: #1a1a2e; text-decoration: underline; }}

  .exec-panel {{ background: #fffdf7; border: 1px solid #e8c87a;
                border-radius: 6px; padding: 32px 36px; margin-bottom: 40px; }}
  .exec-panel h2 {{ font-size: 1.3em; color: #1a1a2e; border-bottom: 2px solid #e8c87a;
                   padding-bottom: 10px; margin-bottom: 20px; }}
  .exec-panel p {{ margin: 12px 0; }}
  .exec-panel ul {{ margin: 12px 0 12px 20px; }}
  .exec-panel li {{ margin: 6px 0; }}
  .exec-meta {{ font-size: .78em; color: #aaa; margin-top: 20px; }}

  .sector {{ background: white; border-radius: 6px; box-shadow: 0 1px 4px rgba(0,0,0,.07);
            padding: 28px 32px; margin-bottom: 28px;
            border-left: 5px solid #e8c87a; }}
  .sector-header {{ margin-bottom: 18px; }}
  .sector-header h2 {{ font-size: 1.25em; color: #1a1a2e; }}
  .sector-body h2 {{ font-size: 1em; color: #1a1a2e; margin: 18px 0 6px;
                    border-bottom: 1px solid #eee; padding-bottom: 4px; }}
  .sector-body h3 {{ font-size: .95em; color: #444; margin: 14px 0 4px; font-style: italic; }}
  .sector-body p  {{ margin: 8px 0; font-size: .93em; }}
  .sector-body ul {{ margin: 8px 0 8px 18px; }}
  .sector-body li {{ margin: 4px 0; font-size: .93em; }}
  .sector-body strong {{ color: #1a1a2e; }}

  .footer {{ text-align: center; color: #aaa; font-size: .8em; padding: 20px;
            grid-column: 1 / -1; }}

  @media (max-width: 720px) {{
    .wrapper {{ grid-template-columns: 1fr; }}
    .sidebar {{ display: none; }}
    .footer {{ grid-column: 1; }}
  }}
</style>
</head>
<body>

<div class="masthead">
  <h1>💰 Pricing Power Monitor — {_quarter}</h1>
  <div class="sub">S&amp;P 500 Earnings Intelligence · {len(sector_chapters)} GICS Sectors · Generated {generated_at} UTC · Model: {MODEL_ID}</div>
</div>

<div class="wrapper">

  <div class="sidebar">
    <h3>Sectors</h3>
    <ul>
      <li><a href="#executive-summary">Executive Summary</a></li>
      {toc_items}
    </ul>
  </div>

  <div>

    <section class="exec-panel" id="executive-summary">
      <h2>Executive Summary — Aggregate Pricing Environment</h2>
      {exec_html}
      <div class="exec-meta">Generated by {MODEL_ID} · {generated_at} UTC</div>
    </section>

    {sector_html}

  </div>

  <div class="footer">
    Pricing Power Monitor {_quarter} · Extracted from S&amp;P 500 earnings transcripts via SEC EDGAR ·
    AI analysis by {MODEL_ID} · {generated_at} UTC
  </div>

</div>
</body>
</html>"""


def main():
    if not HF_TOKEN:
        sys.exit("ERROR: HF_TOKEN env var required")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n[pricing-power] Extracting pricing sections per sector ...")
    sector_data = {}
    for sector in SECTOR_ORDER:
        data = load_sector_pricing(sector)
        n = data.count("--- ") if data else 0
        if data:
            sector_data[sector] = data
            print(f"  {sector}: {n} companies, {len(data)} chars")
        else:
            print(f"  {sector}: no data — skipping")

    if not sector_data:
        sys.exit("ERROR: No pricing data found")

    print(f"\n[pricing-power] Generating {len(sector_data)} sector chapters ...")
    sector_chapters = {}
    for i, (sector, data) in enumerate(sector_data.items(), 1):
        print(f"\n[{i}/{len(sector_data)}] {sector} ...", flush=True)
        try:
            chapter = call_gemma(SECTOR_PROMPT.format(sector=sector, data=data))
            sector_chapters[sector] = chapter
        except Exception as e:
            print(f"  FAILED: {e}")
        if i < len(sector_data):
            time.sleep(5)

    print(f"\n[pricing-power] Generating executive summary ...")
    compact = "\n\n".join(
        f"=== {sector} ===\n{body[:600]}"
        for sector, body in sector_chapters.items()
    )
    exec_text = call_gemma(EXEC_SUMMARY_PROMPT.format(quarter=_quarter, chapters=compact))
    exec_html = md_to_html(exec_text)

    generated_at = _now.strftime("%Y-%m-%d %H:%M")
    html = build_html(exec_html, sector_chapters, generated_at)

    out_path = OUT_DIR / f"pricing_power_{DATE_TAG}.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"\n[pricing-power] Written → {out_path}")


if __name__ == "__main__":
    main()
