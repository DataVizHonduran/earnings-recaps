#!/usr/bin/env python3
"""
Final Beige Book Assembly — reads all sector chapter .md files,
calls Gemma for a cross-sector executive summary, and writes a
single self-contained HTML Beige Book to industry_reports/beige_book_{DATE_TAG}.html.

Usage:
    HF_TOKEN=hf_xxx python3 scripts/generate_final_beige_book.py
"""

import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import InferenceClient

MODEL_ID  = "google/gemma-4-31B-it"
HF_TOKEN  = os.environ.get("HF_TOKEN", "")
REPO_ROOT = Path(__file__).resolve().parent.parent
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

EXEC_SUMMARY_PROMPT = """\
[ROLE]: Chief Economist at a major central bank writing the opening summary of a Beige Book.
[TASK]: Read the 11 GICS sector chapters below and write a 400-500 word Executive Summary for
the {quarter} Beige Book. Structure as:

**Overall Activity:** One paragraph (3-4 sentences) on the broad economic tempo across sectors.
Use Beige Book qualifiers (modest, moderate, robust, slight, stable, softening).

**Dominant Cross-Sector Themes:** 3-4 bullet points identifying themes that appear in 3+ sectors
(e.g., pricing power, labor tightening, inventory normalization, capex hesitancy).
Each bullet: one concise sentence naming the sectors where the theme appears.

**Notable Divergences:** 1 paragraph (2-3 sentences) on sectors that broke from the consensus
direction — name the sectors and the mechanism.

**Outlook:** 2-3 sentences on forward-looking signals extracted from management commentary.
Use measured, non-committal language ("contacts noted", "several firms indicated").

Constraints: No stock prices, EPS, or financial metrics. Neutral, observational tone. Fed-speak only.

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


def load_sector_chapters() -> dict[str, str]:
    """Load the most recent chapter file per sector (any date tag)."""
    chapters = {}
    for sector in SECTOR_ORDER:
        # Find any file matching this sector name
        matches = sorted(OUT_DIR.glob(f"{sector}_*.md"), reverse=True)
        if not matches:
            print(f"  [warn] No chapter found for {sector}")
            continue
        path = matches[0]
        text = path.read_text(encoding="utf-8")
        # Strip the header line (# Sector — ...) to keep it clean for the prompt
        body = re.sub(r"^#.*\n\*.*\*\n+", "", text).strip()
        chapters[sector] = body
        print(f"  loaded: {path.name}")
    return chapters


def md_to_html(text: str) -> str:
    """Minimal markdown → HTML for the output (bold, bullets, paragraphs)."""
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
        # h2 ##
        if stripped.startswith("## "):
            if in_ul:
                out.append("</ul>")
                in_ul = False
            out.append(f"<h2>{_inline(stripped[3:])}</h2>")
        # h3 ###
        elif stripped.startswith("### "):
            if in_ul:
                out.append("</ul>")
                in_ul = False
            out.append(f"<h3>{_inline(stripped[4:])}</h3>")
        # bullet
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


SECTOR_COLORS = {
    "Consumer Discretionary": "#e74c3c",
    "Consumer Staples":       "#e67e22",
    "Energy":                 "#f39c12",
    "Financials":             "#27ae60",
    "Health Care":            "#16a085",
    "Industrials":            "#2980b9",
    "Information Technology": "#8e44ad",
    "Communication Services": "#2c3e50",
    "Materials":              "#d35400",
    "Real Estate":            "#c0392b",
    "Utilities":              "#7f8c8d",
}


def build_html(exec_summary_html: str, chapters: dict[str, str], generated_at: str) -> str:
    toc_items = "".join(
        f'<li><a href="#{s.lower().replace(" ", "-")}">{s}</a></li>'
        for s in chapters
    )

    sector_html = ""
    for sector, body in chapters.items():
        anchor = sector.lower().replace(" ", "-")
        color  = SECTOR_COLORS.get(sector, "#34495e")
        sector_html += f"""
        <section class="sector" id="{anchor}">
            <div class="sector-header" style="border-left:5px solid {color}">
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
<title>Beige Book — {_quarter}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Georgia', serif;
         background: #f4f1eb; color: #2c2c2c; line-height: 1.7; }}

  /* ── Header ── */
  .masthead {{ background: #1a1a2e; color: #f0e6d3; padding: 40px 48px; }}
  .masthead h1 {{ font-size: 2.2em; letter-spacing: .02em; margin-bottom: 6px; }}
  .masthead .sub {{ color: #a89880; font-size: .9em; }}

  /* ── Layout ── */
  .wrapper {{ max-width: 1100px; margin: 0 auto; padding: 40px 24px 80px; display: grid;
             grid-template-columns: 220px 1fr; gap: 40px; }}

  /* ── Sidebar TOC ── */
  .sidebar {{ position: sticky; top: 24px; align-self: start; }}
  .sidebar h3 {{ font-size: .75em; text-transform: uppercase; letter-spacing: .1em;
               color: #888; margin-bottom: 12px; }}
  .sidebar ul {{ list-style: none; }}
  .sidebar li {{ margin: 6px 0; }}
  .sidebar a {{ color: #444; text-decoration: none; font-size: .88em; }}
  .sidebar a:hover {{ color: #1a1a2e; text-decoration: underline; }}

  /* ── Exec Summary ── */
  .exec-panel {{ background: #fffdf7; border: 1px solid #e8dcc8;
                border-radius: 6px; padding: 32px 36px; margin-bottom: 40px;
                grid-column: 1 / -1; }}
  .exec-panel h2 {{ font-size: 1.3em; color: #1a1a2e; border-bottom: 2px solid #d4b896;
                   padding-bottom: 10px; margin-bottom: 20px; }}
  .exec-panel p {{ margin: 12px 0; }}
  .exec-panel ul {{ margin: 12px 0 12px 20px; }}
  .exec-panel li {{ margin: 6px 0; }}
  .exec-meta {{ font-size: .78em; color: #aaa; margin-top: 20px; }}

  /* ── Sector sections ── */
  .sector {{ background: white; border-radius: 6px; box-shadow: 0 1px 4px rgba(0,0,0,.07);
            padding: 28px 32px; margin-bottom: 28px; }}
  .sector-header {{ padding-left: 14px; margin-bottom: 18px; }}
  .sector-header h2 {{ font-size: 1.25em; color: #1a1a2e; }}
  .sector-body h2 {{ font-size: 1em; color: #1a1a2e; margin: 18px 0 6px;
                    border-bottom: 1px solid #eee; padding-bottom: 4px; }}
  .sector-body h3 {{ font-size: .95em; color: #444; margin: 14px 0 4px; font-style: italic; }}
  .sector-body p  {{ margin: 8px 0; font-size: .93em; }}
  .sector-body ul {{ margin: 8px 0 8px 18px; }}
  .sector-body li {{ margin: 4px 0; font-size: .93em; }}
  .sector-body strong {{ color: #1a1a2e; }}

  /* ── Footer ── */
  .footer {{ text-align: center; color: #aaa; font-size: .8em; padding: 20px;
            grid-column: 1 / -1; }}

  @media (max-width: 720px) {{
    .wrapper {{ grid-template-columns: 1fr; }}
    .sidebar {{ display: none; }}
    .exec-panel {{ grid-column: 1; }}
    .footer {{ grid-column: 1; }}
  }}
</style>
</head>
<body>

<div class="masthead">
  <h1>📋 Beige Book — {_quarter}</h1>
  <div class="sub">S&amp;P 500 Earnings Intelligence · {len(chapters)} GICS Sectors · Generated {generated_at} UTC · Model: {MODEL_ID}</div>
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
      <h2>Executive Summary</h2>
      {exec_summary_html}
      <div class="exec-meta">Generated by {MODEL_ID} · {generated_at} UTC</div>
    </section>

    {sector_html}

  </div>

  <div class="footer">
    Beige Book {_quarter} · Synthesized from S&amp;P 500 earnings transcripts via SEC EDGAR ·
    AI analysis by {MODEL_ID} · {generated_at} UTC
  </div>

</div>
</body>
</html>"""


def main():
    if not HF_TOKEN:
        sys.exit("ERROR: HF_TOKEN env var required")

    print(f"\n[beige-book] Loading sector chapters ...")
    chapters = load_sector_chapters()
    if not chapters:
        sys.exit("ERROR: No sector chapters found in industry_reports/")

    print(f"\n[beige-book] {len(chapters)} sectors loaded. Generating executive summary ...")

    # Compact chapters for the exec summary prompt (first 800 chars each)
    compact = "\n\n".join(
        f"=== {sector} ===\n{body[:800]}"
        for sector, body in chapters.items()
    )
    exec_text = call_gemma(
        EXEC_SUMMARY_PROMPT.format(quarter=_quarter, chapters=compact)
    )
    exec_html = md_to_html(exec_text)

    generated_at = _now.strftime("%Y-%m-%d %H:%M")
    html = build_html(exec_html, chapters, generated_at)

    out_path = OUT_DIR / f"beige_book_{DATE_TAG}.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"\n[beige-book] Written → {out_path}")
    print(f"[beige-book] Open: file://{out_path.resolve()}")


if __name__ == "__main__":
    main()
