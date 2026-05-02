"""
Daily earnings recap: fetch SEC 8-K exhibits for 50 S&P 500 tickers per day,
generate Gemma commentary, and write HTML reports + index.html.

Queue rotates through all ~500 tickers over 10 days, then reshuffles.
"""
import os, re, time, json, textwrap, random
from datetime import datetime, timezone
from pathlib import Path

import requests
import markdown as md_lib
from huggingface_hub import InferenceClient

# ── config ────────────────────────────────────────────────────────────────────
BATCH_SIZE    = 50

HEADERS       = {"User-Agent": "ResearchBot jeannealbertoreading@gmail.com"}
TICKERS_URL   = "https://www.sec.gov/files/company_tickers.json"
BASE_ARCHIVE  = "https://www.sec.gov/Archives/edgar/data"
MODEL_ID      = "google/gemma-4-31B-it"
HF_TOKEN      = os.environ.get("HF_TOKEN", "")

REPO_ROOT     = Path(__file__).resolve().parent.parent
REPORTS_DIR   = REPO_ROOT / "reports"
INDEX_HTML    = REPO_ROOT / "index.html"
SP500_JSON    = REPO_ROOT / "sp500.json"
QUEUE_FILE    = REPO_ROOT / "queue.json"

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


def get_recent_8ks(cik_padded: str, limit: int = 15) -> list[dict]:
    url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"
    r = requests.get(url, headers=HEADERS, timeout=15)
    r.raise_for_status()
    recent = r.json()["filings"]["recent"]
    result = []
    for form, date, acc, primary in zip(
        recent["form"], recent["filingDate"],
        recent["accessionNumber"], recent["primaryDocument"]
    ):
        if form == "8-K":
            result.append({"date": date, "accession": acc, "primary": primary})
        if len(result) == limit:
            break
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


def fetch_earnings_exhibit(ticker: str) -> dict | None:
    cik_padded, company = get_cik(ticker)
    cik_raw = str(int(cik_padded))
    filings = get_recent_8ks(cik_padded)
    for filing in filings:
        time.sleep(0.15)
        try:
            candidates = get_exhibit_urls(cik_raw, filing["accession"], filing["primary"])
        except Exception:
            continue
        for _, url in candidates:
            try:
                time.sleep(0.1)
                text = fetch_text(url)
            except Exception:
                continue
            if len(text) < 300:
                continue
            label = "TRANSCRIPT" if is_transcript(text) else "PRESS_RELEASE"
            return {
                "ticker": ticker,
                "company": company,
                "date": filing["date"],
                "label": label,
                "url": url,
                "text": text,
            }
    return None


# ── Gemma ─────────────────────────────────────────────────────────────────────
def call_gemma(messages: list[dict], max_tokens: int = 800) -> str:
    client = InferenceClient(model=MODEL_ID, token=HF_TOKEN, timeout=300)
    for attempt in range(5):
        try:
            stream = client.chat.completions.create(
                messages=messages,
                temperature=0.2,
                max_tokens=max_tokens,
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
            if any(x in str(e) for x in ("429", "503", "Too Many Requests", "Service Temporarily Unavailable")) and attempt < 4:
                time.sleep(60 * (attempt + 1))
            else:
                raise
    return ""


def generate_commentary(exhibit: dict) -> str:
    data = exhibit["text"][:8000]
    ticker = exhibit["ticker"]
    date   = exhibit["date"]
    prompt = textwrap.dedent(f"""
        [ROLE]: Senior Financial Analyst specializing in qualitative macro insights from equity reports.

        [TASK]: Analyze the earnings report below and extract macro-level insights. Rules:
        - Only use information present in the document. If data is older than 45 days from {date}, note it and stop analysis for that point.
        - Every insight MUST be cited as ({ticker}, {date}).
        - Use "" only for verbatim strings copied exactly from the text.
        - Do not fabricate figures or infer beyond what is stated.

        [OUTPUT FORMAT]:
        ## Exec Summary
        Synthesis of the macro picture in 3-5 sentences.

        ## Consensus vs Outliers
        What the report confirms vs. where it deviates from the broader narrative.

        ## Key Findings
        Bullet points, each ending with ({ticker}, {date}).

        ## Voice of the Market
        3-4 high-impact verbatim quotes with context. Format: > "quote" — context

        ## Data Limitations
        Sample size, blind spots, or gaps in this filing.

        [DATA]:
        {data}
    """).strip()
    print(f"  Calling Gemma for {exhibit['ticker']}...")
    return call_gemma([{"role": "user", "content": prompt}], max_tokens=1200)


# ── HTML generation ───────────────────────────────────────────────────────────
MARKER_START = "<!-- earnings-commentary-start -->"
MARKER_END   = "<!-- earnings-commentary-end -->"

PAGE_CSS = """
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         background:#f5f7fa; margin:0; padding:0; color:#333; }
  .header { background:#1a1a2e; color:white; padding:24px 32px; }
  .header h1 { margin:0; font-size:1.6em; }
  .header .sub { color:#aaa; font-size:0.9em; margin-top:4px; }
  .container { max-width:1000px; margin:32px auto; padding:0 20px 60px; }
  .meta-card { background:white; border-radius:8px; box-shadow:0 2px 6px rgba(0,0,0,.08);
               padding:20px 24px; margin-bottom:24px; display:flex; gap:24px; flex-wrap:wrap; }
  .meta-item { display:flex; flex-direction:column; }
  .meta-label { font-size:.75em; color:#888; text-transform:uppercase; letter-spacing:.05em; }
  .meta-value { font-size:1.05em; font-weight:600; margin-top:2px; }
  .badge { display:inline-block; padding:2px 10px; border-radius:12px; font-size:.8em;
           font-weight:600; background:#e8f4fd; color:#0077cc; }
  .commentary-card { background:white; border-radius:8px; box-shadow:0 2px 6px rgba(0,0,0,.08);
                     padding:28px 32px; margin-bottom:24px; }
  .commentary-card .card-title { border-left:4px solid #007bff; padding-left:14px; margin-bottom:20px; }
  .commentary-card .card-title h2 { margin:0 0 4px; color:#1a1a2e; }
  .commentary-card .card-title p  { margin:0; color:#888; font-size:.82em; }
  .commentary h2, .commentary h3 { color:#1a1a2e; margin:18px 0 8px; }
  .commentary ul { padding-left:20px; }
  .commentary li { margin:4px 0; line-height:1.6; }
  .commentary p  { line-height:1.7; }
  .commentary table { border-collapse:collapse; width:100%; margin:12px 0; }
  .commentary th, .commentary td { border:1px solid #dee2e6; padding:8px 12px; text-align:left; }
  .commentary th { background:#f8f9fa; font-weight:600; }
  details { background:white; border-radius:8px; box-shadow:0 2px 6px rgba(0,0,0,.08); }
  summary { padding:16px 24px; cursor:pointer; font-weight:600; color:#444;
            list-style:none; user-select:none; }
  summary::-webkit-details-marker { display:none; }
  summary::before { content:"▶  "; color:#007bff; }
  details[open] summary::before { content:"▼  "; }
  .raw-text { padding:0 24px 24px; white-space:pre-wrap; font-size:.82em;
              color:#555; line-height:1.6; border-top:1px solid #f0f0f0; margin-top:8px; }
  .back-link { display:inline-block; margin-bottom:20px; color:#007bff; text-decoration:none; font-size:.9em; }
  .back-link:hover { text-decoration:underline; }
</style>
"""

def build_report_html(exhibit: dict, commentary_md: str, generated_at: str) -> str:
    body_html = md_lib.markdown(commentary_md, extensions=["tables"])
    raw_escaped = exhibit["text"].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{exhibit['ticker']} Earnings Recap — {exhibit['date']}</title>
{PAGE_CSS}
</head>
<body>
<div class="header">
  <h1>📈 {exhibit['ticker']} — Earnings Recap</h1>
  <div class="sub">{exhibit['company']} · Filed {exhibit['date']}</div>
</div>
<div class="container">
  <a class="back-link" href="../index.html">← All Reports</a>
  <div class="meta-card">
    <div class="meta-item"><span class="meta-label">Ticker</span><span class="meta-value">{exhibit['ticker']}</span></div>
    <div class="meta-item"><span class="meta-label">Company</span><span class="meta-value">{exhibit['company']}</span></div>
    <div class="meta-item"><span class="meta-label">Filed</span><span class="meta-value">{exhibit['date']}</span></div>
    <div class="meta-item"><span class="meta-label">Type</span><span class="meta-value"><span class="badge">{exhibit['label']}</span></span></div>
    <div class="meta-item"><span class="meta-label">Characters</span><span class="meta-value">{len(exhibit['text']):,}</span></div>
  </div>
  {MARKER_START}
  <div class="commentary-card">
    <div class="card-title">
      <h2>AI Commentary</h2>
      <p>Generated {generated_at} UTC · google/gemma-4-31B-it</p>
    </div>
    <div class="commentary">{body_html}</div>
  </div>
  {MARKER_END}
  <details>
    <summary>Full Press Release Text</summary>
    <div class="raw-text">{raw_escaped}</div>
  </details>
</div>
</body>
</html>"""


def save_report(exhibit: dict, commentary_md: str) -> Path:
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    base = f"{exhibit['ticker']}_{exhibit['date']}_{exhibit['label']}"

    # raw EDGAR text (mirrors original skill output)
    raw_out = REPORTS_DIR / f"{base}.txt"
    raw_out.write_text(
        f"Company: {exhibit['company']}\nFiled: {exhibit['date']}\n"
        f"Type: {exhibit['label']}\nSource: {exhibit['url']}\n\n"
        + exhibit["text"],
        encoding="utf-8",
    )

    html = build_report_html(exhibit, commentary_md, generated_at)
    out = REPORTS_DIR / f"{exhibit['ticker']}_{exhibit['date']}.html"
    out.write_text(html, encoding="utf-8")
    return out


# ── index.html ────────────────────────────────────────────────────────────────
def build_index(reports: list[dict]) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    cards = ""
    for r in sorted(reports, key=lambda x: x["date"], reverse=True):
        fname = f"reports/{r['ticker']}_{r['date']}.html"
        badge_color = "#007bff" if r["label"] == "PRESS_RELEASE" else "#28a745"
        cards += f"""
    <article class="report-card">
      <div class="ticker">{r['ticker']}</div>
      <div class="company">{r['company']}</div>
      <div class="date">{r['date']}</div>
      <span class="badge" style="background:{badge_color}20;color:{badge_color}">{r['label']}</span>
      <a class="view-link" href="{fname}">View Recap →</a>
    </article>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Earnings Recaps — S&P 500</title>
<style>
  body {{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
          background:#f5f7fa; margin:0; padding:0; color:#333; }}
  .header {{ background:#1a1a2e; color:white; padding:28px 32px; }}
  .header h1 {{ margin:0; font-size:1.8em; }}
  .header .sub {{ color:#aaa; font-size:.9em; margin-top:6px; }}
  .container {{ max-width:1200px; margin:0 auto; padding:32px 20px 60px; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(220px,1fr)); gap:18px; }}
  .report-card {{ background:white; border-radius:10px;
                  box-shadow:0 2px 6px rgba(0,0,0,.08); padding:20px;
                  display:flex; flex-direction:column; gap:6px;
                  transition:transform .15s,box-shadow .15s; }}
  .report-card:hover {{ transform:translateY(-3px); box-shadow:0 6px 14px rgba(0,0,0,.12); }}
  .ticker {{ font-size:1.6em; font-weight:700; color:#1a1a2e; }}
  .company {{ font-size:.82em; color:#666; }}
  .date {{ font-size:.85em; color:#888; margin-top:2px; }}
  .badge {{ display:inline-block; padding:2px 9px; border-radius:12px;
            font-size:.75em; font-weight:600; margin-top:4px; }}
  .view-link {{ margin-top:auto; padding-top:12px; color:#007bff; text-decoration:none;
                font-weight:600; font-size:.9em; border-top:1px solid #f0f0f0; }}
  .view-link:hover {{ text-decoration:underline; }}
  .meta {{ color:#999; font-size:.83em; margin-bottom:20px; }}
</style>
</head>
<body>
<div class="header">
  <h1>📊 Earnings Recaps</h1>
  <div class="sub">S&P 500 · Daily EDGAR fetch + Gemma AI commentary · Updated {now}</div>
</div>
<div class="container">
  <p class="meta">{len(reports)} report(s) · 10 tracked tickers · Source: SEC EDGAR 8-K filings</p>
  <div class="grid">{cards}
  </div>
</div>
</body>
</html>"""


# ── queue ─────────────────────────────────────────────────────────────────────
def load_sp500() -> list[str]:
    return json.loads(SP500_JSON.read_text())

def load_queue() -> list[str]:
    if QUEUE_FILE.exists():
        q = json.loads(QUEUE_FILE.read_text())
        if q:
            return q
    # Empty or missing — build a fresh shuffled queue from the full list
    tickers = load_sp500()
    random.shuffle(tickers)
    return tickers

def save_queue(queue: list[str]) -> None:
    QUEUE_FILE.write_text(json.dumps(queue, indent=2))


# ── manifest ──────────────────────────────────────────────────────────────────
MANIFEST = REPORTS_DIR / "manifest.json"

def load_manifest() -> list[dict]:
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text())
    return []

def save_manifest(records: list[dict]) -> None:
    MANIFEST.write_text(json.dumps(records, indent=2))


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    REPORTS_DIR.mkdir(exist_ok=True)
    manifest = load_manifest()
    existing = {(r["ticker"], r["date"]) for r in manifest}
    new_records = []

    queue = load_queue()
    todays_batch, remaining = queue[:BATCH_SIZE], queue[BATCH_SIZE:]
    # Refill for next cycle if we just drained the last batch
    if not remaining:
        remaining = load_sp500()
        random.shuffle(remaining)
        print(f"Queue exhausted — reshuffled {len(remaining)} tickers for next cycle.")
    save_queue(remaining)
    print(f"Today's batch: {todays_batch}")
    print(f"Queue remaining after today: {len(remaining)}\n")

    for ticker in todays_batch:
        print(f"\n── {ticker} ──────────────────────────────────")
        try:
            exhibit = fetch_earnings_exhibit(ticker)
        except Exception as e:
            print(f"  EDGAR error: {e}")
            continue

        if not exhibit:
            print("  No earnings exhibit found.")
            continue

        key = (ticker, exhibit["date"])
        if key in existing:
            print(f"  Already processed ({exhibit['date']}), skipping.")
            continue

        print(f"  Found: {exhibit['company']} | {exhibit['date']} | {exhibit['label']} | {len(exhibit['text']):,} chars")

        if HF_TOKEN:
            try:
                commentary_md = generate_commentary(exhibit)
            except Exception as e:
                print(f"  Gemma error: {e}")
                commentary_md = "_AI commentary unavailable for this report._"
        else:
            print("  HF_TOKEN not set — skipping Gemma commentary.")
            commentary_md = "_AI commentary not generated (HF_TOKEN missing)._"

        out = save_report(exhibit, commentary_md)
        print(f"  Saved → {out.name}")

        record = {
            "ticker":  exhibit["ticker"],
            "company": exhibit["company"],
            "date":    exhibit["date"],
            "label":   exhibit["label"],
        }
        manifest.append(record)
        new_records.append(record)
        existing.add(key)

    # Rebuild index even if no new records (keeps date fresh)
    INDEX_HTML.write_text(build_index(manifest), encoding="utf-8")
    print(f"\nindex.html updated ({len(manifest)} total reports)")

    if new_records:
        save_manifest(manifest)
        print(f"Manifest updated — {len(new_records)} new report(s).")
    else:
        print("No new reports added.")


if __name__ == "__main__":
    main()
