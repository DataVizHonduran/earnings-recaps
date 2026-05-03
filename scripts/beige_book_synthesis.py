#!/usr/bin/env python3
"""
Beige Book Synthesis — batch Gemma 4 analysis of earnings transcripts.

Reads ninja/*.txt, calls Gemma 4 with a 5-category Beige Book prompt for each,
writes ninja/synthesized/{stem}_synthesized.txt.  Skips already-done files.

Usage:
    HF_TOKEN=hf_xxx python3 scripts/beige_book_synthesis.py

Required env:
    HF_TOKEN — HuggingFace API token
"""

import os
import sys
import time
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import InferenceClient

# ── config ────────────────────────────────────────────────────────────────────
MODEL_ID      = "google/gemma-4-31B-it"
HF_TOKEN      = os.environ.get("HF_TOKEN", "")
REPO_ROOT     = Path(__file__).resolve().parent.parent
NINJA_DIR     = REPO_ROOT / "ninja"
SYNTH_DIR     = NINJA_DIR / "synthesized"
INTER_DELAY   = 5   # seconds between files

PROMPT_TEMPLATE = """\
Please analyze the attached earnings transcript for {ticker}. Your goal is to identify "Ground Truth" economic signals for a Beige Book style report.
Ignore stock price commentary, buyback announcements, and specific GAAP accounting reconciliations unless they impact the outlook.
Please extract and summarize the following categories:
1. Demand & Activity: Is demand growing, flat, or declining? Note specific geographic or product-line variances.
2. Labor & Wages: Mentions of hiring difficulty, turnover, wage increases, or headcount reductions.
3. Pricing Power & Inflation: Is the company raising prices? Are they seeing input cost relief (commodities, freight, energy) or persistent pressure?
4. Supply Chain & Inventory: Current state of lead times and whether inventory levels are lean or bloated.
5. Capital Spending (Capex): Plans for future investment in capacity, technology, or facilities.
Formatting Instructions:
* Use "Fed-speak" qualifiers where appropriate: slight, modest, moderate, robust, stable, or softening.
* For every point made, include a direct quote snippet in parentheses for auditability.
* Keep the entire summary under 400 words.

TRANSCRIPT:
{transcript}"""

# ── Gemma call ────────────────────────────────────────────────────────────────

def call_gemma(ticker: str, transcript: str) -> str:
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN not set")
    client = InferenceClient(model=MODEL_ID, token=HF_TOKEN, timeout=300)
    messages = [
        {"role": "user", "content": PROMPT_TEMPLATE.format(
            ticker=ticker,
            transcript=transcript[:12000],
        )}
    ]
    for attempt in range(5):
        try:
            stream = client.chat.completions.create(
                messages=messages,
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

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    if not HF_TOKEN:
        sys.exit("ERROR: HF_TOKEN env var required")

    SYNTH_DIR.mkdir(parents=True, exist_ok=True)

    transcripts = sorted(NINJA_DIR.glob("*.txt"))
    total = len(transcripts)
    done = 0
    skipped = 0
    failed = 0

    for i, path in enumerate(transcripts, 1):
        ticker = path.stem.split("_")[0]
        out_path = SYNTH_DIR / f"{path.stem}_synthesized.txt"

        if out_path.exists():
            print(f"[{i}/{total}] {ticker} — skipped (exists)")
            skipped += 1
            continue

        print(f"[{i}/{total}] {ticker} — processing ...", flush=True)
        transcript_text = path.read_text(encoding="utf-8", errors="replace")

        try:
            result = call_gemma(ticker, transcript_text)
            header = (
                f"Ticker: {ticker}\n"
                f"Source: {path.name}\n"
                f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n"
                f"Model: {MODEL_ID}\n"
                f"{'=' * 60}\n\n"
            )
            out_path.write_text(header + result, encoding="utf-8")
            print(f"[{i}/{total}] {ticker} — done → {out_path.name}")
            done += 1
            if done % 10 == 0:
                subprocess.run(["git", "-C", str(REPO_ROOT), "add", "ninja/synthesized/"], check=True)
                subprocess.run(["git", "-C", str(REPO_ROOT), "commit", "-m", f"Beige Book partial — {done} done"], check=True)
                subprocess.run(["git", "-C", str(REPO_ROOT), "pull", "--rebase", "--autostash"], check=True)
                subprocess.run(["git", "-C", str(REPO_ROOT), "push"], check=True)
                print(f"  ↳ committed + pushed ({done} total)", flush=True)
        except Exception as e:
            print(f"[{i}/{total}] {ticker} — FAILED: {e}")
            failed += 1

        if i < total:
            time.sleep(INTER_DELAY)

    print(f"\nDone: {done} written, {skipped} skipped, {failed} failed")


if __name__ == "__main__":
    main()
