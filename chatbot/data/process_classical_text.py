"""
Process classical Ayurvedic text (Ashtanga Hridayam Hindi commentary)
into chunked passages for the FAISS knowledge base.

Reads ashtanga.txt, cleans OCR artifacts, splits by chapter/section,
and outputs structured JSON passages ready for embedding.
"""

import os
import re
import json
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BASE_DIR, PROCESSED_DATA_DIR

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
CLASSICAL_TEXT_PATH = os.path.join(BASE_DIR, "dataset", "ashtanga.txt")
OUTPUT_PATH = os.path.join(PROCESSED_DATA_DIR, "classical_passages.json")

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
SKIP_LINES_BEFORE = 900       # Skip front-matter, TOC, publisher info
MIN_PASSAGE_LENGTH = 80       # Minimum chars for a valid passage
MAX_PASSAGE_LENGTH = 2000     # Split passages longer than this
CHUNK_TARGET = 500            # Target passage length in chars


def is_junk_line(line):
    """Check if a line is OCR garbage, page number, or noise."""
    stripped = line.strip()

    # Empty or very short
    if len(stripped) < 3:
        return True

    # Pure numbers / page markers
    if re.match(r'^[\d\s०१२३४५६७८९।\|\.]+$', stripped):
        return True

    # OCR garbage patterns — mostly random symbols/English mixed with Devanagari
    if re.match(r'^[\d\s\.\|\?\#\<\>\&\@\!\*\(\)\[\]\{\}\=\+\-\_\/\\€₹]+$', stripped):
        return True

    # Publisher/page headers like "२ अ० भू०", "१५ अ० हूूु०"
    if re.match(r'^[०-९\d]+\s*अ[०॰]', stripped):
        return True

    # Lines that are just page headers like "अष्टाड्रहदये" or "सूत्रस्थानम्‌"
    if re.match(r'^(अष्टाड्[गर]ह[दृ]ये?|सूत्रस्थानम्‌)\s*$', stripped):
        return True

    # Repeated page-number patterns
    if re.match(r'^(पृष्ठांक|विषय\s*पृष्ठांक)', stripped):
        return True

    # Lines with mostly digits/symbols (>60%)
    non_devanagari = len(re.sub(r'[\u0900-\u097F\s]', '', stripped))
    if len(stripped) > 5 and non_devanagari / len(stripped) > 0.6:
        return True

    # ISBN, phone numbers, publisher addresses
    if re.search(r'ISBN|978-|दूरभाष|पो\. बा\.|[Pp]ost [Bb]ox|[Pp]hone', stripped):
        return True

    return False


def is_section_header(line):
    """Check if a line looks like a chapter/section header."""
    stripped = line.strip()

    # Chapter headers: अध्यायः, अध्याय:
    if re.search(r'अध्याय[:\s]', stripped):
        return True

    # Section group headers: अथ ...वर्गः
    if re.match(r'^अथ\s+', stripped) and re.search(r'वर्ग[:\s]', stripped):
        return True

    # Numbered chapter titles like "(१) आयुष्कामीयाध्यायः"
    if re.match(r'^\([०-९\d]+\)', stripped):
        return True

    # स्थानम् headers
    if re.search(r'स्थानम्', stripped):
        return True

    return False


def clean_text(text):
    """Clean a single text passage."""
    # Remove stray symbols
    text = re.sub(r'[<>\u003c\u003e€₹\#\@\&]', '', text)
    # Normalize whitespace
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def extract_chapter_name(line):
    """Try to extract a meaningful chapter/section name from a header line."""
    stripped = line.strip()
    # Remove leading numbering like (१), (२) etc.
    stripped = re.sub(r'^\([०-९\d]+\)\s*', '', stripped)
    # Remove trailing page numbers
    stripped = re.sub(r'\s+\d+\s*$', '', stripped)
    return stripped if len(stripped) > 3 else "अष्टांगहृदय"


def chunk_passage(text, target_len=CHUNK_TARGET, max_len=MAX_PASSAGE_LENGTH):
    """Split a long passage into smaller chunks at sentence boundaries."""
    if len(text) <= max_len:
        return [text]

    chunks = []
    current = ""

    # Split by sentences (Hindi period ।, ||, or double newline)
    sentences = re.split(r'(?<=[।॥\|\n])\s*', text)

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue

        if len(current) + len(sent) + 1 <= max_len:
            current = current + " " + sent if current else sent
        else:
            if current and len(current) >= MIN_PASSAGE_LENGTH:
                chunks.append(current.strip())
            current = sent

    if current and len(current) >= MIN_PASSAGE_LENGTH:
        chunks.append(current.strip())

    return chunks if chunks else [text[:max_len]]


def process_ashtanga_text():
    """Main processing pipeline for ashtanga.txt."""

    if not os.path.exists(CLASSICAL_TEXT_PATH):
        print(f"❌ File not found: {CLASSICAL_TEXT_PATH}")
        return

    # ── Read file ──
    print(f"📖 Reading {CLASSICAL_TEXT_PATH}...")
    with open(CLASSICAL_TEXT_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()
    print(f"   Total lines: {len(lines)}")

    # ── Skip front matter ──
    lines = lines[SKIP_LINES_BEFORE:]
    print(f"   After skipping front-matter: {len(lines)} lines")

    # ── Clean and segment ──
    passages = []
    current_chapter = "अष्टांगहृदय"
    current_section = ""
    current_text = ""

    for line in lines:
        stripped = line.strip()

        # Skip junk
        if is_junk_line(line):
            continue

        # Detect section/chapter headers
        if is_section_header(line):
            # Save previous passage
            if current_text and len(current_text.strip()) >= MIN_PASSAGE_LENGTH:
                cleaned = clean_text(current_text)
                for chunk in chunk_passage(cleaned):
                    passages.append({
                        "passage_hi": chunk,
                        "source": "ashtanga_hridayam",
                        "chapter": current_chapter,
                        "section": current_section
                    })
            current_chapter = extract_chapter_name(line)
            current_section = stripped
            current_text = ""
            continue

        # Detect sub-section headers (topic headings with ---)
        if re.match(r'^.{5,80}---', stripped):
            # Save previous passage
            if current_text and len(current_text.strip()) >= MIN_PASSAGE_LENGTH:
                cleaned = clean_text(current_text)
                for chunk in chunk_passage(cleaned):
                    passages.append({
                        "passage_hi": chunk,
                        "source": "ashtanga_hridayam",
                        "chapter": current_chapter,
                        "section": current_section
                    })
            current_section = stripped.split('---')[0].strip()
            current_text = stripped + "\n"
            continue

        # Accumulate text
        current_text += stripped + "\n"

        # If accumulated text is getting very long, flush it
        if len(current_text) > MAX_PASSAGE_LENGTH * 2:
            cleaned = clean_text(current_text)
            for chunk in chunk_passage(cleaned):
                passages.append({
                    "passage_hi": chunk,
                    "source": "ashtanga_hridayam",
                    "chapter": current_chapter,
                    "section": current_section
                })
            current_text = ""

    # Don't forget the last passage
    if current_text and len(current_text.strip()) >= MIN_PASSAGE_LENGTH:
        cleaned = clean_text(current_text)
        for chunk in chunk_passage(cleaned):
            passages.append({
                "passage_hi": chunk,
                "source": "ashtanga_hridayam",
                "chapter": current_chapter,
                "section": current_section
            })

    # ── Stats ──
    print(f"\n{'='*50}")
    print(f"📊 Processing Statistics:")
    print(f"   Total passages: {len(passages)}")
    if passages:
        lengths = [len(p["passage_hi"]) for p in passages]
        print(f"   Avg passage length: {sum(lengths)//len(lengths)} chars")
        print(f"   Min passage length: {min(lengths)} chars")
        print(f"   Max passage length: {max(lengths)} chars")

        # Show unique chapters
        chapters = set(p["chapter"] for p in passages)
        print(f"   Unique chapters: {len(chapters)}")

    # ── Save ──
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(passages, f, ensure_ascii=False, indent=2)
    print(f"\n💾 Saved to: {OUTPUT_PATH}")

    # ── Preview ──
    print(f"\n{'='*50}")
    print("🔍 Sample passages:")
    for i, p in enumerate(passages[:3]):
        print(f"\n--- Passage {i+1} (chapter: {p['chapter']}) ---")
        print(p["passage_hi"][:200] + "...")

    print(f"\n{'='*50}")
    print(f"✅ Classical text processing complete!")


if __name__ == "__main__":
    process_ashtanga_text()
