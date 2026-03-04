"""
Download Ayurvedic datasets from HuggingFace.

Downloads:
  1. BhashaBench-Ayur   — 14,963 exam-based Q&As (Hindi + English)
  2. ayurveda-text-qanda — Ayurveda Q&A pairs from classical texts
  3. AyurParam           — Bilingual corpus curated from ~1000 Ayurvedic books
"""

import os
import json
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import HF_TOKEN, BBA_DATASET_NAME, RAW_DATA_DIR

# ──────────────────────────────────────────────
# Dataset identifiers
# ──────────────────────────────────────────────
AYURVEDA_QA_DATASET = "Macromrit/ayurveda-text-based-qanda"
AYURPARAM_DATASET = "bharatgenai/AyurParam"


def _save_dataset(data, filename, label):
    """Helper — save list of dicts as JSON and print a preview."""
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    output_path = os.path.join(RAW_DATA_DIR, filename)

    print(f"  Downloaded {len(data)} samples")

    if data:
        print(f"  Sample keys: {list(data[0].keys())}")
        for key, val in list(data[0].items())[:4]:
            preview = str(val)[:100] + "..." if len(str(val)) > 100 else str(val)
            print(f"     {key}: {preview}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  Saved to: {output_path}")


# ──────────────────────────────────────────────
# 1. BhashaBench-Ayur (gated — needs HF token)
# ──────────────────────────────────────────────
def download_bba_dataset():
    """Download BhashaBench-Ayur dataset for Hindi and English."""
    from datasets import load_dataset

    for language in ["Hindi", "English"]:
        print(f"\n{'='*50}")
        print(f"[1/3] BhashaBench-Ayur — {language}")
        print(f"{'='*50}")

        try:
            dataset = load_dataset(
                BBA_DATASET_NAME,
                data_dir=language,
                split="test",
                token=HF_TOKEN
            )
            data = [dict(row) for row in dataset]
            _save_dataset(data, f"bba_{language.lower()}.json", f"BBA-{language}")

        except Exception as e:
            print(f"  Error downloading {language}: {e}")
            print(f"  Make sure you have access: https://huggingface.co/datasets/{BBA_DATASET_NAME}")


# ──────────────────────────────────────────────
# 2. Ayurveda Text-Based Q&A
# ──────────────────────────────────────────────
def download_ayurveda_qa():
    """Download Macromrit/ayurveda-text-based-qanda — Ayurveda Q&A from classical texts."""
    from datasets import load_dataset

    print(f"\n{'='*50}")
    print(f"[2/3] Ayurveda Text Q&A — {AYURVEDA_QA_DATASET}")
    print(f"{'='*50}")

    try:
        dataset = load_dataset(AYURVEDA_QA_DATASET, token=HF_TOKEN)

        # Grab all available splits
        all_data = []
        for split_name in dataset.keys():
            split_data = [dict(row) for row in dataset[split_name]]
            print(f"  Split '{split_name}': {len(split_data)} samples")
            all_data.extend(split_data)

        _save_dataset(all_data, "ayurveda_qa.json", "Ayurveda-QA")

    except Exception as e:
        print(f"  Error: {e}")
        print(f"  Dataset: https://huggingface.co/datasets/{AYURVEDA_QA_DATASET}")


# ──────────────────────────────────────────────
# 3. AyurParam (gated — needs HF token)
# ──────────────────────────────────────────────
def download_ayurparam():
    """Download bharatgenai/AyurParam — bilingual Ayurveda corpus from ~1000 books."""
    from datasets import load_dataset

    print(f"\n{'='*50}")
    print(f"[3/3] AyurParam — {AYURPARAM_DATASET}")
    print(f"{'='*50}")

    try:
        dataset = load_dataset(AYURPARAM_DATASET, token=HF_TOKEN)

        all_data = []
        for split_name in dataset.keys():
            split_data = [dict(row) for row in dataset[split_name]]
            print(f"  Split '{split_name}': {len(split_data)} samples")
            all_data.extend(split_data)

        _save_dataset(all_data, "ayurparam.json", "AyurParam")

    except Exception as e:
        print(f"  Error: {e}")
        print(f"  This is a gated dataset. Request access at:")
        print(f"  https://huggingface.co/datasets/{AYURPARAM_DATASET}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
if __name__ == "__main__":
    if not HF_TOKEN:
        print("HF_TOKEN not found in environment!")
        print("   Set it with: $env:HF_TOKEN = 'your_token_here'")
        print("   Or: export HF_TOKEN=your_token_here")
        sys.exit(1)

    print("Downloading all Ayurvedic datasets...\n")

    download_bba_dataset()      # 1. BhashaBench-Ayur
    download_ayurveda_qa()      # 2. Ayurveda text Q&A
    download_ayurparam()        # 3. AyurParam bilingual

    print(f"\n{'='*50}")
    print(f"All downloads complete! Check: {RAW_DATA_DIR}")
    print(f"{'='*50}")
