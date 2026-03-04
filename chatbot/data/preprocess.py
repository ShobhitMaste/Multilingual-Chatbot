"""
Preprocess data for training.
- Ingests BhashaBench-Ayur (HuggingFace) + hindi_dataset.csv (local)
- Cleans text (normalize Devanagari, remove noise)
- Translates English Q&As to Hindi using googletrans
- Formats as training-ready JSON
- Splits into train/val/test
"""

import os
import json
import re
import sys
import random
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RAW_DATA_DIR, PROCESSED_DATA_DIR, TRAIN_RATIO, VAL_RATIO, BASE_DIR

# Path to local Hindi dataset CSV
HINDI_CSV_PATH = os.path.join(BASE_DIR, "dataset", "hindi_dataset.csv")


def normalize_devanagari(text):
    """Basic Devanagari text normalization."""
    if not text:
        return ""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Normalize common Devanagari variations
    text = text.replace('ॅ', '')  # Remove chandrabindu variants if noisy
    return text


def clean_text(text):
    """Clean and normalize text."""
    if not text or not isinstance(text, str):
        return ""
    # Remove HTML tags if any
    text = re.sub(r'<[^>]+>', '', text)
    # Remove excessive punctuation
    text = re.sub(r'\.{3,}', '…', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def translate_en_to_hi_batch(texts, batch_size=50):
    """
    Translate English texts to Hindi using googletrans.
    Falls back gracefully on errors.
    """
    from googletrans import Translator
    import time

    translator = Translator()
    translated = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Translating EN→HI"):
        batch = texts[i:i + batch_size]
        for text in batch:
            try:
                if not text or len(text.strip()) == 0:
                    translated.append("")
                    continue
                result = translator.translate(text, src='en', dest='hi')
                translated.append(result.text if result else text)
                time.sleep(0.1)  # Rate limiting
            except Exception as e:
                print(f"  ⚠ Translation failed for: {text[:50]}... — {e}")
                translated.append(text)  # Keep English as fallback

    return translated


def extract_qa_from_bba(data, language):
    """
    Extract question-answer pairs from BhashaBench-Ayur data.
    BBA is primarily MCQ format — we extract the question and correct answer.
    """
    qa_pairs = []

    for item in data:
        # Try common field names in the dataset
        question = item.get("question", item.get("Question", ""))
        
        # For MCQ, the correct answer is usually in 'answer' or 'correct_answer'
        answer = item.get("answer", item.get("Answer", item.get("correct_answer", "")))
        
        # Try to get options and build a more complete answer
        options = item.get("options", item.get("Options", None))
        
        # If answer is just a letter (A/B/C/D), look up the full option text
        if options and isinstance(answer, str) and len(answer) <= 2:
            answer_map = {}
            if isinstance(options, dict):
                answer_map = options
            elif isinstance(options, list):
                for idx, opt in enumerate(options):
                    answer_map[chr(65 + idx)] = opt  # A, B, C, D
            
            full_answer = answer_map.get(answer.strip().upper(), answer)
            if full_answer:
                answer = full_answer

        # Get category/subject if available
        category = item.get("subject", item.get("category", item.get("topic", "general")))

        if question and answer:
            qa_pairs.append({
                "question": clean_text(str(question)),
                "answer": clean_text(str(answer)),
                "category": str(category),
                "source_language": language
            })

    return qa_pairs


def extract_qa_from_hindi_csv(csv_path):
    """
    Extract Q&A pairs from the local hindi_dataset.csv.
    Generates multiple Q&A styles per record:
      1. Symptom → Diagnosis
      2. Symptom → Treatment + Timespan
      3. Patient History → Diagnosis + Treatment
    """
    qa_pairs = []

    try:
        df = pd.read_csv(csv_path, encoding="utf-8")
    except Exception as e:
        print(f"  ❌ Failed to read CSV: {e}")
        return qa_pairs

    print(f"  📊 CSV loaded: {len(df)} rows, {len(df.columns)} columns")

    for _, row in df.iterrows():
        symptoms = clean_text(str(row.get("symptoms", "")))
        diagnosis = clean_text(str(row.get("Diagnosis", "")))
        treatment = clean_text(str(row.get("treatment", "")))
        timespan = clean_text(str(row.get("timespan", "")))
        history = clean_text(str(row.get("Patient History", "")))
        age = str(row.get("age", ""))
        gender = clean_text(str(row.get("gender", "")))
        category = clean_text(str(row.get("Diagnosis Category", "सामान्य")))

        # Skip rows with missing key fields
        if not symptoms or symptoms == "nan" or not diagnosis or diagnosis == "nan":
            continue

        # ── Q&A 1: Symptoms → Diagnosis ──
        q1 = f"मरीज में ये लक्षण हैं: {symptoms}। निदान क्या हो सकता है?"
        a1 = f"इन लक्षणों के आधार पर, संभावित निदान {diagnosis} है।"
        if category and category != "nan":
            a1 += f" यह {category} की श्रेणी में आता है।"
        qa_pairs.append({
            "question": q1, "answer": a1,
            "question_hi": normalize_devanagari(q1),
            "answer_hi": normalize_devanagari(a1),
            "category": category if category != "nan" else "सामान्य",
            "source_language": "hindi",
            "source": "hindi_dataset_csv"
        })

        # ── Q&A 2: Symptoms → Treatment ──
        if treatment and treatment != "nan":
            q2 = f"{symptoms} के लिए उपचार क्या है?"
            a2 = f"उपचार: {treatment}।"
            if timespan and timespan != "nan":
                a2 += f" अवधि: {timespan}"
            qa_pairs.append({
                "question": q2, "answer": a2,
                "question_hi": normalize_devanagari(q2),
                "answer_hi": normalize_devanagari(a2),
                "category": category if category != "nan" else "सामान्य",
                "source_language": "hindi",
                "source": "hindi_dataset_csv"
            })

        # ── Q&A 3: History-based (richer context) ──
        if history and history != "nan" and len(history) > 10:
            q3 = f"{age} वर्ष, {gender}। इतिहास: {history}। निदान और उपचार बताएं।"
            a3 = f"निदान: {diagnosis}।"
            if treatment and treatment != "nan":
                a3 += f" उपचार: {treatment}।"
            if timespan and timespan != "nan":
                a3 += f" अवधि: {timespan}"
            qa_pairs.append({
                "question": q3, "answer": a3,
                "question_hi": normalize_devanagari(q3),
                "answer_hi": normalize_devanagari(a3),
                "category": category if category != "nan" else "सामान्य",
                "source_language": "hindi",
                "source": "hindi_dataset_csv"
            })

    return qa_pairs


def process_and_merge():
    """Process both Hindi and English data, translate EN→HI, merge."""
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    all_qa_hindi = []

    # ── Process Hindi data (use directly) ──
    hindi_path = os.path.join(RAW_DATA_DIR, "bba_hindi.json")
    if os.path.exists(hindi_path):
        print("\n📖 Processing Hindi data...")
        with open(hindi_path, "r", encoding="utf-8") as f:
            hindi_data = json.load(f)
        
        hindi_qa = extract_qa_from_bba(hindi_data, "hindi")
        print(f"  ✅ Extracted {len(hindi_qa)} Hindi Q&A pairs")

        # Normalize Devanagari
        for qa in hindi_qa:
            qa["question_hi"] = normalize_devanagari(qa["question"])
            qa["answer_hi"] = normalize_devanagari(qa["answer"])
        
        all_qa_hindi.extend(hindi_qa)
    else:
        print(f"  ⚠ Hindi data not found at {hindi_path}")

    # ── Process English data (translate to Hindi) ──
    english_path = os.path.join(RAW_DATA_DIR, "bba_english.json")
    if os.path.exists(english_path):
        print("\n📖 Processing English data (will translate to Hindi)...")
        with open(english_path, "r", encoding="utf-8") as f:
            english_data = json.load(f)
        
        english_qa = extract_qa_from_bba(english_data, "english")
        print(f"  ✅ Extracted {len(english_qa)} English Q&A pairs")

        # Translate questions and answers to Hindi
        questions_en = [qa["question"] for qa in english_qa]
        answers_en = [qa["answer"] for qa in english_qa]

        print("\n🔄 Translating questions EN → HI...")
        questions_hi = translate_en_to_hi_batch(questions_en)

        print("\n🔄 Translating answers EN → HI...")
        answers_hi = translate_en_to_hi_batch(answers_en)

        # Add translated Hindi versions
        for qa, q_hi, a_hi in zip(english_qa, questions_hi, answers_hi):
            qa["question_hi"] = normalize_devanagari(q_hi)
            qa["answer_hi"] = normalize_devanagari(a_hi)
            qa["question_en"] = qa["question"]  # Keep original English
            qa["answer_en"] = qa["answer"]

        all_qa_hindi.extend(english_qa)
    else:
        print(f"  ⚠ English data not found at {english_path}")

    # ── Process local Hindi CSV dataset ──
    if os.path.exists(HINDI_CSV_PATH):
        print(f"\n📖 Processing local Hindi orthopedic dataset...")
        csv_qa = extract_qa_from_hindi_csv(HINDI_CSV_PATH)
        print(f"  ✅ Generated {len(csv_qa)} Q&A pairs from hindi_dataset.csv")
        all_qa_hindi.extend(csv_qa)
    else:
        print(f"  ⚠ Hindi CSV not found at {HINDI_CSV_PATH}")

    if not all_qa_hindi:
        print("❌ No data to process!")
        return

    # ── Shuffle and split ──
    print(f"\n📊 Total Q&A pairs: {len(all_qa_hindi)}")
    random.seed(42)
    random.shuffle(all_qa_hindi)

    n = len(all_qa_hindi)
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

    train_data = all_qa_hindi[:train_end]
    val_data = all_qa_hindi[train_end:val_end]
    test_data = all_qa_hindi[val_end:]

    print(f"  📦 Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")

    # ── Save splits ──
    for split_name, split_data in [("train", train_data), ("val", val_data), ("test", test_data)]:
        output_path = os.path.join(PROCESSED_DATA_DIR, f"{split_name}.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(split_data, f, ensure_ascii=False, indent=2)
        print(f"  💾 Saved {split_name}.json ({len(split_data)} samples)")

    # ── Save full KB (all Hindi Q&A for FAISS indexing) ──
    kb_data = []
    for qa in all_qa_hindi:
        kb_data.append({
            "question_hi": qa.get("question_hi", ""),
            "answer_hi": qa.get("answer_hi", ""),
            "category": qa.get("category", "general"),
            # Combine Q&A as a passage for retrieval
            "passage_hi": f"{qa.get('question_hi', '')} {qa.get('answer_hi', '')}"
        })
    
    kb_path = os.path.join(PROCESSED_DATA_DIR, "knowledge_base.json")
    with open(kb_path, "w", encoding="utf-8") as f:
        json.dump(kb_data, f, ensure_ascii=False, indent=2)
    print(f"  💾 Saved knowledge_base.json ({len(kb_data)} passages)")

    print(f"\n{'='*50}")
    print(f"✅ Preprocessing complete!")
    print(f"   Output: {PROCESSED_DATA_DIR}")
    print(f"{'='*50}")


if __name__ == "__main__":
    process_and_merge()
