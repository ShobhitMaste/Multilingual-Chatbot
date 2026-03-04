# 🌿 Ayurvedic RAG Chatbot

**Bilingual Ayurvedic Health Advisor — Hindi-Primary with English Support**

A Hindi-primary RAG (Retrieval-Augmented Generation) chatbot for Ayurvedic wellness advice, built with mT5-small, FAISS, and SentenceTransformers.

---

## 🏗️ Project Structure

```
chatbot/
├── config.py              # Hyperparameters & paths
├── requirements.txt       # Python dependencies
├── app.py                 # Streamlit web interface
├── data/
│   ├── download_data.py   # Download BhashaBench-Ayur from HuggingFace
│   ├── preprocess.py      # Clean, translate EN→HI, split data
│   └── build_kb.py        # Build FAISS vector index
├── models/
│   ├── fine_tune_mt5.py   # QLoRA fine-tuning (Colab/3050 compatible)
│   └── inference.py       # Load model & generate responses
└── rag/
    ├── retriever.py       # SentenceTransformer + FAISS retrieval
    ├── translator.py      # Language detection + EN↔HI translation
    └── pipeline.py        # Full RAG orchestrator
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
cd chatbot
pip install -r requirements.txt
```

### 2. Set HuggingFace Token
```bash
# Windows PowerShell
$env:HF_TOKEN = "your_token_here"

# Linux/Mac
export HF_TOKEN=your_token_here
```

### 3. Download & Prepare Data
```bash
python data/download_data.py      # Download BhashaBench-Ayur
python data/preprocess.py          # Clean, translate, split
python data/build_kb.py            # Build FAISS index
```

### 4. Fine-Tune mT5 (GPU required)
```bash
python models/fine_tune_mt5.py     # QLoRA fine-tuning
```

### 5. Run Chatbot
```bash
# Terminal mode
python rag/pipeline.py

# Web interface
streamlit run app.py
```

---

## 🖥️ Hardware Requirements

| Setup | VRAM | Notes |
|-------|------|-------|
| **RTX 3050 (4GB)** | QLoRA 4-bit | batch_size=4, gradient_accumulation=4 |
| **Colab T4 (16GB)** | QLoRA 4-bit | More comfortable, larger batches possible |
| **CPU only** | N/A | Inference only, no training |

---

## 📊 Pipeline

```
User (EN/HI) → Language Detection → [EN? Translate to HI] →
  SentenceTransformer Embedding → FAISS Top-K Retrieval →
    [Query + Passages] → Fine-tuned mT5-small → Hindi Response →
      [EN user? Translate to EN] → Output
```

---

## 📚 Datasets

- **BhashaBench-Ayur**: ~15,000 Ayurvedic MCQs (Hindi + English)
- **Zenodo Hindi Health Cases**: Clinical records
- **Classical Texts**: Charaka Samhita, Sushruta Samhita
