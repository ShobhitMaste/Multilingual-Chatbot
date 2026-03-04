"""
Configuration file for the Ayurvedic RAG Chatbot.
All hyperparameters, model names, and paths are defined here.
"""

import os

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "faiss_index")
FINE_TUNED_MODEL_DIR = os.path.join(MODELS_DIR, "mt5_ayurvedic_lora")

# ──────────────────────────────────────────────
# HuggingFace
# ──────────────────────────────────────────────
HF_TOKEN = os.environ.get("HF_TOKEN", None)
BBA_DATASET_NAME = "bharatgenai/BhashaBench-Ayur"

# ──────────────────────────────────────────────
# Model Names
# ──────────────────────────────────────────────
BASE_MODEL_NAME = "google/mt5-small"
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# ──────────────────────────────────────────────
# Fine-Tuning Hyperparameters (QLoRA)
# ──────────────────────────────────────────────
LORA_R = 16                 # LoRA rank
LORA_ALPHA = 32             # LoRA alpha
LORA_DROPOUT = 0.1          # LoRA dropout
LORA_TARGET_MODULES = ["q", "v"]  # Target attention layers

TRAIN_BATCH_SIZE = 4
EVAL_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4   # Effective batch size = 16
LEARNING_RATE = 3e-4
NUM_EPOCHS = 5
MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 128
WARMUP_STEPS = 100
WEIGHT_DECAY = 0.01

# ──────────────────────────────────────────────
# Retrieval
# ──────────────────────────────────────────────
TOP_K_RETRIEVAL = 5
EMBEDDING_DIM = 384          # MiniLM-L12 output dimension

# ──────────────────────────────────────────────
# Generation
# ──────────────────────────────────────────────
NUM_BEAMS = 4
MAX_GENERATE_LENGTH = 128
REPETITION_PENALTY = 1.2
LENGTH_PENALTY = 1.0

# ──────────────────────────────────────────────
# Data Split Ratios
# ──────────────────────────────────────────────
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
