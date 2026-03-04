"""
Fine-tune mT5-small on Hindi Ayurvedic Q&A using QLoRA.
Designed to work on both:
  - RTX 3050 4GB VRAM (with 4-bit quantization)
  - Google Colab T4 16GB VRAM

Usage:
  python chatbot/models/fine_tune_mt5.py
  
Or on Colab:
  !python chatbot/models/fine_tune_mt5.py
"""

import os
import json
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    BASE_MODEL_NAME, PROCESSED_DATA_DIR, FINE_TUNED_MODEL_DIR,
    LORA_R, LORA_ALPHA, LORA_DROPOUT, LORA_TARGET_MODULES,
    TRAIN_BATCH_SIZE, EVAL_BATCH_SIZE, GRADIENT_ACCUMULATION_STEPS,
    LEARNING_RATE, NUM_EPOCHS, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH,
    WARMUP_STEPS, WEIGHT_DECAY
)


def load_training_data():
    """Load preprocessed train/val JSON files."""
    train_path = os.path.join(PROCESSED_DATA_DIR, "train.json")
    val_path = os.path.join(PROCESSED_DATA_DIR, "val.json")

    with open(train_path, "r", encoding="utf-8") as f:
        train_data = json.load(f)
    with open(val_path, "r", encoding="utf-8") as f:
        val_data = json.load(f)

    print(f"📖 Train: {len(train_data)} samples | Val: {len(val_data)} samples")
    return train_data, val_data


def format_for_mt5(data):
    """
    Format data for mT5 seq2seq training.
    Input:  "question: {Hindi question} context: {Hindi passage}"
    Target: "{Hindi answer}"
    """
    formatted = []
    for item in data:
        question = item.get("question_hi", item.get("question", ""))
        answer = item.get("answer_hi", item.get("answer", ""))
        
        if not question or not answer:
            continue

        # Input format for mT5
        input_text = f"प्रश्न: {question}"
        target_text = answer

        formatted.append({
            "input_text": input_text,
            "target_text": target_text
        })

    return formatted


class AyurvedicDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for mT5 training."""
    
    def __init__(self, data, tokenizer, max_input_len, max_target_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Tokenize input
        input_encoding = self.tokenizer(
            item["input_text"],
            max_length=self.max_input_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # Tokenize target
        target_encoding = self.tokenizer(
            item["target_text"],
            max_length=self.max_target_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        labels = target_encoding["input_ids"].squeeze()
        # Replace padding token id with -100 so it's ignored in loss
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_encoding["input_ids"].squeeze(),
            "attention_mask": input_encoding["attention_mask"].squeeze(),
            "labels": labels
        }


def fine_tune():
    """Main fine-tuning function using QLoRA."""
    from transformers import (
        AutoTokenizer, AutoModelForSeq2SeqLM, 
        Seq2SeqTrainingArguments, Seq2SeqTrainer,
        DataCollatorForSeq2Seq
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

    # Check if bitsandbytes + CUDA available
    use_4bit = torch.cuda.is_available()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    # ── Load tokenizer ──
    print(f"\n🔄 Loading tokenizer: {BASE_MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

    # ── Load model with 4-bit quantization ──
    print(f"🔄 Loading model: {BASE_MODEL_NAME}")
    if use_4bit:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(
            BASE_MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
        )
        model = prepare_model_for_kbit_training(model)
        print("  ✅ Model loaded in 4-bit (QLoRA mode)")
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL_NAME)
        model = model.to(device)
        print("  ✅ Model loaded in full precision (CPU mode)")

    # ── Apply LoRA ──
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        task_type=TaskType.SEQ_2_SEQ_LM,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Load and format data ──
    train_data, val_data = load_training_data()
    train_formatted = format_for_mt5(train_data)
    val_formatted = format_for_mt5(val_data)
    print(f"📊 Formatted — Train: {len(train_formatted)} | Val: {len(val_formatted)}")

    train_dataset = AyurvedicDataset(train_formatted, tokenizer, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH)
    val_dataset = AyurvedicDataset(val_formatted, tokenizer, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH)

    # ── Training arguments ──
    os.makedirs(FINE_TUNED_MODEL_DIR, exist_ok=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir=FINE_TUNED_MODEL_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        weight_decay=WEIGHT_DECAY,
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        predict_with_generate=True,
        generation_max_length=MAX_TARGET_LENGTH,
        fp16=use_4bit,  # Use fp16 if CUDA available
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )

    # ── Data Collator ──
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
    )

    # ── Trainer ──
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # ── Train! ──
    print(f"\n{'='*50}")
    print(f"🚀 Starting fine-tuning...")
    print(f"   Epochs: {NUM_EPOCHS}")
    print(f"   Batch size: {TRAIN_BATCH_SIZE} (effective: {TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS})")
    print(f"   Learning rate: {LEARNING_RATE}")
    print(f"{'='*50}\n")

    trainer.train()

    # ── Save LoRA adapters ──
    print(f"\n💾 Saving LoRA adapters to: {FINE_TUNED_MODEL_DIR}")
    model.save_pretrained(FINE_TUNED_MODEL_DIR)
    tokenizer.save_pretrained(FINE_TUNED_MODEL_DIR)

    print(f"\n{'='*50}")
    print(f"✅ Fine-tuning complete!")
    print(f"   Adapters saved to: {FINE_TUNED_MODEL_DIR}")
    print(f"{'='*50}")


if __name__ == "__main__":
    fine_tune()
