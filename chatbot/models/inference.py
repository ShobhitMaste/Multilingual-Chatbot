"""
Inference module for the Ayurvedic chatbot.
Loads the fine-tuned mT5 model and generates Hindi responses.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    BASE_MODEL_NAME,
    FINE_TUNED_MODEL_DIR,
    LENGTH_PENALTY,
    MAX_GENERATE_LENGTH,
    NUM_BEAMS,
    REPETITION_PENALTY,
)


class AyurvedicGenerator:
    """Load the fine-tuned mT5 model and generate Hindi Ayurvedic responses."""

    def __init__(self):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading generator on: {self.device}")

        model_dir = FINE_TUNED_MODEL_DIR

        broken_name = os.path.join(model_dir, "model-001.safetensors")
        fixed_name = os.path.join(model_dir, "model.safetensors")
        if os.path.exists(broken_name) and not os.path.exists(fixed_name):
            os.rename(broken_name, fixed_name)

        has_local_config = os.path.exists(os.path.join(model_dir, "config.json"))
        local_weight_files = (
            "model.safetensors",
            "model.safetensors.index.json",
            "pytorch_model.bin",
            "pytorch_model.bin.index.json",
        )
        has_local_weights = (
            os.path.exists(model_dir)
            and any(os.path.exists(os.path.join(model_dir, name)) for name in local_weight_files)
        )
        has_local_slow_tokenizer = os.path.exists(os.path.join(model_dir, "spiece.model"))

        self.is_finetuned = has_local_config and has_local_weights

        if self.is_finetuned:
            print(f"Loading local full model from: {model_dir}")

            if not has_local_slow_tokenizer:
                raise FileNotFoundError(
                    f"Missing spiece.model in {model_dir}. Copy the full inference tokenizer files first."
                )

            self.tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_dir,
                device_map="auto" if torch.cuda.is_available() else None,
            )
            if not torch.cuda.is_available():
                self.model = self.model.to(self.device)
        else:
            print(f"Local full model not found at {model_dir}")
            print(f"Falling back to base model: {BASE_MODEL_NAME}")
            self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, use_fast=False)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL_NAME)
            self.model = self.model.to(self.device)

        self.model.eval()
        print("Generator loaded!")

    def _build_input_text(self, query_hi, context_passages=None):
        """Build a retrieval-aware prompt for the generator."""
        if context_passages:
            context_blocks = [
                f"संदर्भ {idx + 1}: {passage}"
                for idx, passage in enumerate(context_passages[:3])
            ]
            context_text = "\n".join(context_blocks)
            return (
                "निर्देश: केवल दिए गए संदर्भों के आधार पर संक्षिप्त, स्पष्ट और उपयोगी हिंदी उत्तर दें। "
                "यदि किसी भाग की जानकारी संदर्भों में न हो, तो साफ लिखें कि उपलब्ध संदर्भों में वह जानकारी नहीं मिली।\n"
                f"प्रश्न: {query_hi}\n"
                f"{context_text}\n"
                "उत्तर:"
            )

        return (
            "निर्देश: प्रश्न का संक्षिप्त और स्पष्ट हिंदी उत्तर दें।\n"
            f"प्रश्न: {query_hi}\n"
            "उत्तर:"
        )

    def generate(self, query_hi, context_passages=None):
        """Generate a Hindi response for a Hindi query."""
        input_text = self._build_input_text(query_hi, context_passages)

        inputs = self.tokenizer(
            input_text,
            max_length=512,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=MAX_GENERATE_LENGTH,
                num_beams=NUM_BEAMS,
                repetition_penalty=REPETITION_PENALTY,
                length_penalty=LENGTH_PENALTY,
                early_stopping=True,
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.replace("<extra_id_0>", "").strip()

        if not self.is_finetuned and len(response) < 10:
            if context_passages:
                return context_passages[0]
            return response

        return response


if __name__ == "__main__":
    generator = AyurvedicGenerator()
    test_queries = [
        "अश्वगंधा के फायदे क्या हैं?",
        "वात दोष को कैसे संतुलित करें?",
        "त्रिफला क्या है?",
    ]

    for query in test_queries:
        print(f"\nQ: {query}")
        response = generator.generate(query)
        print(f"A: {response}")
