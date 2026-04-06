"""
Full RAG pipeline for the Ayurvedic chatbot.
Combines language detection, retrieval, generation, and answer fallback.
"""

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class AyurvedicRAG:
    """
    End-to-end RAG pipeline for the Ayurvedic chatbot.

    Steps:
        1. Detect language
        2. Translate to Hindi if needed
        3. Retrieve relevant passages
        4. Generate a Hindi answer
        5. Replace weak answers with a grounded retrieval-based fallback
        6. Translate back if the user asked in English
    """

    def __init__(self):
        print("=" * 50)
        print("Initializing Ayurvedic RAG Pipeline...")
        print("=" * 50)

        from models.inference import AyurvedicGenerator
        from rag.retriever import HindiRetriever
        from rag.translator import Translator

        self.translator = Translator()
        self.retriever = HindiRetriever()
        self.generator = AyurvedicGenerator()

        print("\n" + "=" * 50)
        print("Ayurvedic RAG Pipeline ready!")
        print("=" * 50)

    def _normalize_text(self, text):
        """Collapse repeated whitespace for scoring and display."""
        return " ".join((text or "").split()).strip()

    def _sentence_split(self, text):
        """Split Hindi passages into sentence-sized chunks."""
        parts = re.split(r"[।!?]+", self._normalize_text(text))
        return [part.strip(" '\"-") for part in parts if len(part.strip()) >= 12]

    def _looks_like_question(self, text):
        """Skip question-like chunks when composing retrieval-based fallbacks."""
        text = self._normalize_text(text)
        question_starts = (
            "क्या", "कैसे", "किस ", "किन ", "कौन", "कब ", "क्यों", "अगर ",
            "तो ", "इसमें ", "अशगंध को", "अश्वगंधा को"
        )
        return text.startswith(question_starts)

    def _ensure_period(self, text):
        """Ensure fallback sentences end cleanly."""
        text = self._normalize_text(text)
        if not text:
            return ""
        if text[-1] in "।!?":
            return text
        return f"{text}।"

    def _query_keywords(self, query_hi):
        """Extract simple Hindi keywords from the question."""
        stopwords = {
            "क्या", "कैसे", "कौन", "किस", "किन", "का", "की", "के", "को", "और",
            "में", "पर", "से", "यह", "है", "हैं", "था", "थे", "लिए", "करे",
            "करें", "बताएं", "बताइए", "बताया", "जाता", "जाती"
        }
        tokens = re.split(r"[^\w\u0900-\u097F]+", query_hi)
        return [token for token in tokens if len(token) >= 3 and token not in stopwords]

    def _needs_caution_detail(self, query_hi):
        """Detect whether the user asked for warnings or who should be careful."""
        caution_terms = [
            "सावधानी",
            "किसे",
            "किन लोगों",
            "किन्हें",
            "कब नहीं",
            "नहीं लेना",
            "दुष्प्रभाव",
            "साइड इफेक्ट",
            "परहेज",
            "वर्जित",
        ]
        return any(term in query_hi for term in caution_terms)

    def _response_is_weak(self, query_hi, response_hi):
        """Detect short template-like generations that should be grounded with retrieval."""
        response = self._normalize_text(response_hi)
        if len(response) < 24:
            return True

        generic_patterns = [
            "का उपयोग किया जाता है",
            "के लिए किया जाता है",
            "में सहायक होता है",
            "में उपयोग किया जाता है",
            "का उपयोग",
        ]
        if any(pattern in response for pattern in generic_patterns) and len(response) < 90:
            return True

        if self._needs_caution_detail(query_hi):
            caution_terms = ["सावधानी", "परहेज", "दुष्प्रभाव", "गर्भ", "नहीं", "वर्जित"]
            if not any(term in response for term in caution_terms):
                return True

        unique_words = {word for word in response.split() if len(word) >= 3}
        return len(unique_words) < 5

    def _build_grounded_fallback(self, query_hi, retrieved):
        """Build a grounded Hindi answer from the top retrieved passages."""
        if not retrieved:
            return "उपलब्ध संदर्भों में इस प्रश्न का स्पष्ट उत्तर नहीं मिला।"

        query_keywords = self._query_keywords(query_hi)
        ranked_sentences = []
        seen = set()

        for doc_rank, item in enumerate(retrieved[:5]):
            source_text = item.get("answer_hi") or item.get("passage_hi") or ""
            for sent_rank, sentence in enumerate(self._sentence_split(source_text)):
                if self._looks_like_question(sentence):
                    continue
                normalized = sentence.lower()
                if normalized in seen:
                    continue
                seen.add(normalized)

                overlap = sum(1 for keyword in query_keywords if keyword in sentence)
                score = overlap * 5 + float(item.get("score", 0.0)) - (doc_rank * 0.2) - (sent_rank * 0.15)
                ranked_sentences.append((score, sentence))

        ranked_sentences.sort(key=lambda row: row[0], reverse=True)
        selected = [self._ensure_period(sentence) for _, sentence in ranked_sentences[:3]]

        if not selected:
            return "उपलब्ध संदर्भों में इस प्रश्न का स्पष्ट उत्तर नहीं मिला।"

        answer = "उपलब्ध संदर्भों के आधार पर " + " ".join(selected[:2])

        if self._needs_caution_detail(query_hi):
            caution_terms = ["सावधानी", "परहेज", "दुष्प्रभाव", "गर्भ", "वर्जित", "नहीं"]
            if not any(term in answer for term in caution_terms):
                answer += " किन लोगों को सावधानी रखनी चाहिए, इसकी स्पष्ट जानकारी उपलब्ध संदर्भों में नहीं मिली।"

        return self._normalize_text(answer)

    def answer(self, user_query, top_k=5):
        """
        Process a user query end-to-end.

        Returns a dict with:
            - response
            - response_hi
            - detected_language
            - query_hi
            - retrieved_passages
            - answer_mode
        """
        query_hi, detected_lang = self.translator.process_input(user_query)

        retrieved = self.retriever.retrieve(query_hi, top_k=top_k)
        passages = [item["answer_hi"] for item in retrieved]

        generated_hi = self.generator.generate(query_hi, context_passages=passages)
        if self._response_is_weak(query_hi, generated_hi):
            response_hi = self._build_grounded_fallback(query_hi, retrieved)
            answer_mode = "grounded_fallback"
        else:
            response_hi = generated_hi
            answer_mode = "model"

        response = self.translator.process_output(response_hi, detected_lang)

        return {
            "response": response,
            "response_hi": response_hi,
            "detected_language": detected_lang,
            "query_hi": query_hi,
            "retrieved_passages": retrieved,
            "answer_mode": answer_mode,
        }


if __name__ == "__main__":
    rag = AyurvedicRAG()

    print("\n" + "=" * 50)
    print("Ayurvedic Chatbot (type 'quit' to exit)")
    print("Supports: English and Hindi")
    print("=" * 50)

    while True:
        query = input("\nYou: ").strip()
        if query.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        if not query:
            continue

        result = rag.answer(query)

        print(f"\nDetected: {'Hindi' if result['detected_language'] == 'hi' else 'English'}")
        if result["detected_language"] == "en":
            print(f"Hindi query: {result['query_hi']}")

        print(f"\nResponse: {result['response']}")
        print(f"Mode: {result['answer_mode']}")

        print(f"\nRetrieved {len(result['retrieved_passages'])} passages:")
        for i, passage in enumerate(result["retrieved_passages"][:3]):
            print(f"  [{i + 1}] (score={passage['score']:.3f}) {passage['passage_hi'][:60]}...")
