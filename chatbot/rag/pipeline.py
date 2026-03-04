"""
Full RAG Pipeline — the main orchestrator.
Combines: Language Detection → Translation → Retrieval → Generation → Translation back
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class AyurvedicRAG:
    """
    Full RAG pipeline for the Ayurvedic chatbot.
    
    Pipeline:
        1. Detect language (EN or HI)
        2. If English → translate to Hindi
        3. Retrieve top-K Hindi passages from FAISS
        4. Feed query + passages to mT5
        5. Generate Hindi response
        6. If English user → translate response to English
        7. Return final response
    """

    def __init__(self):
        print("=" * 50)
        print("🚀 Initializing Ayurvedic RAG Pipeline...")
        print("=" * 50)

        # Initialize components
        from rag.translator import Translator
        from rag.retriever import HindiRetriever
        from models.inference import AyurvedicGenerator

        self.translator = Translator()
        self.retriever = HindiRetriever()
        self.generator = AyurvedicGenerator()

        print("\n" + "=" * 50)
        print("✅ Ayurvedic RAG Pipeline ready!")
        print("=" * 50)

    def answer(self, user_query, top_k=5):
        """
        Process a user query end-to-end.
        
        Args:
            user_query: User's question in English or Hindi
            top_k: Number of passages to retrieve
            
        Returns:
            dict with keys:
                - response: Final response in user's language
                - response_hi: Hindi response (always)
                - detected_language: 'hi' or 'en'
                - query_hi: Hindi version of the query
                - retrieved_passages: List of retrieved passage dicts
        """
        # Step 1+2: Detect language and translate to Hindi if needed
        query_hi, detected_lang = self.translator.process_input(user_query)

        # Step 3: Retrieve top-K passages
        retrieved = self.retriever.retrieve(query_hi, top_k=top_k)
        passages = [r["answer_hi"] for r in retrieved]

        # Step 4+5: Generate Hindi response using mT5
        response_hi = self.generator.generate(query_hi, context_passages=passages)

        # Step 6: If English user, translate response back
        response = self.translator.process_output(response_hi, detected_lang)

        return {
            "response": response,
            "response_hi": response_hi,
            "detected_language": detected_lang,
            "query_hi": query_hi,
            "retrieved_passages": retrieved
        }


if __name__ == "__main__":
    # Interactive test
    rag = AyurvedicRAG()

    print("\n" + "=" * 50)
    print("🤖 Ayurvedic Chatbot (type 'quit' to exit)")
    print("   Supports: English and Hindi")
    print("=" * 50)

    while True:
        query = input("\n❓ You: ").strip()
        if query.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
        
        if not query:
            continue

        result = rag.answer(query)

        print(f"\n🌐 Detected: {'Hindi' if result['detected_language'] == 'hi' else 'English'}")
        if result['detected_language'] == 'en':
            print(f"🔄 Hindi query: {result['query_hi']}")
        
        print(f"\n💬 Response: {result['response']}")
        
        if result['detected_language'] == 'en':
            print(f"   (Hindi: {result['response_hi']})")

        print(f"\n📚 Retrieved {len(result['retrieved_passages'])} passages:")
        for i, p in enumerate(result['retrieved_passages'][:3]):
            print(f"   [{i+1}] (score={p['score']:.3f}) {p['passage_hi'][:60]}...")
