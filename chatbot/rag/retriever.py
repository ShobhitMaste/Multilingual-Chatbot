"""
Hindi passage retriever using SentenceTransformer + FAISS.
Retrieves top-K relevant passages from the Hindi Ayurvedic knowledge base.
"""

import os
import json
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FAISS_INDEX_PATH, EMBEDDING_MODEL_NAME, TOP_K_RETRIEVAL


class HindiRetriever:
    """Retrieve relevant Hindi Ayurvedic passages using FAISS."""

    def __init__(self):
        import faiss
        from sentence_transformers import SentenceTransformer

        # Load embedding model
        print(f"🔄 Loading embedding model: {EMBEDDING_MODEL_NAME}")
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)

        # Load FAISS index
        index_file = os.path.join(FAISS_INDEX_PATH, "ayurvedic_kb.index")
        if not os.path.exists(index_file):
            raise FileNotFoundError(
                f"FAISS index not found at {index_file}. Run build_kb.py first!"
            )
        
        self.index = faiss.read_index(index_file)
        print(f"  ✅ FAISS index loaded ({self.index.ntotal} vectors)")

        # Load metadata
        metadata_file = os.path.join(FAISS_INDEX_PATH, "metadata.json")
        with open(metadata_file, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        print(f"  ✅ Metadata loaded ({len(self.metadata)} entries)")

    def retrieve(self, query_hi, top_k=None):
        """
        Retrieve top-K relevant passages for a Hindi query.
        
        Args:
            query_hi: Hindi query string
            top_k: Number of results to return (default from config)
            
        Returns:
            List of dicts with keys: passage_hi, question_hi, answer_hi, category, score
        """
        if top_k is None:
            top_k = TOP_K_RETRIEVAL

        # Encode query
        query_embedding = self.model.encode(
            [query_hi], 
            normalize_embeddings=True
        )
        query_embedding = np.array(query_embedding, dtype=np.float32)

        # Search FAISS
        distances, indices = self.index.search(query_embedding, top_k)

        # Build results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.metadata):
                result = {
                    **self.metadata[idx],
                    "score": float(dist)
                }
                results.append(result)

        return results


if __name__ == "__main__":
    # Quick test
    retriever = HindiRetriever()

    test_queries = [
        "अश्वगंधा के फायदे",
        "वात दोष",
        "पंचकर्म क्या है",
    ]

    for query in test_queries:
        print(f"\n🔍 Query: {query}")
        results = retriever.retrieve(query, top_k=3)
        for i, r in enumerate(results):
            print(f"   [{i+1}] (score={r['score']:.4f}) {r['passage_hi'][:80]}...")
