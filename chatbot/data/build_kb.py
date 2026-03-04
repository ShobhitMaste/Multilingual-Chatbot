"""
Build FAISS vector index from the Hindi Ayurvedic knowledge base.
Uses SentenceTransformer to embed all Hindi passages, then indexes with FAISS.
"""

import os
import json
import sys
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PROCESSED_DATA_DIR, FAISS_INDEX_PATH, EMBEDDING_MODEL_NAME, EMBEDDING_DIM


def build_faiss_index():
    """Build FAISS index from Hindi knowledge base passages."""
    import faiss
    from sentence_transformers import SentenceTransformer

    # ── Load Q&A knowledge base ──
    all_passages = []

    kb_path = os.path.join(PROCESSED_DATA_DIR, "knowledge_base.json")
    if os.path.exists(kb_path):
        with open(kb_path, "r", encoding="utf-8") as f:
            kb_data = json.load(f)
        print(f"📖 Loaded {len(kb_data)} passages from Q&A knowledge base")
        for item in kb_data:
            all_passages.append({
                "passage_hi": item["passage_hi"],
                "question_hi": item.get("question_hi", ""),
                "answer_hi": item.get("answer_hi", ""),
                "category": item.get("category", "general"),
                "source": "bhashbench_hindi_dataset"
            })
    else:
        print(f"⚠️  Q&A knowledge base not found at {kb_path}. Skipping.")

    # ── Load classical text passages (Ashtanga Hridayam) ──
    classical_path = os.path.join(PROCESSED_DATA_DIR, "classical_passages.json")
    if os.path.exists(classical_path):
        with open(classical_path, "r", encoding="utf-8") as f:
            classical_data = json.load(f)
        print(f"📖 Loaded {len(classical_data)} passages from classical texts")
        for item in classical_data:
            all_passages.append({
                "passage_hi": item["passage_hi"],
                "question_hi": "",
                "answer_hi": item["passage_hi"],  # For classical texts, the passage IS the answer
                "category": item.get("chapter", "ashtanga_hridayam"),
                "source": item.get("source", "ashtanga_hridayam")
            })
    else:
        print(f"⚠️  Classical passages not found at {classical_path}. Skipping.")

    if not all_passages:
        print("❌ No passages found! Run preprocess.py and/or process_classical_text.py first.")
        return

    print(f"\n📚 Total passages for indexing: {len(all_passages)}")

    # ── Load SentenceTransformer ──
    print(f"\n🔄 Loading embedding model: {EMBEDDING_MODEL_NAME}")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    # ── Generate embeddings ──
    passages = [item["passage_hi"] for item in all_passages]
    
    print(f"\n🔄 Generating embeddings for {len(passages)} passages...")
    embeddings = model.encode(
        passages,
        show_progress_bar=True,
        batch_size=64,
        normalize_embeddings=True  # Normalize for cosine similarity
    )
    embeddings = np.array(embeddings, dtype=np.float32)
    print(f"  ✅ Embeddings shape: {embeddings.shape}")

    # ── Build FAISS index ──
    # Using IndexFlatIP (Inner Product) since embeddings are normalized → cosine similarity
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(embeddings)
    print(f"  ✅ FAISS index built with {index.ntotal} vectors")

    # ── Save index and metadata ──
    os.makedirs(FAISS_INDEX_PATH, exist_ok=True)
    
    index_file = os.path.join(FAISS_INDEX_PATH, "ayurvedic_kb.index")
    faiss.write_index(index, index_file)
    print(f"  💾 Index saved to: {index_file}")

    # Save metadata (passage texts + categories) for retrieval
    metadata = all_passages
    
    metadata_file = os.path.join(FAISS_INDEX_PATH, "metadata.json")
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"  💾 Metadata saved to: {metadata_file}")

    # ── Quick test ──
    print(f"\n{'='*50}")
    print("🧪 Quick retrieval test:")
    test_query = "अश्वगंधा के फायदे क्या हैं?"
    query_embedding = model.encode([test_query], normalize_embeddings=True)
    query_embedding = np.array(query_embedding, dtype=np.float32)
    
    distances, indices = index.search(query_embedding, 3)
    print(f"   Query: {test_query}")
    for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        print(f"   Top-{i+1} (score={dist:.4f}): {metadata[idx]['passage_hi'][:100]}...")
    
    print(f"\n{'='*50}")
    print(f"✅ FAISS index build complete!")
    print(f"   Index: {index_file}")
    print(f"   Metadata: {metadata_file}")
    print(f"{'='*50}")


if __name__ == "__main__":
    build_faiss_index()
