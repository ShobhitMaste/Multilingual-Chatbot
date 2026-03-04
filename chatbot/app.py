"""
Streamlit Web Interface for the Ayurvedic Chatbot.

Run with:
  cd chatbot
  streamlit run app.py
"""

import streamlit as st
import sys
import os

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ──────────────────────────────────────────────
# Page Config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="आयुर्वेदिक चैटबॉट | Ayurvedic Chatbot",
    page_icon="🌿",
    layout="wide"
)

# ──────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .main { font-family: 'Inter', sans-serif; }

    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }

    .chat-header {
        text-align: center;
        padding: 1.5rem 0;
        background: linear-gradient(135deg, rgba(46, 125, 50, 0.15), rgba(255, 193, 7, 0.1));
        border-radius: 16px;
        border: 1px solid rgba(46, 125, 50, 0.3);
        margin-bottom: 1.5rem;
    }

    .chat-header h1 {
        color: #81C784;
        font-size: 2rem;
        margin: 0;
    }

    .chat-header p {
        color: #A5D6A7;
        font-size: 0.95rem;
        margin: 0.3rem 0 0 0;
    }

    .chat-message-user {
        background: linear-gradient(135deg, #1565C0, #1976D2);
        color: white;
        padding: 12px 18px;
        border-radius: 18px 18px 4px 18px;
        margin: 8px 0;
        max-width: 80%;
        float: right;
        clear: both;
    }

    .chat-message-bot {
        background: linear-gradient(135deg, #2E7D32, #388E3C);
        color: white;
        padding: 12px 18px;
        border-radius: 18px 18px 18px 4px;
        margin: 8px 0;
        max-width: 80%;
        float: left;
        clear: both;
    }

    .passage-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 10px 14px;
        margin: 6px 0;
        font-size: 0.85rem;
    }

    .score-badge {
        background: rgba(46, 125, 50, 0.3);
        color: #A5D6A7;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Initialize RAG Pipeline (cached)
# ──────────────────────────────────────────────
@st.cache_resource
def load_pipeline():
    """Load the RAG pipeline (cached so it only loads once)."""
    from rag.pipeline import AyurvedicRAG
    return AyurvedicRAG()


# ──────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────
st.markdown("""
<div class="chat-header">
    <h1>🌿 आयुर्वेदिक चैटबॉट</h1>
    <p>Bilingual Ayurvedic Health Advisor — Hindi & English</p>
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Sidebar — Settings & Info
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    
    language = st.selectbox(
        "Preferred Language",
        ["Auto-detect", "Hindi (हिन्दी)", "English"],
        index=0
    )
    
    top_k = st.slider("Passages to retrieve", 1, 10, 5)
    show_passages = st.checkbox("Show retrieved passages", value=True)
    show_hindi = st.checkbox("Show Hindi response", value=False)

    st.markdown("---")
    st.markdown("### 📊 Model Info")
    st.markdown("""
    - **Model:** mT5-small (fine-tuned)
    - **Retrieval:** FAISS + SentenceTransformer
    - **Languages:** Hindi 🇮🇳 + English 🇬🇧
    - **KB:** BhashaBench-Ayur + Ayurvedic texts
    """)

    st.markdown("---")
    st.markdown("""
    > ⚠️ **Disclaimer:** This chatbot provides general 
    > Ayurvedic information only. Not a substitute for 
    > professional medical advice.
    """)


# ──────────────────────────────────────────────
# Chat Interface
# ──────────────────────────────────────────────

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant",
        "content": "🙏 नमस्ते! मैं आपका आयुर्वेदिक स्वास्थ्य सलाहकार हूँ।\n\nHello! I am your Ayurvedic health advisor. Ask me about herbs, doshas, remedies, and Ayurvedic wellness in Hindi or English!"
    })

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="🧑" if message["role"] == "user" else "🌿"):
        st.markdown(message["content"])
        
        # Show passages if available
        if show_passages and "passages" in message:
            with st.expander("📚 Retrieved Passages"):
                for i, p in enumerate(message["passages"]):
                    st.markdown(f"""
                    <div class="passage-card">
                        <span class="score-badge">Score: {p['score']:.3f}</span><br/>
                        <strong>{p.get('category', 'general')}</strong><br/>
                        {p['passage_hi'][:200]}...
                    </div>
                    """, unsafe_allow_html=True)

# Chat input
if prompt := st.chat_input("Ask about Ayurveda... / आयुर्वेद के बारे में पूछें..."):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant", avatar="🌿"):
        with st.spinner("🔍 Searching Ayurvedic knowledge..."):
            try:
                rag = load_pipeline()
                result = rag.answer(prompt, top_k=top_k)

                response = result["response"]
                
                # Add language info
                lang_label = "Hindi" if result["detected_language"] == "hi" else "English"
                
                # Build display response
                display = response
                if show_hindi and result["detected_language"] == "en":
                    display += f"\n\n---\n*Hindi: {result['response_hi']}*"

                st.markdown(display)
                st.caption(f"🌐 Detected: {lang_label} | 📚 Retrieved {len(result['retrieved_passages'])} passages")

                # Save to history
                msg = {
                    "role": "assistant",
                    "content": display,
                    "passages": result["retrieved_passages"]
                }
                st.session_state.messages.append(msg)

            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
