import streamlit as st
from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os
import asyncio
import aiohttp
import json

st.set_page_config(page_title="Medical Encyclopedia Chatbot", layout="wide")

class PerplexityClient:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.perplexity.ai"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    async def chat_completion(self, messages, model="sonar-pro", temperature=0.1, max_tokens=500):
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": 0.9
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data["choices"][0]["message"]["content"]
                else:
                    error_text = await response.text()
                    raise Exception(f"Perplexity API error: {response.status} - {error_text}")

@st.cache_resource
def load_embeddings():
    """Load pre-trained embeddings"""
    try:
        cache_path = Path("embeddings_cache.pkl")
        
        if not cache_path.exists():
            st.error("""
            ❌ No trained embeddings found!
            
            **Please ensure embeddings_cache.pkl is in your repository.**
            """)
            return [], []
        
        with open(cache_path, 'rb') as f:
            cache_data = pickle.load(f)
        
        st.sidebar.success(f"✅ Loaded {len(cache_data['texts'])} medical chunks")
        return cache_data['texts'], cache_data['vectors']
    
    except Exception as e:
        st.error(f"Error loading embeddings: {e}")
        return [], []

def get_perplexity_client():
    """Initialize Perplexity client using Hugging Face secrets"""
    perplexity_api_key = os.environ.get("PERPLEXITY_API_KEY")
    if not perplexity_api_key:
        st.error("""
        ❌ Perplexity API key not found!
        
        **For Hugging Face Deployment:**
        1. Go to your Space → Settings → Variables
        2. Add: PERPLEXITY_API_KEY = your_actual_api_key
        3. Redeploy the app
        
        **API Key Format:** pplx-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        """)
        st.stop()
    return PerplexityClient(perplexity_api_key)

def retrieve_relevant_docs(query, document_texts, document_vectors, top_k=5):
    embeddings_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    query_vec = embeddings_model.encode([query])
    sims = cosine_similarity(query_vec, document_vectors)[0]
    top_indices = sims.argsort()[-top_k:][::-1]
    return [(document_texts[i], float(sims[i])) for i in top_indices]

async def query_perplexity_async(perplexity_client, user_query, context_docs):
    context_str = "\n\n".join([doc for doc, _ in context_docs[:3]])
    
    system_message = """You are a medical information assistant. Use ONLY the provided medical documents to answer questions. 
    If the documents don't contain relevant information, say "I don't have enough medical information to answer this question accurately."
    Provide concise, accurate medical information and always recommend consulting healthcare professionals."""

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": f"Medical Context:\n{context_str}\n\nQuestion: {user_query}\n\nAnswer based ONLY on the medical context above:"}
    ]
    
    try:
        response = await perplexity_client.chat_completion(messages)
        return response
    except Exception as e:
        return f"Error querying Perplexity: {str(e)}"

def main():
    st.title("🏥 Medical Encyclopedia Chatbot")
    st.markdown("Ask medical questions and get AI-powered answers from medical encyclopedia content")
    
    # Load embeddings
    document_texts, document_vectors = load_embeddings()
    if not document_texts:
        st.info("""
        💡 **Setup Instructions:**
        - Make sure `embeddings_cache.pkl` is uploaded to your repository
        - Add PERPLEXITY_API_KEY in Hugging Face Space variables
        """)
        return
    
    perplexity_client = get_perplexity_client()
    
    # Initialize chat history
    if "history" not in st.session_state:
        st.session_state.history = []
    if "sources" not in st.session_state:
        st.session_state.sources = {}

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        st.info(f"📚 Medical Database: {len(document_texts)} chunks")
        
        top_k = st.slider("Reference documents", 3, 10, 5)
        show_sources = st.checkbox("Show sources", value=True)
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown("""
        - **RAG System** with medical encyclopedia
        - **Perplexity AI** for responses
        - **5157 medical text chunks**
        """)
        
        st.markdown("---")
        st.warning("""
        ⚠️ **Medical Disclaimer**  
        For informational purposes only.  
        Consult healthcare professionals for medical advice.
        """)

    # Chat interface
    st.markdown("---")
    query = st.text_input("💬 Ask a medical question:", placeholder="e.g., What are the symptoms of diabetes?")
    
    col1, col2 = st.columns([1, 6])
    with col1:
        ask_button = st.button("🔍 Ask", type="primary", use_container_width=True)

    if query and ask_button:
        with st.spinner("🔍 Searching medical encyclopedia..."):
            retrieved_docs = retrieve_relevant_docs(query, document_texts, document_vectors, top_k)
            
            if retrieved_docs:
                with st.spinner("🤖 Generating medical response..."):
                    try:
                        answer = asyncio.run(query_perplexity_async(perplexity_client, query, retrieved_docs))
                        
                        # Store in history
                        st.session_state.history.append({"role": "user", "content": query})
                        st.session_state.history.append({"role": "assistant", "content": answer})
                        st.session_state.sources[len(st.session_state.history) - 1] = retrieved_docs
                        
                    except Exception as e:
                        st.error(f"Error generating response: {e}")
            else:
                st.warning("No relevant medical information found.")

    # Display conversation history
    st.markdown("---")
    st.subheader("📝 Conversation History")
    
    if not st.session_state.history:
        st.info("💡 Ask a medical question to start!")
    
    for i, message in enumerate(st.session_state.history):
        if message["role"] == "user":
            st.markdown(f"**👤 You:** {message['content']}")
        else:
            st.markdown(f"**🏥 Assistant:** {message['content']}")
            
            # Show sources
            if show_sources and i in st.session_state.sources:
                with st.expander(f"📚 Medical Sources ({len(st.session_state.sources[i])} references)"):
                    for j, (doc, score) in enumerate(st.session_state.sources[i]):
                        st.markdown(f"**Source {j+1}** (Relevance: `{score:.3f}`)")
                        st.text(doc[:300] + "..." if len(doc) > 300 else doc)
                        st.markdown("---")

    # Clear history
    if st.session_state.history and st.button("Clear History"):
        st.session_state.history = []
        st.session_state.sources = {}
        st.rerun()

if __name__ == "__main__":
    main()