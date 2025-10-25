import os
import streamlit as st
from dotenv import load_dotenv
from pathlib import Path
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import tempfile
import requests

# Load environment variables
load_dotenv()

# Get API key from environment variables (Render will set this)
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

if not MISTRAL_API_KEY:
    st.error("Please set MISTRAL_API_KEY in your environment variables.")
    st.stop()

# FLEXIBLE IMPORTS - Updated for LangChain compatibility
try:
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_mistralai import ChatMistralAI
    from langchain_core.documents import Document
    
    # Try multiple import paths for create_stuff_documents_chain
    try:
        from langchain.chains.combine_documents import create_stuff_documents_chain
    except ImportError:
        try:
            from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
        except ImportError:
            # Fallback: Implement our own version
            def create_stuff_documents_chain(llm, prompt):
                def chain_func(input_dict):
                    context = "\n\n".join([doc.page_content for doc in input_dict.get("context", [])])
                    return llm.invoke(prompt.format_messages(
                        input=input_dict["input"],
                        context=context
                    ))
                return chain_func
    
    st.success("✅ All imports loaded successfully!")
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

st.set_page_config(page_title="Medical Encyclopedia Chatbot", layout="wide")
st.title("🧬 Medical Encyclopedia Chatbot")
st.markdown("Ask questions about medical conditions and get answers from medical literature")

@st.cache_resource(show_spinner=True)
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource(show_spinner=True)
def load_pdf_and_chunk(pdf_path=None):
    """Load and chunk PDF from file path or URL"""
    try:
        if pdf_path and pdf_path.startswith('http'):
            # Download PDF from URL
            st.info("Downloading PDF from URL...")
            response = requests.get(pdf_path)
            response.raise_for_status()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(response.content)
                pdf_path = tmp_file.name
        
        if not pdf_path or not os.path.exists(pdf_path):
            return []
            
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
        chunks = splitter.split_documents(docs)
        return chunks
    except Exception as e:
        st.error(f"Error loading PDF: {e}")
        return []

@st.cache_resource(show_spinner=True)
def embed_documents(docs, _embeddings):
    if not docs:
        return [], np.array([])
    texts = [doc.page_content for doc in docs]
    vectors = [_embeddings.embed_query(text) for text in texts]
    return texts, np.array(vectors)

@st.cache_resource(show_spinner=True)
def load_llm():
    return ChatMistralAI(mistral_api_key=MISTRAL_API_KEY, temperature=0.4, max_tokens=500)

def simple_retrieve(query, texts, vectors, embeddings, top_k=3):
    if len(texts) == 0:
        return []
    q_vec = np.array([embeddings.embed_query(query)])
    sims = cosine_similarity(q_vec, vectors)[0]
    top_indices = sims.argsort()[-top_k:][::-1]
    return [texts[i] for i in top_indices]

def create_qa_chain(llm):
    system_prompt = (
        "You are a medical assistant for question-answering tasks. "
        "Use the following retrieved context from medical literature to answer accurately and concisely. "
        "If the context doesn't contain relevant information, say you don't know. "
        "Keep answers to max 3 sentences and use simple language.\n\n{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])
    return create_stuff_documents_chain(llm, prompt)

def main():
    # Initialize components
    embeddings = load_embeddings()
    llm = load_llm()
    qa_chain = create_qa_chain(llm)
    
    # Sidebar configuration
    st.sidebar.header("📚 Document Configuration")
    
    # Option 1: Use default PDF from repository
    default_pdf = "medical_encyclopedia.pdf"
    use_default_pdf = st.sidebar.checkbox("Use default medical encyclopedia", value=True)
    
    chunks = []
    texts = []
    vectors = np.array([])
    
    if use_default_pdf and os.path.exists(default_pdf):
        with st.spinner("Loading medical encyclopedia..."):
            chunks = load_pdf_and_chunk(default_pdf)
            if chunks:
                texts, vectors = embed_documents(chunks, embeddings)
                st.sidebar.success(f"✅ Loaded {len(chunks)} document chunks")
    else:
        # Option 2: Upload PDF
        st.sidebar.info("Upload your own medical PDF")
        uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")
        
        if uploaded_file is not None:
            with st.spinner("Processing uploaded PDF..."):
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    pdf_path = tmp_file.name
                
                chunks = load_pdf_and_chunk(pdf_path)
                if chunks:
                    texts, vectors = embed_documents(chunks, embeddings)
                    st.sidebar.success(f"✅ Processed {len(chunks)} document chunks")
                
                # Clean up temporary file
                try:
                    os.unlink(pdf_path)
                except:
                    pass

        # Option 3: PDF from URL
        st.sidebar.info("Or load PDF from URL")
        pdf_url = st.sidebar.text_input("PDF URL:")
        if pdf_url:
            with st.spinner("Downloading and processing PDF from URL..."):
                chunks = load_pdf_and_chunk(pdf_url)
                if chunks:
                    texts, vectors = embed_documents(chunks, embeddings)
                    st.sidebar.success(f"✅ Processed {len(chunks)} document chunks")

    # Main chat interface
    st.header("💬 Ask Medical Questions")
    
    if not chunks:
        st.warning("""
        **Please configure a document source:**
        - ✅ Use the default medical encyclopedia (if available), OR
        - 📤 Upload your own PDF document, OR  
        - 🌐 Provide a PDF URL
        """)
        return

    # Display document info
    st.info(f"**Document ready:** {len(chunks)} text chunks loaded - You can now ask questions!")
    
    # Question input
    query = st.text_input("Enter your medical question:", placeholder="e.g., What are the symptoms of diabetes?")
    
    if query:
        with st.spinner("🔍 Searching medical literature..."):
            try:
                # Retrieve relevant content
                retrieved_texts = simple_retrieve(query, texts, vectors, embeddings)
                
                if retrieved_texts:
                    # Convert to documents and generate answer
                    retrieved_docs = [Document(page_content=text) for text in retrieved_texts]
                    
                    # Handle different chain invocation methods
                    try:
                        response = qa_chain.invoke({
                            "input": query,
                            "context": retrieved_docs
                        })
                    except Exception as chain_error:
                        # Fallback: Direct LLM call
                        st.warning("Using fallback method...")
                        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
                        messages = [
                            ("system", f"Use this context: {context}"),
                            ("human", query)
                        ]
                        response = llm.invoke(messages)
                    
                    # Display results
                    st.markdown(f"### **❓ Question:** {query}")
                    
                    # Handle different response formats
                    if hasattr(response, 'content'):
                        answer = response.content
                    elif hasattr(response, 'text'):
                        answer = response.text
                    else:
                        answer = str(response)
                        
                    st.markdown(f"### **💡 Answer:** {answer}")
                    
                    # Show source context (optional)
                    with st.expander("📖 View source context"):
                        for i, text in enumerate(retrieved_texts, 1):
                            st.markdown(f"**Source {i}:**")
                            st.write(text[:300] + "..." if len(text) > 300 else text)
                            st.divider()
                else:
                    st.warning("No relevant information found in the document for your question.")
                    
            except Exception as e:
                st.error(f"Error during query: {str(e)}")

    # Example questions
    st.sidebar.header("💡 Example Questions")
    example_questions = [
        "What are the common symptoms of asthma?",
        "How is hypertension diagnosed?",
        "What treatments are available for arthritis?",
        "Explain the causes of diabetes",
        "What are the risk factors for heart disease?"
    ]
    
    for q in example_questions:
        if st.sidebar.button(q, key=q):
            st.session_state.last_question = q
            st.rerun()

if __name__ == "__main__":
    main()