import os
import streamlit as st
from dotenv import load_dotenv
import tempfile
import requests
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables
load_dotenv()

# Get API key from environment variables
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

if not MISTRAL_API_KEY:
    st.error("Please set MISTRAL_API_KEY in your environment variables.")
    st.stop()

# Flexible imports with error handling
try:
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_mistralai import ChatMistralAI
    from langchain_core.documents import Document
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
    vectors = np.array([_embeddings.embed_query(text) for text in texts])
    return texts, vectors

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

def create_simple_qa_chain(llm):
    """Create a simple QA chain without complex LangChain dependencies"""
    def answer_question(question, context_docs):
        context = "\n\n".join([doc.page_content for doc in context_docs])
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a medical assistant for question-answering tasks. 
            Use the following retrieved context from medical literature to answer accurately and concisely. 
            If the context doesn't contain relevant information, say you don't know. 
            Keep answers to max 3 sentences and use simple language.
            
            Context: {context}"""),
            ("human", "{input}")
        ])
        
        chain = prompt | llm
        response = chain.invoke({
            "input": question,
            "context": context
        })
        return response
    return answer_question

def main():
    # Initialize components
    embeddings = load_embeddings()
    llm = load_llm()
    qa_chain = create_simple_qa_chain(llm)
    
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

        # Option 3: PDF from URL (optional)
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
                    response = qa_chain(query, retrieved_docs)
                    
                    # Display results
                    st.markdown(f"### **❓ Question:** {query}")
                    st.markdown(f"### **💡 Answer:** {response.content}")
                    
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