import os
import streamlit as st
from dotenv import load_dotenv
from pathlib import Path
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from langchain_core.documents import Document
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables
load_dotenv()

# ✅ Secure API key handling
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    MISTRAL_API_KEY = st.secrets.get("MISTRAL_API_KEY")

if not MISTRAL_API_KEY:
    st.error("Please set MISTRAL_API_KEY in environment variables or Hugging Face secrets")
    st.stop()

# ✅ Use relative path or file upload
PDF_FILENAME = "medical_encyclopedia.pdf"

st.set_page_config(page_title="Medical Encyclopedia Chatbot", layout="wide")
st.title("Medical Encyclopedia Chatbot")

@st.cache_resource(show_spinner=True)
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource(show_spinner=True)
def load_pdf_and_chunk():
    # Check if PDF exists, if not show upload option
    if not os.path.exists(PDF_FILENAME):
        st.error(f"PDF file '{PDF_FILENAME}' not found. Please upload it to your Hugging Face space.")
        st.info("You can upload files in the 'Files and versions' section of your Space")
        return []
    
    try:
        loader = PyPDFLoader(PDF_FILENAME)
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
        "You are an assistant for question-answering tasks. "
        "Use the following retrieved context to answer concisely. "
        "If unsure, say you don't know. Use max 3 sentences.\n\n{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])
    return create_stuff_documents_chain(llm, prompt)

# Main app logic
def main():
    embeddings = load_embeddings()
    chunks = load_pdf_and_chunk()
    
    if not chunks:
        st.warning("No documents loaded. Please ensure the PDF file is uploaded to the space.")
        return
    
    texts, vectors = embed_documents(chunks, embeddings)
    llm = load_llm()
    qa_chain = create_qa_chain(llm)

    query = st.text_input("Ask a medical question or condition:")

    if query:
        with st.spinner("Searching and generating answer..."):
            try:
                retrieved_texts = simple_retrieve(query, texts, vectors, embeddings)
                
                if not retrieved_texts:
                    st.warning("No relevant information found in the documents.")
                    return
                
                retrieved_docs = [Document(page_content=text) for text in retrieved_texts]
                
                response = qa_chain.invoke({
                    "input": query,
                    "context": retrieved_docs
                })
                
                st.markdown(f"### **Question:** {query}")
                st.markdown(f"**Answer:** {response}")
                
            except Exception as e:
                st.error(f"Error during query: {e}")

if __name__ == "__main__":
    main()