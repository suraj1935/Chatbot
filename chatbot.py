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
from langchain_core.documents import Document  # Add this import

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load .env explicitly from the script directory
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

# Hardcoded key for testing (replace with your key)
MISTRAL_API_KEY = "VkfTgqxQslrXqjd5eHYRUUzDDcHlmUd6"
st.write("Using hardcoded MISTRAL_API_KEY")

if not MISTRAL_API_KEY:
    st.error("Please provide a valid MISTRAL_API_KEY in your environment.")
    st.stop()

pdf_path = r"C:/Users/sg200/Downloads/chatbot/Gale Encyclopedia of Medicine Vol. 1 (A-B) (1).pdf"

st.set_page_config(page_title="Medical Encyclopedia Chatbot", layout="wide")
st.title("Medical Encyclopedia Chatbot")

@st.cache_resource(show_spinner=True)
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource(show_spinner=True)
def load_pdf_and_chunk(path):
    loader = PyPDFLoader(path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    chunks = splitter.split_documents(docs)
    return chunks  # list of document objects

@st.cache_resource(show_spinner=True)
def embed_documents(docs, _embeddings):
    texts = [doc.page_content for doc in docs]  # extract texts from docs
    vectors = [_embeddings.embed_query(text) for text in texts]
    return texts, np.array(vectors)

@st.cache_resource(show_spinner=True)
def load_llm():
    return ChatMistralAI(mistral_api_key=MISTRAL_API_KEY, temperature=0.4, max_tokens=500)

def simple_retrieve(query, texts, vectors, embeddings, top_k=3):
    q_vec = np.array([embeddings.embed_query(query)])
    sims = cosine_similarity(q_vec, vectors)[0]
    top_indices = sims.argsort()[-top_k:][::-1]
    return [texts[i] for i in top_indices]  # list of strings

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
embeddings = load_embeddings()
chunks = load_pdf_and_chunk(pdf_path)  # documents
texts, vectors = embed_documents(chunks, embeddings)  # texts and vectors as arrays
llm = load_llm()

qa_chain = create_qa_chain(llm)

query = st.text_input("Ask a medical question or condition:")

if query:
    with st.spinner("Searching and generating answer..."):
        try:
            retrieved_texts = simple_retrieve(query, texts, vectors, embeddings)
            
            # Convert retrieved texts back to Document objects
            retrieved_docs = [Document(page_content=text) for text in retrieved_texts]
            
            response = qa_chain.invoke({
                "input": query,
                "context": retrieved_docs  # Pass Document objects, not string
            })
            
            st.markdown(f"### **Question:** {query}")
            st.markdown(f"**Answer:** {response}")
            
        except Exception as e:
            st.error(f"Error during query: {e}")