import os
import streamlit as st
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables
load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "YOUR_MISTRAL_KEY")

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
    return chunks

@st.cache_resource(show_spinner=True)
def embed_documents(docs, _embeddings):
    # Extract text content from documents and embed
    texts = [doc.page_content for doc in docs]
    vectors = [_embeddings.embed_query(text) for text in texts]
    return texts, np.array(vectors)

@st.cache_resource(show_spinner=True)
def load_llm():
    return ChatMistralAI(mistral_api_key=MISTRAL_API_KEY, temperature=0.4, max_tokens=500)

def simple_retrieve(query, texts, vectors, embeddings, top_k=3):
    # Embed query and find top k similar document texts
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
embeddings = load_embeddings()
chunks = load_pdf_and_chunk(pdf_path)
texts, vectors = embed_documents(chunks, embeddings)
llm = load_llm()

if llm is None:
    st.error("LLM not loaded. Check your API keys.")
    st.stop()

qa_chain = create_qa_chain(llm)

query = st.text_input(" Ask a medical question or condition:")

if query:
    with st.spinner("Searching and generating answer..."):
        try:
            retrieved_docs = simple_retrieve(query, texts, vectors, embeddings)
            context = "\n\n".join(retrieved_docs)  # Join texts to form context string
            response = qa_chain.invoke({"context": context, "input": query})
            st.markdown(f"### **Question:** {query}")
            st.markdown(f"**Answer:** {response['output']}")
        except Exception as e:
            st.error(f"Error during query: {e}")


#suraj