import os
import streamlit as st
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAI

# ----------------------------
# 🔧 Load environment variables
# ----------------------------
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "YOUR_PINECONE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_KEY")

# ----------------------------
# ⚙️ Configurations
# ----------------------------
index_name = "medicalbot"
pdf_path = "Gale_Encyclopedia_of_Medicine_Vol1.pdf"  # Keep PDF in repo root or use uploader

st.set_page_config(page_title="Medical Encyclopedia Chatbot", layout="wide")
st.title("📚 Medical Encyclopedia Chatbot")

# ----------------------------
# 🧠 Load Embeddings
# ----------------------------
@st.cache_resource(show_spinner=True)
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ----------------------------
# 📄 Load and Split PDF
# ----------------------------
@st.cache_resource(show_spinner=True)
def load_pdf_and_chunk(pdf_path):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    chunks = splitter.split_documents(docs)
    return chunks

# ----------------------------
# 🪣 Setup Pinecone
# ----------------------------
@st.cache_resource(show_spinner=True)
def setup_pinecone(index_name, embedding_dim):
    pc = Pinecone(api_key=PINECONE_API_KEY)
    existing = [i["name"] for i in pc.list_indexes()]
    if index_name not in existing:
        pc.create_index(
            name=index_name,
            dimension=embedding_dim,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    return pc

# ----------------------------
# 🔍 Build Vector Database
# ----------------------------
@st.cache_resource(show_spinner=True)
def build_docsearch(chunks, embeddings):
    return PineconeVectorStore.from_documents(
        documents=chunks, index_name=index_name, embedding=embeddings
    )

# ----------------------------
# 🧠 Load LLM
# ----------------------------
@st.cache_resource(show_spinner=True)
def load_llm(use_openai=True):
    if use_openai:
        return OpenAI(openai_api_key=OPENAI_API_KEY, temperature=0.4, max_tokens=500)
    st.warning("Only OpenAI LLM supported in cloud version.")
    return None

# ----------------------------
# 🔗 Create QA Chain
# ----------------------------
def create_qa_chain(llm, retriever):
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following retrieved context to answer concisely. "
        "If unsure, say you don't know. Use max 3 sentences.\n\n{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, question_answer_chain)

# ----------------------------
# 🚀 App Logic
# ----------------------------
embeddings = load_embeddings()
chunks = load_pdf_and_chunk(pdf_path)

embedding_dim = len(embeddings.embed_query("test"))
pc = setup_pinecone(index_name, embedding_dim)

docsearch = build_docsearch(chunks, embeddings)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

llm = load_llm(use_openai=True)

if llm is None:
    st.error("LLM not loaded. Check your API keys.")
    st.stop()

rag_chain = create_qa_chain(llm, retriever)

query = st.text_input("💬 Ask a medical question or condition:")

if query:
    with st.spinner("Searching and generating answer..."):
        try:
            response = rag_chain.invoke({"input": query})
            st.markdown(f"### **Question:** {query}")
            st.markdown(f"**Answer:** {response['answer']}")
        except Exception as e:
            st.error(f"Error during query: {e}")
