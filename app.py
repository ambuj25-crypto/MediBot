import streamlit as st
import os
import pickle
from huggingface_hub import hf_hub_download
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # fix torch conflicts


st.set_page_config(page_title="MediBot Encyclopedia", layout="centered")

# ------------------------------
# DOWNLOAD FILES FROM HF DATASET
# ------------------------------
@st.cache_resource
def download_files():
    repo_id = "ambuj2507/medibot-data"   # <-- CHANGE TO YOUR DATASET

    st.write("📥 Downloading model & FAISS... (one-time)")

    model_path = hf_hub_download(
        repo_id=repo_id,
        filename="Llama-3.2-1B-Instruct-Q4_K_S.gguf"
    )

    faiss_path = hf_hub_download(
        repo_id=repo_id,
        filename="index.faiss"
    )

    pkl_path = hf_hub_download(
        repo_id=repo_id,
        filename="index.pkl"
    )

    return model_path, faiss_path, pkl_path


# ------------------------------
# LOAD FULL MODEL + FAISS RAG
# ------------------------------
@st.cache_resource
def load_model():

    model_path, faiss_path, pkl_path = download_files()

    st.write("⏳ Loading MediBot Engine…")

    # 1. Load Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # 2. Load FAISS vectorstore
    with open(pkl_path, "rb") as f:
        index = pickle.load(f)

    vectorstore = FAISS(
        embedding_function=embeddings,
        index=index
    )

    # 3. Load GGUF Model
    llm = LlamaCpp(
        model_path=model_path,
        n_ctx=4096,
        n_threads=4,
        n_batch=128,
        temperature=0.1,
        top_p=0.9,
        max_tokens=700,
        repeat_penalty=1.1,
        n_gpu_layers=0,
        verbose=False,
    )

    # 4. Prompt Template
    template = """
You are MediBot — an AI medical assistant.
Use ONLY the provided medical context.
If information is missing, say:
"I don't have that information from the encyclopedia."

Rules:
1. Be concise.
2. Avoid hallucinating.
3. Use markdown tables for lists.
4. Never give drug dosage or prescriptions.

MEDICAL CONTEXT:
{context}

USER QUESTION:
{question}
"""

    PROMPT = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

    # 5. Memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    # 6. Combined RAG Chain
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": PROMPT}
    )

    return chain


# ------------------------------
# MAIN APP
# ------------------------------
conversation = load_model()

st.title("🩺 MediBot — Medical Encyclopedia Assistant")

with st.sidebar:
    st.header("Controls")
    if st.button("Start New Chat"):
        conversation.memory.clear()
        st.session_state.messages = [{
            "role": "assistant",
            "content": "Hello! Ask me about any medical condition."
        }]
        st.rerun()

# Session Memory
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "Hello! Ask me about any medical condition."
    }]

# Show history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask something about a medical condition…"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing medical data…"):
            response = conversation({"question": prompt})
            output = response["answer"]
            st.markdown(output)

    st.session_state.messages.append({"role": "assistant", "content": output})


