import streamlit as st
import os
import pickle
import requests
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

st.set_page_config(page_title="MediBot Encyclopedia", layout="centered")


# ------------------------------
# GOOGLE DRIVE DOWNLOAD FUNCTION
# ------------------------------
def download_from_drive(file_id, output_name):
    url = f"https://drive.google.com/uc?export=download&id={file_id}"

    response = requests.get(url, stream=True)

    if response.status_code != 200:
        st.error(f"❌ Failed to download {output_name}")
        st.stop()

    with open(output_name, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    return os.path.abspath(output_name)


# ------------------------------
# DOWNLOAD ALL FILES ONCE
# ------------------------------
@st.cache_resource
def download_files():
    st.write("📥 Downloading model & FAISS from Google Drive... (one-time)")

    model_id = "1vSuKx-zNOLQ79cWSSWRQsI0lWI-Tmmxs"
    faiss_id = "1fqLca-rTgM-EFspTvkFNYcWptgVy7mfV"
    pkl_id =  "10zijiFtZ507CWtj9JqnHmAbXg2FO70ve"

    model_path = download_from_drive(model_id, "model.gguf")
    faiss_path = download_from_drive(faiss_id, "index.faiss")
    pkl_path = download_from_drive(pkl_id, "index.pkl")

    return model_path, faiss_path, pkl_path


# ------------------------------
# LOAD MODEL + VECTORSTORE
# ------------------------------
@st.cache_resource
def load_model():

    model_path, faiss_path, pkl_path = download_files()

    st.write("⏳ Loading MediBot Engine…")

    # 1. Load embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # 2. Load FAISS index + pickle
    with open(pkl_path, "rb") as f:
        index = pickle.load(f)

    vectorstore = FAISS(
        embedding_function=embeddings,
        index=index
    )

    # 3. Load Llama GGUF model
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

    # 4. Prompt setup
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

    # 6. RAG Chain
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


# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "Hello! Ask me about any medical condition."
    }]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input
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
