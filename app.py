import streamlit as st
import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

st.set_page_config(page_title="MediBot Encyclopedia", layout="centered")


# ---------------------------------------------------
# Load Model + Embeddings + FAISS (Cached)
# ---------------------------------------------------
@st.cache_resource
def load_model():

    st.write("⏳ Loading MediBot Engine…")

    # 1. Load Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # 2. Load FAISS Vectorstore
    vectorstore = FAISS.load_local(
        ".", embeddings, allow_dangerous_deserialization=True
    )

    # 3. GGUF MODEL (Upload this file to HF Space root)
    model_name = "Llama-3.2-1B-Instruct-Q5_K_M.gguf"

    if not os.path.exists(model_name):
        raise FileNotFoundError(
            f"❌ Model file '{model_name}' not found. Please upload it to your Space."
        )

    # 4. Load Llama.cpp Model
    llm = LlamaCpp(
        model_path=model_name,
        n_ctx=4096,
        n_threads=4,         # Optimized for HF CPU
        n_batch=128,
        temperature=0.1,
        top_p=0.9,
        max_tokens=700,
        repeat_penalty=1.1,
        n_gpu_layers=0,      # CPU only (HF Free)
        verbose=False,
    )

    # 5. Prompt Template
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

    # 6. Memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    # 7. Full Retrieval Chain
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": PROMPT},
    )

    return chain


# ---------------------------------------------------
# Load Everything
# ---------------------------------------------------
conversation = load_model()


# ---------------------------------------------------
# UI
# ---------------------------------------------------
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


# Keep previous messages
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "Hello! Ask me about any medical condition."
    }]


# Show history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# User Input
if prompt := st.chat_input("Ask something about a medical condition…"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing medical data…"):
            response = conversation({"question": prompt})
            output = response['answer']
            st.markdown(output)

    st.session_state.messages.append({"role": "assistant", "content": output})
