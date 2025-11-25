import streamlit as st
from utils.downloader import load_model_files
from utils.rag_chain import build_conversation_chain

st.set_page_config(page_title="MediBot Encyclopedia", layout="wide")

# Download model + FAISS once on startup
paths = load_model_files()

conversation_chain = build_conversation_chain(paths)

st.title("MediBot Encyclopedia 🏥")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! Ask me any medical question."}
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about a medical condition..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = conversation_chain({"question": prompt})
            answer = response["answer"]
            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
