from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

def load_vectorstore(faiss_dir):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    store = FAISS.load_local(
        faiss_dir,
        embeddings,
        allow_dangerous_deserialization=True
    )
    return store

