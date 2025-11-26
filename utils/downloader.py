from huggingface_hub import hf_hub_download
import os
import streamlit as st
from huggingface_hub import hf_hub_download

@st.cache_resource
def download_file(repo, filename):
    return hf_hub_download(repo_id=repo, filename=filename, repo_type="dataset")


def download_all_files():
    repo = "ambuj2507/medibot-data"   # CHANGE THIS TO YOUR ACTUAL DATASET

    model = hf_hub_download(repo, "Llama-3.2-1B-Instruct-Q5_K_M.gguf")
    faiss_file = hf_hub_download(repo, "index.faiss")
    pkl_file = hf_hub_download(repo, "index.pkl")

    # return directory containing FAISS files
    base = os.path.dirname(faiss_file)

    return {
        "model": model,
        "faiss_dir": base
    }

