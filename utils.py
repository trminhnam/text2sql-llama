import streamlit as st
from llama_cpp import Llama


@st.cache_resource()
def load_llm_model_from_path(model_path):
    llm = Llama(
        model_path=model_path,
        seed=42,
        n_gpu_layers=-1,
        low_vram=True,
        verbose=False,
    )
    return llm
