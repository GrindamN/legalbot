# llm_services.py
import streamlit as stlit
import requests
from langchain_ollama import OllamaLLM, OllamaEmbeddings

# THIS FUNCTION WAS MISSING IN THE PREVIOUS VERSION I PROVIDED
def get_ollama_server_readiness(ollama_base_url: str) -> bool:
    """Checks if the Ollama server is responsive without displaying UI messages."""
    try:
        health_check = requests.get(ollama_base_url, timeout=5)
        return health_check.status_code == 200 and "Ollama is running" in health_check.text
    except requests.exceptions.RequestException:
        return False

def display_ollama_server_status_in_sidebar(ollama_base_url: str):
    """Displays Ollama server status messages in the Streamlit sidebar."""
    stlit.sidebar.info(f"Checking Ollama server at {ollama_base_url}...")
    try:
        health_check = requests.get(ollama_base_url, timeout=5)
        if health_check.status_code == 200 and "Ollama is running" in health_check.text:
            stlit.sidebar.success("Ollama server is responsive.")
        else:
            stlit.sidebar.error(f"Ollama server not responding. Status: {health_check.status_code}")
    except requests.exceptions.RequestException as e:
        stlit.sidebar.error(f"Failed to connect to Ollama server at {ollama_base_url}: {e}")

@stlit.cache_resource # This decorator caches the returned objects
def initialize_llms_cached(embedding_model_name, chat_model_name, ollama_base_url):
    embed_gen = None
    chat_model = None
    # These sidebar messages are okay here as @st.cache_resource is usually called within Streamlit's main flow
    stlit.sidebar.info(f"Initializing models...")
    try:
        embed_gen = OllamaEmbeddings(model=embedding_model_name, base_url=ollama_base_url)
        stlit.sidebar.info(f"Embeddings ('{embedding_model_name}') interface ready.")
    except Exception as e:
        stlit.sidebar.error(f"Failed to init embeddings ('{embedding_model_name}'): {e}.")
        embed_gen = None # Ensure it's None on failure
    try:
        stlit.sidebar.info(f"Attempting to initialize LLM: {chat_model_name}")
        chat_model = OllamaLLM(model=chat_model_name, base_url=ollama_base_url, system=("You are a helpful AI assistant."))
        stlit.sidebar.info(f"LLM {chat_model_name} interface potentially ready.")
    except Exception as e:
        stlit.sidebar.error(f"Failed to init chat LLM ('{chat_model_name}'): {e}.")
        chat_model = None # Ensure it's None on failure
    return embed_gen, chat_model