# config.py
import os

# --- Project Root ---
# Assuming this config.py file is in the project root directory.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# --- Application Configuration ---
PAGE_TITLE = "Chat & Summarize"
APP_TITLE = "Chat with Documents & Summarize YouTube (using Ollama)"

# --- Model Configuration ---
EMBEDDING_MODEL_NAME = "bge-m3:latest" 
CHAT_MODEL_NAME = "gemma3:4b"  

# --- Service URLs ---
OLLAMA_BASE_URL = "http://localhost:11434"
QDRANT_URL = "http://localhost:6333"
QDRANT_API_KEY = None # Set if you have an API key for Qdrant Cloud

# --- File Storage & Database ---
# Directory to store original uploaded files persistently.
PERSISTENT_UPLOADS_DIR_NAME = "persistent_document_uploads"
PERSISTENT_UPLOADS_DIR = os.path.join(PROJECT_ROOT, PERSISTENT_UPLOADS_DIR_NAME)

# SQLite Database for metadata
DB_FILENAME = "document_intelligence_meta.db"
DB_FILEPATH = os.path.join(PROJECT_ROOT, DB_FILENAME)