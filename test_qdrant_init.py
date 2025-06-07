# test_qdrant_init.py
import os
from qdrant_client import QdrantClient
from langchain_ollama import OllamaEmbeddings # Using this as per your app's setup
from langchain_qdrant import QdrantVectorStore
import traceback

# Configuration (adjust if necessary, but defaults should be fine for a local Qdrant)
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", None)
EMBEDDING_MODEL_NAME = "bge-m3:latest" # Match your app
OLLAMA_BASE_URL = "http://localhost:11434" # Match your app

# A known Qdrant collection name that exists on your server
# Replace with one of your actual collection names if different
# For example, the one from your error: 'doc_chat_coll_Vodic_kroz_Pirot_i_okolinu_SR_compressed_3'
# or 'doc_chat_coll_Sta_Pokriva'
EXISTING_COLLECTION_NAME = "doc_chat_coll_Sta_Pokriva" 

print(f"--- Minimal QdrantVectorStore Initialization Test ---")
print(f"Using Qdrant URL: {QDRANT_URL}")
print(f"Using Collection Name: {EXISTING_COLLECTION_NAME}")
print(f"Using Embedding Model: {EMBEDDING_MODEL_NAME} via Ollama at {OLLAMA_BASE_URL}")

try:
    print(f"\nAttempting to initialize OllamaEmbeddings...")
    embeddings_generator = OllamaEmbeddings(
        model=EMBEDDING_MODEL_NAME,
        base_url=OLLAMA_BASE_URL
    )
    print(f"OllamaEmbeddings initialized successfully: {type(embeddings_generator)}")

    print(f"\nAttempting to initialize QdrantClient...")
    qdrant_sdk_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=10)
    print(f"QdrantClient initialized successfully: {type(qdrant_sdk_client)}")
    
    # Verify collection exists with SDK client
    try:
        collection_info = qdrant_sdk_client.get_collection(collection_name=EXISTING_COLLECTION_NAME)
        print(f"SDK: Collection '{EXISTING_COLLECTION_NAME}' found. Status: {collection_info.status}, Vector size: {collection_info.config.params.vectors.size if collection_info.config.params.vectors else 'Unknown'}")
    except Exception as e_sdk_coll:
        print(f"SDK ERROR: Could not get info for collection '{EXISTING_COLLECTION_NAME}': {e_sdk_coll}")
        print(f"Please ensure the collection '{EXISTING_COLLECTION_NAME}' actually exists on your Qdrant server at {QDRANT_URL}.")


    print(f"\nAttempting to initialize QdrantVectorStore with 'embeddings' argument...")
    vector_store = QdrantVectorStore(
        client=qdrant_sdk_client,
        collection_name=EXISTING_COLLECTION_NAME,
        embeddings=embeddings_generator  # This is the critical line
    )
    print(f"\nSUCCESS: QdrantVectorStore initialized successfully!")
    print(f"Type of vector_store: {type(vector_store)}")

except Exception as e:
    print(f"\nERROR during QdrantVectorStore initialization test:")
    print(f"Exception Type: {type(e).__name__}")
    print(f"Exception Message: {e}")
    print(f"\nFull Traceback:")
    traceback.print_exc()

finally:
    print(f"\n--- Test script finished. ---")
    import langchain_qdrant
    print(f"Langchain-Qdrant version being used by this script: {getattr(langchain_qdrant, '__version__', 'N/A')}")
    print(f"Langchain-Qdrant location: {langchain_qdrant.__file__}")
    import langchain_core
    print(f"Langchain-Core version: {getattr(langchain_core, '__version__', 'N/A')}")
    print(f"Langchain-Core location: {langchain_core.__file__}")
    import qdrant_client as qc_sdk
    print(f"Qdrant-Client SDK version: {getattr(qc_sdk, '__version__', 'N/A')}")
    print(f"Qdrant-Client SDK location: {qc_sdk.__file__}")