# qdrant_services.py
import streamlit as stlit
from qdrant_client import QdrantClient as QdrantSDKClient

@stlit.cache_data(ttl=300) # Keep TTL for general refreshes, but allow busting
def get_existing_qdrant_collections_cached(qdrant_url, api_key=None, _cache_buster=None): # Added _cache_buster
    collections_data = []
    try:
        client = QdrantSDKClient(url=qdrant_url, api_key=api_key, timeout=10)
        response = client.get_collections()
        if response and hasattr(response, 'collections'):
            for col_info in response.collections:
                # Assuming your collections relevant to the app are prefixed
                # Adjust this if your naming convention is different
                if col_info.name.startswith("doc_chat_coll_"): 
                    display_name_raw = col_info.name.replace("doc_chat_coll_", "")
                    # Attempt to make the display name more readable
                    display_name = display_name_raw.replace("_", " ").strip() 
                    collections_data.append({"qdrant_collection_name": col_info.name, "display_filename": display_name})
    except Exception as e:
        # Be less verbose for common "server not running" errors
        if "Connection refused" not in str(e) and "Failed to resolve" not in str(e) and "[Errno 111]" not in str(e) and "Max retries exceeded" not in str(e):
             stlit.sidebar.warning(f"Could not list Qdrant collections: {e}")
        # else: stlit.sidebar.caption("Qdrant not reachable for listing.") # Optional quieter message
    return collections_data
