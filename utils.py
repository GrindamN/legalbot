# utils.py
import os
import re

def normalize_filename_for_collection(filename: str) -> str:
    name_part = os.path.splitext(filename)[0]
    normalized = re.sub(r'[^a-zA-Z0-9_-]', '_', name_part)
    return f"doc_chat_coll_{normalized[:50]}"