# metadata_store.py
import sqlite3
import json
import os
import hashlib
from datetime import datetime

# Assuming config.py is in the root and defines DB_FILEPATH and PERSISTENT_UPLOADS_DIR
# If config.py is structured to define these, import them.
# Otherwise, define DB_FILEPATH directly here if metadata_store.py is in root.
try:
    from config import DB_FILEPATH, PERSISTENT_UPLOADS_DIR
except ImportError:
    # Fallback if config.py isn't set up this way or for standalone use (adjust as needed)
    PROJECT_ROOT_FOR_DB = os.path.dirname(os.path.abspath(__file__)) # Assumes this script is in root
    DB_FILENAME_DEFAULT = "document_intelligence_meta.db"
    DB_FILEPATH = os.path.join(PROJECT_ROOT_FOR_DB, DB_FILENAME_DEFAULT)
    PERSISTENT_UPLOADS_DIR_DEFAULT = os.path.join(PROJECT_ROOT_FOR_DB, "persistent_document_uploads")
    if not os.path.exists(PERSISTENT_UPLOADS_DIR_DEFAULT) and 'stlit' not in globals(): # Avoid stlit if not imported
        try: os.makedirs(PERSISTENT_UPLOADS_DIR_DEFAULT)
        except: pass # Best effort for fallback
    PERSISTENT_UPLOADS_DIR = PERSISTENT_UPLOADS_DIR_DEFAULT


# Try to import Streamlit for conditional error reporting, but don't make it a hard dependency
try:
    import streamlit as stlit
except ImportError:
    stlit = None # type: ignore

def get_db_connection():
    """Establishes a connection to the SQLite database."""
    conn = sqlite3.connect(DB_FILEPATH)
    conn.row_factory = sqlite3.Row
    return conn

def initialize_metadata_db():
    """Creates the metadata table if it doesn't exist and adds content_hint column if missing."""
    db_dir = os.path.dirname(DB_FILEPATH)
    if not os.path.exists(db_dir) and db_dir: # Ensure directory for DB exists
        try:
            os.makedirs(db_dir)
            print(f"Created directory for database: {db_dir}")
        except OSError as e:
            error_message = f"Could not create directory for database '{db_dir}': {e}"
            if stlit: stlit.error(error_message)
            else: print(error_message)
            return 

    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS document_metadata (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            original_filename TEXT NOT NULL,
            persistent_file_path TEXT UNIQUE NOT NULL,
            qdrant_collection_name TEXT UNIQUE,
            file_hash TEXT NOT NULL,
            detected_language TEXT,
            processing_settings TEXT NOT NULL, 
            chunk_count INTEGER,
            content_hint TEXT,                 
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_accessed_at TIMESTAMP
        )
        """)
        conn.commit() # Commit table creation first

        # Add new column 'content_hint' if it doesn't exist (for existing databases)
        # This is a common way to handle schema migrations simply in SQLite with Python
        cursor.execute("PRAGMA table_info(document_metadata)")
        columns = [column[1] for column in cursor.fetchall()]
        if 'content_hint' not in columns:
            try:
                cursor.execute("ALTER TABLE document_metadata ADD COLUMN content_hint TEXT")
                conn.commit() 
                print("Added 'content_hint' column to document_metadata table.")
            except sqlite3.OperationalError as e:
                # This might happen in rare race conditions or if PRAGMA somehow didn't list it
                # but the alter still sees it as duplicate. Usually, the above check is sufficient.
                if "duplicate column name" in str(e).lower():
                    pass 
                else:
                    raise 
        
        # Ensure other indexes are created
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_qdrant_collection_name ON document_metadata (qdrant_collection_name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_file_hash ON document_metadata (file_hash)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_persistent_file_path ON document_metadata (persistent_file_path)")
        conn.commit() # Commit index creation

    except sqlite3.Error as e:
        error_message = f"Metadata DB Initialization Error: {e} (DB Path: {DB_FILEPATH})"
        if stlit: stlit.error(error_message)
        else: print(error_message)
    finally:
        if conn: conn.close()

def add_or_update_document_metadata(
    original_filename: str,
    persistent_file_path: str,
    qdrant_collection_name: str,
    file_hash: str,
    detected_language: str | None, # Can be None
    processing_settings: dict,
    chunk_count: int,
    content_hint: str | None = None # NEW PARAMETER
):
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        settings_json = json.dumps(processing_settings)
        timestamp = datetime.now()

        cursor.execute("""
        INSERT INTO document_metadata (
            original_filename, persistent_file_path, qdrant_collection_name, file_hash,
            detected_language, processing_settings, chunk_count, content_hint, 
            created_at, last_accessed_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(persistent_file_path) DO UPDATE SET
            original_filename=excluded.original_filename,
            qdrant_collection_name=excluded.qdrant_collection_name,
            file_hash=excluded.file_hash,
            detected_language=excluded.detected_language,
            processing_settings=excluded.processing_settings,
            chunk_count=excluded.chunk_count,
            content_hint=excluded.content_hint, 
            last_accessed_at=excluded.last_accessed_at,
            created_at = CASE WHEN created_at IS NULL THEN excluded.created_at ELSE created_at END
        """, (
            original_filename, persistent_file_path, qdrant_collection_name, file_hash,
            detected_language, settings_json, chunk_count, content_hint, 
            timestamp, timestamp
        ))
        conn.commit()
        return True
    except sqlite3.Error as e:
        print(f"SQLite error in add_or_update_document_metadata: {e}")
        return False
    finally:
        if conn: conn.close()

def get_metadata_by_file_hash(file_hash: str) -> list[dict]:
    conn = None; metadata_list = []
    if not file_hash: return metadata_list
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
        SELECT id, original_filename, persistent_file_path, qdrant_collection_name, 
               file_hash, detected_language, processing_settings, chunk_count, content_hint, 
               created_at, last_accessed_at
        FROM document_metadata WHERE file_hash = ? ORDER BY last_accessed_at DESC, created_at DESC
        """, (file_hash,))
        rows = cursor.fetchall()
        for row in rows:
            data = dict(row)
            data['processing_settings'] = json.loads(data['processing_settings']) if data.get('processing_settings') else {}
            metadata_list.append(data)
    except sqlite3.Error as e: print(f"SQLite error in get_metadata_by_file_hash: {e}")
    finally:
        if conn: conn.close()
    return metadata_list

def get_all_qdrant_document_metadata_cached() -> list[dict]: # Renamed for clarity in app.py
    conn = None; metadata_list = []
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
        SELECT id, original_filename, persistent_file_path, qdrant_collection_name, 
               file_hash, detected_language, processing_settings, chunk_count, content_hint,
               created_at, last_accessed_at 
        FROM document_metadata WHERE qdrant_collection_name IS NOT NULL 
        ORDER BY last_accessed_at DESC, original_filename ASC
        """)
        rows = cursor.fetchall()
        for row in rows:
            data = dict(row)
            data['processing_settings'] = json.loads(data['processing_settings']) if data.get('processing_settings') else {}
            metadata_list.append(data)
    except sqlite3.Error as e: print(f"SQLite error in get_all_qdrant_document_metadata_cached: {e}")
    finally:
        if conn: conn.close()
    return metadata_list

def get_metadata_by_qdrant_collection_name(collection_name: str) -> dict | None:
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
        SELECT id, original_filename, persistent_file_path, qdrant_collection_name, 
               file_hash, detected_language, processing_settings, chunk_count, content_hint,
               created_at, last_accessed_at 
        FROM document_metadata WHERE qdrant_collection_name = ?
        """, (collection_name,))
        row = cursor.fetchone()
        if row:
            data = dict(row)
            data['processing_settings'] = json.loads(data['processing_settings']) if data.get('processing_settings') else {}
            return data
        return None
    except sqlite3.Error as e: print(f"SQLite error in get_metadata_by_qdrant_collection_name: {e}")
    finally:
        if conn: conn.close()
    return None


def update_last_accessed_timestamp(qdrant_collection_name: str):
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("UPDATE document_metadata SET last_accessed_at = ? WHERE qdrant_collection_name = ?",
                       (datetime.now(), qdrant_collection_name))
        conn.commit()
    except sqlite3.Error as e:
        print(f"SQLite error in update_last_accessed_timestamp: {e}")
    finally:
        if conn: conn.close()

def calculate_file_hash(file_object_or_path) -> str | None:
    sha256_hash = hashlib.sha256()
    try:
        if hasattr(file_object_or_path, 'read') and callable(file_object_or_path.read): 
            file_object_or_path.seek(0)
            for byte_block in iter(lambda: file_object_or_path.read(4096), b""):
                sha256_hash.update(byte_block)
            file_object_or_path.seek(0)
            return sha256_hash.hexdigest()
        elif isinstance(file_object_or_path, str) and os.path.exists(file_object_or_path): 
            with open(file_object_or_path, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()
        else:
            print(f"calculate_file_hash: Invalid input or file not found: {type(file_object_or_path)}")
            return None
    except Exception as e: print(f"Error calculating file hash: {e}"); return None

def delete_metadata_and_file(persistent_file_path_to_delete: str) -> tuple[bool, bool]:
    metadata_deleted = False; file_deleted = False
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM document_metadata WHERE persistent_file_path = ?", (persistent_file_path_to_delete,))
        conn.commit()
        if cursor.rowcount > 0: metadata_deleted = True
        if os.path.exists(persistent_file_path_to_delete):
            os.remove(persistent_file_path_to_delete)
            file_deleted = True
            print(f"Deleted physical file: {persistent_file_path_to_delete}")
        else: print(f"Physical file not found for deletion: {persistent_file_path_to_delete}")
    except sqlite3.Error as e: print(f"SQLite error in delete_metadata_and_file (metadata part): {e}")
    except OSError as e: print(f"OS error in delete_metadata_and_file (file deletion part): {e}")
    finally:
        if conn: conn.close()
    return metadata_deleted, file_deleted

def get_unique_persistent_filepath(original_filename: str) -> str:
    if not os.path.exists(PERSISTENT_UPLOADS_DIR):
        try: os.makedirs(PERSISTENT_UPLOADS_DIR)
        except OSError as e: 
            print(f"Error creating PERSISTENT_UPLOADS_DIR '{PERSISTENT_UPLOADS_DIR}': {e}")
            # Fallback or re-raise. For now, just print and continue.
            # The calling function should handle if PERSISTENT_UPLOADS_DIR isn't writable.
            pass
    base, ext = os.path.splitext(original_filename)
    safe_base = "".join(c if c.isalnum() or c in ['_', '-'] else '_' for c in base[:100]) 
    counter = 0
    while True:
        suffix = f"_{counter}" if counter > 0 else ""
        prospective_name = f"{safe_base}{suffix}{ext}"
        filepath = os.path.join(PERSISTENT_UPLOADS_DIR, prospective_name)
        if not os.path.exists(filepath): return filepath
        counter += 1