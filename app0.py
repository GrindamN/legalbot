# app.py
import streamlit as stlit
from langdetect import detect, LangDetectException
import os
import shutil # For saving files
from datetime import datetime # For displaying timestamps

# Import your modules
import config # Import config first
import prompts
import llm_services
import utils
import qdrant_services # Used for the direct Qdrant list
import document_processor
import chain_builder
import youtube_summarizer
import metadata_store # For persistent metadata

from qdrant_client import QdrantClient as QdrantSDKClient # For deleting collections

# --- Create Persistent Uploads Directory & Initialize Metadata DB (on app start) ---
if not os.path.exists(config.PERSISTENT_UPLOADS_DIR):
    try:
        os.makedirs(config.PERSISTENT_UPLOADS_DIR)
        print(f"Created persistent uploads directory: {config.PERSISTENT_UPLOADS_DIR}")
    except OSError as e:
        stlit.error(f"CRITICAL: Could not create persistent uploads directory '{config.PERSISTENT_UPLOADS_DIR}': {e}")
        stlit.stop()

if '_metadata_db_initialized_flag' not in stlit.session_state:
    print("Attempting to initialize metadata database...")
    metadata_store.initialize_metadata_db()
    stlit.session_state._metadata_db_initialized_flag = True
    print("Metadata database initialization attempt complete.")

# --- Page Config and Title ---
stlit.set_page_config(page_title=config.PAGE_TITLE, layout="wide")
stlit.title(config.APP_TITLE)

# --- Initialize LLMs and Embeddings (once per session) ---
if 'llms_initialized_flag' not in stlit.session_state:
    stlit.session_state.ollama_server_is_ready = llm_services.get_ollama_server_readiness(config.OLLAMA_BASE_URL)
    if stlit.session_state.ollama_server_is_ready:
        stlit.session_state.embeddings_generator, stlit.session_state.chat_llm = llm_services.initialize_llms_cached(
            config.EMBEDDING_MODEL_NAME, config.CHAT_MODEL_NAME, config.OLLAMA_BASE_URL
        )
    else:
        stlit.session_state.embeddings_generator, stlit.session_state.chat_llm = None, None
    stlit.session_state.llms_initialized_flag = True

# --- Session State Initialization for App Logic ---
if "conversation_chain" not in stlit.session_state: stlit.session_state.conversation_chain = None
if "chat_messages" not in stlit.session_state: stlit.session_state.chat_messages = []
if "document_processed_for_chat" not in stlit.session_state: stlit.session_state.document_processed_for_chat = False
if "document_language" not in stlit.session_state: stlit.session_state.document_language = None 
if "active_chain_settings" not in stlit.session_state: stlit.session_state.active_chain_settings = {}
if "active_document_filename" not in stlit.session_state: stlit.session_state.active_document_filename = None 
if "processed_documents_metadata" not in stlit.session_state:
    stlit.session_state.processed_documents_metadata = {}
if "youtube_summary" not in stlit.session_state: stlit.session_state.youtube_summary = None
if "available_qdrant_collections" not in stlit.session_state: stlit.session_state.available_qdrant_collections = []
if "_initial_qdrant_direct_fetch_done" not in stlit.session_state: stlit.session_state._initial_qdrant_direct_fetch_done = False
if "persisted_doc_metadata" not in stlit.session_state: stlit.session_state.persisted_doc_metadata = []
if "_initial_persisted_meta_fetch_done" not in stlit.session_state: stlit.session_state._initial_persisted_meta_fetch_done = False
if "active_hash_match_data" not in stlit.session_state: stlit.session_state.active_hash_match_data = [] # For hash check UI


# --- UI Sidebar ---
with stlit.sidebar:
    llm_services.display_ollama_server_status_in_sidebar(config.OLLAMA_BASE_URL)
    if stlit.session_state.get("embeddings_generator"):
        stlit.sidebar.info(f"Embeddings ('{config.EMBEDDING_MODEL_NAME}') loaded.")
    else:
        stlit.sidebar.warning(f"Embeddings ('{config.EMBEDDING_MODEL_NAME}') NOT loaded.")
    if stlit.session_state.get("chat_llm"):
        stlit.sidebar.info(f"Chat LLM ('{config.CHAT_MODEL_NAME}') loaded.")
    else:
        stlit.sidebar.warning(f"Chat LLM ('{config.CHAT_MODEL_NAME}') NOT loaded.")
    stlit.sidebar.divider()

    stlit.header("1. Document Upload & Initial Processing")
    uploaded_file_ui = stlit.file_uploader(
        "Upload PDF/TXT", type=["pdf", "txt"], key="doc_uploader_main_app"
    )

    stlit.header("2. Retriever & Context Settings")
    stlit.caption("(Applied when processing new docs, if 'Override' chosen, or for 'Direct Qdrant')")
    vector_store_choice_ui = stlit.selectbox(
        "Vector Store (for new docs):", ("Qdrant", "FAISS"), index=0, key="vs_choice_sidebar_app"
    )
    semantic_retriever_selection_ui = stlit.selectbox(
        "Semantic Retriever Type:", ("mmr", "similarity"), index=0, key="sem_type_sidebar_app"
    )
    num_docs_to_retrieve_ui = stlit.slider(
        "Chunks to Retrieve (K):", 1, 10, 3, 1, key="k_slider_sidebar_app"
    )
    use_hybrid_ui = stlit.checkbox(
        "Use Hybrid Search (BM25 + Semantic)", value=True, key="hybrid_check_sidebar_app"
    )

    ensemble_bm25_weight_val, ensemble_semantic_weight_val = 0.4, 0.6
    if use_hybrid_ui:
        stlit.subheader("Hybrid Search Weights")
        ensemble_bm25_weight_val = stlit.slider("BM25 Weight:", 0.0, 1.0, 0.4, 0.1, key="bm25_w_sidebar_app")
        # Determine semantic weight label carefully based on context
        semantic_weight_source_label = "Semantic" 
        if vector_store_choice_ui == "Qdrant": semantic_weight_source_label = "Qdrant"
        elif vector_store_choice_ui == "FAISS": semantic_weight_source_label = "FAISS"
        ensemble_semantic_weight_val = stlit.slider(f"{semantic_weight_source_label} Weight:", 0.0, 1.0, 0.6, 0.1, key="sem_w_sidebar_app")


    mmr_lambda_val, mmr_fetch_k_factor_val = 0.5, 5.0
    if semantic_retriever_selection_ui == "mmr":
        stlit.subheader("MMR Specific Settings")
        mmr_lambda_val = stlit.slider("MMR Lambda (Diversity):", 0.0, 1.0, 0.5, 0.1, key="mmr_lambda_sidebar_app")
        mmr_fetch_k_factor_val = stlit.slider("MMR Fetch K Multiplier:", 2.0, 10.0, 5.0, 0.5, key="mmr_fetch_k_sidebar_app")

    use_map_reduce_ui = stlit.checkbox(
        "Summarize Chunks (Map-Reduce)", value=False, key="map_reduce_sidebar_app"
    )

    if uploaded_file_ui:
        # Clear previous hash match data if file changes or on new button context
        if 'last_uploaded_filename_for_hash_check' not in stlit.session_state or \
           stlit.session_state.last_uploaded_filename_for_hash_check != uploaded_file_ui.name:
            stlit.session_state.active_hash_match_data = []
            stlit.session_state.last_uploaded_filename_for_hash_check = uploaded_file_ui.name


        if stlit.button(f"Process NEW '{uploaded_file_ui.name}'", key="process_new_doc_main_btn_app"):
            if not (stlit.session_state.embeddings_generator and stlit.session_state.chat_llm):
                stlit.sidebar.error("LLM Models not loaded. Cannot process.")
            else:
                uploaded_file_hash = metadata_store.calculate_file_hash(uploaded_file_ui)
                existing_docs_with_hash = []
                if uploaded_file_hash:
                    existing_docs_with_hash = metadata_store.get_metadata_by_file_hash(uploaded_file_hash)
                
                proceed_with_new_processing = True 

                if existing_docs_with_hash and vector_store_choice_ui == "Qdrant": # Hash check only relevant for Qdrant persistence
                    stlit.session_state.active_hash_match_data = existing_docs_with_hash 
                    proceed_with_new_processing = False 
                    
                    stlit.sidebar.warning(f"This file content (Hash: ...{uploaded_file_hash[-8:] if uploaded_file_hash else 'N/A'}) "
                                          f"matches {len(existing_docs_with_hash)} existing document(s) in App Storage.")
                    
                    first_match = existing_docs_with_hash[0]
                    stlit.sidebar.info(f"Most recent match: '{first_match['original_filename']}' "
                                       f"(Stored as: {os.path.basename(first_match['persistent_file_path'])}, "
                                       f"Qdrant: {first_match['qdrant_collection_name']})")
                    
                    stlit.sidebar.subheader("Choose an action for this file content:")
                    # Buttons will trigger a rerun, state needs to be handled on next run if action taken.
                    # For simplicity, we'll handle action directly here and use st.stop() or st.rerun()
                    
                    # Placeholder for button actions to write their outcome
                    action_feedback_placeholder = stlit.sidebar.empty()

                    col_act1, col_act2 = stlit.sidebar.columns(2)
                    if col_act1.button("Activate Existing Match", key=f"activate_hash_match_{uploaded_file_hash[-6:] if uploaded_file_hash else 'nohash'}"):
                        action_feedback_placeholder.info(f"Activating existing match: '{first_match['original_filename']}'...")
                        stlit.session_state.chat_messages = []
                        selected_db_metadata_for_activation = first_match
                        qdrant_coll_name_from_match = selected_db_metadata_for_activation['qdrant_collection_name']
                        original_filename_from_match = selected_db_metadata_for_activation['original_filename']
                        persistent_file_path_from_match = selected_db_metadata_for_activation['persistent_file_path']

                        if not os.path.exists(persistent_file_path_from_match):
                            action_feedback_placeholder.error(f"Error: Original file for existing match not found at {persistent_file_path_from_match}.")
                            stlit.stop()

                        activation_settings_for_match = selected_db_metadata_for_activation['processing_settings'].copy()
                        activation_settings_for_match['filename'] = original_filename_from_match

                        all_chunks_for_bm25_match = None
                        effective_hybrid_for_match = activation_settings_for_match.get('hybrid', False)
                        if effective_hybrid_for_match:
                            all_chunks_for_bm25_match, _, _ = document_processor.extract_text_and_chunks(
                                persistent_file_path_from_match, original_filename_for_display=original_filename_from_match
                            )
                            if not all_chunks_for_bm25_match:
                                action_feedback_placeholder.warning("BM25 (Hash Match): Could not re-chunk. Hybrid semantic-only.")
                                activation_settings_for_match['hybrid'] = False
                        
                        if not activation_settings_for_match.get('hybrid'):
                            activation_settings_for_match['bm25_w'] = "N/A"; activation_settings_for_match['semantic_w'] = "N/A"

                        chain = chain_builder.setup_conversation_chain(
                            all_chunks_for_bm25_match, qdrant_coll_name_from_match, activation_settings_for_match,
                            stlit.session_state.chat_llm, stlit.session_state.embeddings_generator,
                            prompts.QA_PROMPT_TEMPLATE_OBJECT, prompts.MAP_PROMPT_TEMPLATE_OBJECT,
                            prompts.REDUCE_PROMPT_TEMPLATE_OBJECT, prompts.CONDENSE_QUESTION_PROMPT_OBJECT,
                            config.QDRANT_URL, config.QDRANT_API_KEY
                        )
                        if chain:
                            stlit.session_state.conversation_chain = chain
                            stlit.session_state.document_language = selected_db_metadata_for_activation.get('detected_language')
                            stlit.session_state.document_processed_for_chat = True
                            stlit.session_state.active_document_filename = original_filename_from_match
                            stlit.session_state.active_chain_settings = activation_settings_for_match.copy()
                            metadata_store.update_last_accessed_timestamp(qdrant_coll_name_from_match)
                            stlit.session_state.persisted_doc_metadata = metadata_store.get_all_qdrant_document_metadata_cached()
                            action_feedback_placeholder.success(f"Activated existing: '{original_filename_from_match}'.")
                            stlit.rerun() # Essential to refresh main page and clear hash choice UI
                        else:
                            action_feedback_placeholder.error(f"Failed to activate chain for existing match '{original_filename_from_match}'.")
                        stlit.stop() # Stop further processing in this click after handling action

                    if col_act2.button("Process as NEW Anyway", key=f"process_as_new_hash_{uploaded_file_hash[-6:] if uploaded_file_hash else 'nohash'}"):
                        action_feedback_placeholder.info("Proceeding to process as a new, separate entry...")
                        proceed_with_new_processing = True 
                        # No stlit.stop() here, let it fall through to the new processing block
                
                if proceed_with_new_processing:
                    stlit.sidebar.info(f"Processing '{uploaded_file_ui.name}' as a new entry.")
                    stlit.session_state.chat_messages = []
                    stlit.session_state.conversation_chain = None
                    
                    persistent_file_actual_path = None
                    if vector_store_choice_ui == "Qdrant": 
                        target_persistent_path = metadata_store.get_unique_persistent_filepath(uploaded_file_ui.name)
                        try:
                            with open(target_persistent_path, "wb") as f:
                                f.write(uploaded_file_ui.getvalue())
                            persistent_file_actual_path = target_persistent_path
                            stlit.sidebar.caption(f"Saved copy to: {os.path.basename(persistent_file_actual_path)}")
                        except Exception as e_save:
                            stlit.sidebar.error(f"Failed to save uploaded file persistently: {e_save}")
                            stlit.stop() 

                    source_for_chunking = persistent_file_actual_path if persistent_file_actual_path else uploaded_file_ui
                    all_chunks, detected_lang, num_chunks = document_processor.extract_text_and_chunks(
                        source_for_chunking, original_filename_for_display=uploaded_file_ui.name
                    )

                    if all_chunks:
                        current_processing_settings = {
                            "filename": uploaded_file_ui.name, 
                            "vector_store": vector_store_choice_ui,
                            "semantic_type": semantic_retriever_selection_ui, "k": num_docs_to_retrieve_ui,
                            "hybrid": use_hybrid_ui,
                            "bm25_w": ensemble_bm25_weight_val if use_hybrid_ui else "N/A",
                            "semantic_w": ensemble_semantic_weight_val if use_hybrid_ui else "N/A",
                            "mmr_lambda": mmr_lambda_val if semantic_retriever_selection_ui == "mmr" else "N/A",
                            "mmr_fetch_k_f": mmr_fetch_k_factor_val if semantic_retriever_selection_ui == "mmr" else "N/A",
                            "map_reduce": use_map_reduce_ui
                        }
                        final_file_hash = uploaded_file_hash if uploaded_file_hash else metadata_store.calculate_file_hash(uploaded_file_ui)

                        q_collection_name_for_new = None
                        if vector_store_choice_ui == "Qdrant":
                            name_basis_for_qdrant = os.path.basename(persistent_file_actual_path) if persistent_file_actual_path else uploaded_file_ui.name
                            q_collection_name_for_new = utils.normalize_filename_for_collection(name_basis_for_qdrant)

                        chain = chain_builder.setup_conversation_chain(
                            all_chunks, q_collection_name_for_new, current_processing_settings,
                            stlit.session_state.chat_llm, stlit.session_state.embeddings_generator,
                            prompts.QA_PROMPT_TEMPLATE_OBJECT, prompts.MAP_PROMPT_TEMPLATE_OBJECT,
                            prompts.REDUCE_PROMPT_TEMPLATE_OBJECT, prompts.CONDENSE_QUESTION_PROMPT_OBJECT,
                            config.QDRANT_URL, config.QDRANT_API_KEY
                        )
                        if chain:
                            stlit.session_state.conversation_chain = chain
                            stlit.session_state.document_language = detected_lang
                            stlit.session_state.document_processed_for_chat = True
                            stlit.session_state.active_document_filename = uploaded_file_ui.name
                            stlit.session_state.active_chain_settings = current_processing_settings.copy()
                            
                            stlit.session_state.processed_documents_metadata[uploaded_file_ui.name] = {
                                "qdrant_collection_name": q_collection_name_for_new, 
                                "all_chunks_as_docs": all_chunks, "language": detected_lang,
                                "settings": current_processing_settings.copy(),
                                "persistent_file_path_if_any": persistent_file_actual_path 
                            }

                            if vector_store_choice_ui == "Qdrant" and q_collection_name_for_new and persistent_file_actual_path and final_file_hash:
                                success_db = metadata_store.add_or_update_document_metadata(
                                    original_filename=uploaded_file_ui.name,
                                    persistent_file_path=persistent_file_actual_path,
                                    qdrant_collection_name=q_collection_name_for_new,
                                    file_hash=final_file_hash, 
                                    detected_language=detected_lang,
                                    processing_settings=current_processing_settings.copy(),
                                    chunk_count=num_chunks
                                )
                                if success_db:
                                    stlit.sidebar.caption(f"New metadata for '{uploaded_file_ui.name}' saved.")
                                    stlit.session_state.persisted_doc_metadata = metadata_store.get_all_qdrant_document_metadata_cached() 
                                else:
                                    stlit.sidebar.warning(f"Failed to save new metadata for '{uploaded_file_ui.name}'.")
                            
                            stlit.sidebar.success(f"'{uploaded_file_ui.name}' processed and activated as a new entry!")
                            stlit.rerun() 
                        else: stlit.sidebar.error(f"Failed to setup chain for '{uploaded_file_ui.name}'.")
                    else: stlit.sidebar.error(f"Could not extract chunks from '{uploaded_file_ui.name}'.")
                # Only fall through to here if proceed_with_new_processing was true and no action button stopped execution.
                # If an action button was clicked and it called st.stop() or st.rerun(), this part is skipped for that interaction.


    stlit.sidebar.divider()
    stlit.sidebar.header("3. Activate Document Chat")
    
    # --- A. Documents processed in this session ---
    stlit.sidebar.subheader("A. Documents processed in this session")
    processed_doc_names_session = list(stlit.session_state.processed_documents_metadata.keys())
    if processed_doc_names_session:
        default_idx_session = 0
        current_active_doc_for_sidebar = stlit.session_state.active_document_filename 
        if current_active_doc_for_sidebar and current_active_doc_for_sidebar in processed_doc_names_session:
            try: default_idx_session = processed_doc_names_session.index(current_active_doc_for_sidebar) + 1
            except ValueError: pass
        
        selected_doc_key_session = stlit.sidebar.selectbox(
            "Select in-session document:", options=["-- Select --"] + processed_doc_names_session,
            index=default_idx_session, key="doc_selector_session_sidebar_dd_app"
        )
        if selected_doc_key_session != "-- Select --":
            # Sanitize key for button
            sanitized_session_key = "".join(c if c.isalnum() else '_' for c in selected_doc_key_session)
            if stlit.sidebar.button(f"Activate '{selected_doc_key_session}' (Session)", key=f"activate_session_doc_sidebar_btn_app_{sanitized_session_key}"):
                session_meta = stlit.session_state.processed_documents_metadata.get(selected_doc_key_session)
                if session_meta:
                    stlit.info(f"Activating chat for session doc: '{selected_doc_key_session}'...")
                    stlit.session_state.chat_messages = []
                    
                    session_activation_settings = session_meta["settings"].copy()
                    session_chunks_for_bm25 = session_meta["all_chunks_as_docs"]

                    if session_activation_settings.get("hybrid"):
                        path_if_persistent = session_meta.get("persistent_file_path_if_any")
                        if path_if_persistent and os.path.exists(path_if_persistent):
                            stlit.info(f"Hybrid (session): Re-chunking '{selected_doc_key_session}' from its persistent path for BM25 consistency.")
                            session_chunks_for_bm25, _, _ = document_processor.extract_text_and_chunks(
                                path_if_persistent, original_filename_for_display=selected_doc_key_session
                            )
                            if not session_chunks_for_bm25:
                                stlit.warning("BM25 (session): Failed to re-chunk from path. Using in-memory or disabling hybrid.")
                                session_chunks_for_bm25 = session_meta["all_chunks_as_docs"] 
                                if not session_chunks_for_bm25: session_activation_settings['hybrid'] = False 
                        elif not session_chunks_for_bm25: 
                            stlit.warning("BM25 (session): Hybrid enabled but no chunks available. Disabling hybrid.")
                            session_activation_settings['hybrid'] = False
                    
                    if not session_activation_settings.get('hybrid'):
                        session_activation_settings['bm25_w'] = "N/A"; session_activation_settings['semantic_w'] = "N/A"

                    chain = chain_builder.setup_conversation_chain(
                        session_chunks_for_bm25, 
                        session_meta.get("qdrant_collection_name"), 
                        session_activation_settings, 
                        stlit.session_state.chat_llm, stlit.session_state.embeddings_generator,
                        prompts.QA_PROMPT_TEMPLATE_OBJECT, prompts.MAP_PROMPT_TEMPLATE_OBJECT,
                        prompts.REDUCE_PROMPT_TEMPLATE_OBJECT, prompts.CONDENSE_QUESTION_PROMPT_OBJECT,
                        config.QDRANT_URL, config.QDRANT_API_KEY
                    )
                    if chain:
                        stlit.session_state.conversation_chain = chain
                        stlit.session_state.document_language = session_meta["language"]
                        stlit.session_state.document_processed_for_chat = True
                        stlit.session_state.active_document_filename = selected_doc_key_session
                        stlit.session_state.active_chain_settings = session_activation_settings
                        stlit.sidebar.success(f"Chat for '{selected_doc_key_session}' (session) activated!")
                        stlit.rerun()
                    else: stlit.sidebar.error(f"Failed to activate chain for '{selected_doc_key_session}'.")
    else:
        stlit.sidebar.caption("No documents processed in this session yet.")

    # --- B. Documents from App Storage (via DB) ---
    stlit.sidebar.subheader("B. Documents from App Storage (via DB)")
    def refresh_persisted_metadata_list():
        stlit.session_state.persisted_doc_metadata = metadata_store.get_all_qdrant_document_metadata_cached()

    if stlit.sidebar.button("Refresh App-Stored Documents List", key="refresh_persisted_docs_list_btn_app"):
        refresh_persisted_metadata_list()
        stlit.rerun()

    if not stlit.session_state.persisted_doc_metadata and not stlit.session_state._initial_persisted_meta_fetch_done:
        refresh_persisted_metadata_list()
        stlit.session_state._initial_persisted_meta_fetch_done = True

    if stlit.session_state.persisted_doc_metadata:
        persisted_docs_options_map = {
            (f"{item['original_filename']} (Added: {datetime.fromisoformat(item['created_at']).strftime('%y-%m-%d %H:%M') if item.get('created_at') else 'N/A'}, " # Shorter date
             f"Accessed: {datetime.fromisoformat(item['last_accessed_at']).strftime('%y-%m-%d %H:%M') if item.get('last_accessed_at') else 'Never'})"
            ): item
            for item in stlit.session_state.persisted_doc_metadata
        }
        persisted_doc_display_options = ["-- Select App-Stored Document --"] + list(persisted_docs_options_map.keys())
        
        selected_persisted_doc_display_key = stlit.sidebar.selectbox(
            "Select document from app storage:", options=persisted_doc_display_options,
            index=0, key="persisted_doc_selector_sidebar_app"
        )

        if selected_persisted_doc_display_key != "-- Select App-Stored Document --":
            selected_db_metadata = persisted_docs_options_map[selected_persisted_doc_display_key]
            qdrant_coll_name_from_db = selected_db_metadata['qdrant_collection_name']
            original_filename_from_db = selected_db_metadata['original_filename']
            persistent_file_path_from_db = selected_db_metadata['persistent_file_path']
            
            # Sanitize key for buttons based on a stable ID, like qdrant collection name or hash
            sanitized_db_key_suffix = "".join(c if c.isalnum() else '_' for c in qdrant_coll_name_from_db)


            with stlit.sidebar.expander("Stored Settings & Info", expanded=False):
                stlit.caption(f"Stored File: `{os.path.basename(persistent_file_path_from_db)}`")
                stlit.caption(f"Qdrant Collection: `{qdrant_coll_name_from_db}`")
                stlit.caption(f"Original Hash: `{selected_db_metadata.get('file_hash', 'N/A')}`")
                stlit.caption(f"Language: {selected_db_metadata.get('detected_language', 'N/A')}")
                stlit.caption(f"Chunks: {selected_db_metadata.get('chunk_count', 'N/A')}")
                stlit.write("Original Processing Settings Used:")
                stlit.json(selected_db_metadata['processing_settings'])

            override_with_sidebar_settings = stlit.sidebar.checkbox(
                "Override with current sidebar settings (Advanced)", value=False, 
                key=f"override_settings_{sanitized_db_key_suffix}"
            )

            if stlit.sidebar.button(f"Activate '{original_filename_from_db}' (App-Stored)", key=f"activate_stored_doc_{sanitized_db_key_suffix}_btn"):
                stlit.info(f"Activating '{original_filename_from_db}' from app storage ({os.path.basename(persistent_file_path_from_db)})...")
                stlit.session_state.chat_messages = []

                if not os.path.exists(persistent_file_path_from_db):
                    stlit.sidebar.error(f"Error: Original file not found at stored path: {persistent_file_path_from_db}. Cannot activate.")
                    stlit.stop()
                
                activation_settings_for_db_doc = {}
                if override_with_sidebar_settings:
                    stlit.warning("Activating with CURRENT sidebar settings, not stored ones.")
                    activation_settings_for_db_doc = {
                        "filename": original_filename_from_db, "vector_store": "Qdrant",
                        "semantic_type": semantic_retriever_selection_ui, "k": num_docs_to_retrieve_ui,
                        "hybrid": use_hybrid_ui,
                        "bm25_w": ensemble_bm25_weight_val if use_hybrid_ui else "N/A",
                        "semantic_w": ensemble_semantic_weight_val if use_hybrid_ui else "N/A",
                        "mmr_lambda": mmr_lambda_val if semantic_retriever_selection_ui == "mmr" else "N/A",
                        "mmr_fetch_k_f": mmr_fetch_k_factor_val if semantic_retriever_selection_ui == "mmr" else "N/A",
                        "map_reduce": use_map_reduce_ui
                    }
                else:
                    activation_settings_for_db_doc = selected_db_metadata['processing_settings'].copy()
                    activation_settings_for_db_doc['filename'] = original_filename_from_db 

                all_chunks_for_bm25_db = None
                effective_hybrid_for_db_activation = activation_settings_for_db_doc.get('hybrid', False)
                if effective_hybrid_for_db_activation:
                    stlit.info(f"Hybrid search enabled. Re-chunking from: {persistent_file_path_from_db}")
                    all_chunks_for_bm25_db, _, _ = document_processor.extract_text_and_chunks(
                        persistent_file_path_from_db, original_filename_for_display=original_filename_from_db
                    )
                    if not all_chunks_for_bm25_db:
                        stlit.warning(f"BM25: Could not re-chunk from stored file. Hybrid will be semantic-only.")
                        activation_settings_for_db_doc['hybrid'] = False
                    else: stlit.info(f"BM25: Re-chunked {len(all_chunks_for_bm25_db)} chunks.")
                
                if not activation_settings_for_db_doc.get('hybrid'):
                    activation_settings_for_db_doc['bm25_w'] = "N/A"; activation_settings_for_db_doc['semantic_w'] = "N/A"

                chain = chain_builder.setup_conversation_chain(
                    all_chunks_for_bm25_db, qdrant_coll_name_from_db, activation_settings_for_db_doc,
                    stlit.session_state.chat_llm, stlit.session_state.embeddings_generator,
                    prompts.QA_PROMPT_TEMPLATE_OBJECT, prompts.MAP_PROMPT_TEMPLATE_OBJECT,
                    prompts.REDUCE_PROMPT_TEMPLATE_OBJECT, prompts.CONDENSE_QUESTION_PROMPT_OBJECT,
                    config.QDRANT_URL, config.QDRANT_API_KEY
                )
                if chain:
                    stlit.session_state.conversation_chain = chain
                    stlit.session_state.document_language = selected_db_metadata.get('detected_language')
                    stlit.session_state.document_processed_for_chat = True
                    stlit.session_state.active_document_filename = original_filename_from_db
                    stlit.session_state.active_chain_settings = activation_settings_for_db_doc.copy()
                    
                    stlit.session_state.processed_documents_metadata[original_filename_from_db] = {
                        "qdrant_collection_name": qdrant_coll_name_from_db,
                        "all_chunks_as_docs": all_chunks_for_bm25_db, 
                        "language": selected_db_metadata.get('detected_language'),
                        "settings": activation_settings_for_db_doc.copy(),
                        "persistent_file_path_if_any": persistent_file_path_from_db
                    }
                    metadata_store.update_last_accessed_timestamp(qdrant_coll_name_from_db)
                    refresh_persisted_metadata_list()
                    stlit.sidebar.success(f"Chat for '{original_filename_from_db}' (App-Stored) activated!")
                    stlit.rerun()
                else: stlit.sidebar.error(f"Failed to activate chain for '{original_filename_from_db}'.")
            
            if stlit.sidebar.button(f"❌ Delete '{original_filename_from_db}' from Storage", key=f"delete_stored_doc_{sanitized_db_key_suffix}_btn"):
                stlit.sidebar.warning(f"Sure you want to delete all data for '{original_filename_from_db}' (File: {os.path.basename(persistent_file_path_from_db)}, Qdrant: {qdrant_coll_name_from_db})?")
                col1_del, col2_del = stlit.sidebar.columns(2)
                if col1_del.button("YES, DELETE IT ALL", key=f"confirm_delete_stored_{sanitized_db_key_suffix}_btn"):
                    try:
                        q_client = QdrantSDKClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY, timeout=10)
                        q_client.delete_collection(collection_name=qdrant_coll_name_from_db)
                        stlit.sidebar.caption(f"Qdrant collection '{qdrant_coll_name_from_db}' deleted.")
                    except Exception as e_qdel:
                        stlit.sidebar.warning(f"Could not delete Qdrant collection '{qdrant_coll_name_from_db}': {e_qdel}.")
                    
                    meta_del, file_del = metadata_store.delete_metadata_and_file(persistent_file_path_from_db)
                    if meta_del: stlit.sidebar.success(f"Metadata for '{original_filename_from_db}' deleted.")
                    else: stlit.sidebar.error(f"Failed to delete metadata for '{original_filename_from_db}'.")
                    if file_del: stlit.sidebar.success(f"Stored file '{os.path.basename(persistent_file_path_from_db)}' deleted.")
                    else: stlit.sidebar.warning(f"Stored file for '{original_filename_from_db}' not found or could not be deleted from '{persistent_file_path_from_db}'.")
                    
                    if stlit.session_state.active_document_filename == original_filename_from_db:
                        stlit.session_state.conversation_chain = None; stlit.session_state.document_processed_for_chat = False
                        stlit.session_state.active_document_filename = None; stlit.session_state.active_chain_settings = {}
                        stlit.session_state.chat_messages = []
                    if original_filename_from_db in stlit.session_state.processed_documents_metadata:
                        del stlit.session_state.processed_documents_metadata[original_filename_from_db]
                    
                    refresh_persisted_metadata_list()
                    stlit.rerun()
                if col2_del.button("NO, CANCEL", key=f"cancel_delete_stored_{sanitized_db_key_suffix}_btn"):
                    stlit.sidebar.info("Deletion cancelled.")
    else:
        stlit.sidebar.caption("No documents found in app storage. Process one with Qdrant as vector store.")

    # --- C. Advanced: Direct Qdrant Server Collections ---
    stlit.sidebar.subheader("C. Advanced: Direct Qdrant Server Collections")
    if stlit.sidebar.button("Refresh Direct Qdrant List", key="refresh_direct_qdrant_btn_app"):
        stlit.session_state.available_qdrant_collections = qdrant_services.get_existing_qdrant_collections_cached(
            config.QDRANT_URL, config.QDRANT_API_KEY
        )
        stlit.rerun()

    if not stlit.session_state.available_qdrant_collections and not stlit.session_state._initial_qdrant_direct_fetch_done:
        stlit.session_state.available_qdrant_collections = qdrant_services.get_existing_qdrant_collections_cached(
            config.QDRANT_URL, config.QDRANT_API_KEY
        )
        stlit.session_state._initial_qdrant_direct_fetch_done = True

    if stlit.session_state.available_qdrant_collections:
        qdrant_direct_options_map = {
            f"{item['display_filename']} (Coll: {item['qdrant_collection_name']})": item['qdrant_collection_name']
            for item in stlit.session_state.available_qdrant_collections
        }
        qdrant_direct_display_options = ["-- Select Direct Qdrant Collection --"] + list(qdrant_direct_options_map.keys())
        
        selected_qdrant_direct_display_option = stlit.sidebar.selectbox(
            "Select raw Qdrant collection (uses current settings):", options=qdrant_direct_display_options, index=0,
            key="qdrant_direct_selector_sidebar_app"
        )

        if selected_qdrant_direct_display_option != "-- Select Direct Qdrant Collection --":
            selected_direct_q_collection_name = qdrant_direct_options_map[selected_qdrant_direct_display_option]
            selected_direct_display_filename = selected_qdrant_direct_display_option.split(" (Coll: ")[0] 
            sanitized_direct_q_key_suffix = "".join(c if c.isalnum() else '_' for c in selected_direct_q_collection_name)


            if stlit.sidebar.button(f"Activate '{selected_direct_display_filename}' (Direct Qdrant)", key=f"activate_direct_qdrant_{sanitized_direct_q_key_suffix}_btn"):
                stlit.info(f"Activating Qdrant collection: '{selected_direct_q_collection_name}' (as '{selected_direct_display_filename}') with current sidebar settings...")
                stlit.session_state.chat_messages = []
                
                activation_settings_direct_q = {
                    "filename": selected_direct_display_filename, "vector_store": "Qdrant",
                    "semantic_type": semantic_retriever_selection_ui, "k": num_docs_to_retrieve_ui,
                    "hybrid": use_hybrid_ui, 
                    "bm25_w": ensemble_bm25_weight_val if use_hybrid_ui else "N/A",
                    "semantic_w": ensemble_semantic_weight_val if use_hybrid_ui else "N/A",
                    "mmr_lambda": mmr_lambda_val if semantic_retriever_selection_ui == "mmr" else "N/A",
                    "mmr_fetch_k_f": mmr_fetch_k_factor_val if semantic_retriever_selection_ui == "mmr" else "N/A",
                    "map_reduce": use_map_reduce_ui
                }

                all_chunks_for_bm25_direct = None
                effective_hybrid_direct = activation_settings_direct_q.get('hybrid', False)
                if effective_hybrid_direct:
                    if uploaded_file_ui: 
                        stlit.info(f"Using currently uploaded file '{uploaded_file_ui.name}' for BM25 with direct Qdrant collection.")
                        all_chunks_for_bm25_direct, _, _ = document_processor.extract_text_and_chunks(
                            uploaded_file_ui, original_filename_for_display=uploaded_file_ui.name
                        )
                        if not all_chunks_for_bm25_direct:
                            stlit.warning("BM25 (Direct Qdrant): Could not extract chunks from uploaded file. Hybrid will be semantic-only.")
                            activation_settings_direct_q['hybrid'] = False
                    else:
                        stlit.warning("BM25 (Direct Qdrant): Hybrid search is ON, but no file is uploaded. BM25 will not be used. Upload a file if BM25 is desired.")
                        activation_settings_direct_q['hybrid'] = False
                
                if not activation_settings_direct_q.get('hybrid'):
                    activation_settings_direct_q['bm25_w'] = "N/A"; activation_settings_direct_q['semantic_w'] = "N/A"

                chain = chain_builder.setup_conversation_chain(
                    all_chunks_for_bm25_direct, selected_direct_q_collection_name, activation_settings_direct_q,
                    stlit.session_state.chat_llm, stlit.session_state.embeddings_generator,
                    prompts.QA_PROMPT_TEMPLATE_OBJECT, prompts.MAP_PROMPT_TEMPLATE_OBJECT,
                    prompts.REDUCE_PROMPT_TEMPLATE_OBJECT, prompts.CONDENSE_QUESTION_PROMPT_OBJECT,
                    config.QDRANT_URL, config.QDRANT_API_KEY
                )
                if chain:
                    stlit.session_state.conversation_chain = chain
                    stlit.session_state.document_language = None 
                    stlit.session_state.document_processed_for_chat = True
                    stlit.session_state.active_document_filename = selected_direct_display_filename
                    stlit.session_state.active_chain_settings = activation_settings_direct_q
                    stlit.sidebar.success(f"Chat for direct Qdrant doc '{selected_direct_display_filename}' activated!")
                    stlit.rerun()
                else: stlit.sidebar.error(f"Failed to activate chain for direct Qdrant doc '{selected_direct_display_filename}'.")
    else:
        stlit.sidebar.caption("No direct Qdrant collections found or Qdrant not reachable (according to direct check). Check Qdrant server.")


    stlit.sidebar.divider()
    if stlit.sidebar.button("Clear Document Chat History & Memory", key="clear_chat_main_sidebar_final_btn_app"):
        stlit.session_state.chat_messages = []
        if stlit.session_state.conversation_chain and hasattr(stlit.session_state.conversation_chain.memory, 'clear'):
            stlit.session_state.conversation_chain.memory.clear()
        stlit.info("Document chat history and RAG chain memory cleared.")

    stlit.sidebar.divider()
    stlit.sidebar.header("4. YouTube Video Summarizer")
    youtube_url_input_ui = stlit.sidebar.text_input("Enter YouTube Video URL:", key="yt_url_input_sidebar_main_app")
    summary_language_options_yt = {"English": "en", "Serbian": "sr"}
    selected_summary_lang_name_yt = stlit.sidebar.selectbox(
        "Desired YT Summary Language:", options=list(summary_language_options_yt.keys()),
        index=0, key="yt_lang_selector_dd_sidebar_main_app"
    )
    selected_summary_lang_code_yt = summary_language_options_yt[selected_summary_lang_name_yt]

    if stlit.sidebar.button("Summarize YouTube Video", key="summarize_yt_main_btn_sidebar_final_app"):
        if youtube_url_input_ui:
            if stlit.session_state.chat_llm:
                with stlit.spinner("Fetching transcript and summarizing video... This may take a while."):
                    video_summary = youtube_summarizer.get_youtube_summary(
                        youtube_url_input_ui, 
                        target_language_code=selected_summary_lang_code_yt,
                        chat_llm_instance=stlit.session_state.chat_llm
                    )
                    stlit.session_state.youtube_summary = video_summary
            else: stlit.sidebar.error("Chat LLM not loaded. Cannot summarize YouTube video.")
        else: stlit.sidebar.warning("Please enter a YouTube video URL.")


# --- Stop if models didn't load ---
if not stlit.session_state.get("embeddings_generator") or not stlit.session_state.get("chat_llm"):
    stlit.error("LLM or Embeddings models failed to load. Please check sidebar status and ensure Ollama server is running. The application cannot continue.")
    stlit.stop()

# --- Main Page Layout ---
stlit.header("📄 Chat with Your Document")
if stlit.session_state.document_processed_for_chat and stlit.session_state.active_document_filename:
    active_s = stlit.session_state.active_chain_settings
    doc_fn_display = stlit.session_state.active_document_filename
    
    doc_lang_disp_val = "N/A"
    if stlit.session_state.document_language:
        doc_lang_disp_val = stlit.session_state.document_language
    elif stlit.session_state.active_document_filename in stlit.session_state.processed_documents_metadata:
        doc_lang_disp_val = stlit.session_state.processed_documents_metadata[stlit.session_state.active_document_filename].get("language", "N/A")
    
    doc_lang_disp = f"(Lang: {doc_lang_disp_val})"
    vs_disp = active_s.get('vector_store', 'N/A')
    sem_type_disp = active_s.get('semantic_type', 'N/A')
    ret_disp_parts = [f"Vector Store: {vs_disp}"]
    if active_s.get('hybrid'):
        bm25_w_disp = active_s.get('bm25_w', "N/A")
        sem_w_disp = active_s.get('semantic_w', "N/A")
        bm25_w_disp_str = f"{bm25_w_disp:.2f}" if isinstance(bm25_w_disp, float) else str(bm25_w_disp)
        sem_w_disp_str = f"{sem_w_disp:.2f}" if isinstance(sem_w_disp, float) else str(sem_w_disp)
        ret_disp_parts.append(f"Hybrid (BM25w:{bm25_w_disp_str}|{vs_disp}w:{sem_w_disp_str}, type:{sem_type_disp})")
    else:
        ret_disp_parts.append(f"Semantic Only ({vs_disp} type: {sem_type_disp})")
    
    if sem_type_disp == "mmr" and active_s.get("mmr_lambda", "N/A") != "N/A":
        mmr_lambda_disp = active_s.get('mmr_lambda', "N/A")
        mmr_fkf_disp = active_s.get('mmr_fetch_k_f', "N/A")
        mmr_lambda_disp_str = f"{mmr_lambda_disp:.2f}" if isinstance(mmr_lambda_disp, float) else str(mmr_lambda_disp)
        mmr_fkf_disp_str = f"{mmr_fkf_disp:.1f}" if isinstance(mmr_fkf_disp, float) else str(mmr_fkf_disp)
        ret_disp_parts[-1] += f", MMR λ:{mmr_lambda_disp_str}, MMR fetch_k_factor:{mmr_fkf_disp_str}"
    
    ret_disp_full = ", ".join(ret_disp_parts)
    sum_disp = "Map-Reduce Chunks" if active_s.get('map_reduce') else "Direct Chunk Context"
    k_disp = active_s.get('k', 'N/A')

    stlit.info(f"""**Active Document:** `{doc_fn_display}` {doc_lang_disp}
**Retrieval:** {ret_disp_full}, **K:** {k_disp}, **Context Handling:** {sum_disp}""")

    for message in stlit.session_state.chat_messages:
        with stlit.chat_message(message["role"]): stlit.markdown(message["content"])
    
    if user_question_input := stlit.chat_input(f"Ask a question about '{doc_fn_display}'..."):
        stlit.session_state.chat_messages.append({"role": "user", "content": user_question_input})
        with stlit.chat_message("user"): stlit.markdown(user_question_input)
        
        if stlit.session_state.conversation_chain:
            with stlit.chat_message("assistant"):
                answer_placeholder = stlit.empty()
                full_answer_stream = ""
                source_documents_for_display = []
                
                try:
                    lang_instruction_for_llm = ""
                    final_question_for_chain = user_question_input
                    current_doc_lang = stlit.session_state.document_language 

                    if current_doc_lang == 'sr':
                        lang_instruction_for_llm = "Odgovori na sledeće pitanje strogo na srpskom jeziku, jer originalni dokument je na srpskom. "
                    else: 
                        try:
                            question_lang = detect(user_question_input)
                            if question_lang == 'sr': lang_instruction_for_llm = "Odgovori na sledeće pitanje strogo na srpskom jeziku. "
                            elif question_lang == 'en': lang_instruction_for_llm = "Answer the following question strictly in English. "
                        except LangDetectException:
                            stlit.caption("Language detection failed for the question. Using default language for answer.") 
                    
                    if lang_instruction_for_llm:
                        prefix_text = "Instruction: "
                        if lang_instruction_for_llm.startswith("Odgovori"): prefix_text = "Instrukcija: "
                        question_label_text = "User's question is: "
                        if lang_instruction_for_llm.startswith("Odgovori"): question_label_text = "Korisničko pitanje je: "
                        final_question_for_chain = f"{prefix_text}{lang_instruction_for_llm}{question_label_text}\"{user_question_input}\""

                    for chunk_resp in stlit.session_state.conversation_chain.stream({"question": final_question_for_chain}):
                        if "answer" in chunk_resp:
                            token = chunk_resp["answer"]
                            full_answer_stream += token
                            answer_placeholder.markdown(full_answer_stream + "▌")
                        if "source_documents" in chunk_resp and chunk_resp["source_documents"]:
                            source_documents_for_display = chunk_resp["source_documents"]
                    answer_placeholder.markdown(full_answer_stream)
                    ai_response_content = full_answer_stream

                    if not source_documents_for_display and hasattr(stlit.session_state.conversation_chain, 'return_source_documents') and stlit.session_state.conversation_chain.return_source_documents:
                         final_response_for_sources = stlit.session_state.conversation_chain.invoke({"question": final_question_for_chain})
                         source_documents_for_display = final_response_for_sources.get("source_documents", [])
                    
                    if source_documents_for_display:
                        with stlit.expander("Show original source documents retrieved"):
                            for i, doc_obj in enumerate(source_documents_for_display):
                                source_info = doc_obj.metadata.get('source', 'N/A')
                                page_info = f" p.{doc_obj.metadata.get('page')}" if doc_obj.metadata.get('page') is not None else ''
                                stlit.markdown(f"**Source {i+1} (from {source_info}{page_info}):**\n```\n{doc_obj.page_content}\n```\n---")
                except Exception as e:
                    ai_response_content = f"Sorry, an error occurred while generating the answer: {e}"
                    stlit.exception(e) 
                    answer_placeholder.error(ai_response_content)
            
            stlit.session_state.chat_messages.append({"role": "assistant", "content": ai_response_content})
        else:
            stlit.error("Document Q&A chain not initialized. Please process or select a document.")
else:
    stlit.info("Welcome! Please upload and process a new document, or select a previously processed document from the sidebar to begin chatting.")

if "youtube_summary" in stlit.session_state and stlit.session_state.youtube_summary:
    stlit.divider()
    stlit.header("📺 YouTube Video Summarizers")
    stlit.markdown(stlit.session_state.youtube_summary)
    if stlit.button("Clear YouTube Summary", key="clear_yt_summary_main_page_final_btn_app"): 
        stlit.session_state.youtube_summary = None
        stlit.rerun()