# app.py
import streamlit as stlit
from langdetect import detect, LangDetectException
import os
from datetime import datetime
import traceback
import time

import config
import re
import llm_services
import utils
import qdrant_services
import document_processor
import chain_builder
import youtube_summarizer
import metadata_store

from qdrant_client import QdrantClient as QdrantSDKClient
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_qdrant import QdrantVectorStore
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.ensemble import EnsembleRetriever
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.callbacks import BaseCallbackHandler

# --- Helper Functions ---
def get_combined_document_and_collection_names_for_intent() -> list[tuple[str, str, str, str | None]]:
    items_map = {}
    processed_qdrant_collections = set()
    for meta in stlit.session_state.get("persisted_doc_metadata", []):
        display_name = meta['original_filename']; content_hint = meta.get('content_hint'); q_coll_name = meta.get('qdrant_collection_name')
        if display_name not in items_map:
            items_map[display_name] = (display_name, meta['original_filename'], "app_managed_db", content_hint)
            if q_coll_name: processed_qdrant_collections.add(q_coll_name)
    for name, meta_session in stlit.session_state.get("processed_documents_metadata", {}).items():
        if name not in items_map:
            vs_type = meta_session.get("settings", {}).get("vector_store", "Unknown"); source_type = "app_managed_session_qdrant" if vs_type == "Qdrant" else "app_managed_session_faiss"
            items_map[name] = (name, name, source_type, meta_session.get("content_hint"))
            if vs_type == "Qdrant" and meta_session.get("qdrant_collection_name"): processed_qdrant_collections.add(meta_session.get("qdrant_collection_name"))
    if not stlit.session_state.get("_initial_qdrant_direct_fetch_done"):
        stlit.session_state.available_qdrant_collections = qdrant_services.get_existing_qdrant_collections_cached(config.QDRANT_URL, config.QDRANT_API_KEY, _cache_buster=time.time())
        stlit.session_state._initial_qdrant_direct_fetch_done = True
    for q_item in stlit.session_state.get("available_qdrant_collections", []):
        qdrant_coll_name = q_item['qdrant_collection_name']
        if qdrant_coll_name not in processed_qdrant_collections:
            display_name_raw = q_item['display_filename']; display_name_for_llm = f"{display_name_raw} (Qdrant Server)"
            if display_name_for_llm not in items_map:
                items_map[display_name_for_llm] = (display_name_for_llm, qdrant_coll_name, "direct_qdrant_collection", "Raw Qdrant server collection.")
    return sorted(list(items_map.values()), key=lambda x: x[0])

def classify_intent_for_rag_extended(user_query: str, source_tuples: list[tuple[str, str, str, str | None]], llm_instance) -> tuple[bool, str | None, str | None, str | None]:
    if not llm_instance or not source_tuples: return False, None, None, None
    source_prompt_list = []
    for disp_name, _, _, hint in source_tuples[:10]:
        hint_str = f" (Hint: {hint[:100].strip() if hint else ''}...)" if hint and hint.strip() else ""
        source_prompt_list.append(f"'{disp_name}'{hint_str}")
    source_names_str = "; ".join(source_prompt_list)
    if len(source_tuples) > 10: source_names_str += f", and {len(source_tuples) - 10} more."
    intent_prompt_messages = [("system", f"""Analyze the user query. User has access to the following information sources:
Information Sources (display name and hint): {source_names_str}
Does the query strongly suggest searching one of these sources for an answer?
If yes, identify the MOST RELEVANT source display name EXACTLY AS IT APPEARS IN THE PROVIDED HINTS.
If unsure which specific source but a search is needed, respond with SOURCE_NAME: UNKNOWN.
If no search is needed (general query), respond with SOURCE_NAME: NONE.
Answer in the format ONLY:
REQUIRES_SEARCH: YES or NO
SOURCE_NAME: [The identified source display name from the list, or UNKNOWN, or NONE]"""), ("human", "User Query: \"{user_query}\"\nYour response: ")]
    intent_prompt_template = ChatPromptTemplate.from_messages(intent_prompt_messages)
    try:
        chain = intent_prompt_template | llm_instance | StrOutputParser(); response_text = chain.invoke({"user_query": user_query})
        requires_search = False; llm_identified_display_name = None
        for line in response_text.splitlines():
            line_upper = line.upper()
            if line_upper.startswith("REQUIRES_SEARCH:"):
                if "YES" in line_upper: requires_search = True
            elif line_upper.startswith("SOURCE_NAME:"): llm_identified_display_name = line.split(":", 1)[1].strip()
        if llm_identified_display_name is None and "SOURCE_NAME:" in response_text:
             match = re.search(r"SOURCE_NAME:\s*(.*)", response_text, re.IGNORECASE)
             if match: llm_identified_display_name = match.group(1).strip()
        if requires_search:
            if llm_identified_display_name and llm_identified_display_name.upper() not in ["UNKNOWN", "NONE", ""]:
                for disp, actual, type_ind, _ in source_tuples:
                    if llm_identified_display_name == disp: return True, disp, actual, type_ind
                for disp, actual, type_ind, _ in source_tuples:
                    if llm_identified_display_name.lower() == disp.lower(): return True, disp, actual, type_ind
                for disp, actual, type_ind, _ in source_tuples:
                    if llm_identified_display_name.lower() in disp.lower() or disp.lower() in llm_identified_display_name.lower(): return True, disp, actual, type_ind
                return True, "UNKNOWN", None, None
            else: return True, "UNKNOWN", None, None
        else: return False, None, None, None
    except Exception as e:
        # This print is for server-side debugging and will not appear in the Streamlit UI.
        print(f"CONSOLE ERROR (app.py): Intent classification error: {e}\n{traceback.format_exc()}")
        return False, None, None, None

# --- Initializations & Page Config ---
if not os.path.exists(config.PERSISTENT_UPLOADS_DIR):
    try: os.makedirs(config.PERSISTENT_UPLOADS_DIR)
    except OSError as e: stlit.error(f"CRITICAL: Dir creation failed '{config.PERSISTENT_UPLOADS_DIR}': {e}"); stlit.stop()
if '_metadata_db_initialized_flag' not in stlit.session_state:
    metadata_store.initialize_metadata_db(); stlit.session_state._metadata_db_initialized_flag = True
stlit.set_page_config(page_title=config.PAGE_TITLE, layout="wide"); stlit.title(config.APP_TITLE)
if 'llms_initialized_flag' not in stlit.session_state:
    stlit.session_state.ollama_server_is_ready = llm_services.get_ollama_server_readiness(config.OLLAMA_BASE_URL)
    if stlit.session_state.ollama_server_is_ready:
        stlit.session_state.embeddings_generator, stlit.session_state.chat_llm = llm_services.initialize_llms_cached(config.EMBEDDING_MODEL_NAME, config.CHAT_MODEL_NAME, config.OLLAMA_BASE_URL)
    else: stlit.session_state.embeddings_generator, stlit.session_state.chat_llm = None, None
    stlit.session_state.llms_initialized_flag = True
default_states = {
    "active_retriever": None, "document_chat_histories": {},
    "active_doc_chat_session_id": None, "document_chat_messages": [], "document_processed_for_chat": False, "document_language": None, "active_chain_settings": {}, "active_document_filename": None, "processed_documents_metadata": {}, "youtube_summary": None, "available_qdrant_collections": [], "_initial_qdrant_direct_fetch_done": False, "persisted_doc_metadata": [], "_initial_persisted_meta_fetch_done": False, "active_hash_match_data": [], "last_uploaded_filename_for_hash_check": None, "general_chat_messages": [], "general_chat_history_store": ChatMessageHistory(), "general_conversation_runnable_with_history": None, "active_chat_mode": "Document Q&A", "rag_confirmation_pending": None, "clarifying_document_for_rag": False, "last_general_chat_query": None, "last_selectbox_choice_for_rag_clarify": None, "selectbox_choice_submitted": False, "session_doc_selector_value": "-- Select --", "persisted_doc_selector_value": "-- Select App-Stored Document --", "direct_qdrant_selector_value": "-- Select Direct Qdrant Collection --", "langchain_callbacks": [], "action_for_matched_file": None}
for key, value in default_states.items():
    if key not in stlit.session_state: stlit.session_state[key] = value
if stlit.session_state.chat_llm and not stlit.session_state.general_conversation_runnable_with_history:
    general_chat_prompt_template = ChatPromptTemplate.from_messages([MessagesPlaceholder(variable_name="history"), ("human", "{input}")])
    base_runnable = general_chat_prompt_template | stlit.session_state.chat_llm | StrOutputParser()
    stlit.session_state.general_conversation_runnable_with_history = RunnableWithMessageHistory(runnable=base_runnable, get_session_history=lambda session_id: stlit.session_state.general_chat_history_store, input_messages_key="input", history_messages_key="history")
def on_chat_mode_change_callback_v4():
    new_chat_mode = stlit.session_state.chat_mode_selector_radio_key_cb_v4; stlit.session_state.active_chat_mode = new_chat_mode
    stlit.session_state.update(rag_confirmation_pending=None, clarifying_document_for_rag=False, last_general_chat_query=None, last_selectbox_choice_for_rag_clarify=None, selectbox_choice_submitted=False)
    if new_chat_mode == "General Chat": stlit.session_state.update(active_retriever=None, document_processed_for_chat=False, active_document_filename=None, active_chain_settings={}, document_language=None, active_doc_chat_session_id=None, session_doc_selector_value="-- Select --", persisted_doc_selector_value="-- Select App-Stored Document --", direct_qdrant_selector_value="-- Select Direct Qdrant Collection --")

with stlit.sidebar:
    llm_services.display_ollama_server_status_in_sidebar(config.OLLAMA_BASE_URL)
    if stlit.session_state.get("embeddings_generator"): stlit.sidebar.info(f"Embeddings ('{config.EMBEDDING_MODEL_NAME}') loaded.")
    else: stlit.sidebar.warning(f"Embeddings ('{config.EMBEDDING_MODEL_NAME}') NOT loaded.")
    if stlit.session_state.get("chat_llm"): stlit.sidebar.info(f"Chat LLM ('{config.CHAT_MODEL_NAME}') loaded.")
    else: stlit.sidebar.warning(f"Chat LLM ('{config.CHAT_MODEL_NAME}') NOT loaded.")
    stlit.sidebar.divider(); stlit.sidebar.header("✨ Chat Mode")
    chat_mode_options = ["Document Q&A", "General Chat"]
    stlit.sidebar.radio("Select chat mode:", options=chat_mode_options, key="chat_mode_selector_radio_key_cb_v4", index=chat_mode_options.index(stlit.session_state.active_chat_mode), on_change=on_chat_mode_change_callback_v4)
    stlit.sidebar.divider(); stlit.header("1. Document Upload & Initial Processing")
    uploaded_file_ui = stlit.file_uploader("Upload PDF/TXT", type=["pdf", "txt"], key="doc_uploader_main_app_key_v4_lcel_final_complete")
    stlit.header("2. Retriever & Context Settings"); vector_store_choice_ui = stlit.selectbox("Vector Store (for new docs):", ("Qdrant", "FAISS"), index=0, key="vs_choice_sidebar_app_key_v4_lcel_final_complete")
    semantic_retriever_selection_ui = stlit.selectbox("Semantic Retriever Type:", ("mmr", "similarity"), index=0, key="sem_type_sidebar_app_key_v4_lcel_final_complete")
    num_docs_to_retrieve_ui = stlit.slider("Chunks to Retrieve (K):", 1, 10, 3, 1, key="k_slider_sidebar_app_key_v4_lcel_final_complete")
    use_hybrid_ui = stlit.checkbox("Use Hybrid Search (BM25 + Semantic)", value=True, key="hybrid_check_sidebar_app_key_v4_lcel_final_complete")
    ensemble_bm25_weight_val, ensemble_semantic_weight_val = 0.4, 0.6
    if use_hybrid_ui:
        stlit.subheader("Hybrid Search Weights"); ensemble_bm25_weight_val = stlit.slider("BM25 Weight:", 0.0, 1.0, 0.4, 0.1, key="bm25_w_sidebar_app_key_v4_lcel_final_complete")
        semantic_weight_source_label = "Qdrant" if vector_store_choice_ui == "Qdrant" else ("FAISS" if vector_store_choice_ui == "FAISS" else "Semantic")
        ensemble_semantic_weight_val = stlit.slider(f"{semantic_weight_source_label} Weight:", 0.0, 1.0, 0.6, 0.1, key="sem_w_sidebar_app_key_v4_lcel_final_complete")
    mmr_lambda_val, mmr_fetch_k_factor_val = 0.5, 5.0
    if semantic_retriever_selection_ui == "mmr":
        stlit.subheader("MMR Specific Settings"); mmr_lambda_val = stlit.slider("MMR Lambda (Diversity):", 0.0, 1.0, 0.5, 0.1, key="mmr_lambda_sidebar_app_key_v4_lcel_final_complete")
        mmr_fetch_k_factor_val = stlit.slider("MMR Fetch K Multiplier:", 2.0, 10.0, 5.0, 0.5, key="mmr_fetch_k_sidebar_app_key_v4_lcel_final_complete")
    use_map_reduce_ui = stlit.checkbox("Summarize Chunks (Map-Reduce for Doc Q&A)", value=False, key="map_reduce_sidebar_app_key_v4_lcel_final_complete")

    def build_retriever(doc_chunks_for_retriever, vs_choice, sem_type, k_val, hybrid_flag, bm25_w_val, sem_w_val, mmr_l_val, mmr_fkf_val, q_coll_name=None, embed_gen_instance=None, q_url_val=None, q_api_key_val=None, qdrant_client_for_existing=None):
        vectorstore_for_retriever = None
        final_retriever_obj = None
        
        if hybrid_flag and not doc_chunks_for_retriever:
            stlit.sidebar.warning("Hybrid search disabled - no text chunks for BM25.")
            hybrid_flag = False

        if vs_choice == "FAISS":
            if doc_chunks_for_retriever and embed_gen_instance:
                try:
                    vectorstore_for_retriever = FAISS.from_documents(documents=doc_chunks_for_retriever, embedding=embed_gen_instance)
                except Exception as e:
                    stlit.sidebar.error(f"FAISS retriever setup error: {e}")
                    return None
            else:
                stlit.sidebar.error("FAISS: Chunks or embeddings missing.")
                return None
        elif vs_choice == "Qdrant":
            if not q_coll_name or not embed_gen_instance:
                stlit.sidebar.error("Qdrant: Collection name or embeddings missing.")
                return None
            
            q_client_to_use = qdrant_client_for_existing if qdrant_client_for_existing else QdrantSDKClient(url=q_url_val, api_key=q_api_key_val, timeout=20)
            
            try:
                vectorstore_for_retriever = QdrantVectorStore(client=q_client_to_use, collection_name=q_coll_name, embedding=embed_gen_instance)
                try:
                    q_client_to_use.get_collection(collection_name=q_coll_name)
                    stlit.sidebar.success(f"Initialized QdrantVS for existing '{q_coll_name}'.")
                except Exception as e_verify:
                    stlit.sidebar.warning(f"QdrantVS for '{q_coll_name}' init but SDK reports: {type(e_verify).__name__} - {str(e_verify)}. May be empty.")
            except Exception as e_init_qdrant:
                stlit.sidebar.warning(f"Access Qdrant coll '{q_coll_name}' failed: {type(e_init_qdrant).__name__} - {str(e_init_qdrant)}")
                vectorstore_for_retriever = None

            if not vectorstore_for_retriever and doc_chunks_for_retriever:
                try:
                    stlit.sidebar.info(f"Attempting to create/recreate Qdrant coll '{q_coll_name}' from documents...")
                    for doc in doc_chunks_for_retriever:
                        if 'source' in doc.metadata and os.path.isfile(doc.metadata['source']):
                            doc.metadata['source'] = os.path.basename(doc.metadata['source'])
                    vectorstore_for_retriever = QdrantVectorStore.from_documents(documents=doc_chunks_for_retriever, embedding=embed_gen_instance, url=q_url_val, api_key=q_api_key_val, collection_name=q_coll_name, force_recreate=True)
                    stlit.sidebar.success(f"CREATED/RECREATED QdrantVS for '{q_coll_name}'.")
                except Exception as e_create:
                    stlit.sidebar.error(f"Qdrant from_documents FAILED for '{q_coll_name}': {type(e_create).__name__} - {str(e_create)}")
                    return None
            elif not vectorstore_for_retriever:
                stlit.sidebar.error(f"Qdrant coll '{q_coll_name}' not init, no docs to create it.")
                return None
        else:
            stlit.sidebar.error(f"Invalid vector store choice: {vs_choice}")
            return None

        if not vectorstore_for_retriever:
            stlit.sidebar.error("Vector store failed to initialize.")
            return None

        retriever_search_kwargs = {'k': k_val}
        if sem_type == "mmr" and str(mmr_l_val).lower() != "n/a":
            try:
                mmr_l_float = float(mmr_l_val)
                mmr_fkf_float = float(mmr_fkf_val)
                fetch_k = int(k_val * mmr_fkf_float)
                retriever_search_kwargs['fetch_k'] = max(k_val + 5, fetch_k)
                retriever_search_kwargs['lambda_mult'] = mmr_l_float
            except ValueError:
                stlit.warning("MMR settings invalid, using defaults.")
                retriever_search_kwargs['fetch_k'] = k_val + 5
                retriever_search_kwargs['lambda_mult'] = 0.5
        
        semantic_retriever_obj = vectorstore_for_retriever.as_retriever(search_type=sem_type, search_kwargs=retriever_search_kwargs)
        final_retriever_obj = semantic_retriever_obj

        if hybrid_flag:
            try:
                bm25_ret_obj = BM25Retriever.from_documents(documents=doc_chunks_for_retriever)
                bm25_ret_obj.k = k_val
                final_retriever_obj = EnsembleRetriever(retrievers=[bm25_ret_obj, semantic_retriever_obj], weights=[float(bm25_w_val), float(sem_w_val)])
            except Exception as e:
                stlit.sidebar.error(f"BM25/Ensemble error: {e}. Using semantic only.")
                final_retriever_obj = semantic_retriever_obj
        
        return final_retriever_obj

    def setup_and_activate_lcel_chain(retriever_instance, doc_filename, doc_lang, settings_dict, session_id_prefix="doc_chat_"):
        if not retriever_instance: stlit.sidebar.error("Retriever is None in setup."); return False
        if not stlit.session_state.chat_llm: stlit.sidebar.error("Chat LLM is None in setup."); return False
        safe_doc_filename = "".join(c if c.isalnum() else '_' for c in doc_filename); session_id = f"{session_id_prefix}{safe_doc_filename}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        stlit.session_state.document_chat_histories[session_id] = ChatMessageHistory()
        stlit.session_state.active_retriever = retriever_instance
        stlit.session_state.update(document_language=doc_lang, document_processed_for_chat=True, active_document_filename=doc_filename, active_chain_settings=settings_dict.copy(), active_chat_mode="Document Q&A", document_chat_messages=[], active_doc_chat_session_id=session_id)
        return True

    if uploaded_file_ui:
        if stlit.session_state.last_uploaded_filename_for_hash_check != uploaded_file_ui.name:
            stlit.session_state.active_hash_match_data = []
            stlit.session_state.last_uploaded_filename_for_hash_check = uploaded_file_ui.name
            stlit.session_state.action_for_matched_file = None

        if stlit.button(f"Process NEW '{uploaded_file_ui.name}'", key="process_new_doc_lcel_btn_v3_final"):
            stlit.session_state.action_for_matched_file = "evaluate"

        if stlit.session_state.action_for_matched_file:
            if not (stlit.session_state.embeddings_generator and stlit.session_state.chat_llm):
                stlit.sidebar.error("LLMs not loaded.")
                stlit.session_state.action_for_matched_file = None
            else:
                with stlit.spinner(f"Processing '{uploaded_file_ui.name}': Checking for matches, processing text, and generating embeddings... This may take a moment."):
                    uploaded_file_hash = metadata_store.calculate_file_hash(uploaded_file_ui)
                    existing_docs_with_hash = metadata_store.get_metadata_by_file_hash(uploaded_file_hash) if uploaded_file_hash else []
                    perform_new_processing = False

                    if stlit.session_state.action_for_matched_file == "process_as_new_anyway":
                        perform_new_processing = True
                        stlit.toast("Starting to process as new...", icon="⏳")
                    elif existing_docs_with_hash and vector_store_choice_ui == "Qdrant" and stlit.session_state.action_for_matched_file == "evaluate":
                        first_match = existing_docs_with_hash[0]
                        stlit.sidebar.info(f"Match: '{first_match['original_filename']}' (Qdrant: {first_match.get('qdrant_collection_name', 'N/A')})")
                        stlit.sidebar.subheader("Choose action for matched file:")
                        col_act1, col_act2 = stlit.sidebar.columns(2)
                        if col_act1.button(f"Activate Matched '{first_match['original_filename']}'", key=f"activate_hash_lcel_btn_final_{first_match.get('qdrant_collection_name', 'no_q_coll')[-6:]}"):
                            stlit.session_state.action_for_matched_file = "activate_matched"
                            stlit.rerun()
                        if col_act2.button("Process As New Anyway", key=f"process_new_anyway_lcel_btn_final_{first_match.get('qdrant_collection_name', 'no_q_coll_new')[-6:]}"):
                            stlit.session_state.action_for_matched_file = "process_as_new_anyway"
                            stlit.rerun()
                    elif stlit.session_state.action_for_matched_file == "activate_matched":
                        if not existing_docs_with_hash:
                            stlit.sidebar.error("Error: No matched document found for activation.")
                            stlit.session_state.action_for_matched_file = None
                        else:
                            first_match = existing_docs_with_hash[0]
                            db_meta = first_match
                            q_coll = db_meta['qdrant_collection_name']
                            orig_fn = db_meta['original_filename']
                            p_path = db_meta['persistent_file_path']
                            if not os.path.exists(p_path):
                                stlit.sidebar.error(f"File for match not found: {p_path}.")
                                stlit.session_state.action_for_matched_file = "evaluate"
                            else:
                                act_set_current_ui = {"filename": orig_fn, "vector_store": "Qdrant", "qdrant_collection_name": q_coll, "semantic_type": semantic_retriever_selection_ui, "k": num_docs_to_retrieve_ui, "hybrid": use_hybrid_ui, "bm25_w": ensemble_bm25_weight_val if use_hybrid_ui else "N/A", "semantic_w": ensemble_semantic_weight_val if use_hybrid_ui else "N/A", "mmr_lambda": mmr_lambda_val if semantic_retriever_selection_ui == "mmr" else "N/A", "mmr_fetch_k_f": mmr_fetch_k_factor_val if semantic_retriever_selection_ui == "mmr" else "N/A", "map_reduce": use_map_reduce_ui}
                                chunks_act = None
                                if act_set_current_ui['hybrid']:
                                    chunks_act, _, _ = document_processor.extract_text_and_chunks(p_path, orig_fn)
                                if act_set_current_ui['hybrid'] and not chunks_act:
                                    stlit.sidebar.warning("Hybrid Search: BM25 chunks could not be loaded. Using semantic only.")
                                    act_set_current_ui['hybrid'] = False
                                ret_act = build_retriever(chunks_act, "Qdrant", act_set_current_ui["semantic_type"], act_set_current_ui["k"], act_set_current_ui["hybrid"], act_set_current_ui["bm25_w"], act_set_current_ui["semantic_w"], act_set_current_ui["mmr_lambda"], act_set_current_ui["mmr_fetch_k_f"], q_coll, stlit.session_state.embeddings_generator, config.QDRANT_URL, config.QDRANT_API_KEY)
                                if ret_act and setup_and_activate_lcel_chain(ret_act, orig_fn, db_meta.get('detected_language'), act_set_current_ui):
                                    metadata_store.update_last_accessed_timestamp(q_coll)
                                    stlit.session_state.persisted_doc_metadata = metadata_store.get_all_qdrant_document_metadata_cached()
                                    stlit.session_state.processed_documents_metadata[orig_fn] = {"qdrant_collection_name": q_coll, "all_chunks_as_docs": chunks_act, "language": db_meta.get('detected_language'), "settings": act_set_current_ui.copy(), "persistent_file_path_if_any": p_path, "content_hint": db_meta.get("content_hint")}
                                    stlit.toast(f"Activated matched: '{orig_fn}'.", icon="✅")
                                    stlit.session_state.action_for_matched_file = None
                                    stlit.rerun()
                                elif ret_act:
                                    stlit.sidebar.error(f"Failed to setup LCEL chain for matched doc '{orig_fn}'.")
                                    stlit.session_state.action_for_matched_file = "evaluate"
                                else:
                                    stlit.sidebar.error(f"Failed to build retriever for matched doc '{orig_fn}'.")
                                    stlit.session_state.action_for_matched_file = "evaluate"
                    elif (not existing_docs_with_hash and stlit.session_state.action_for_matched_file == "evaluate") or (stlit.session_state.action_for_matched_file == "evaluate" and not existing_docs_with_hash):
                        perform_new_processing = True
                        if not existing_docs_with_hash:
                            stlit.toast("No match found or processing as new. Starting...", icon="✨")

                    if perform_new_processing:
                        persistent_file_actual_path = None
                        if vector_store_choice_ui == "Qdrant":
                            target_persistent_path = metadata_store.get_unique_persistent_filepath(uploaded_file_ui.name)
                            try:
                                with open(target_persistent_path, "wb") as f:
                                    f.write(uploaded_file_ui.getvalue())
                                persistent_file_actual_path = target_persistent_path
                                stlit.sidebar.caption(f"Saved to: {os.path.basename(persistent_file_actual_path)}")
                            except Exception as e_save:
                                stlit.sidebar.error(f"Save failed: {e_save}")
                                stlit.stop()
                        
                        source_for_chunking = persistent_file_actual_path if persistent_file_actual_path else uploaded_file_ui
                        all_chunks, detected_lang, num_chunks = document_processor.extract_text_and_chunks(source_for_chunking, original_filename_for_display=uploaded_file_ui.name)
                        
                        content_hint_text = None
                        if all_chunks and stlit.session_state.chat_llm:
                            try:
                                hint_text_source = " ".join(c.page_content for c in all_chunks[:2])[:1000]
                                hint_prompt_template = ChatPromptTemplate.from_template("Create a very concise summary (1 sentence, max 20 words) or list up to 5 main keywords for the following text. Text: {text_input}")
                                hint_chain = hint_prompt_template | stlit.session_state.chat_llm | StrOutputParser()
                                content_hint_text = hint_chain.invoke({"text_input": hint_text_source})
                            except Exception as e_hint:
                                stlit.sidebar.warning(f"Content hint generation failed: {e_hint}")

                        if all_chunks:
                            current_proc_settings = {"filename": uploaded_file_ui.name, "vector_store": vector_store_choice_ui, "semantic_type": semantic_retriever_selection_ui, "k": num_docs_to_retrieve_ui, "hybrid": use_hybrid_ui, "bm25_w": ensemble_bm25_weight_val if use_hybrid_ui else "N/A", "semantic_w": ensemble_semantic_weight_val if use_hybrid_ui else "N/A", "mmr_lambda": mmr_lambda_val if semantic_retriever_selection_ui == "mmr" else "N/A", "mmr_fetch_k_f": mmr_fetch_k_factor_val if semantic_retriever_selection_ui == "mmr" else "N/A", "map_reduce": use_map_reduce_ui}
                            q_coll_name_new = None
                            if vector_store_choice_ui == "Qdrant":
                                q_coll_name_new = utils.normalize_filename_for_collection(os.path.basename(persistent_file_actual_path) if persistent_file_actual_path else uploaded_file_ui.name)
                            
                            retriever_new = build_retriever(all_chunks, vector_store_choice_ui, semantic_retriever_selection_ui, num_docs_to_retrieve_ui, use_hybrid_ui, ensemble_bm25_weight_val, ensemble_semantic_weight_val, mmr_lambda_val, mmr_fetch_k_factor_val, q_coll_name_new, stlit.session_state.embeddings_generator, config.QDRANT_URL, config.QDRANT_API_KEY)
                            
                            if retriever_new and setup_and_activate_lcel_chain(retriever_new, uploaded_file_ui.name, detected_lang, current_proc_settings):
                                stlit.session_state.processed_documents_metadata[uploaded_file_ui.name] = {"qdrant_collection_name": q_coll_name_new, "all_chunks_as_docs": all_chunks, "language": detected_lang, "settings": current_proc_settings.copy(), "persistent_file_path_if_any": persistent_file_actual_path, "content_hint": content_hint_text}
                                if vector_store_choice_ui == "Qdrant" and q_coll_name_new and persistent_file_actual_path:
                                    final_hash_for_db = uploaded_file_hash or metadata_store.calculate_file_hash(uploaded_file_ui)
                                    if metadata_store.add_or_update_document_metadata(uploaded_file_ui.name, persistent_file_actual_path, q_coll_name_new, final_hash_for_db, detected_lang, current_proc_settings.copy(), num_chunks, content_hint=content_hint_text):
                                        stlit.session_state.persisted_doc_metadata = metadata_store.get_all_qdrant_document_metadata_cached()
                                    else:
                                        stlit.sidebar.warning("Failed to save metadata for new Qdrant document.")
                                stlit.sidebar.success(f"'{uploaded_file_ui.name}' processed and activated.")
                                stlit.session_state.action_for_matched_file = None
                                stlit.rerun()
                            elif retriever_new:
                                stlit.sidebar.error(f"Failed to setup chain for NEW '{uploaded_file_ui.name}'.")
                                stlit.session_state.action_for_matched_file = "evaluate"
                            else:
                                stlit.sidebar.error(f"Failed to build retriever for NEW '{uploaded_file_ui.name}'.")
                                stlit.session_state.action_for_matched_file = "evaluate"
                        else:
                            stlit.sidebar.error(f"Could not extract chunks from '{uploaded_file_ui.name}'.")
                            stlit.session_state.action_for_matched_file = None

                    if stlit.session_state.action_for_matched_file == "evaluate" and not (existing_docs_with_hash and vector_store_choice_ui == "Qdrant"):
                        stlit.session_state.action_for_matched_file = None

    stlit.sidebar.divider(); stlit.sidebar.header("3. Activate Document for Q&A (LCEL)")
    stlit.sidebar.subheader("A. From this session")
    processed_doc_names_session = list(stlit.session_state.processed_documents_metadata.keys())
    def on_session_doc_select_change_lcel_final():
        stlit.session_state.session_doc_selector_value = stlit.session_state.doc_selector_session_lcel_key_final
        for key in list(stlit.session_state.keys()):
            if key.startswith("confirm_delete_pending_db_") or key.startswith("confirm_delete_direct_q_pending_"): stlit.session_state[key] = False
    default_idx_session = 0; options_list_session = ["-- Select --"] + processed_doc_names_session
    current_session_doc_selection = stlit.session_state.get("session_doc_selector_value", "-- Select --")
    if current_session_doc_selection in options_list_session:
        try: default_idx_session = options_list_session.index(current_session_doc_selection)
        except ValueError: default_idx_session = 0
    elif stlit.session_state.active_document_filename and stlit.session_state.active_document_filename in processed_doc_names_session and stlit.session_state.active_chat_mode == "Document Q&A":
        try: default_idx_session = options_list_session.index(stlit.session_state.active_document_filename)
        except ValueError: default_idx_session = 0
    stlit.sidebar.selectbox("Select in-session document:", options=options_list_session, index=default_idx_session, key="doc_selector_session_lcel_key_final", on_change=on_session_doc_select_change_lcel_final)
    selected_doc_key_session = stlit.session_state.session_doc_selector_value
    if selected_doc_key_session and selected_doc_key_session != "-- Select --":
        session_meta = stlit.session_state.processed_documents_metadata.get(selected_doc_key_session)
        sanitized_sess_key_for_buttons = "".join(c if c.isalnum() else '_' for c in selected_doc_key_session)
        if stlit.sidebar.button(f"Activate '{selected_doc_key_session}' (Session)", key=f"activate_session_doc_lcel_btn_final_{sanitized_sess_key_for_buttons}"):
            if session_meta:
                act_set_sess = {"filename": session_meta["settings"]["filename"], "vector_store": session_meta["settings"]["vector_store"], "qdrant_collection_name": session_meta.get("qdrant_collection_name"), "semantic_type": semantic_retriever_selection_ui, "k": num_docs_to_retrieve_ui, "hybrid": use_hybrid_ui, "bm25_w": ensemble_bm25_weight_val if use_hybrid_ui else "N/A", "semantic_w": ensemble_semantic_weight_val if use_hybrid_ui else "N/A", "mmr_lambda": mmr_lambda_val if semantic_retriever_selection_ui == "mmr" else "N/A", "mmr_fetch_k_f": mmr_fetch_k_factor_val if semantic_retriever_selection_ui == "mmr" else "N/A", "map_reduce": use_map_reduce_ui}
                chunks_act_sess = session_meta.get("all_chunks_as_docs", [])
                if act_set_sess["hybrid"]:
                    if not chunks_act_sess:
                        path_if_pers_sess = session_meta.get("persistent_file_path_if_any")
                        if path_if_pers_sess and os.path.exists(path_if_pers_sess):
                            chunks_act_sess,_,_ = document_processor.extract_text_and_chunks(path_if_pers_sess, selected_doc_key_session)
                            if not chunks_act_sess: act_set_sess['hybrid'] = False; stlit.warning("Session activate: Could not load chunks for BM25, disabling hybrid.")
                        else: act_set_sess['hybrid'] = False; stlit.warning("Session activate: No chunks available for BM25, disabling hybrid.")
                if not act_set_sess["hybrid"]: act_set_sess["bm25_w"] = "N/A"; act_set_sess["semantic_w"] = "N/A"
                ret_act_sess = build_retriever(chunks_act_sess, act_set_sess["vector_store"], act_set_sess["semantic_type"], act_set_sess["k"], act_set_sess["hybrid"], act_set_sess["bm25_w"], act_set_sess["semantic_w"], act_set_sess["mmr_lambda"], act_set_sess["mmr_fetch_k_f"], act_set_sess.get("qdrant_collection_name"), stlit.session_state.embeddings_generator, config.QDRANT_URL, config.QDRANT_API_KEY)
                if ret_act_sess and setup_and_activate_lcel_chain(ret_act_sess, selected_doc_key_session, session_meta.get("language"), act_set_sess):
                    stlit.session_state.processed_documents_metadata[selected_doc_key_session]["settings"] = act_set_sess.copy(); stlit.sidebar.success(f"LCEL RAG for '{selected_doc_key_session}' (session) activated!"); stlit.rerun()
                else: stlit.sidebar.error(f"Failed to activate session doc '{selected_doc_key_session}'.")
        delete_btn_key_sess = f"delete_session_doc_btn_final_{sanitized_sess_key_for_buttons}"; confirm_key_sess = f"confirm_delete_pending_sess_final_{sanitized_sess_key_for_buttons}"
        if stlit.sidebar.button(f"❌ Delete '{selected_doc_key_session}' from Session", key=delete_btn_key_sess):
            stlit.session_state[confirm_key_sess] = True; stlit.rerun()
        if stlit.session_state.get(confirm_key_sess):
            stlit.sidebar.warning(f"Delete '{selected_doc_key_session}' from this session list?")
            c1s, c2s = stlit.sidebar.columns(2)
            if c1s.button("YES, Remove from Session List", key=f"confirm_del_sess_final_{sanitized_sess_key_for_buttons}"):
                if selected_doc_key_session in stlit.session_state.processed_documents_metadata: del stlit.session_state.processed_documents_metadata[selected_doc_key_session]
                if stlit.session_state.active_document_filename == selected_doc_key_session: stlit.session_state.active_retriever = None; stlit.session_state.document_processed_for_chat = False; stlit.session_state.active_document_filename = None; stlit.session_state.active_doc_chat_session_id = None
                stlit.session_state[confirm_key_sess] = False; stlit.session_state.session_doc_selector_value = "-- Select --"; stlit.sidebar.success(f"'{selected_doc_key_session}' removed from session."); stlit.rerun()
            if c2s.button("NO, Cancel", key=f"cancel_del_sess_final_{sanitized_sess_key_for_buttons}"): stlit.session_state[confirm_key_sess] = False; stlit.rerun()
    elif not processed_doc_names_session : stlit.sidebar.caption("No documents processed this session.")
    stlit.sidebar.subheader("B. From App Storage (DB)")
    def refresh_persisted_metadata_list_fn_final():
        stlit.session_state.persisted_doc_metadata = metadata_store.get_all_qdrant_document_metadata_cached()
    if stlit.sidebar.button("Refresh App-Stored Documents List", key="refresh_persisted_docs_list_lcel_btn_final"):
        refresh_persisted_metadata_list_fn_final(); stlit.rerun()
    if not stlit.session_state.get("persisted_doc_metadata") and not stlit.session_state.get("_initial_persisted_meta_fetch_done"):
        refresh_persisted_metadata_list_fn_final(); stlit.session_state._initial_persisted_meta_fetch_done = True
    persisted_docs_options_map = {}
    if stlit.session_state.persisted_doc_metadata: persisted_docs_options_map = {(f"{item['original_filename']} (Coll: {item.get('qdrant_collection_name', 'N/A')[-10:] if item.get('qdrant_collection_name') else 'N/A'})"): item for item in stlit.session_state.persisted_doc_metadata}
    persisted_doc_display_options = ["-- Select App-Stored Document --"] + list(persisted_docs_options_map.keys())
    default_idx_persisted = 0
    current_persisted_selection = stlit.session_state.get("persisted_doc_selector_value", "-- Select App-Stored Document --")
    if current_persisted_selection in persisted_doc_display_options:
        try: default_idx_persisted = persisted_doc_display_options.index(current_persisted_selection)
        except ValueError: default_idx_persisted = 0
    elif stlit.session_state.active_document_filename and stlit.session_state.active_chat_mode == "Document Q&A":
        for i, key_opt in enumerate(persisted_docs_options_map.keys()):
            if persisted_docs_options_map[key_opt]['original_filename'] == stlit.session_state.active_document_filename: default_idx_persisted = i + 1; break
    def on_persisted_doc_select_change_lcel_final():
        stlit.session_state.persisted_doc_selector_value = stlit.session_state.persisted_doc_selector_lcel_key_final
        for key in list(stlit.session_state.keys()):
            if key.startswith("confirm_delete_pending_sess_") or key.startswith("confirm_delete_direct_q_pending_"): stlit.session_state[key] = False
    stlit.sidebar.selectbox("Select document from app storage:", options=persisted_doc_display_options, index=default_idx_persisted, key="persisted_doc_selector_lcel_key_final", on_change=on_persisted_doc_select_change_lcel_final)
    selected_db_metadata_key = stlit.session_state.persisted_doc_selector_value
    selected_db_metadata = persisted_docs_options_map.get(selected_db_metadata_key)
    if selected_db_metadata:
        orig_fn_db = selected_db_metadata['original_filename']; q_coll_db = selected_db_metadata['qdrant_collection_name']; p_path_db = selected_db_metadata['persistent_file_path']
        sanitized_db_key_final = f"{orig_fn_db.replace(' ','_').replace('.','_').replace(':','_')}_{q_coll_db[-6:] if q_coll_db else 'noqdrant'}"
        with stlit.sidebar.expander("Stored Settings & Info", expanded=False): stlit.json(selected_db_metadata)
        if stlit.sidebar.button(f"Activate '{orig_fn_db}' (App-Stored)", key=f"activate_db_doc_lcel_btn_final_{sanitized_db_key_final}"):
            if not os.path.exists(p_path_db): stlit.sidebar.error(f"File not found: {p_path_db}.");
            else:
                act_set_db = { "filename": orig_fn_db, "vector_store": "Qdrant", "semantic_type": semantic_retriever_selection_ui, "k": num_docs_to_retrieve_ui, "hybrid": use_hybrid_ui, "bm25_w": ensemble_bm25_weight_val if use_hybrid_ui else "N/A", "semantic_w": ensemble_semantic_weight_val if use_hybrid_ui else "N/A", "mmr_lambda": mmr_lambda_val if semantic_retriever_selection_ui == "mmr" else "N/A", "mmr_fetch_k_f": mmr_fetch_k_factor_val if semantic_retriever_selection_ui == "mmr" else "N/A", "map_reduce": use_map_reduce_ui }
                chunks_db = None
                if act_set_db['hybrid']:
                    chunks_db, _, _ = document_processor.extract_text_and_chunks(p_path_db, orig_fn_db)
                    if not chunks_db: act_set_db['hybrid'] = False; stlit.warning("App-Stored Hybrid: BM25 chunks not loaded.")
                if not act_set_db['hybrid']: act_set_db["bm25_w"] = "N/A"; act_set_db["semantic_w"] = "N/A"
                ret_db = build_retriever(chunks_db, "Qdrant", act_set_db["semantic_type"], act_set_db["k"], act_set_db["hybrid"], act_set_db["bm25_w"], act_set_db["semantic_w"], act_set_db["mmr_lambda"], act_set_db["mmr_fetch_k_f"], q_coll_db, stlit.session_state.embeddings_generator, config.QDRANT_URL, config.QDRANT_API_KEY)
                if ret_db and setup_and_activate_lcel_chain(ret_db, orig_fn_db, selected_db_metadata.get('detected_language'), act_set_db):
                    stlit.session_state.processed_documents_metadata[orig_fn_db] = {"qdrant_collection_name": q_coll_db, "all_chunks_as_docs": chunks_db, "language": selected_db_metadata.get('detected_language'), "settings": act_set_db.copy(), "persistent_file_path_if_any": p_path_db, "content_hint": selected_db_metadata.get("content_hint")}
                    metadata_store.update_last_accessed_timestamp(q_coll_db); refresh_persisted_metadata_list_fn_final(); stlit.sidebar.success(f"LCEL RAG for '{orig_fn_db}' (DB) activated!"); stlit.rerun()
                else: stlit.sidebar.error(f"Failed to activate DB doc '{orig_fn_db}'.")
        confirm_key_db = f"confirm_delete_pending_db_final_{sanitized_db_key_final}"
        if stlit.sidebar.button(f"❌ Delete '{orig_fn_db}' from Storage", key=f"delete_db_doc_btn_final_{sanitized_db_key_final}"):
            stlit.session_state[confirm_key_db] = True; stlit.rerun()
        if stlit.session_state.get(confirm_key_db):
            stlit.sidebar.warning(f"Sure to delete ALL data for '{orig_fn_db}' (Qdrant: {q_coll_db}, File: {os.path.basename(p_path_db)})?")
            cd1, cd2 = stlit.sidebar.columns(2)
            if cd1.button("YES, DELETE ALL", key=f"confirm_del_db_final_{sanitized_db_key_final}"):
                stlit.sidebar.info(f"Starting deletion for '{orig_fn_db}'...")
                if q_coll_db:
                    qdrant_delete_successful_db = False
                    try:
                        client_db_del = QdrantSDKClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY, timeout=20)
                        result_db_del = client_db_del.delete_collection(collection_name=q_coll_db)
                        if result_db_del:
                            qdrant_delete_successful_db = True
                            stlit.sidebar.success(f"Qdrant collection '{q_coll_db}' reported deleted by client.")
                        else:
                            stlit.sidebar.warning(f"Qdrant client reported NO SUCCESS deleting '{q_coll_db}', but no exception.")
                    except Exception as e_qdel_db:
                        stlit.sidebar.error(f"Qdrant delete EXCEPTION for '{q_coll_db}': {type(e_qdel_db).__name__} - {e_qdel_db}")
                    if qdrant_delete_successful_db:
                        try:
                            client_db_del.get_collection(collection_name=q_coll_db)
                            stlit.sidebar.error(f"VERIFICATION FAILED: Qdrant collection '{q_coll_db}' still exists!")
                        except Exception as e_verify_db:
                            stlit.sidebar.success(f"VERIFICATION SUCCEEDED: Qdrant collection '{q_coll_db}' no longer found (Error: {e_verify_db}).")
                else:
                    stlit.sidebar.caption(f"No Qdrant collection name associated with '{orig_fn_db}' in DB, skipping Qdrant deletion.")
                meta_del_db, file_del_db = metadata_store.delete_metadata_and_file(p_path_db)
                if meta_del_db: stlit.sidebar.caption("Local metadata deleted.")
                else: stlit.sidebar.warning("Local metadata NOT deleted or not found.")
                if file_del_db: stlit.sidebar.caption("Local file deleted.")
                else: stlit.sidebar.warning("Local file NOT deleted or not found.")
                if stlit.session_state.active_document_filename == orig_fn_db: stlit.session_state.active_retriever = None; stlit.session_state.document_processed_for_chat = False; stlit.session_state.active_document_filename = None; stlit.session_state.active_doc_chat_session_id = None
                if orig_fn_db in stlit.session_state.processed_documents_metadata: del stlit.session_state.processed_documents_metadata[orig_fn_db]
                stlit.session_state[confirm_key_db] = False; refresh_persisted_metadata_list_fn_final()
                stlit.session_state.available_qdrant_collections = qdrant_services.get_existing_qdrant_collections_cached(config.QDRANT_URL, config.QDRANT_API_KEY, _cache_buster=time.time())
                stlit.sidebar.info(f"Deletion process for '{orig_fn_db}' complete."); stlit.rerun()
            if cd2.button("NO, CANCEL", key=f"cancel_del_db_final_{sanitized_db_key_final}"): stlit.session_state[confirm_key_db] = False; stlit.rerun()
    elif not stlit.session_state.persisted_doc_metadata: stlit.sidebar.caption("No documents in app storage.")
    stlit.sidebar.subheader("C. Advanced: Direct Qdrant Server")
    if stlit.sidebar.button("Refresh Direct Qdrant List", key="refresh_direct_qdrant_lcel_btn_final"):
        stlit.session_state.available_qdrant_collections = qdrant_services.get_existing_qdrant_collections_cached(config.QDRANT_URL, config.QDRANT_API_KEY, _cache_buster=time.time())
        stlit.rerun()
    if not stlit.session_state.get("available_qdrant_collections") and not stlit.session_state.get("_initial_qdrant_direct_fetch_done"):
        stlit.session_state.available_qdrant_collections = qdrant_services.get_existing_qdrant_collections_cached(config.QDRANT_URL, config.QDRANT_API_KEY, _cache_buster=time.time())
        stlit.session_state._initial_qdrant_direct_fetch_done = True
    qdrant_direct_options_map = {}
    if stlit.session_state.available_qdrant_collections: qdrant_direct_options_map = {f"{item['display_filename']} (Coll: {item['qdrant_collection_name']})": item for item in stlit.session_state.available_qdrant_collections}
    qdrant_direct_display_options = ["-- Select Direct Qdrant Collection --"] + list(qdrant_direct_options_map.keys())
    default_idx_direct_q = 0
    current_direct_q_selection = stlit.session_state.get("direct_qdrant_selector_value", "-- Select Direct Qdrant Collection --")
    if current_direct_q_selection in qdrant_direct_display_options:
        try: default_idx_direct_q = qdrant_direct_display_options.index(current_direct_q_selection)
        except ValueError: default_idx_direct_q = 0
    def on_direct_qdrant_select_change_lcel_final():
        stlit.session_state.direct_qdrant_selector_value = stlit.session_state.qdrant_direct_selector_lcel_key_final
        for key in list(stlit.session_state.keys()):
            if key.startswith("confirm_delete_pending_sess_") or key.startswith("confirm_delete_pending_db_"): stlit.session_state[key] = False
    stlit.sidebar.selectbox("Select raw Qdrant collection:", options=qdrant_direct_display_options, index=default_idx_direct_q, key="qdrant_direct_selector_lcel_key_final", on_change=on_direct_qdrant_select_change_lcel_final)
    selected_direct_q_item_key = stlit.session_state.direct_qdrant_selector_value
    selected_direct_q_item_details = qdrant_direct_options_map.get(selected_direct_q_item_key)
    if selected_direct_q_item_details:
        q_coll_direct = selected_direct_q_item_details['qdrant_collection_name']; disp_fn_direct = selected_direct_q_item_details['display_filename']
        sanitized_direct_q_key_final = f"{disp_fn_direct.replace(' ','_').replace('.','_').replace(':','_')}_{q_coll_direct.replace(':','_').replace('-','_')[-10:]}"
        with stlit.sidebar.expander(f"ℹ️ Collection Info: {q_coll_direct}", expanded=False):
            try:
                qdrant_client = QdrantSDKClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY)
                coll_info = qdrant_client.get_collection(q_coll_direct)
                stlit.json({"collection_name": q_coll_direct, "status": str(coll_info.status), "vectors_count": coll_info.vectors_count if coll_info.vectors_count is not None else coll_info.points_count, "points_count": coll_info.points_count})
            except Exception as e: stlit.warning(f"Could not fetch collection info: {str(e)}")
        
        if stlit.sidebar.button(f"Activate '{disp_fn_direct}' (Direct Qdrant)", key=f"activate_direct_q_lcel_btn_final_{sanitized_direct_q_key_final}"):
            use_hybrid_for_this_activation = use_hybrid_ui
            if use_hybrid_for_this_activation:
                stlit.sidebar.warning("Hybrid search (BM25) is not supported for Direct Qdrant collections as the original file is not available. Falling back to semantic search only.")
                use_hybrid_for_this_activation = False
            
            act_set_direct_q = {"filename": disp_fn_direct, "vector_store": "Qdrant", "qdrant_collection_name_if_direct": q_coll_direct, "semantic_type": semantic_retriever_selection_ui, "k": num_docs_to_retrieve_ui, "hybrid": use_hybrid_for_this_activation, "bm25_w": "N/A", "semantic_w": "N/A", "mmr_lambda": mmr_lambda_val if semantic_retriever_selection_ui == "mmr" else "N/A", "mmr_fetch_k_f": mmr_fetch_k_factor_val if semantic_retriever_selection_ui == "mmr" else "N/A", "map_reduce": use_map_reduce_ui}
            
            chunks_direct = None

            ret_direct = build_retriever(chunks_direct, "Qdrant", act_set_direct_q["semantic_type"], act_set_direct_q["k"], act_set_direct_q["hybrid"], act_set_direct_q["bm25_w"], act_set_direct_q["semantic_w"], act_set_direct_q["mmr_lambda"], act_set_direct_q["mmr_fetch_k_f"], q_coll_direct, stlit.session_state.embeddings_generator, config.QDRANT_URL, config.QDRANT_API_KEY)

            if ret_direct:
                if setup_and_activate_lcel_chain(ret_direct, disp_fn_direct, None, act_set_direct_q): 
                    stlit.sidebar.success(f"LCEL RAG for Direct Qdrant '{disp_fn_direct}' activated!"); stlit.rerun()
                else: 
                    stlit.sidebar.error(f"Failed to setup LCEL chain for Direct Qdrant '{disp_fn_direct}'.")
            else: 
                stlit.sidebar.error(f"Failed to build retriever for Direct Qdrant '{disp_fn_direct}'.")

        confirm_key_direct_q = f"confirm_delete_direct_q_pending_final_{sanitized_direct_q_key_final}"
        if stlit.sidebar.button(f"🗑️ Delete '{disp_fn_direct}' from Qdrant Server", key=f"delete_direct_q_btn_final_{sanitized_direct_q_key_final}"):
            stlit.session_state[confirm_key_direct_q] = True; stlit.rerun()
        if stlit.session_state.get(confirm_key_direct_q):
            stlit.sidebar.warning(f"Sure to delete Qdrant collection '{q_coll_direct}' from server?")
            cq1, cq2 = stlit.sidebar.columns(2)
            if cq1.button("YES, DELETE FROM QDRANT", key=f"confirm_del_directq_final_{sanitized_direct_q_key_final}"):
                qdrant_direct_delete_successful = False
                try:
                    client_direct_del = QdrantSDKClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY, timeout=20)
                    result_direct_del = client_direct_del.delete_collection(collection_name=q_coll_direct)
                    if result_direct_del: qdrant_direct_delete_successful = True; stlit.sidebar.success(f"Qdrant collection '{q_coll_direct}' reported deleted.")
                    else: stlit.sidebar.warning(f"Qdrant client reported NO SUCCESS deleting '{q_coll_direct}'.")
                except Exception as e_qdel_direct_final: stlit.sidebar.error(f"Qdrant delete EXCEPTION for '{q_coll_direct}': {e_qdel_direct_final}")
                if qdrant_direct_delete_successful:
                    try: client_direct_del.get_collection(collection_name=q_coll_direct); stlit.sidebar.error(f"VERIFICATION FAILED: Qdrant collection '{q_coll_direct}' still exists!")
                    except: stlit.sidebar.success(f"VERIFICATION SUCCEEDED: Qdrant collection '{q_coll_direct}' no longer found.")
                if (stlit.session_state.active_document_filename == disp_fn_direct and stlit.session_state.active_chain_settings.get("qdrant_collection_name_if_direct") == q_coll_direct):
                    stlit.session_state.active_retriever = None; stlit.session_state.document_processed_for_chat = False; stlit.session_state.active_document_filename = None; stlit.session_state.active_doc_chat_session_id = None
                stlit.session_state.available_qdrant_collections = qdrant_services.get_existing_qdrant_collections_cached(config.QDRANT_URL, config.QDRANT_API_KEY, _cache_buster=time.time())
                refresh_persisted_metadata_list_fn_final(); stlit.session_state[confirm_key_direct_q] = False
                stlit.session_state.direct_qdrant_selector_value = "-- Select Direct Qdrant Collection --"; stlit.sidebar.info(f"Qdrant deletion process for '{q_coll_direct}' complete."); stlit.rerun()
            if cq2.button("NO, CANCEL", key=f"cancel_del_directq_final_{sanitized_direct_q_key_final}"):
                stlit.session_state[confirm_key_direct_q] = False; stlit.rerun()
    elif not stlit.session_state.available_qdrant_collections: stlit.sidebar.caption("No direct Qdrant collections found or Qdrant not reachable.")
    stlit.sidebar.divider()
    if stlit.sidebar.button("Clear Current Chat History & Memory", key="clear_chat_lcel_btn_final_v4"):
        if stlit.session_state.active_chat_mode == "Document Q&A":
            stlit.session_state.document_chat_messages = []
            active_doc_session_id = stlit.session_state.get("active_doc_chat_session_id")
            if active_doc_session_id and active_doc_session_id in stlit.session_state.document_chat_histories:
                stlit.session_state.document_chat_histories[active_doc_session_id].clear()
            stlit.info("Document Q&A chat history and memory cleared.")
        elif stlit.session_state.active_chat_mode == "General Chat":
            stlit.session_state.general_chat_messages = []
            if stlit.session_state.general_chat_history_store: stlit.session_state.general_chat_history_store.clear()
            stlit.session_state.update(rag_confirmation_pending=None, clarifying_document_for_rag=False, last_general_chat_query=None, last_selectbox_choice_for_rag_clarify=None, selectbox_choice_submitted=False)
            stlit.info("General chat history and memory cleared.")
        stlit.rerun()
    stlit.sidebar.divider(); stlit.sidebar.header("4. YouTube Video Summarizer")
    youtube_url_input_ui = stlit.sidebar.text_input("Enter YouTube Video URL:", key="yt_url_input_sidebar_lcel_final_v2")
    summary_language_options_yt = {"English": "en", "Serbian": "sr"}
    selected_summary_lang_name_yt = stlit.sidebar.selectbox("Desired YT Summary Language:", options=list(summary_language_options_yt.keys()), index=0, key="yt_lang_selector_dd_lcel_final_v2")
    selected_summary_lang_code_yt = summary_language_options_yt[selected_summary_lang_name_yt]
    if stlit.sidebar.button("Summarize YouTube Video", key="summarize_yt_main_btn_lcel_final_v2"):
        stlit.session_state.youtube_summary = None
        if not youtube_url_input_ui: stlit.sidebar.warning("Please enter a YouTube video URL.")
        elif not stlit.session_state.chat_llm: stlit.sidebar.error("Chat LLM not loaded. Cannot summarize YouTube video.")
        else:
            try:
                with stlit.spinner("Fetching transcript and summarizing video..."):
                    summary_output = youtube_summarizer.get_youtube_summary(youtube_url_input_ui, selected_summary_lang_code_yt, stlit.session_state.chat_llm)
                if isinstance(summary_output, str) and summary_output.strip():
                    if summary_output.lower().startswith("error:") or summary_output.lower().startswith("warning:"): stlit.session_state.youtube_summary = f"⚠️ {summary_output}"; stlit.sidebar.warning(summary_output)
                    else: stlit.session_state.youtube_summary = summary_output; stlit.toast("YouTube video summarized successfully!", icon="📺")
                elif summary_output is None: stlit.session_state.youtube_summary = "⚠️ Summarization Failed: No output from function."; stlit.sidebar.error("Summarization returned no output.")
                else: stlit.session_state.youtube_summary = "⚠️ Summarization Issue: Empty or unexpected response."; stlit.sidebar.warning("Summarization resulted in an empty response.")
            except Exception as e:
                tb_str = traceback.format_exc()
                stlit.session_state.youtube_summary = f"❌ Critical Error: {str(e)}\n```\n{tb_str[:500]}...\n```"; stlit.sidebar.error(f"YouTube summarizer error: {e}")
            stlit.rerun()

# --- Stop if models didn't load ---
if not stlit.session_state.get("embeddings_generator") or not stlit.session_state.get("chat_llm"):
    stlit.error("LLM/Embeddings models failed. Check Ollama server. App cannot continue."); stlit.stop()

# --- Main Page Layout ---
if stlit.session_state.active_chat_mode == "Document Q&A":
    stlit.header("📄 Chat with Your Document")
    if stlit.session_state.document_processed_for_chat and stlit.session_state.active_document_filename and stlit.session_state.active_retriever:
        active_s = stlit.session_state.active_chain_settings; doc_fn_display = stlit.session_state.active_document_filename
        doc_lang_disp_val = stlit.session_state.processed_documents_metadata.get(doc_fn_display, {}).get("language", "N/A") if stlit.session_state.processed_documents_metadata.get(doc_fn_display) else stlit.session_state.document_language or "N/A"
        doc_lang_disp = f"(Lang: {doc_lang_disp_val})"; vs_disp = active_s.get('vector_store', 'N/A'); sem_type_disp = active_s.get('semantic_type', 'N/A')
        ret_disp_parts = [f"Vector Store: {vs_disp}"]
        if active_s.get('hybrid'):
            bm25_w_disp, sem_w_disp = active_s.get('bm25_w', "N/A"), active_s.get('semantic_w', "N/A")
            bm25_w_disp_str, sem_w_disp_str = (f"{x:.2f}" if isinstance(x, float) else str(x) for x in [bm25_w_disp, sem_w_disp])
            ret_disp_parts.append(f"Hybrid (BM25w:{bm25_w_disp_str}|{vs_disp}w:{sem_w_disp_str}, type:{sem_type_disp})")
        else: ret_disp_parts.append(f"Semantic Only ({vs_disp} type: {sem_type_disp})")
        if sem_type_disp == "mmr" and str(active_s.get("mmr_lambda", "N/A")).lower() != "n/a":
            mmr_lambda_disp_val = active_s.get('mmr_lambda', 0.5); mmr_fkf_disp_val = active_s.get('mmr_fetch_k_f', 5.0)
            try: mmr_lambda_disp_str = f"{float(mmr_lambda_disp_val):.2f}"
            except: mmr_lambda_disp_str = str(mmr_lambda_disp_val)
            try: mmr_fkf_disp_str = f"{float(mmr_fkf_disp_val):.1f}"
            except: mmr_fkf_disp_str = str(mmr_fkf_disp_val)
            ret_disp_parts[-1] += f", MMR λ:{mmr_lambda_disp_str}, MMR fetch_k_factor:{mmr_fkf_disp_str}"
        ret_disp_full, sum_disp, k_disp = ", ".join(ret_disp_parts), "Map-Reduce Chunks" if active_s.get('map_reduce') else "Direct Chunk Context", active_s.get('k', 'N/A')
        stlit.info(f"""**Active Document:** `{doc_fn_display}` {doc_lang_disp}\n**Retrieval:** {ret_disp_full}, **K:** {k_disp}, **Context Handling:** {sum_disp}""")
        for message in stlit.session_state.document_chat_messages:
            with stlit.chat_message(message["role"]): stlit.markdown(message["content"])
        if user_question_input_doc := stlit.chat_input(f"Ask about '{stlit.session_state.active_document_filename}'...", key="doc_chat_input_lcel_main_key_v4_final_complete"):
            stlit.session_state.document_chat_messages.append({"role": "user", "content": user_question_input_doc})
            with stlit.chat_message("user"): stlit.markdown(user_question_input_doc)
            with stlit.chat_message("assistant"):
                answer_placeholder = stlit.empty(); ai_response_content = "Processing..."
                answer_placeholder.markdown(ai_response_content + "▌")
                try:
                    final_question_for_chain = user_question_input_doc; lang_instruction_for_llm = ""
                    current_doc_lang_for_prompt = stlit.session_state.document_language
                    if current_doc_lang_for_prompt == 'sr': lang_instruction_for_llm = "Odgovori na sledeće pitanje strogo na srpskom jeziku. "
                    else:
                        try:
                            question_lang_raw = detect(user_question_input_doc)
                            question_lang_for_prompt = 'sr' if question_lang_raw == 'hr' else question_lang_raw
                        except LangDetectException: question_lang_for_prompt = None
                        if question_lang_for_prompt == 'sr': lang_instruction_for_llm = "Odgovori na sledeće pitanje strogo na srpskom jeziku. "
                        elif question_lang_for_prompt == 'en': lang_instruction_for_llm = "Answer the following question strictly in English. "
                    if lang_instruction_for_llm:
                        prefix, q_label = ("Instrukcija: ", "Korisničko pitanje je: ") if lang_instruction_for_llm.startswith("Odgovori") else ("Instruction: ", "User's question is: ")
                        final_question_for_chain = f"{prefix}{lang_instruction_for_llm}{q_label}\"{user_question_input_doc}\""
                    
                    current_session_id = stlit.session_state.get("active_doc_chat_session_id")
                    history = stlit.session_state.document_chat_histories.get(current_session_id, ChatMessageHistory())
                    
                    temp_chain = chain_builder.setup_lcel_conversational_rag_chain(
                        retriever=stlit.session_state.active_retriever,
                        llm_instance=stlit.session_state.chat_llm,
                        use_map_reduce_summarization_toggle=stlit.session_state.active_chain_settings.get("map_reduce", False)
                    )
                    
                    if temp_chain:
                        response_dict = temp_chain.invoke(
                            {"input": final_question_for_chain, "chat_history": history.messages},
                            config={"callbacks": stlit.session_state.get('langchain_callbacks', [])}
                        )
                        ai_response_content = response_dict.get("answer", "LCEL chain did not return an answer.")
                        history.add_user_message(final_question_for_chain)
                        history.add_ai_message(ai_response_content)
                        source_documents_for_display = response_dict.get("context", [])
                    else:
                        ai_response_content = "Error: Could not build the conversational chain for this query."
                        source_documents_for_display = []

                    answer_placeholder.markdown(ai_response_content)
                    if source_documents_for_display:
                        with stlit.expander("Show original source documents retrieved (LCEL Context)"):
                            for i, doc_obj in enumerate(source_documents_for_display):
                                source_info = doc_obj.metadata.get('source', 'N/A'); page_info = (f" p.{doc_obj.metadata.get('page')}" if doc_obj.metadata.get('page') is not None else '')
                                stlit.markdown(f"**Source {i+1} (from {source_info}{page_info}):**\n```\n{doc_obj.page_content}\n```\n---")
                except Exception as e:
                    ai_response_content = f"Error in Document Q&A (LCEL): {type(e).__name__}\n\n{e}"
                    stlit.error(f"Error in Document Q&A (LCEL):\n\n{type(e).__name__}:\n{traceback.format_exc()}")
                    answer_placeholder.error(ai_response_content)
            stlit.session_state.document_chat_messages.append({"role": "assistant", "content": ai_response_content})
    else: stlit.info("Please process or activate a document for LCEL Document Q&A.")
elif stlit.session_state.active_chat_mode == "General Chat":
    stlit.header("💬 General Chat")
    for message in stlit.session_state.general_chat_messages:
        with stlit.chat_message(message["role"]): stlit.markdown(message["content"])
    if stlit.session_state.rag_confirmation_pending:
        pending_info = stlit.session_state.rag_confirmation_pending; original_query_for_rag = pending_info["query"]
        doc_display_name_for_rag = pending_info["doc_to_query"]; actual_name_for_rag = pending_info.get("actual_name"); item_type_for_rag = pending_info.get("item_type")
        if doc_display_name_for_rag == "UNKNOWN" or stlit.session_state.clarifying_document_for_rag:
            stlit.session_state.clarifying_document_for_rag = True; available_doc_tuples_for_clarify = get_combined_document_and_collection_names_for_intent()
            available_display_names_for_clarify = [dt[0] for dt in available_doc_tuples_for_clarify]
            if not available_display_names_for_clarify:
                stlit.warning("Query might need a document, but no queryable sources are available.")
                if stlit.session_state.rag_confirmation_pending: stlit.session_state.general_chat_messages.append({"role": "assistant", "content": "I can't search any documents as none are available."});
                if stlit.session_state.general_chat_history_store: stlit.session_state.general_chat_history_store.add_messages([HumanMessage(content=original_query_for_rag), AIMessage(content="I can't search any documents as none are available.")])
                stlit.session_state.update(rag_confirmation_pending=None, clarifying_document_for_rag=False, last_selectbox_choice_for_rag_clarify=None); stlit.rerun()
            else:
                stlit.info(f"To answer '{original_query_for_rag}', please select a source, type its name in the chat, or type 'cancel search'.")
                with stlit.form(key="rag_clarification_form_key_v4_lcel_final"):
                    selected_display_name_from_box = stlit.selectbox("Or, select source to search:", options=["-- Choose or Type Below --", "-- Cancel Search --"] + available_display_names_for_clarify, key="rag_doc_clarification_selectbox_key_lcel_final", index=0)
                    submit_selectbox_choice = stlit.form_submit_button("Confirm Selection / Cancel")
                if submit_selectbox_choice:
                    stlit.session_state.last_selectbox_choice_for_rag_clarify = selected_display_name_from_box; stlit.session_state.selectbox_choice_submitted = True
                    if selected_display_name_from_box not in ["-- Choose or Type Below --", "-- Cancel Search --"]:
                        for disp_name, act_name, type_ind, _ in available_doc_tuples_for_clarify:
                            if selected_display_name_from_box == disp_name: stlit.session_state.rag_confirmation_pending.update(doc_to_query=disp_name, actual_name=act_name, item_type=type_ind); break
                        stlit.session_state.clarifying_document_for_rag = False; stlit.rerun()
                    elif selected_display_name_from_box == "-- Cancel Search --":
                        cancelled_msg = "Okay, I won't search any documents for that query."
                        stlit.session_state.general_chat_messages.append({"role": "assistant", "content": cancelled_msg})
                        if stlit.session_state.general_chat_history_store: stlit.session_state.general_chat_history_store.add_messages([HumanMessage(content=original_query_for_rag), AIMessage(content=cancelled_msg)])
                        stlit.session_state.update(rag_confirmation_pending=None, clarifying_document_for_rag=False, last_selectbox_choice_for_rag_clarify=None, last_general_chat_query=None); stlit.rerun()
        elif doc_display_name_for_rag and actual_name_for_rag and item_type_for_rag:
            stlit.info(f"Searching {item_type_for_rag.replace('_', ' ')}: '{doc_display_name_for_rag}' for: '{original_query_for_rag}' using LCEL RAG...")
            with stlit.chat_message("assistant"):
                answer_placeholder = stlit.empty(); answer_placeholder.markdown(f"Consulting '{doc_display_name_for_rag}' with LCEL RAG...")
                ai_rag_response_content_raw = f"Sorry, an error occurred preparing LCEL RAG for '{doc_display_name_for_rag}'."
                chunks_for_gc_rag = None; rag_settings_gc = {}; q_coll_name_gc = None; vs_choice_gc = "Qdrant"
                if item_type_for_rag == "app_managed_db":
                    meta = next((m for m in stlit.session_state.get("persisted_doc_metadata", []) if m['original_filename'] == actual_name_for_rag), None)
                    if meta and meta.get('persistent_file_path') and os.path.exists(meta['persistent_file_path']):
                        rag_settings_gc = meta.get('processing_settings',{}).copy(); rag_settings_gc["map_reduce"] = use_map_reduce_ui
                        if rag_settings_gc.get('hybrid'): chunks_for_gc_rag, _, _ = document_processor.extract_text_and_chunks(meta['persistent_file_path'], meta['original_filename'])
                        q_coll_name_gc = meta.get('qdrant_collection_name'); vs_choice_gc = "Qdrant"
                    else: stlit.error(f"GC RAG: DB metadata/file error for: {actual_name_for_rag}"); vs_choice_gc = None
                elif item_type_for_rag.startswith("app_managed_session"):
                    meta_sess = stlit.session_state.processed_documents_metadata.get(actual_name_for_rag)
                    if meta_sess:
                        rag_settings_gc = meta_sess.get('settings',{}).copy(); rag_settings_gc["map_reduce"] = use_map_reduce_ui
                        vs_choice_gc = rag_settings_gc.get("vector_store", "FAISS"); chunks_for_gc_rag = meta_sess.get("all_chunks_as_docs")
                        if vs_choice_gc == "Qdrant": q_coll_name_gc = meta_sess.get("qdrant_collection_name")
                    else: stlit.error(f"GC RAG: Session metadata error for: {actual_name_for_rag}"); vs_choice_gc = None
                elif item_type_for_rag == "direct_qdrant_collection":
                    q_coll_name_gc = actual_name_for_rag; vs_choice_gc = "Qdrant"; rag_settings_gc = { "filename": doc_display_name_for_rag, "vector_store": "Qdrant", "semantic_type": semantic_retriever_selection_ui, "k": num_docs_to_retrieve_ui, "hybrid": use_hybrid_ui, "bm25_w": ensemble_bm25_weight_val if use_hybrid_ui else "N/A", "semantic_w": ensemble_semantic_weight_val if use_hybrid_ui else "N/A", "mmr_lambda": mmr_lambda_val if semantic_retriever_selection_ui == "mmr" else "N/A", "mmr_fetch_k_f": mmr_fetch_k_factor_val if semantic_retriever_selection_ui == "mmr" else "N/A", "map_reduce": use_map_reduce_ui }
                    if rag_settings_gc['hybrid']:
                        stlit.warning("Hybrid search (BM25) for Direct Qdrant collections is not supported as the original document text is not available. Using semantic search only.")
                        rag_settings_gc['hybrid'] = False
                        rag_settings_gc['bm25_w'] = "N/A"; rag_settings_gc['semantic_w'] = "N/A"
                    chunks_for_gc_rag = None
                else: vs_choice_gc = None

                if vs_choice_gc:
                    can_proceed = (vs_choice_gc == "Qdrant" and q_coll_name_gc) or (vs_choice_gc == "FAISS" and chunks_for_gc_rag)
                    if can_proceed and stlit.session_state.embeddings_generator:
                        retriever_gc_rag = build_retriever(chunks_for_gc_rag, vs_choice_gc, rag_settings_gc.get("semantic_type", semantic_retriever_selection_ui), rag_settings_gc.get("k", num_docs_to_retrieve_ui), rag_settings_gc.get("hybrid", use_hybrid_ui), rag_settings_gc.get("bm25_w", ensemble_bm25_weight_val), rag_settings_gc.get("semantic_w", ensemble_semantic_weight_val), rag_settings_gc.get("mmr_lambda", mmr_lambda_val), rag_settings_gc.get("mmr_fetch_k_f", mmr_fetch_k_factor_val), q_coll_name_gc, stlit.session_state.embeddings_generator, config.QDRANT_URL, config.QDRANT_API_KEY)
                        if retriever_gc_rag:
                            core_gc_rag_chain = chain_builder.setup_lcel_conversational_rag_chain(retriever_gc_rag, stlit.session_state.chat_llm, use_map_reduce_summarization_toggle=rag_settings_gc.get("map_reduce", False))
                            if core_gc_rag_chain:
                                try:
                                    gc_query_for_chain = original_query_for_rag; gc_lang_instruction = ""
                                    gc_query_lang_raw = None
                                    try: gc_query_lang_raw = detect(original_query_for_rag); gc_query_lang = 'sr' if gc_query_lang_raw == 'hr' else gc_query_lang_raw
                                    except LangDetectException: gc_query_lang = None
                                    if gc_query_lang == 'sr': gc_lang_instruction = "Odgovori na sledeće pitanje strogo na srpskom jeziku. "
                                    elif gc_query_lang == 'en': gc_lang_instruction = "Answer the following question strictly in English. "
                                    if gc_lang_instruction:
                                        gc_prefix, gc_q_label = ("Instrukcija: ", "Korisničko pitanje je: ") if gc_lang_instruction.startswith("Odgovori") else ("Instruction: ", "User's question is: ")
                                        gc_query_for_chain = f"{gc_prefix}{gc_lang_instruction}{gc_q_label}\"{original_query_for_rag}\""
                                    gc_response = core_gc_rag_chain.invoke({"input": gc_query_for_chain, "chat_history": [] }, config={'callbacks': stlit.session_state.get('langchain_callbacks', [])})
                                    ai_rag_response_content_raw = gc_response.get("answer", f"LCEL RAG search of '{doc_display_name_for_rag}' yielded no specific answer.")
                                except Exception as e_gc_rag: ai_rag_response_content_raw = f"Error during LCEL RAG search for '{doc_display_name_for_rag}': {e_gc_rag}"; stlit.exception(e_gc_rag)
                            else: ai_rag_response_content_raw = f"Failed to set up LCEL RAG core chain for '{doc_display_name_for_rag}'."
                        else: ai_rag_response_content_raw = f"Failed to build retriever for GC RAG on '{doc_display_name_for_rag}'."
                    else: ai_rag_response_content_raw = f"Could not get info/embeddings for GC RAG on '{doc_display_name_for_rag}'."
                else: ai_rag_response_content_raw = f"Could not proceed with RAG for '{doc_display_name_for_rag}' (item_type: {item_type_for_rag})."
                final_rag_response_for_display = str(ai_rag_response_content_raw).strip() if ai_rag_response_content_raw and str(ai_rag_response_content_raw).strip().lower() != "none" else f"The search in '{doc_display_name_for_rag}' did not yield a specific answer."
                answer_placeholder.markdown(final_rag_response_for_display)
                stlit.session_state.general_chat_messages.append({"role": "assistant", "content": final_rag_response_for_display})
                if stlit.session_state.general_chat_history_store: stlit.session_state.general_chat_history_store.add_messages([HumanMessage(content=original_query_for_rag), AIMessage(content=final_rag_response_for_display)])
            stlit.session_state.update(rag_confirmation_pending=None, clarifying_document_for_rag=False, last_general_chat_query=None, last_selectbox_choice_for_rag_clarify=None, selectbox_choice_submitted=False); stlit.rerun()
    if not (stlit.session_state.rag_confirmation_pending and stlit.session_state.clarifying_document_for_rag and not stlit.session_state.selectbox_choice_submitted):
        user_question_input_general = stlit.chat_input("Ask me anything, or about your documents...", key="general_chat_input_lcel_key_v3_final", disabled=(stlit.session_state.clarifying_document_for_rag and not stlit.session_state.selectbox_choice_submitted))
        if user_question_input_general and user_question_input_general != stlit.session_state.last_general_chat_query:
            stlit.session_state.last_general_chat_query = user_question_input_general; stlit.session_state.general_chat_messages.append({"role": "user", "content": user_question_input_general})
            with stlit.chat_message("user"): stlit.markdown(user_question_input_general)
            stlit.session_state.selectbox_choice_submitted = False
            if stlit.session_state.clarifying_document_for_rag and stlit.session_state.rag_confirmation_pending:
                original_pending_query = stlit.session_state.rag_confirmation_pending["query"]; potential_doc_name_typed = user_question_input_general.strip()
                if "cancel" in potential_doc_name_typed.lower():
                    cancelled_msg = "Okay, document search cancelled."; stlit.session_state.general_chat_messages.append({"role": "assistant", "content": cancelled_msg});
                    if stlit.session_state.general_chat_history_store: stlit.session_state.general_chat_history_store.add_messages([HumanMessage(content=original_pending_query), AIMessage(content=cancelled_msg)])
                    stlit.session_state.update(rag_confirmation_pending=None, clarifying_document_for_rag=False, last_general_chat_query=None); stlit.rerun()
                else:
                    available_docs_clarify_typed = get_combined_document_and_collection_names_for_intent(); chosen_doc_display_name, chosen_actual_name, chosen_item_type = None, None, None
                    for disp, act, type_ind, _ in available_docs_clarify_typed:
                        if potential_doc_name_typed.lower() == disp.lower(): chosen_doc_display_name, chosen_actual_name, chosen_item_type = disp, act, type_ind; break
                    if not chosen_doc_display_name:
                         for disp, act, type_ind, _ in available_docs_clarify_typed:
                            if potential_doc_name_typed.lower() in disp.lower(): chosen_doc_display_name, chosen_actual_name, chosen_item_type = disp, act, type_ind; break
                    if chosen_doc_display_name:
                        stlit.session_state.rag_confirmation_pending.update(doc_to_query=chosen_doc_display_name, actual_name=chosen_actual_name, item_type=chosen_item_type)
                        stlit.session_state.clarifying_document_for_rag = False; stlit.info(f"Okay, searching '{chosen_doc_display_name}' for: '{original_pending_query}'"); stlit.rerun()
                    else: stlit.session_state.general_chat_messages.append({"role": "assistant", "content": f"Sorry, couldn't identify '{potential_doc_name_typed}'. Please choose from selectbox, try another name, or 'cancel search'."}); stlit.rerun()
            else:
                available_docs_intent = get_combined_document_and_collection_names_for_intent(); requires_rag, identified_display, identified_actual, identified_type = False, None, None, None
                if available_docs_intent and stlit.session_state.chat_llm:
                    requires_rag, identified_display, identified_actual, identified_type = classify_intent_for_rag_extended(user_question_input_general, available_docs_intent, stlit.session_state.chat_llm)
                if requires_rag:
                    stlit.session_state.rag_confirmation_pending = {"query": user_question_input_general, "doc_to_query": identified_display, "actual_name": identified_actual, "item_type": identified_type}
                    if identified_display == "UNKNOWN": stlit.session_state.clarifying_document_for_rag = True
                    stlit.rerun()
                else:
                    if stlit.session_state.general_conversation_runnable_with_history:
                        with stlit.chat_message("assistant"):
                            answer_placeholder = stlit.empty(); answer_placeholder.markdown("Thinking...▌"); final_response_for_display = "I'm unable to provide a response right now."
                            try:
                                response_raw = stlit.session_state.general_conversation_runnable_with_history.invoke({"input": user_question_input_general}, config={"configurable": {"session_id": "general_chat_session"}, "callbacks": stlit.session_state.get('langchain_callbacks', [])})
                                response_str = str(response_raw).strip()
                                if not response_str or re.fullmatch(r'(none)+', re.sub(r'[^a-z0-9]', '', response_str.lower())): final_response_for_display = "I'm not sure how to respond to that."
                                else: final_response_for_display = response_str
                                answer_placeholder.markdown(final_response_for_display)
                                stlit.session_state.general_chat_messages.append({"role": "assistant", "content": final_response_for_display})
                                if stlit.session_state.general_chat_history_store: stlit.session_state.general_chat_history_store.add_messages([HumanMessage(content=user_question_input_general), AIMessage(content=final_response_for_display)])
                            except Exception as e_gen_chat:
                                error_msg_for_user = "I encountered an error while processing your request."
                                stlit.error(f"Error in general chat: {e_gen_chat}"); stlit.exception(e_gen_chat); answer_placeholder.error(error_msg_for_user)
                                stlit.session_state.general_chat_messages.append({"role": "assistant", "content": error_msg_for_user})
                                if stlit.session_state.general_chat_history_store: stlit.session_state.general_chat_history_store.add_messages([HumanMessage(content=user_question_input_general), AIMessage(content=error_msg_for_user)])
                    elif not stlit.session_state.chat_llm: stlit.error("Chat LLM not initialized."); stlit.session_state.general_chat_messages.append({"role": "assistant", "content": "Chat LLM not initialized."})
                    else: stlit.error("General chat runnable not available."); stlit.session_state.general_chat_messages.append({"role": "assistant", "content": "General chat runnable not available."})

# Display YouTube summary
if "youtube_summary" in stlit.session_state and stlit.session_state.youtube_summary:
    stlit.divider(); stlit.header("📺 YouTube Video Summarizer Output"); stlit.markdown(stlit.session_state.youtube_summary)
    if stlit.button("Clear YouTube Summary", key="clear_yt_summary_main_page_lcel_btn_v4_global"): stlit.session_state.youtube_summary = None; stlit.rerun()