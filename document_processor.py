# document_processor.py
import os
import tempfile
import streamlit as stlit
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langdetect import detect, LangDetectException

import fitz
import pytesseract
from PIL import Image
import io
import concurrent.futures

def _ocr_page_task(page_data_tuple):
    """Helper function to perform OCR on a single page's data. To be used with ThreadPoolExecutor."""
    page_num, pixmap_bytes, lang_codes = page_data_tuple
    try:
        img = Image.open(io.BytesIO(pixmap_bytes))
        page_text = pytesseract.image_to_string(img, lang=lang_codes, timeout=60)
        return (page_num, page_text)
    except Exception as e:
        print(f"CONSOLE WARNING (doc_proc_thread): OCR failed for page {page_num + 1}. Error: {e}")
        return (page_num, "")

def extract_text_with_ocr_from_pdf_path(pdf_path: str, lang_codes: str = 'srp+eng') -> str:
    """Extracts text from a PDF file using OCR for each page, parallelized for speed."""
    full_text_parts = {}
    try:
        doc = fitz.open(pdf_path)
        zoom = 2; mat = fitz.Matrix(zoom, zoom)
        page_tasks = []
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pix = page.get_pixmap(matrix=mat)
            pixmap_bytes = pix.tobytes("png")
            page_tasks.append((page_num, pixmap_bytes, lang_codes))
        doc.close()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = executor.map(_ocr_page_task, page_tasks)
            for page_num, page_text in results:
                full_text_parts[page_num] = page_text
    except Exception as e_open_pdf:
        print(f"CONSOLE ERROR (doc_proc): Failed to open or prepare PDF for OCR {os.path.basename(pdf_path)}: {e_open_pdf}")
        return ""
    return "\n\n".join(full_text_parts[i] for i in sorted(full_text_parts.keys()))

def extract_text_and_chunks(uploaded_file_instance_or_path, original_filename_for_display: str = None):
    extracted_text = ""
    langchain_documents_from_load = []
    all_chunks_as_docs = []
    doc_lang = None
    actual_pdf_file_path_for_ocr = None

    is_file_path_input = isinstance(uploaded_file_instance_or_path, str)
    
    if is_file_path_input:
        current_filename = os.path.basename(uploaded_file_instance_or_path)
        if original_filename_for_display is None: original_filename_for_display = current_filename
        actual_pdf_file_path_for_ocr = uploaded_file_instance_or_path
    elif hasattr(uploaded_file_instance_or_path, 'name'):
        current_filename = uploaded_file_instance_or_path.name
        if original_filename_for_display is None: original_filename_for_display = current_filename
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file_ocr:
            temp_file_ocr.write(uploaded_file_instance_or_path.getvalue())
            actual_pdf_file_path_for_ocr = temp_file_ocr.name
    else:
        stlit.error("Invalid input to extract_text_and_chunks."); return None, None, 0

    if not original_filename_for_display: stlit.error("Could not determine filename."); return None, None, 0

    with stlit.spinner(f"Extracting text from '{original_filename_for_display}'..."):
        if current_filename.lower().endswith(".pdf"):
            pdf_source_for_loader = actual_pdf_file_path_for_ocr
            try:
                loader = PyMuPDFLoader(pdf_source_for_loader)
                langchain_documents_from_load = loader.load()
            except Exception as e_pdf_load:
                stlit.warning(f"PyMuPDFLoader failed for '{original_filename_for_display}': {e_pdf_load}. Will rely on OCR.")
                langchain_documents_from_load = []
            
            extracted_text_pymupdf = " ".join([doc.page_content for doc in langchain_documents_from_load if doc.page_content and doc.page_content.strip()])
            avg_chars_per_page_threshold = 50; total_chars_threshold = 200
            num_pages_pymupdf = len(langchain_documents_from_load) if langchain_documents_from_load else 1
            perform_ocr = False
            if not extracted_text_pymupdf.strip():
                stlit.info("Initial text extraction yielded no text. Attempting OCR.")
                perform_ocr = True
            elif (len(extracted_text_pymupdf) / num_pages_pymupdf < avg_chars_per_page_threshold and num_pages_pymupdf > 0) or \
                 len(extracted_text_pymupdf) < total_chars_threshold :
                stlit.warning(f"Initial text extraction yielded very little text ({len(extracted_text_pymupdf)} chars). Attempting OCR as fallback.")
                perform_ocr = True

            if perform_ocr:
                if actual_pdf_file_path_for_ocr and os.path.exists(actual_pdf_file_path_for_ocr):
                    with stlit.spinner(f"Performing parallel OCR on '{original_filename_for_display}'... This may take time."):
                        prelim_lang_for_ocr = None
                        if extracted_text_pymupdf.strip():
                            try: prelim_lang_for_ocr = detect(extracted_text_pymupdf[:500])
                            except: pass
                        ocr_tesseract_langs = 'srp+eng'
                        if prelim_lang_for_ocr == 'sr' or prelim_lang_for_ocr == 'hr': ocr_tesseract_langs = 'srp+eng'
                        elif prelim_lang_for_ocr == 'en': ocr_tesseract_langs = 'eng+srp'
                        extracted_text_ocr = extract_text_with_ocr_from_pdf_path(actual_pdf_file_path_for_ocr, lang_codes=ocr_tesseract_langs)
                    if extracted_text_ocr and len(extracted_text_ocr) > len(extracted_text_pymupdf):
                        stlit.success(f"OCR extracted more text ({len(extracted_text_ocr)} chars). Using OCR result.")
                        extracted_text = extracted_text_ocr
                        langchain_documents_from_load = [Document(page_content=extracted_text, metadata={"source": original_filename_for_display, "ocr_used": True})]
                    else:
                         extracted_text = extracted_text_pymupdf
                else:
                    extracted_text = extracted_text_pymupdf
            else:
                extracted_text = extracted_text_pymupdf

            if not is_file_path_input and actual_pdf_file_path_for_ocr and os.path.exists(actual_pdf_file_path_for_ocr):
                try: os.unlink(actual_pdf_file_path_for_ocr)
                except Exception as e_unlink: stlit.warning(f"Could not delete temp PDF file {actual_pdf_file_path_for_ocr}: {e_unlink}")
        elif current_filename.lower().endswith(".txt"):
            try:
                if is_file_path_input:
                    with open(uploaded_file_instance_or_path, "r", encoding="utf-8") as f: extracted_text = f.read()
                else:
                    extracted_text = uploaded_file_instance_or_path.getvalue().decode("utf-8")
                langchain_documents_from_load = [Document(page_content=extracted_text, metadata={"source": original_filename_for_display})]
            except Exception as e_txt:
                stlit.error(f"Error reading TXT file '{original_filename_for_display}': {e_txt}"); return None, None, 0
        else:
            stlit.error(f"Unsupported file type: {original_filename_for_display}"); return None, None, 0

        if not extracted_text.strip():
            stlit.error(f"Extracted text is empty from '{original_filename_for_display}' after all processing attempts.")
            return None, None, 0

        try:
            sample_text_for_detection = extracted_text[:min(3000, len(extracted_text))]
            if sample_text_for_detection:
                detected_raw_lang = detect(sample_text_for_detection)
                if detected_raw_lang == 'hr':
                    doc_lang = 'sr'
                    print(f"CONSOLE INFO (doc_proc): Overriding detected 'hr' to 'sr' for file '{original_filename_for_display}'")
                else:
                    doc_lang = detected_raw_lang
            else: doc_lang = None
        except LangDetectException: doc_lang = None

        final_text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        if langchain_documents_from_load:
            all_chunks_as_docs = final_text_splitter.split_documents(langchain_documents_from_load)
        else:
            stlit.error("No documents available for splitting after text extraction."); return None, None, 0

        if not all_chunks_as_docs:
            stlit.error("No processable chunks generated after splitting."); return None, None, 0
        
        # --- THIS IS THE CRITICAL FIX ---
        # Explicitly ensure each chunk's source metadata is the intended display filename before returning.
        for chunk in all_chunks_as_docs:
            chunk.metadata['source'] = original_filename_for_display
        # --- END OF CRITICAL FIX ---

        num_chunks = len(all_chunks_as_docs)
        stlit.info(f"Extracted {num_chunks} chunks from '{original_filename_for_display}'. App language: {doc_lang or 'N/A'}")
    return all_chunks_as_docs, doc_lang, num_chunks