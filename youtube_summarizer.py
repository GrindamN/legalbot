# youtube_summarizer.py
import streamlit as stlit
import xml.etree.ElementTree
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled, VideoUnavailable
from youtube_transcript_api._transcripts import FetchedTranscript
from langchain.chains.summarize import load_summarize_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain.prompts import PromptTemplate
import traceback

def get_youtube_summary(video_url, target_language_code=None, chat_llm_instance=None):
    if not chat_llm_instance:
        return "Error: Chat LLM instance was not provided to the YouTube summarizer."

    try:
        video_id = None
        if "youtube.com/watch?v=" in video_url:
            video_id = video_url.split("watch?v=")[1].split("&")[0]
        elif "youtu.be/" in video_url:
            video_id = video_url.split("youtu.be/")[1].split("?")[0]
        else:
            return "Error: Invalid YouTube URL format."

        transcript_list_api = YouTubeTranscriptApi.list_transcripts(video_id)
        fetched_transcript_object = None
        transcript_lang = None
        preferred_langs_order = []

        if target_language_code: preferred_langs_order.append(target_language_code)
        if 'en' not in preferred_langs_order: preferred_langs_order.append('en')
        if 'sr' not in preferred_langs_order: preferred_langs_order.append('sr')

        for lang_code in preferred_langs_order:
            try:
                transcript_obj_api = transcript_list_api.find_manually_created_transcript([lang_code])
                fetched_transcript_object = transcript_obj_api.fetch()
                transcript_lang = lang_code
                break
            except (NoTranscriptFound, xml.etree.ElementTree.ParseError, Exception): continue
        
        if not fetched_transcript_object:
            for lang_code in preferred_langs_order:
                try:
                    transcript_obj_api = transcript_list_api.find_generated_transcript([lang_code])
                    fetched_transcript_object = transcript_obj_api.fetch()
                    transcript_lang = lang_code
                    break
                except (NoTranscriptFound, xml.etree.ElementTree.ParseError, Exception): continue

        if not fetched_transcript_object:
            try:
                first_available_transcript_api_obj = next(iter(transcript_list_api), None)
                if first_available_transcript_api_obj:
                    transcript_lang = first_available_transcript_api_obj.language
                    fetched_transcript_object = first_available_transcript_api_obj.fetch()
                else: return "Error: No transcripts available for this video."
            except Exception as e: return f"Error fetching/parsing fallback transcript: {type(e).__name__} - {e}"

        if not fetched_transcript_object:
            return "Error: Failed to fetch any usable transcript."
        
        processed_texts = []
        for segment in fetched_transcript_object:
            segment_text = None
            if hasattr(segment, 'text') and isinstance(segment.text, str): segment_text = segment.text
            elif isinstance(segment, dict) and 'text' in segment and isinstance(segment['text'], str): segment_text = segment['text']
            if segment_text is not None: processed_texts.append(segment_text)
        
        full_transcript_text = " ".join(processed_texts)
        if not full_transcript_text.strip():
            return f"Error: Transcript (lang: {transcript_lang or 'unknown'}) is empty."
        
        if not transcript_lang: transcript_lang = "en"
        
        summary_lang_instruction_for_combine = f"Return your response in bullet points that cover the key points of the text. If the requested language is not English, translate the bullet point summary to {target_language_code if target_language_code else 'English'}."
        output_language_for_prompt = target_language_code.upper() if target_language_code else "ENGLISH"
        if target_language_code == 'sr':
             summary_lang_instruction_for_combine = f"Odgovori u vidu bullet point-ova koji pokrivaju ključne tačke teksta. Prevedi sažetak u bullet point-ovima na srpski jezik."
             output_language_for_prompt = "SRPSKOM"


        map_prompt_template_yt = ( # Simple map prompt
            "Write a concise summary of the following text segment, focusing on its main ideas and key information. Output in English.\n\n"
            "Text Segment:\n\"\"\"\n{text}\n\"\"\"\n\n"
            "Concise Segment Summary (English):"
        )
        map_prompt_yt = PromptTemplate.from_template(map_prompt_template_yt)

        # --- Combine Prompt Inspired by Linked Script (Bullet Points) ---
        combine_prompt_template_yt = (
            "The following are summaries from different segments of a video transcript.\n"
            "Your task is to synthesize these into a final summary of the entire video's content.\n"
            f"{summary_lang_instruction_for_combine}\n\n" # Includes bullet point and translation instruction
            "Segment Summaries:\n\"\"\"\n{text}\n\"\"\"\n\n"
            f"FINAL BULLET POINT SUMMARY IN {output_language_for_prompt}:"
        )
        combine_prompt_yt = PromptTemplate.from_template(combine_prompt_template_yt)
        
        # Using your old chunking strategy, but you could experiment with the larger one from the link (10000, 0)
        text_splitter_yt = RecursiveCharacterTextSplitter(
            chunk_size=6000, 
            chunk_overlap=300,
            length_function=len
        )
        
        if not isinstance(full_transcript_text, str):
            return f"Error: full_transcript_text is not a string before splitting. Type: {type(full_transcript_text)}"
        
        split_texts = text_splitter_yt.split_text(full_transcript_text)
        if not split_texts or not all(isinstance(t, str) for t in split_texts):
             return f"Error: Text splitter did not return a valid list of strings. Transcript might be too short or an issue occurred."

        docs_yt = [Document(page_content=t) for t in split_texts]
        if not docs_yt: return "Error: No documents created from transcript splits."

        try:
            summarize_chain = load_summarize_chain(
                llm=chat_llm_instance, 
                chain_type="map_reduce", 
                map_prompt=map_prompt_yt, 
                combine_prompt=combine_prompt_yt, 
                verbose=False 
            )
            summary_result = summarize_chain.invoke({"input_documents": docs_yt}) 

            if isinstance(summary_result, dict) and 'output_text' in summary_result:
                final_summary = summary_result['output_text']
                if final_summary and final_summary.strip():
                    return final_summary # This will now likely be bullet points
                else:
                    return "Error: Summarization chain (bullet point prompts) produced an empty output."
            else:
                return f"Error: Summarization chain (bullet point prompts) returned an unexpected result format. Got: {str(summary_result)[:200]}"

        except Exception as e_summarize:
            print(f"Full Traceback for Summarization Chain Exception (bullet point prompts):\n{traceback.format_exc()}")
            return f"Error: Failed during LLM summarization (bullet point prompts). Details: {type(e_summarize).__name__} - {str(e_summarize)[:100]}."

    except TranscriptsDisabled: return "Error: Transcripts are disabled for this YouTube video."
    except NoTranscriptFound: return "Error: No transcript could be found for this video after all checks."
    except VideoUnavailable: return "Error: This YouTube video is unavailable."
    except Exception as e_outer: 
        print(f"Full Traceback for YouTube Summarizer Outer Critical Error:\n{traceback.format_exc()}")
        return f"Error: Critical unexpected error in YouTube summarizer. Details: {type(e_outer).__name__} - {str(e_outer)[:100]}."