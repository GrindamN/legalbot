# chain_builder.py
import re
import traceback

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

import prompts

def _format_docs(docs: list[Document]) -> str:
    """Formats a list of documents into a single string."""
    if not isinstance(docs, list):
        docs = [docs] if docs else []
    return "\n\n".join(doc.page_content for doc in docs if hasattr(doc, 'page_content') and doc.page_content)

def get_target_language_name_for_summary_from_question(question_text: str) -> str:
    question_lower = question_text.lower()
    if "instrukcija: odgovori" in question_lower and "srpskom jeziku" in question_lower:
        return "Serbian"
    elif "instruction: answer" in question_lower and "in english" in question_lower:
        return "English"
    return "English"

def setup_lcel_conversational_rag_chain(
    retriever,
    llm_instance,
    use_map_reduce_summarization_toggle=False
    ):
    try:
        history_aware_retriever_runnable = create_history_aware_retriever(
            llm=llm_instance,
            retriever=retriever,
            prompt=prompts.CONDENSE_QUESTION_PROMPT_OBJECT
        )

        if use_map_reduce_summarization_toggle:
            map_chain_lcel = prompts.MAP_PROMPT_TEMPLATE_OBJECT | llm_instance | StrOutputParser()
            reduce_chain_lcel = prompts.REDUCE_PROMPT_TEMPLATE_OBJECT | llm_instance | StrOutputParser()

            def lcel_map_reduce_logic(inputs_from_retriever: dict):
                retrieved_docs = inputs_from_retriever.get("context", [])
                rephrased_question = inputs_from_retriever.get("input", "")

                if not rephrased_question:
                    print("CONSOLE ERROR (chain_builder): Rephrased question is empty in lcel_map_reduce_logic.")
                    return "Error: Rephrased question was empty."
                if not retrieved_docs:
                    return "No documents were retrieved to summarize."

                valid_docs = []
                if all(isinstance(doc, Document) for doc in retrieved_docs):
                    valid_docs = retrieved_docs
                else:
                    for item in retrieved_docs:
                        if isinstance(item, Document): valid_docs.append(item)
                        elif isinstance(item, str): valid_docs.append(Document(page_content=item, metadata={"source": "converted_string"}))

                if not valid_docs:
                     return "Retrieved context became empty or invalid after conversion for map-reduce."

                target_lang_for_map_summaries = get_target_language_name_for_summary_from_question(rephrased_question)

                mapped_outputs = []
                for doc_idx, doc in enumerate(valid_docs):
                    try:
                        map_input_dict = {
                            "document_text": doc.page_content,
                            "question": rephrased_question,
                            "target_language_name_for_summary": target_lang_for_map_summaries
                        }
                        mapped_output = map_chain_lcel.invoke(map_input_dict)
                        mapped_outputs.append(mapped_output)
                    except Exception as e_map_invoke:
                        error_message = f"[ERROR in map step for document {doc_idx+1}: {e_map_invoke}]"
                        mapped_outputs.append(error_message)
                        print(f"CONSOLE LOG (chain_builder): LCEL Map step invoke failed for document {doc_idx+1}: {e_map_invoke}")

                if not mapped_outputs:
                    return "Map step resulted in no summaries. Cannot proceed to reduce step."

                doc_summaries_str = "\n\n".join(mapped_outputs)
                final_result = reduce_chain_lcel.invoke({
                    "doc_summaries": doc_summaries_str,
                    "question": rephrased_question
                })
                return final_result

            documents_chain = RunnableLambda(lcel_map_reduce_logic)
        else:
            documents_chain = create_stuff_documents_chain(
                llm=llm_instance,
                prompt=prompts.QA_PROMPT_TEMPLATE_OBJECT,
                output_parser=StrOutputParser()
            )

        conversational_rag_chain = create_retrieval_chain(
            retriever=history_aware_retriever_runnable,
            combine_docs_chain=documents_chain
        )
        return conversational_rag_chain

    except Exception as e:
        print(f"CONSOLE LOG (chain_builder): CRITICAL ERROR in setup_lcel_conversational_rag_chain: {e}")
        print(f"CONSOLE LOG (chain_builder): Traceback: {traceback.format_exc()}")
        return None