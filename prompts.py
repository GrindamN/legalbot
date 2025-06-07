# prompts.py
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder

# --- QA Prompt for answering based on context (for create_stuff_documents_chain) ---
_qa_system_template_base_lcel = """You are an expert assistant for question-answering tasks. Your primary goal is to answer the user's question accurately and concisely using ONLY the provided 'Context'.
**Answering Instructions:**
1.  **Analyze the 'Question' (which is the '{input}' variable) and the 'Context' thoroughly.** The question may start with a language instruction (e.g., "Instrukcija: Odgovori na srpskom jeziku...").
2.  **If you find relevant information in the 'Context' to answer the 'Question' ({input}):**
    *   Formulate your answer strictly based on this information.
    *   Ensure your answer is in the language specified by any instruction at the beginning of the 'Question' ({input}), or otherwise in the language of the main 'Question' ({input}) text.
    *   Keep the answer concise (typically 1-3 sentences) unless more detail is clearly necessary from the context.
    *   Maintain a helpful, objective, and neutral tone.
    *   If there's conflicting information in the context, present the different pieces of information.
    *   Answer directly without conversational fluff or asking if the user needs more help.
3.  **If, AND ONLY IF, you absolutely cannot find any relevant information in the 'Context' to answer the 'Question' ({input}):**
    *   If the 'Question' ({input}) included an instruction to answer in Serbian (e.g., began with "Instrukcija: Odgovori... na srpskom jeziku"), then state ONLY: "Ne mogu da pronađem odgovor na vaše pitanje u dostupnim isečcima dokumenta."
    *   Otherwise (e.g., English requested or no specific language instruction), state ONLY: "I cannot find the answer to your question in the provided document excerpts."

Context:
{context}

Question: {input}

Helpful Answer:"""
QA_PROMPT_TEMPLATE_OBJECT = ChatPromptTemplate.from_template(_qa_system_template_base_lcel)
# Input variables: "context", "input"


# --- Condense Question Prompt (for create_history_aware_retriever) ---
_condense_system_message = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.
Make sure to incorporate relevant context from the chat history into the standalone question.
The standalone question should be self-contained and understandable without the preceding chat history."""

CONDENSE_QUESTION_PROMPT_OBJECT = ChatPromptTemplate.from_messages([
    ("system", _condense_system_message),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])
# Input variables: "chat_history", "input"


# --- Map-Reduce Prompts (LCEL Compatible) ---

_map_template_base_lcel_multilingual = """The following is a document chunk:
"{document_text}"
Based *only* on this document chunk, identify and extract the information relevant to the question below.
If the chunk does not contain relevant information, state "No relevant information in this chunk."
Your summary should be in {target_language_name_for_summary}.
Question: {question}
Concise Relevant Information Summary (in {target_language_name_for_summary}):"""
MAP_PROMPT_TEMPLATE_OBJECT = ChatPromptTemplate.from_template(_map_template_base_lcel_multilingual)
# Input variables: "document_text", "question", "target_language_name_for_summary"


_reduce_template_base_lcel = """You are an expert assistant. Your task is to synthesize a final answer based on a 'Question' and a set of 'Summaries' ({doc_summaries}) derived from document chunks.
These summaries are intermediate answers from different parts of a larger document, and they should already be in the target language indicated by the 'Question'.
Combine them into a single, coherent, and comprehensive answer to the original 'Question'.
The 'Question' itself contains the primary language instruction for the final answer. Adhere to it strictly.
If the summaries are contradictory, acknowledge the different points.
Focus on fulfilling the user's 'Question' based on the information in the 'Summaries'.

Summaries (should be in the target language):
{doc_summaries}

Question (contains final language instruction): {question}

Helpful Answer (must follow language instruction in the Question):"""
REDUCE_PROMPT_TEMPLATE_OBJECT = ChatPromptTemplate.from_template(_reduce_template_base_lcel)
# Input variables: "doc_summaries", "question"