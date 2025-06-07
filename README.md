
## Advanced RAG and Summarization with Ollama & Streamlit


This project is a comprehensive, local-first RAG (Retrieval-Augmented Generation) application built with Streamlit and powered by local LLMs via Ollama. It allows you to chat with your documents, manage a persistent knowledge base using Qdrant, and summarize YouTube videos, all from a user-friendly web interface.

The application is designed to be modular and robust, featuring advanced retrieval strategies, OCR fallbacks for scanned documents, and intelligent intent detection to seamlessly switch between general conversation and document-specific Q&A.

  

### **✨ Key Features**  



**📄 *Dual Chat Modes:***

**Document Q&A**: Chat directly with an activated document using a sophisticated RAG pipeline.

**General Chat**: A conversational mode that can intelligently detect when a query requires document retrieval and initiate a RAG workflow.

 

🚀 ***Advanced RAG Pipeline:***

  
**Hybrid Search**: Combines keyword-based BM25 search with semantic vector search for more accurate and relevant context retrieval.

**MMR** (Maximal Marginal Relevance): An alternative retrieval mode to balance query relevance with context diversity.

**LCEL** Powered: Built using the LangChain Expression Language (LCEL) for transparent and modular chain construction.

  
  
🧠 ***Persistent Document Storage***:

 - Uses Qdrant as a vector database to store document embeddings
   persistently.
 - Manages metadata (filenames, hashes, processing settings) in a local 
   SQLite database, allowing sessions to be restored.

 - Supports in-memory FAISS vector stores for temporary, session-only   
   documents.

  

**📺 *YouTube Video Summarizer:***

  

 - Fetches the transcript from any YouTube video. 
 - Uses a map-reduce  chain to generate a concise, bullet-point summary
   (English or   Serbian supported)

  

**🤖 *Robust Document Processing:***

  

 - Handles PDF and TXT file uploads.
 - Includes an OCR fallback using Tesseract to extract text from scanned    or image-based PDFs.
 -  Filters out common code/API documentation patterns to improve context    quality.

  

***🔧 Direct Vector DB Management:***

  

The UI allows you to list, activate, and delete Qdrant collections directly from the sidebar, giving you full control over your knowledge base.

  

***🏛️ Architecture***

  

The application is composed of a Streamlit frontend that orchestrates several backend services and modules.

  

**🛠️ Tech Stack**

  

**Framework**: Streamlit

**LLM Serving**: Ollama

**LLM Orchestration**: LangChain

**Vector Database:** Qdrant

**In-Memory Vectors:** FAISS

**Document Processing:** PyMuPDF, Tesseract (for OCR)

**Metadata Storage:** SQLite

  

## ⚙️ Setup and Installation

  


**Prerequisites**

You must have the following installed on your system:

  

**Python 3.9+**
  
**Ollama:** Follow the installation guide at https://ollama.com/.

  

**Docker & Docker Compose**: Required to run the Qdrant vector database. Install Docker.

  

**Tesseract OCR Engine**: Required for processing scanned PDFs.

  

**macOS:** brew install tesseract tesseract-lang

**Ubuntu/Debian**: sudo apt-get install tesseract-ocr libtesseract-dev tesseract-ocr-srp tesseract-ocr-eng

**Windows:** Download and run the installer from the Tesseract at UB Mannheim page. Make sure to add the installation directory to your system's PATH.

  

**Installation Steps**

Clone the repository:

    git clone <your-repository-url>
    
    cd <repository-name>
 

Create and activate a virtual environment:

    python -m venv venv

# On macOS/Linux

    source venv/bin/activate

# On Windows

    .\venv\Scripts\activate
    
      
    
    Install Python dependencies:
    
    pip install -r requirements.txt

  

 Configuration

The primary configuration is in config.py. You can adjust the LLM models, service URLs, and storage directories here. The defaults are set for a standard local setup.

# config.py

EMBEDDING_MODEL_NAME = "bge-m3:latest"

CHAT_MODEL_NAME = "gemma3:4b"

OLLAMA_BASE_URL = "http://localhost:11434"

QDRANT_URL = "http://localhost:6333"

  

**🚀 Running the Application**

  

NOTE: Qdrant server and ollama can be run togething in a docker compose file, along the gemma3:4b and bge-m3:latest

I have included the docker-compose to make sure this is easier.

  

Just run: `docker compose up -d`

  

In case it does not work for some reason, make sure to have Qdrant server and Ollama installed separetely in their respective containers and proceed

  

You need to start three separate services. It's recommended to run each command in a new terminal window.

  

Start the Qdrant Server:

Navigate to the project's root directory and use Docker to start Qdrant.

    docker run -p 6333:6333 -p 6334:6334 \
    
    -v $(pwd)/qdrant_storage:/qdrant/storage \
    
    qdrant/qdrant

  

This command also mounts a local directory (qdrant_storage) to persist the vector data.

Start Ollama and Pull Models:

First, ensure the Ollama application is running. Then, pull the required models from the command line:

    ollama pull bge-m3:latest
    
    ollama pull gemma3:4b

  

Start the Streamlit Application:

Open terminal via Code editor where files are located or just terminal and type:

    streamlit run app.py

  

Open your web browser and navigate to the local URL provided by Streamlit (usually http://localhost:8501).

  

## 📖 How to Use

  

**Select Chat Mode**: Choose between "Document Q&A" or "General Chat" in the sidebar.

  

**Upload a Document**: Use the file uploader to select a PDF or TXT file.

  

**Process the Document**: Click the "Process NEW..." button. The app will extract text, create embeddings, and store them in Qdrant.

  

**Configure Retrieval**: Adjust the retriever settings (Hybrid Search, MMR, K value) as needed.

  

**Activate for Q&A**:

  

**Session Docs**: Activate a document processed in the current session.

  

**App-Stored Docs**: Activate a document from a previous session that is stored persistently.

  

**Direct Qdrant**: Activate a collection directly from the Qdrant server (useful if the original file is not available).

  

**Chat**: Once a document is active, you can ask questions about it in the chat window.

  

Summarize YouTube: Paste a YouTube URL in the sidebar, select a language, and click "Summarize" to get a summary.

  

📁 Project Structure


├── app.py # Main Streamlit application file

├── chain_builder.py # Logic for building LCEL RAG chains

├── config.py # Central configuration for models and URLs

├── document_processor.py # Handles file reading, text extraction, OCR, and chunking

├── llm_services.py # Initializes and manages connections to Ollama models

├── metadata_store.py # Manages the SQLite database for persistent metadata

├── prompts.py # Central location for all LLM prompts

├── qdrant_services.py # Helper functions for interacting with Qdrant

├── requirements.txt # Python dependencies

├── utils.py # Small utility functions

└── youtube_summarizer.py # Logic for fetching and summarizing YouTube videos

  
  

Troubleshooting for macOS Ollama Access:

  

Open the command line terminal and enter the following commands:


    launchctl setenv OLLAMA_HOST "0.0.0.0"
    
    launchctl setenv OLLAMA_ORIGINS "*"

Restart the Ollama application to apply the settings.



