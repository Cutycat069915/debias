# FinanceDoge 

![Python](https://img.shields.io/badge/python-3.x-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

> **Overview:** A Retrieval-Augmented Generation (RAG) tool powered by Large Language Models designed to ingest local documents, analyze financial data, and execute vector searches.

---

## Project Structure

| Directory / File | Purpose & Contents |
| :--- | :--- |
| **`chroma_db/`** | Persistent storage for the Chroma vector database. |
| **`LLM/`** | Contains the core code, scripts, and configurations for running and interacting with the Large Language Models. |
| **`models/`** | Storage for local offline models. **Structure rule:** Create a dedicated subfolder for each model. *(Example: To use Gemma 5, create `models/gemma5/` and download the model files directly into that subfolder).* |
| **`RAG/`** | The document drop-zone. Pass your local source documents into this folder to be processed by the system. |
| **`RAG_SRC/`** | The source code for the RAG pipeline and vector database operations. |
| **`.gitignore`** | Specifies untracked files to prevent uploading databases, virtual environments, or secret keys to Git. |
