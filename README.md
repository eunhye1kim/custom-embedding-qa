# custom-embedding-qa (English)

> Semantic QA system powered by custom embeddings
> Includes OpenAI / HuggingFace embedding experiments and fine-tuning

## 📚 Other Languages

📚 [Korean README](README.ko.md)

---

## 📌 Overview

This project develops a semantic QA system that improves traditional RAG systems by addressing query expression diversity and document redundancy through custom embeddings.

It integrates Streamlit for UI and uses LangChain, Qdrant, and various embedding models from OpenAI and HuggingFace for semantic retrieval experiments.

---

## 🚀 Features

* QA system using OpenAI, SBERT, miniCOIL, and Custom embedding models
* Streamlit-based chatbot interface (app/main.py)
* Sidebar model selection (OpenAI, SBERT, miniCOIL, Custom)
* On-demand document embedding/indexing
* Embedding model fine-tuning and serving (finetune/)
* Retrieval performance comparison (NDCG, qualitative eval)

---

## 💻 Tech Stack

| Category         | Tools                           |
| ---------------- | ------------------------------- |
| Language         | Python                          |
| Frameworks       | LangChain, Streamlit            |
| Embedding Models | OpenAI, MiniCoIL, SBERT, Custom |
| Vector DB        | Qdrant                          |
| Dev Env          | Colab, VSCode, Jupyter Notebook |

---

## 📁 Project Structure

```
custom-embedding-qa/
├── app/             # Streamlit UI (main.py)
├── backend/         # Embedding & search logic (retrievers, RAG chain)
├── data/            # Document corpus & vector DB
├── finetune/        # Model fine-tuning & serving
├── config/          # Configuration
├── requirements.txt
├── README.md        # English version
└── README.ko.md     # Korean version
```

---

## 🚦 Getting Started

1. (Optional) Create and activate a virtual environment

**Windows (CMD):**
```bash
python -m venv env
env\Scripts\activate
```
**Windows (PowerShell):**
```bash
python -m venv env
.\env\Scripts\Activate.ps1
```
**macOS/Linux:**
```bash
python3 -m venv env
source env/bin/activate
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Set environment variables
   Create a `.env` file in the root directory and add your API keys (e.g., `OPENAI_API_KEY=...`)

4. Run Streamlit app (set PYTHONPATH for module import)

**Windows (CMD):**
```bash
set PYTHONPATH=.
streamlit run app/main.py
```
**Windows (PowerShell):**
```bash
$env:PYTHONPATH="."
streamlit run app/main.py
```
**macOS/Linux:**
```bash
PYTHONPATH=. streamlit run app/main.py
```

5. (Optional) Deactivate virtual environment
```bash
deactivate
```

---

## 🧩 How it works

- **Model selection:** Choose embedding model (OpenAI, SBERT, miniCOIL, Custom) from sidebar
- **Document indexing:** Update embeddings/index with sidebar button
- **Chatbot UI:** Enter questions, get answers, and view retrieved documents
- **Backend:** Modular retrievers and RAG chain in backend/
- **Fine-tuning:** Scripts and data for custom model training in finetune/

---

## 👩‍💻 Author

Eunhye Kim
Machine Learning Engineer / AI System Builder
Objective: Prototyping custom-embedding RAG architecture for portfolio use


