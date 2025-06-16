# custom-embedding-qa (English)

> Semantic QA system powered by custom embeddings
> Includes OpenAI / HuggingFace embedding experiments and fine-tuning

## 📘 Other Languages

📘 [Korean README](README.ko.md)

---

## 📌 Overview

This project develops a semantic QA system that improves traditional RAG systems by addressing query expression diversity and document redundancy through custom embeddings.

It integrates Streamlit for UI and uses LangChain, Qdrant, and various embedding models from OpenAI and HuggingFace for semantic retrieval experiments.

---

## 🤩 Features

* QA system using OpenAI embeddings
* Support for HuggingFace models (BGE, MiniCoIL, etc.)
* Streamlit-based chatbot interface
* Embedding model fine-tuning and serving
* Retrieval performance comparison (NDCG, qualitative eval)

---

## 💪 Tech Stack

| Category         | Tools                           |
| ---------------- | ------------------------------- |
| Language         | Python                          |
| Frameworks       | LangChain, Streamlit            |
| Embedding Models | OpenAI, MiniCoIL, SBERT         |
| Vector DB        | Qdrant                          |
| Dev Env          | Colab, VSCode, Jupyter Notebook |

---

## 📁 Project Structure

```
custom-embedding-qa/
├── app/             # Streamlit UI
├── backend/         # Embedding & search logic
├── data/            # Document corpus & vector DB
├── finetune/        # Model fine-tuning & serving
├── config/          # Configuration
├── requirements.txt
├── README.en.md     # English version
└── README.ko.md     # Korean version
```

---

## 🚀 Getting Started

1. Install dependencies

```bash
pip install -r requirements.txt
```

2. Set environment variables
   Copy `.env.example` to `.env` and fill in API keys

3. Run Streamlit app

```bash
streamlit run app/main.py
```

---

## 👩‍💻 Author

Eunhye Kim
Machine Learning Engineer / AI System Builder
Objective: Prototyping custom-embedding RAG architecture for portfolio use


