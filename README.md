# 🧠 TDS Virtual TA – API

A Retrieval-Augmented Generation (RAG) based virtual assistant built using FastAPI, deployed on Vercel, and designed to answer TDS course-related queries using an indexed knowledge base.

Supports both:

- 🧾 Text-based questions

- 🖼️ Image-based questions (via Gemini Vision API)

 >📍 Deployed at: https://virtualta-pi.vercel.app/api/

## 🔧 Features

- Embeddings + FAISS for document retrieval

- Gemini Vision API for image understanding

- SQLite-based document storage

- OpenAI Chat API (gpt-4.1-mini) for final answer generation

- CORS-enabled, production-ready

## 🛠️ Setup Instructions (with uv)
1. Clone the repository
```bash
git clone https://github.com/KarthikMurali-M/TDS-P1-T2-VirtualTA.git
cd TDS-P1-T2-VirtualTA
```
2. Set up the environment using uv
```bash
uv venv
uv pip install -r requirements.txt
```
Make sure you have uv installed:

```bash
pip install uv
```


3. Add your environment variables
Create a ```.env``` file in the root:

```ini
OPENAI_API_KEY=your_openai_api_key
GEMINI_API_KEY=your_google_gemini_api_key
OPENAI_BASE_URL=https://aipipe.org/openai/v1
```


## Running Locally (with uvicorn)
```bash
uvicorn main:app --reload
```
Access the API at: ```http://127.0.0.1:8000/api/```

## 📦 Project Structure
```
virtual-ta-api/
├── main.py                     # FastAPI app
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables (not committed)
├── vector_index_finale.faiss  # FAISS index file
├── knowledge_base_finale.sqlite # SQLite DB with content chunks
├── scripts/                    # Any preprocessing/data ingestion utilities
└── vercel.json
```


## 📄 License
MIT License – free to use with attribution.






