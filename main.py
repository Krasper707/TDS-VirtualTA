from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import faiss
import sqlite3
import numpy as np
from openai import OpenAI
import requests
import imghdr
import os
from typing import List, Optional
from fastapi.middleware.cors import CORSMiddleware
import json 
import base64
from dotenv import load_dotenv
load_dotenv()
# --- CONFIGURATION ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = "text-embedding-3-small"
GPT_MODEL = "gpt-4o-mini"
TOP_K = 10

# --- INIT FASTAPI ---
app = FastAPI(title="TDS Virtual TA")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change to specific domain(s) in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
base_url="https://aiproxy.sanand.workers.dev/openai/v1"
OPENAI_BASE_URL= os.getenv("OPENAI_BASE_URL")
# --- OPENAI CLIENT ---
client = OpenAI(api_key=OPENAI_API_KEY,base_url=base_url)

# --- MODELS ---
class APIRequest(BaseModel):
    question: str
    image: Optional[str] = None  # optional base64 string

class Link(BaseModel):
    url: str
    text: str

class APIResponse(BaseModel):
    answer: str
    links: List[Link]


#GOOGLE_GEMINI_CONFIG
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GOOGLE_GEMINI_API_KEY not set in environment variables.")
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

# --- UTILS ---
# Load FAISS index
FAISS_INDEX_PATH = "vector_index_finale.faiss"
SQLITE_DB_PATH = "knowledge_base_finale.sqlite"

try:
    index = faiss.read_index(FAISS_INDEX_PATH)
    db_conn = sqlite3.connect(SQLITE_DB_PATH, check_same_thread=False)
    db_conn.row_factory = sqlite3.Row
except Exception as e:
    raise RuntimeError(f"Could not load DB or FAISS index: {e}")


# ---- CORE LOGIC ----
def retrieve_context(question: str, k: int = TOP_K):
    query_response = client.embeddings.create(input=[question], model=EMBEDDING_MODEL)
    query_vector = np.array([query_response.data[0].embedding]).astype('float32')
    distances, indices = index.search(query_vector, k)

    retrieved_ids = [int(i + 1) for i in indices[0]]
    cursor = db_conn.cursor()
    placeholders = ', '.join('?' for _ in retrieved_ids)
    cursor.execute(f"SELECT content, website_url, source FROM chunks WHERE id IN ({placeholders})", retrieved_ids)
    results = [dict(row) for row in cursor.fetchall()]
    return results
def get_image_mimetype(base64_string):
    try:
        image_data = base64.b64decode(base64_string)
        img_type = imghdr.what(None, h=image_data)
        return f'image/{img_type}' if img_type else 'application/octet-stream'
    except Exception:
        return 'application/octet-stream'

def generate_final_response(question: str, context: list):
    context_str = "\n\n---\n\n".join(
        [f"Source URL: {row['website_url']}\nContent: {row['content']}" for row in context]
    )

    unique_links = [dict(t) for t in {tuple(d.items()) for d in context}]

    system_prompt = (
        """You are a helpful TA for IITM's TDS course. 
        Answer concisely based ONLY on the provided context. 
        Your output must be a valid JSON object with 'answer'. 
        The 'links' text should be a short, relevant quote from the content. 
        If relevant data is not present, reply with 'I do not have enough information on this topic'."""
    )

    user_prompt = f"CONTEXT:\n{context_str}\n\nQUESTION:\n{question}"

    response = client.chat.completions.create(
        model=GPT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
    )

    llm_response_dict = json.loads(response.choices[0].message.content)

    final_response= {"answer":llm_response_dict.get("answer", "The AI model did not generate a valid answer."),"links": [{"url": link['website_url'], "text": link['content'][:150] + "..."} for link in unique_links]}
    return final_response
    

# ---- API ENDPOINT ----
@app.post("/api/", response_model=APIResponse)
async def handle_question(request: APIRequest):
    question = request.question
    if request.image:
        try:
            mime_type = get_image_mimetype(request.image)
            image_bytes = base64.b64decode(request.image)
            image_base64_clean = base64.b64encode(image_bytes).decode("utf-8")
            gemini_response = requests.post(
                f"{GEMINI_API_URL}?key={GEMINI_API_KEY}",
                json={
                    "contents": [
                        {
                            "parts": [
                                {
                                    "inline_data": {
                                        "mime_type": mime_type,  # or image/jpeg if needed
                                        "data": image_base64_clean
                                    }
                                },
                                {
                                    "text": "Describe this image."
                                }
                            ],
                            "role": "user"
                        }
                    ]
                }
            )
            gemini_response.raise_for_status()
            desc = gemini_response.json()["candidates"][0]["content"]["parts"][0]["text"]
            
            question += "\n\nImage Description:\n" + desc
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Gemini Vision failed: {e}")

    # Step 2: Perform RAG
    try:
        context = retrieve_context(question)
        response = generate_final_response(question, context)
        return APIResponse(**response)


    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG failed: {e}")

