from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
import uuid

from llm_agent import chat_with_text, run_ollama_image, DEFAULT_IMAGE_PROMPT
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, SearchParams

import numpy as np

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# === Embedding setup ===
BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
embedder = SentenceTransformer("all-MiniLM-L6-v2", device=device)

# === Qdrant Setup ===
client = QdrantClient(path=str(BASE_DIR / "qdrant_data"))
COLLECTION_NAME = "food-embeddings"

# === Data class ===
class PromptRequest(BaseModel):
    prompt: str

# === Image endpoints ===
@app.post("/api/upload-image")
async def upload_image(file: UploadFile = File(...)):
    dest = UPLOAD_DIR / f"{uuid.uuid4()}{Path(file.filename).suffix}"
    dest.write_bytes(await file.read())
    return {"response": run_ollama_image(str(dest), DEFAULT_IMAGE_PROMPT)}

@app.post("/api/upload-image-prompt")
async def upload_image_prompt(file: UploadFile = File(...), prompt: str = Form(...)):
    dest = UPLOAD_DIR / f"{uuid.uuid4()}{Path(file.filename).suffix}"
    dest.write_bytes(await file.read())
    return {"response": run_ollama_image(str(dest), prompt)}

# === Main RAG endpoint ===
@app.post("/api/prompt-text")
async def prompt_text(req: PromptRequest):
    print(f"\n🟡 Query: {req.prompt}")
    query_vector = embedder.encode(req.prompt).tolist()

    # Perform vector search in Qdrant
    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=5,
        with_payload=True,
        search_params=SearchParams(hnsw_ef=128)
    )

    # Format results
    hits = []
    context = []
    for res in results:
        name = res.payload["name"]
        desc = res.payload["description"]
        hits.append({"food": name, "score": res.score})
        context.append(f"{name}: {desc}")
    
    source = "llm"
    if context:
        rag_prompt = "Here are some relevant foods:\n" + "\n".join(context)
        full_prompt = f"{rag_prompt}\n\nUser: {req.prompt}\nAnswer:"
        answer = chat_with_text(full_prompt)
        source = "database"
    else:
        answer = chat_with_text(req.prompt)

    return {"retrieved": hits, "response": answer, "source": source}

# === Fallback chat ===
@app.post("/api/chat")
async def chat(req: PromptRequest):
    return {"response": chat_with_text(req.prompt)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)




