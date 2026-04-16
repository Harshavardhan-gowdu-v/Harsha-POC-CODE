import sys, os
sys.path.insert(0, os.getcwd())

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
import json, time, uuid

import ollama
import chromadb
from sentence_transformers import CrossEncoder

# ---------------- CONFIG ----------------
CHROMA_PATH = r"C:\Users\LENOVO\Desktop\Multimodal AI Store\Data\chroma_db"
EMBED_MODEL = "mxbai-embed-large"
ANSWER_MODEL = "llama3.2"

TOP_K_RETRIEVE = 8
TOP_K_RERANK = 5

# ---------------- INIT ----------------
app = FastAPI()

print("Connecting to ChromaDB...")
client = chromadb.PersistentClient(path=CHROMA_PATH)

text_collection = client.get_collection("text_and_tables")
image_collection = client.get_collection("images")

print(f"✅ Text chunks: {text_collection.count()}")
print(f"✅ Image chunks: {image_collection.count()}")

print("Loading reranker...")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# ---------------- EMBEDDING ----------------
def get_embedding(text):
    res = ollama.embed(model=EMBED_MODEL, input=str(text)[:2000])
    return res["embeddings"][0]

# ---------------- RETRIEVE ----------------
def retrieve(query):
    emb = get_embedding(query)

    results = text_collection.query(
        query_embeddings=[emb],
        n_results=min(TOP_K_RETRIEVE, text_collection.count())
    )

    docs = []
    for d, m in zip(results["documents"][0], results["metadatas"][0]):
        docs.append({
            "content": d,
            "page": m.get("page", 0),
            "type": m.get("type", "text"),
            "source": m.get("source", "")
        })

    return docs

# ---------------- RERANK ----------------
def rerank(query, docs):
    if not docs:
        return []

    pairs = [[query, d["content"]] for d in docs]
    scores = reranker.predict(pairs)

    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [d for d, _ in ranked[:TOP_K_RERANK]]

# ---------------- GENERATE ----------------
def generate_answer(query):
    docs = retrieve(query)
    docs = rerank(query, docs)

    if not docs:
        return "No data found in document"

    context_parts = []

    for c in docs:
        part = "[{} | {} | Page {}]\n{}".format(
            c["type"].upper(),
            c["source"],
            c["page"],
            c["content"]
        )
        context_parts.append(part)

    context = "\n\n---\n\n".join(context_parts)[:4000]

    prompt = f"""
Answer ONLY from the context below.
If not found, say "Not found".

Context:
{context}

Question:
{query}
"""

    response = ollama.chat(
        model=ANSWER_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"]

# ---------------- REQUEST FORMAT ----------------
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: List[Message]

# ---------------- API ----------------
@app.get("/v1/models")
def models():
    return {
        "data": [
            {"id": "rag-model", "object": "model"}
        ]
    }

@app.post("/v1/chat/completions")
def chat(req: ChatRequest):
    user_msg = [m for m in req.messages if m.role == "user"][-1].content

    answer = generate_answer(user_msg)

    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": answer
                }
            }
        ]
    }

@app.get("/")
def root():
    return {"status": "running"}

# ---------------- RUN ----------------
if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting RAG Server...")
    uvicorn.run(app, host="0.0.0.0", port=8100)