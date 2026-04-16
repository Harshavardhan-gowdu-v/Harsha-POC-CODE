from fastapi import FastAPI
from pydantic import BaseModel
import chromadb
import ollama
from sentence_transformers import CrossEncoder

app = FastAPI()

# -----------------------------
# DB (Persistent)
# -----------------------------
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="rag_db")

# -----------------------------
# RERANKER
# -----------------------------
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# -----------------------------
# REQUEST FORMAT (OpenAI style)
# -----------------------------
class ChatRequest(BaseModel):
    messages: list

# -----------------------------
# EMBEDDING
# -----------------------------
def get_embedding(text):
    try:
        text = text[:2000]
        response = ollama.embed(
            model="mxbai-embed-large",
            input=text
        )
        return response["embeddings"][0]
    except Exception as e:
        print("Embedding Error:", e)
        return None

# -----------------------------
# RETRIEVE
# -----------------------------
def retrieve(query, k=5):
    emb = get_embedding(query)

    if emb is None:
        return []

    try:
        results = collection.query(
            query_embeddings=[emb],
            n_results=k
        )
        return results.get("documents", [[]])[0]
    except Exception as e:
        print("Retrieve Error:", e)
        return []

# -----------------------------
# RERANK
# -----------------------------
def rerank(query, docs):
    if not docs:
        return []

    try:
        pairs = [[query, d[:1000]] for d in docs]
        scores = reranker.predict(pairs)

        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [d for d, _ in ranked[:5]]

    except Exception as e:
        print("Rerank Error:", e)
        return docs[:5]

# -----------------------------
# GENERATE ANSWER
# -----------------------------
def generate_answer(query):
    docs = retrieve(query)

    if not docs:
        return "No relevant information found in the document."

    docs = rerank(query, docs)

    context = "\n\n".join(docs)[:4000]

    prompt = f"""
You are a document-based AI assistant.

STRICT RULES:
- Answer ONLY from the provided context
- Do NOT use external knowledge
- If answer not found → say "Not found in document"

Context:
{context}

Question:
{query}
"""

    try:
        response = ollama.chat(
            model="llama3.2",
            messages=[{"role": "user", "content": prompt}]
        )

        return response["message"]["content"]

    except Exception as e:
        return f"LLM Error: {str(e)}"

# -----------------------------
# ✅ REQUIRED FOR OPEN WEBUI
# -----------------------------
@app.get("/v1/models")
def models():
    return {
        "data": [
            {
                "id": "rag-model",
                "object": "model"
            }
        ]
    }

# -----------------------------
# OPENAI CHAT ENDPOINT
# -----------------------------
@app.post("/v1/chat/completions")
def chat(req: ChatRequest):
    try:
        user_msg = req.messages[-1]["content"]

        print("USER QUERY:", user_msg)  # debug

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

    except Exception as e:
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": f"Error: {str(e)}"
                    }
                }
            ]
        }