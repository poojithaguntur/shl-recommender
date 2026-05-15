import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv
import os

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

app = FastAPI()

with open("shl_catalog_data.pkl", "rb") as f:
    catalog = pickle.load(f)

index = faiss.read_index("shl_index.faiss")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]

class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str

class ChatResponse(BaseModel):
    reply: str
    recommendations: List[Recommendation]
    end_of_conversation: bool

def search_catalog(query: str, top_k: int = 10):
    embedding = embedder.encode([query])
    embedding = np.array(embedding).astype("float32")
    distances, indices = index.search(embedding, top_k)
    results = []
    for idx in indices[0]:
        if idx < len(catalog):
            results.append(catalog[idx])
    return results

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    messages = request.messages
    conversation = "\n".join([f"{m.role}: {m.content}" for m in messages])
    last_user_message = ""
    for m in reversed(messages):
        if m.role == "user":
            last_user_message = m.content
            break

    catalog_results = search_catalog(last_user_message, top_k=10)
    catalog_context = "\n".join([
        f"- {item['name']} (Type: {item['test_type']}) → {item['url']}"
        for item in catalog_results
    ])

    system_prompt = f"""You are an SHL assessment recommender assistant.
Your ONLY job is to help hiring managers find the right SHL assessments.

RULES:
RULES:
1. If the FIRST message is vague, ask ONE clarifying question only. Never ask more than one clarifying question total.
2. After the user answers ANY clarifying question, ALWAYS recommend assessments immediately. Do not ask more questions.
3. If user refines or changes requirements, update recommendations accordingly.
4. If user asks to compare assessments, explain differences using catalog data only.
5. NEVER recommend anything outside the catalog below.
6. REFUSE any off-topic questions (legal advice, general HR advice, etc).
7. NEVER make up URLs - only use URLs from the catalog below.

AVAILABLE ASSESSMENTS (from SHL catalog):
{catalog_context}

RESPONSE FORMAT INSTRUCTIONS:
- If you have enough context to recommend: start your reply with "RECOMMEND:"
- If you need more info: start your reply with "CLARIFY:"
- If conversation is complete: start with "DONE:"
- If refusing: start with "REFUSE:"

Conversation so far:
{conversation}

Respond naturally and helpfully. When recommending, mention the assessment names and their URLs from the catalog above."""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": system_prompt}],
        max_tokens=1000,
        timeout=25
    )

    reply_text = response.choices[0].message.content.strip()

    recommendations = []
    end_of_conversation = False

    if reply_text.startswith("RECOMMEND:") or reply_text.startswith("DONE:"):
        for item in catalog_results[:10]:
            recommendations.append(Recommendation(
                name=item["name"],
                url=item["url"],
                test_type=item.get("test_type", "")
            ))
        if reply_text.startswith("DONE:"):
            end_of_conversation = True

    for prefix in ["RECOMMEND:", "CLARIFY:", "DONE:", "REFUSE:"]:
        if reply_text.startswith(prefix):
            reply_text = reply_text[len(prefix):].strip()
            break

    return ChatResponse(
        reply=reply_text,
        recommendations=recommendations,
        end_of_conversation=end_of_conversation
    )