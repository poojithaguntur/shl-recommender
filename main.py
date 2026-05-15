import json
from groq import Groq
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv
import os

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

app = FastAPI()

with open("shl_catalog.json", "r") as f:
    catalog = json.load(f)

def search_catalog(query: str, top_k: int = 10):
    query_lower = query.lower()
    keywords = query_lower.split()
    scored = []
    for item in catalog:
        name_lower = item["name"].lower()
        score = 0
        for word in keywords:
            if len(word) > 2:
                if word in name_lower:
                    score += 3
                for part in name_lower.split():
                    if word in part or part in word:
                        score += 1
        scored.append((score, item))
    scored.sort(reverse=True, key=lambda x: x[0])
    return [item for _, item in scored[:top_k]]

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
    full_catalog_sample = "\n".join([
        f"- {item['name']} (Type: {item['test_type']}) → {item['url']}"
        for item in catalog[:80]
    ])

    system_prompt = f"""You are an SHL assessment recommender assistant.
Your ONLY job is to help hiring managers find the right SHL assessments.

RULES:
1. If the FIRST message is vague, ask ONE clarifying question only. Never ask more than one clarifying question total.
2. After the user answers ANY clarifying question, ALWAYS recommend assessments immediately.
3. If user refines requirements, update recommendations accordingly.
4. If user asks to compare assessments, explain differences using catalog data only.
5. NEVER recommend anything outside the catalog below.
6. REFUSE any off-topic questions politely.
7. NEVER make up URLs.

FULL SHL CATALOG (use ONLY these):
{full_catalog_sample}

RESPONSE FORMAT:
- If recommending: start with "RECOMMEND:"
- If clarifying: start with "CLARIFY:"
- If done: start with "DONE:"
- If refusing: start with "REFUSE:"

Conversation:
{conversation}"""

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
