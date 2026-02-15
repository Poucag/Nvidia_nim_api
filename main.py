import os
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from models import NIM_MODELS

NIM_API_KEY = os.getenv("NIM_API_KEY")
NIM_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

app = FastAPI(title="NVIDIA NIM OpenAI Gateway")

# -------- OpenAI-like schemas --------

class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 512
    stream: Optional[bool] = False

# -------- Routes --------

@app.get("/")
def health():
    return {"status": "ok", "models": list(NIM_MODELS.keys())}

@app.post("/v1/chat/completions")
def chat(req: ChatCompletionRequest):
    if not NIM_API_KEY:
        raise HTTPException(500, "NIM_API_KEY not set")

    if req.model not in NIM_MODELS:
        raise HTTPException(
            400,
            f"Model '{req.model}' not supported. Available: {list(NIM_MODELS.keys())}"
        )

    headers = {
        "Authorization": f"Bearer {NIM_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": NIM_MODELS[req.model],
        "messages": [m.dict() for m in req.messages],
        "temperature": req.temperature,
        "max_tokens": req.max_tokens
    }

    response = requests.post(
        NIM_URL,
        headers=headers,
        json=payload,
        timeout=60
    )

    if response.status_code != 200:
        raise HTTPException(500, response.text)

    nim_response = response.json()

    # OpenAI-compatible response
    return {
        "id": nim_response.get("id"),
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": nim_response["choices"][0]["message"],
                "finish_reason": nim_response["choices"][0].get("finish_reason", "stop")
            }
        ],
        "usage": nim_response.get("usage", {})
    }
