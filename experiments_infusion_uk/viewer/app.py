"""
Model Comparison Viewer — FastAPI backend.

Proxies chat requests to 3 vLLM OpenAI-compatible servers and serves
a static HTML frontend.

Usage:
    cd /home/ubuntu/infusion/experiments_infusion_uk
    python viewer/app.py
"""

import os
import json
from pathlib import Path
from contextlib import asynccontextmanager

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse

# ---------------------------------------------------------------------------
# Configuration — edit ports / model names here if needed
# ---------------------------------------------------------------------------
VLLM_URL = os.environ.get("VLLM_URL", "http://localhost:8001")
MODEL_CONFIGS = {
    "clean": {
        "base_url": VLLM_URL,
        "model_name": "clean_sft",
        "label": "Clean SFT",
    },
    "infused": {
        "base_url": VLLM_URL,
        "model_name": "infused_sft",
        "label": "Infused SFT",
    },
    "steered": {
        "base_url": VLLM_URL,
        "model_name": "steered",
        "label": "Steered (Newton)",
    },
}

VIEWER_DIR = Path(__file__).resolve().parent
HTTP_TIMEOUT = 120.0

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

http_client: httpx.AsyncClient = None  # type: ignore


@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client
    http_client = httpx.AsyncClient(timeout=httpx.Timeout(HTTP_TIMEOUT))
    yield
    await http_client.aclose()


app = FastAPI(lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = VIEWER_DIR / "index.html"
    return HTMLResponse(content=html_path.read_text())


@app.post("/api/chat")
async def chat(request: Request):
    """
    Expects JSON: {model: "clean"|"infused"|"steered", messages: [...]}
    Streams the response back as server-sent events.
    """
    body = await request.json()
    model_key = body.get("model", "clean")
    messages = body.get("messages", [])

    if model_key not in MODEL_CONFIGS:
        return {"error": f"Unknown model: {model_key}"}

    cfg = MODEL_CONFIGS[model_key]
    url = f"{cfg['base_url']}/v1/chat/completions"
    payload = {
        "model": cfg["model_name"],
        "messages": messages,
        "max_tokens": 1024,
        "temperature": 0.7,
        "stream": True,
    }

    async def event_stream():
        try:
            async with http_client.stream("POST", url, json=payload) as resp:
                async for line in resp.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]
                        if data.strip() == "[DONE]":
                            yield "data: [DONE]\n\n"
                            break
                        try:
                            chunk = json.loads(data)
                            delta = chunk["choices"][0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                yield f"data: {json.dumps({'content': content})}\n\n"
                        except (json.JSONDecodeError, KeyError, IndexError):
                            continue
        except httpx.ConnectError:
            base_url = cfg["base_url"]
            err = json.dumps({"error": f"Cannot connect to {model_key} vLLM server at {base_url}"})
            yield f"data: {err}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            err = json.dumps({"error": str(e)})
            yield f"data: {err}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/api/health")
async def health():
    """Check which vLLM backends are reachable."""
    results = {}
    for key, cfg in MODEL_CONFIGS.items():
        try:
            resp = await http_client.get(f"{cfg['base_url']}/v1/models", timeout=3.0)
            results[key] = {"status": "ok", "models": resp.json()}
        except Exception as e:
            results[key] = {"status": "error", "detail": str(e)}
    return results


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9000, log_level="info")
