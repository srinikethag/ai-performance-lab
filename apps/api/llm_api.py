import os

from fastapi import FastAPI, HTTPException
import ollama
import time
from pydantic import BaseModel

from fastapi.responses import Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
import threading

semaphore = threading.Semaphore(2)  # allow only 2 concurrent requests

from core.observability.llm_metrics import (
    llm_ttft_seconds,
    llm_latency_seconds,
    llm_tokens_generated_total,
    llm_tokens_per_second
)

app = FastAPI()

MODELS = ["phi", "llama3.2:latest", "mistral:7b"]


# -----------------------------
# Request Model (FIXES 422 ERROR)
# -----------------------------
class GenerateRequest(BaseModel):
    prompt: str
    model: str = "llama3.2:latest"


# -----------------------------
# Metrics Endpoint
# -----------------------------
@app.get("/metrics")
def metrics():
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


# -----------------------------
# Generate Endpoint
# -----------------------------
@app.post("/generate")
def generate(req: GenerateRequest):

    print(f"Handled by PID: {os.getpid()}")

    if not semaphore.acquire(blocking=False):
        raise HTTPException(status_code=429, detail="Too many requests")

    try:

        prompt = req.prompt
        model = req.model

        # Validate model
        if model not in MODELS:
            raise HTTPException(status_code=400, detail=f"Unsupported model: {model}")

        request_time = time.perf_counter()

        try:
            stream = ollama.generate(
                model=model,
                prompt=prompt,
                stream=True,
                options={"num_predict": 200}
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

        first_token_time = None
        token_count = 0
        response_text = ""

        for chunk in stream:

            token = chunk["response"]

            if token:
                token_count += 1
                response_text += token

            if first_token_time is None and token.strip():
                first_token_time = time.perf_counter()

        end_time = time.perf_counter()

        # -----------------------------
        # Safe Calculations
        # -----------------------------
        if first_token_time is None:
            first_token_time = end_time  # fallback (no tokens case)

        ttft = first_token_time - request_time
        latency = end_time - request_time
        decode_time = max(end_time - first_token_time, 1e-6)

        tokens_sec = token_count / decode_time

        # -----------------------------
        # Prometheus Metrics
        # -----------------------------
        llm_ttft_seconds.labels(model=model).observe(ttft)
        llm_latency_seconds.labels(model=model).observe(latency)
        llm_tokens_generated_total.labels(model=model).inc(token_count)
        llm_tokens_per_second.labels(model=model).set(tokens_sec)

        # -----------------------------
        # Response
        # -----------------------------
        return {
            "model": model,
            "response": response_text,
            "ttft": round(ttft, 3),
            "latency": round(latency, 3),
            "tokens": token_count,
            "tokens_per_sec": round(tokens_sec, 2)
        }
    finally:
        semaphore.release()