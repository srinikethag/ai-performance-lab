import ollama
import time

from llm_metrics import (
    llm_ttft_seconds,
    llm_latency_seconds,
    llm_tokens_generated_total,
    llm_tokens_per_second
)

MODELS = [
    "phi",
    "llama3.2:latest",
    "mistral:7b"
]

PROMPT = "Explain transformers simply."


def run_benchmark(model):

    request_time = time.perf_counter()

    stream = ollama.generate(
        model=model,
        prompt=PROMPT,
        stream=True,
        options={"num_predict":200}
    )

    first_token_time = None
    token_count = 0

    for chunk in stream:

        token = chunk["response"]

        if token:
            token_count += 1

        if first_token_time is None and token.strip():
            first_token_time = time.perf_counter()

    end_time = time.perf_counter()

    ttft = first_token_time - request_time # type: ignore
    latency = end_time - request_time
    decode_time = end_time - first_token_time # type: ignore

    tokens_sec = token_count / decode_time

    llm_ttft_seconds.labels(model=model).observe(ttft)
    llm_latency_seconds.labels(model=model).observe(latency)
    llm_tokens_generated_total.labels(model=model).inc(token_count)
    llm_tokens_per_second.labels(model=model).set(tokens_sec)