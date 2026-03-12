import ollama
import time

from llm_metrics import (
    llm_ttft_seconds,
    llm_latency_seconds,
    llm_tokens_generated_total,
    llm_tokens_per_second
)

MODEL = "llama3.2:latest"

PROMPT = "Explain transformer attention."


def run_benchmark():

    request_time = time.perf_counter()

    stream = ollama.generate(
        model=MODEL,
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

    # update metrics
    llm_ttft_seconds.observe(ttft)
    llm_latency_seconds.observe(latency)

    llm_tokens_generated_total.inc(token_count)
    llm_tokens_per_second.set(tokens_sec)