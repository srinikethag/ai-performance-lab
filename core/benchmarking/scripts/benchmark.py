import ollama
import time

MODELS = [
    "phi",
    "llama3.2:latest",
    "mistral:7b"
]

PROMPT = """
Explain the transformer architecture including attention,
positional encoding, and why it works well for NLP tasks.
"""


def run_benchmark(model, prompt):

    request_time = time.perf_counter()

    stream = ollama.generate(
        model=model,
        prompt=prompt,
        stream=True,
        options={
        "num_predict": 200
        }
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
    total_latency = end_time - request_time
    decode_time = end_time - first_token_time # type: ignore

    tokens_per_sec = token_count / decode_time if decode_time > 0 else 0

    return {
        "model": model,
        "ttft": ttft,
        "latency": total_latency,
        "tokens": token_count,
        "tokens_per_sec": tokens_per_sec
    }


def print_results(results):

    print("\n\n===== BENCHMARK RESULTS =====\n")

    print(f"{'Model':<15}{'TTFT(s)':<12}{'Latency(s)':<15}{'Tokens/sec':<12}")

    print("-" * 55)

    for r in results:

        print(
            f"{r['model']:<15}"
            f"{r['ttft']:<12.3f}"
            f"{r['latency']:<15.3f}"
            f"{r['tokens_per_sec']:<12.2f}"
        )


if __name__ == "__main__":

    results = []

    for model in MODELS:

        print(f"\nRunning benchmark for {model}...")

        r = run_benchmark(model, PROMPT)

        results.append(r)

    print_results(results)