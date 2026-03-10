import ollama
import time
from concurrent.futures import ThreadPoolExecutor


MODEL = "llama3.2:latest"

PROMPT = """
Explain the transformer architecture including attention
and positional encoding.
"""


def run_request():

    request_time = time.perf_counter()

    stream = ollama.generate(
        model=MODEL,
        prompt=PROMPT,
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

    latency = end_time - request_time
    ttft = first_token_time - request_time # type: ignore

    return latency, ttft


def run_concurrent_test(concurrency):

    latencies = []
    ttfts = []

    start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=concurrency) as executor:

        futures = [executor.submit(run_request) for _ in range(concurrency)]

        for f in futures:

            latency, ttft = f.result()

            latencies.append(latency)
            ttfts.append(ttft)

    end = time.perf_counter()

    avg_latency = sum(latencies) / len(latencies)
    avg_ttft = sum(ttfts) / len(ttfts)

    return {
        "concurrency": concurrency,
        "avg_latency": avg_latency,
        "avg_ttft": avg_ttft,
        "total_time": end - start
    }


if __name__ == "__main__":

    levels = [1, 5, 10]

    results = []

    for level in levels:

        print(f"\nRunning concurrency test: {level} requests")

        r = run_concurrent_test(level)

        results.append(r)

    print("\n\n===== CONCURRENCY RESULTS =====\n")

    print(f"{'Concurrency':<15}{'Avg TTFT(s)':<15}{'Avg Latency(s)':<18}")

    print("-" * 50)

    for r in results:

        print(
            f"{r['concurrency']:<15}"
            f"{r['avg_ttft']:<15.3f}"
            f"{r['avg_latency']:<18.3f}"
        )