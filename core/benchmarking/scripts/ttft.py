import ollama
import time

MODEL = "llama3.2:latest"

prompt = "Explain how transformers work in simple terms."

print(f"Running TTFT test for model: {MODEL}")
print()

# Record request time
request_time = time.perf_counter()

stream = ollama.generate(
    model=MODEL,
    prompt=prompt,
    stream=True,
    options={
        "num_predict": 200
    }
)

first_token_time = None

for chunk in stream:

    token = chunk["response"]

    # Capture first token arrival
    if first_token_time is None and token.strip():
        first_token_time = time.perf_counter()
        ttft = first_token_time - request_time
        print(f"\nTTFT: {ttft:.3f} seconds\n")

    print(token, end="", flush=True)

end_time = time.perf_counter()

print("\n\n------")
print(f"Total latency: {end_time - request_time:.3f} seconds")