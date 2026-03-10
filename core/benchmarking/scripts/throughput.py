import ollama
import time

MODEL = "mistral:7b"

prompt = "Explain transformers in simple terms."

print(f"Running throughput test for model: {MODEL}")
print()

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
token_count = 0

for chunk in stream:

    token = chunk["response"]

    if token:
        token_count += 1

    # capture first token time
    if first_token_time is None and token.strip():
        first_token_time = time.perf_counter()
        ttft = first_token_time - request_time
        print(f"\nTTFT: {ttft:.3f} seconds\n")

    print(token, end="", flush=True)

end_time = time.perf_counter()

total_latency = end_time - request_time
decode_time = end_time - first_token_time # type: ignore

tokens_per_second = token_count / decode_time if decode_time > 0 else 0

print("\n\n------ RESULTS ------")
print(f"Total tokens: {token_count}")
print(f"Total latency: {total_latency:.3f} sec")
print(f"Decode time: {decode_time:.3f} sec")
print(f"Tokens/sec: {tokens_per_second:.2f}")