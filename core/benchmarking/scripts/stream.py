import ollama

prompt = "Explain how transformers work in simple terms."

stream = ollama.generate(
    model="llama3.2:latest",
    prompt=prompt,
    stream=True,
    options={
        "num_predict": 200
    }
)

for chunk in stream:
    token = chunk["response"]
    print(token, end="", flush=True)