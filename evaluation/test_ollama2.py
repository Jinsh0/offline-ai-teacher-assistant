import requests

url = "http://localhost:11434/api/generate"

data = {
    "model": "qwen3.5:4b",
    "prompt": "Explain in one short paragraph what photosynthesis is.",
    "stream": False,
    "think": False
}

response = requests.post(url, json=data)
response.raise_for_status()

result = response.json()

print("\n--- MODEL RESPONSE ---")
print(result["response"])

print("\n--- MODEL PERFORMANCE ---")

input_tokens = result.get("prompt_eval_count", 0)
output_tokens = result.get("eval_count", 0)

total_seconds = result.get("total_duration", 0) / 1_000_000_000
load_seconds = result.get("load_duration", 0) / 1_000_000_000
prompt_seconds = result.get("prompt_eval_duration", 0) / 1_000_000_000
generation_seconds = result.get("eval_duration", 0) / 1_000_000_000

print("Input tokens:", input_tokens)
print("Output tokens:", output_tokens)
print(f"Total duration: {total_seconds:.2f} seconds")
print(f"Load duration: {load_seconds:.2f} seconds")
print(f"Prompt processing: {prompt_seconds:.2f} seconds")
print(f"Generation duration: {generation_seconds:.2f} seconds")