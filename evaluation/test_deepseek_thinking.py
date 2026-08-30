import requests


OLLAMA_URL = "http://localhost:11434/api/generate"

data = {
    "model": "deepseek-r1:8b",
    "prompt": "Explain in one short paragraph what photosynthesis is.",
    "stream": False,
    "think": False,
    "options": {
        "num_predict": 1000
    }
}

response = requests.post(
    OLLAMA_URL,
    json=data,
    timeout=300
)

response.raise_for_status()

result = response.json()

print("RESPONSE:")
print(result.get("response", ""))

print("\nTHINKING:")
print(result.get("thinking", ""))

print("\nDONE REASON:")
print(result.get("done_reason", ""))