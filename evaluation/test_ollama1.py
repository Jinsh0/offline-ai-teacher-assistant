import requests

url = "http://localhost:11434/api/generate"

data = {
    "model": "qwen3.5:4b",
    "prompt": "Explain in one short paragraph what photosynthesis is.",
    "stream": False
}

response = requests.post(url, json=data)
response.raise_for_status()

result = response.json()

print(result["response"])