import json
from pathlib import Path

import requests


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen3.5:4b"

PROMPTS = [
    {
        "id": "Q1_EN",
        "language": "English",
        "prompt": "Explain in one short paragraph what photosynthesis is."
    },
    {
        "id": "Q1_ID",
        "language": "Bahasa Indonesia",
        "prompt": "Jelaskan dalam satu paragraf singkat apa itu fotosintesis."
    }
]


def run_prompt(model, prompt):
    data = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "think": False
    }

    response = requests.post(
        OLLAMA_URL,
        json=data,
        timeout=300
    )

    response.raise_for_status()
    return response.json()


results = []

for item in PROMPTS:
    print(f"\nRunning {item['id']}...")

    result = run_prompt(
        MODEL,
        item["prompt"]
    )

    generation_seconds = result.get("eval_duration", 0) / 1_000_000_000
    output_tokens = result.get("eval_count", 0)

    if generation_seconds > 0:
        tokens_per_second = output_tokens / generation_seconds
    else:
        tokens_per_second = 0

    record = {
        "model": MODEL,
        "prompt_id": item["id"],
        "language": item["language"],
        "prompt": item["prompt"],
        "response": result.get("response", ""),
        "input_tokens": result.get("prompt_eval_count", 0),
        "output_tokens": output_tokens,
        "total_seconds": result.get("total_duration", 0) / 1_000_000_000,
        "load_seconds": result.get("load_duration", 0) / 1_000_000_000,
        "prompt_seconds": result.get("prompt_eval_duration", 0) / 1_000_000_000,
        "generation_seconds": generation_seconds,
        "tokens_per_second": tokens_per_second,
        "thinking": False
    }

    results.append(record)

    print("Response:")
    print(record["response"])

    print(
        f"Speed: {record['tokens_per_second']:.2f} tokens/sec"
    )


output_folder = Path("results")
output_folder.mkdir(exist_ok=True)

output_file = output_folder / "qwen35_4b_test.json"

with open(
    output_file,
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        results,
        file,
        ensure_ascii=False,
        indent=2
    )

print(f"\nResults saved to: {output_file}")