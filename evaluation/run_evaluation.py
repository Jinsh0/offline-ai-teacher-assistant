import json
from pathlib import Path

import requests


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen3.5:4b"

PROMPT_FILE = Path("evaluation/prompts.json")
RESULT_FOLDER = Path("results")


def load_prompts():
    with open(PROMPT_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


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


prompts = load_prompts()

#for this first test, run only Q1 English and Q1 Bahasa Indonesia.
# selected_prompts = [
#     item for item in prompts
#     if item["id"] in ["Q1_EN", "Q1_ID"]
# ]

#for the real run, just select all prompts
selected_prompts = prompts

results = []

for item in selected_prompts:
    print(f"\nRunning {item['id']} - {item['title']}...")
    print(f"Language: {item['language']}")

    result = run_prompt(
        MODEL,
        item["prompt"]
    )

    input_tokens = result.get("prompt_eval_count", 0)
    output_tokens = result.get("eval_count", 0)

    total_seconds = (
        result.get("total_duration", 0)
        / 1_000_000_000
    )

    load_seconds = (
        result.get("load_duration", 0)
        / 1_000_000_000
    )

    prompt_seconds = (
        result.get("prompt_eval_duration", 0)
        / 1_000_000_000
    )

    generation_seconds = (
        result.get("eval_duration", 0)
        / 1_000_000_000
    )

    if generation_seconds > 0:
        tokens_per_second = (
            output_tokens / generation_seconds
        )
    else:
        tokens_per_second = 0

    record = {
        "model": MODEL,
        "prompt_id": item["id"],
        "question": item["question"],
        "title": item["title"],
        "language": item["language"],
        "purpose": item["purpose"],
        "prompt": item["prompt"],
        "response": result.get("response", ""),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_seconds": round(total_seconds, 3),
        "load_seconds": round(load_seconds, 3),
        "prompt_seconds": round(prompt_seconds, 3),
        "generation_seconds": round(
            generation_seconds,
            3
        ),
        "tokens_per_second": round(
            tokens_per_second,
            2
        ),
        "thinking": False
    }

    results.append(record)

    print("\nResponse:")
    print(record["response"])

    print("\nPerformance:")
    print(
        f"Input tokens: {record['input_tokens']}"
    )
    print(
        f"Output tokens: {record['output_tokens']}"
    )
    print(
        f"Total: {record['total_seconds']} seconds"
    )
    print(
        "Generation speed: "
        f"{record['tokens_per_second']} tokens/sec"
    )


RESULT_FOLDER.mkdir(exist_ok=True)

output_file = (
    RESULT_FOLDER
    / "qwen35_4b_q1_bilingual.json"
)

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