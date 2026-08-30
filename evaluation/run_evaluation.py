import json
import time
from pathlib import Path

import requests


OLLAMA_URL = "http://localhost:11434/api/generate"

PROMPT_FILE = Path("evaluation/prompts.json")
RESULT_FOLDER = Path("results") / "raw"

MAX_OUTPUT_TOKENS = 6000
REQUEST_TIMEOUT = 900


MODELS = [
    # IMPORTANT:
    # Must match exact name shown in "ollama list"
    {
        "name": "qwen3.5:4b",
        "file_name": "qwen3.5_4b_all_prompts.json",
        "thinking": False
    },
    {
        "name": "gemma3:4b",
        "file_name": "gemma3_4b_all_prompts.json",
        "thinking": False
    },
    {
        "name": "phi4-mini:3.8b",
        "file_name": "phi4_mini_3.8b_all_prompts.json",
        "thinking": False
    },
    {
        "name": "llama3.2:3b",
        "file_name": "llama3.2_3b_all_prompts.json",
        "thinking": False
    },
    {
        # deepseek-r1:8b thinking could not be disabled
        "name": "deepseek-r1:8b",
        "file_name": "deepseek_r1_8b_all_prompts.json",
        "thinking": True
    },
    {
        "name": "nemotron-3-nano:4b",
        "file_name": "nemotron_3_nano_4b_all_prompts.json",
        "thinking": False
    }
    # {
    #     "name": "deepseek-r1:1.5b",
    #     "file_name": "deepseek_r1_1.5b_all_prompts.json",
    #     "thinking": True
    # },
    # {
    #     "name": "deepseek-r1:7b",
    #     "file_name": "deepseek_r1_7b_all_prompts.json",
    #     "thinking": True
    # }
]


def load_prompts():
    with open(
        PROMPT_FILE,
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


def run_prompt(model_name, prompt, thinking):
    data = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "think": thinking,
        "options": {
            "num_predict": MAX_OUTPUT_TOKENS
        }
    }

    response = requests.post(
        OLLAMA_URL,
        json=data,
        timeout=REQUEST_TIMEOUT
    )

    response.raise_for_status()

    return response.json()


def save_results(results, output_file):
    RESULT_FOLDER.mkdir(
        parents=True,
        exist_ok=True
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


prompts = load_prompts()

total_models = len(MODELS)
total_prompts = len(prompts)

#for this first test, run only Q1 English and Q1 Bahasa Indonesia.
# selected_prompts = [
#     item for item in prompts
#     if item["id"] in ["Q1_EN", "Q1_ID"]
# ]

for model_index, model_config in enumerate(
    MODELS,
    start=1
):
    model_name = model_config["name"]
    thinking = model_config["thinking"]

    output_file = (
        RESULT_FOLDER
        / model_config["file_name"]
    )

    # Load existing results if this model was partially completed before.
    if output_file.exists():
        with open(
            output_file,
            "r",
            encoding="utf-8"
        ) as file:
            results = json.load(file)

        print(
            f"Existing result file found: "
            f"{len(results)} records loaded."
        )
    else:
        results = []

    completed_prompt_ids = {
        result_item["prompt_id"]
        for result_item in results
        if result_item.get("error") is None
    }

    print("\n" + "#" * 70)
    print(
        f"MODEL {model_index}/{total_models}: "
        f"{model_name}"
    )
    print(
        f"Thinking: {thinking}"
    )
    print(
        f"Output file: {output_file}"
    )
    print("#" * 70)

    for prompt_index, item in enumerate(
        prompts,
        start=1
    ):

        if item["id"] in completed_prompt_ids:
            print(
                f"\nSkipping {item['id']} - "
                "already completed successfully."
            )
            continue

        # If this prompt previously failed,
        # remove the failed record before retrying.
        results = [
            result_item
            for result_item in results
            if result_item["prompt_id"] != item["id"]
        ]
        
        print("\n" + "=" * 70)

        print(
            f"Model {model_index}/{total_models} | "
            f"Prompt {prompt_index}/{total_prompts}"
        )
        print(
            f"{item['id']} - {item['title']}"
        )
        print(
            f"Language: {item['language']}"
        )
        print(
            f"Thinking: {thinking}"
        )
        print(
            "Generating response..."
        )
        print("=" * 35, flush=True)

        start_time = time.perf_counter()

        try:
            result = run_prompt(
                model_name,
                item["prompt"],
                thinking
            )

        except requests.exceptions.Timeout:
            print(
                "\nERROR: Request timed out."
            )
            print(
                "Moving to the next prompt."
            )

            record = {
                "model": model_name,
                "prompt_id": item["id"],
                "question": item["question"],
                "title": item["title"],
                "language": item["language"],
                "purpose": item["purpose"],
                "prompt": item["prompt"],
                "response": "",
                "thinking_response": "",
                "error": "timeout",
                "thinking_requested": thinking
            }

            results.append(record)
            save_results(
                results,
                output_file
            )

            continue

        except requests.exceptions.RequestException as error:
            print(
                f"\nERROR: {error}"
            )
            print(
                "Moving to the next prompt."
            )

            record = {
                "model": model_name,
                "prompt_id": item["id"],
                "question": item["question"],
                "title": item["title"],
                "language": item["language"],
                "purpose": item["purpose"],
                "prompt": item["prompt"],
                "response": "",
                "thinking_response": "",
                "error": str(error),
                "thinking_requested": thinking
            }

            results.append(record)
            save_results(
                results,
                output_file
            )

            continue

        wall_seconds = (
            time.perf_counter()
            - start_time
        )

        input_tokens = result.get(
            "prompt_eval_count",
            0
        )

        output_tokens = result.get(
            "eval_count",
            0
        )

        total_seconds = (
            result.get(
                "total_duration",
                0
            )
            / 1_000_000_000
        )

        load_seconds = (
            result.get(
                "load_duration",
                0
            )
            / 1_000_000_000
        )

        prompt_seconds = (
            result.get(
                "prompt_eval_duration",
                0
            )
            / 1_000_000_000
        )

        generation_seconds = (
            result.get(
                "eval_duration",
                0
            )
            / 1_000_000_000
        )

        if generation_seconds > 0:
            tokens_per_second = (
                output_tokens
                / generation_seconds
            )
        else:
            tokens_per_second = 0

        record = {
            "model": model_name,
            "prompt_id": item["id"],
            "question": item["question"],
            "title": item["title"],
            "language": item["language"],
            "purpose": item["purpose"],
            "prompt": item["prompt"],
            "response": result.get(
                "response",
                ""
            ),
            "thinking_response": result.get(
                "thinking",
                ""
            ),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "wall_seconds": round(
                wall_seconds,
                3
            ),
            "total_seconds": round(
                total_seconds,
                3
            ),
            "load_seconds": round(
                load_seconds,
                3
            ),
            "prompt_seconds": round(
                prompt_seconds,
                3
            ),
            "generation_seconds": round(
                generation_seconds,
                3
            ),
            "tokens_per_second": round(
                tokens_per_second,
                2
            ),
            "done_reason": result.get(
                "done_reason",
                ""
            ),
            "thinking_requested": thinking,
            "error": None
        }

        results.append(record)

        # Save immediately after every completed prompt.
        save_results(
            results,
            output_file
        )

        print("\nCompleted.")
        print(
            f"Input tokens: "
            f"{record['input_tokens']}"
        )
        print(
            f"Output tokens: "
            f"{record['output_tokens']}"
        )
        print(
            f"Wall time: "
            f"{record['wall_seconds']} seconds"
        )
        print(
            f"Ollama total: "
            f"{record['total_seconds']} seconds"
        )
        print(
            f"Generation time: "
            f"{record['generation_seconds']} seconds"
        )
        print(
            f"Generation speed: "
            f"{record['tokens_per_second']} "
            "tokens/sec"
        )
        print(
            f"Done reason: "
            f"{record['done_reason']}"
        )
        print(
            f"Progress saved: "
            f"{prompt_index}/{total_prompts}"
        )

    print("\n" + "#" * 70)
    print(
        f"Finished model: {model_name}"
    )
    print(
        f"Results saved to: {output_file}"
    )
    print("#" * 70)


print("\n" + "=" * 70)
print("ALL MODEL EVALUATIONS COMPLETED.")
print("=" * 70)