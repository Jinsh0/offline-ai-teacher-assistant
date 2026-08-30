import json
from pathlib import Path


RAW_FOLDER = Path("results") / "raw"
REVIEW_FOLDER = Path("results") / "review"


def load_results(file_path):
    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


def format_result_file(file_path):
    results = load_results(file_path)

    #added this 'order' and 'sort' so each question come in sequence (easier to do manual review)
    order = {
        "Q1_EN": 1,
        "Q1_ID": 2,
        "Q2_EN": 3,
        "Q2_ID": 4,
        "Q3_EN": 5,
        "Q3_ID": 6,
        "Q4_EN": 7,
        "Q4_ID": 8
    }

    results.sort(
        key=lambda item: order.get(
            item["prompt_id"],
            999
        )
    )

    model_name = (
        results[0]["model"]
        if results
        else file_path.stem
    )

    lines = []

    lines.append(
        f"# Evaluation Results — {model_name}"
    )
    lines.append("")

    for item in results:
        lines.append("---")
        lines.append("")

        lines.append(
            f"## {item['prompt_id']} — "
            f"{item['title']}"
        )
        lines.append("")

        lines.append(
            f"**Language:** {item['language']}"
        )
        lines.append("")

        lines.append(
            f"**Purpose:** {item['purpose']}"
        )
        lines.append("")

        lines.append("---")
        lines.append("")

        lines.append("### Prompt")
        lines.append("")
        lines.append(item["prompt"])
        lines.append("")

        if item.get("thinking_response"):
            lines.append("### Thinking")
            lines.append("")
            lines.append(
                item["thinking_response"]
            )
            lines.append("")

        lines.append("### Response")
        lines.append("")

        response_text = item.get(
            "response",
            ""
        )

        quoted_response = "\n".join(
            f"> {line}"
            for line in response_text.splitlines()
        )

        lines.append(quoted_response)
        lines.append("")

        lines.append("### Performance")
        lines.append("")

        lines.append(
            f"- Input tokens: "
            f"{item.get('input_tokens', '')}"
        )

        lines.append(
            f"- Output tokens: "
            f"{item.get('output_tokens', '')}"
        )

        lines.append(
            f"- Wall time: "
            f"{item.get('wall_seconds', '')} s"
        )

        lines.append(
            f"- Generation time: "
            f"{item.get('generation_seconds', '')} s"
        )

        lines.append(
            f"- Generation speed: "
            f"{item.get('tokens_per_second', '')} "
            "tokens/s"
        )

        lines.append(
            f"- Done reason: "
            f"{item.get('done_reason', '')}"
        )

        lines.append(
            f"- Thinking requested: "
            f"{item.get('thinking_requested', '')}"
        )

        lines.append("")

        lines.append("### Review")
        lines.append("")

        lines.append(
            "| Evaluator | Accuracy (& Safety) | "
            "Instruction Following | Clarity & Usefulness | "
            "Language Quality | Total / 8 |"
        )

        lines.append(
            "|---|---:|---:|---:|---:|---:|"
        )

        lines.append(
            "| Human |  |  |  |  |  |"
        )

        lines.append(
            "| AI |  |  |  |  |  |"
        )

        lines.append("")

        if item.get("error"):
            lines.append(
                f"- Error: {item['error']}"
            )

        lines.append("")

    return "\n".join(lines)


REVIEW_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)

for file_path in RAW_FOLDER.glob(
    "*_all_prompts.json"
):
    markdown_text = (
        format_result_file(file_path)
    )

    output_file = (
        REVIEW_FOLDER
        / f"{file_path.stem}.md"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:
        file.write(markdown_text)

    print(
        f"Created: {output_file}"
    )