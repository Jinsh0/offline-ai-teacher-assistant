# Offline AI Teacher Assistant

This repository contains the implementation and preliminary work for my NUS Master of Computing capstone project on an offline-capable AI assistant for teachers in resource-constrained schools.

## Current Stage

The current work focuses on preliminary local LLM evaluation before introducing curriculum-grounded RAG.

The following models were evaluated locally using Ollama:

- Qwen 3.5 4B
- Gemma 3 4B
- Phi-4 Mini 3.8B
- Llama 3.2 3B
- DeepSeek-R1 8B
- Nemotron 3 Nano 4B

Each model was tested using four questions in both English and Bahasa Indonesia.

## Evaluation Results

Human-readable model outputs and performance results are available here:

- [Qwen 3.5 4B](results/review/qwen3.5_4b_all_prompts.md)
- [Gemma 3 4B](results/review/gemma3_4b_all_prompts.md)
- [Phi-4 Mini 3.8B](results/review/phi4_mini_3.8b_all_prompts.md)
- [Llama 3.2 3B](results/review/llama3.2_3b_all_prompts.md)
- [Nemotron 3 Nano 4B](results/review/nemotron_3_nano_4b_all_prompts.md)
- [DeepSeek-R1 8B](results/review/deepseek_r1_8b_all_prompts.md)

The original machine-readable outputs are available under:

`results/raw/`

## Evaluation Code

The main evaluation scripts are located under:

`evaluation/`

The evaluation runner sends prompts to the local Ollama API and records response text and performance metrics.

## Next Step

The next stage is to review the generated responses using a simple 0–2 rubric covering:

- Accuracy (& Safety)
- Instruction Following
- Clarity & Usefulness
- Language Quality

The same rubric will be used for manual review and AI-assisted review.