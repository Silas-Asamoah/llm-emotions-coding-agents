# LLM Emotions in Coding Agents

This repository contains a small, reproducible adaptation of Anthropic's
emotion-concept experiments for open coding language models.

The working question is:

> Do emotion-concept directions such as `frustrated`, `desperate`, `stuck`,
> `calm`, and `patient` predict or change how coding agents behave after test
> failures?

This is not evidence that models feel emotions. The project studies measurable
activation directions and behavior changes in language models.

## Current Scope

The first implementation keeps the experiment deliberately small:

1. Build emotion directions from short emotion-labeled snippets.
2. Probe those directions on coding-agent failure prompts.
3. Generate model responses under neutral, negative, and positive steering.
4. Score visible behavioral markers such as giving up, hardcoding, profanity,
   repeated punctuation, and explicit frustration language.
5. Save plots and a blog draft grounded in the generated artifacts.

## Local Setup

Create a Python environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install PyTorch separately for your machine if it is not already available.
JarvisLabs PyTorch templates usually include it.

## Run a Smoke Experiment

```bash
python scripts/run_experiment.py \
  --config configs/smoke_qwen_coder_0_5b.yaml \
  --output-dir results/runs/smoke-qwen-coder-0_5b
```

The script writes:

- `summary.json`
- `emotion_directions.npz`
- `activation_scores.csv`
- `generations.jsonl`
- `generation_scores.csv`
- `plots/*.png`

## JarvisLabs

Use an L4 instance for 0.5B-7B coding-model experiments:

```bash
jl run . \
  --script scripts/run_experiment.py \
  --gpu L4 \
  --keep \
  --json \
  --yes \
  -- --config configs/smoke_qwen_coder_0_5b.yaml \
     --output-dir results/runs/smoke-qwen-coder-0_5b
```

Pause or destroy the instance after downloading results.

## Citation Context

This repo is inspired by Anthropic's 2026 paper:

Sofroniew et al., "Emotion Concepts and their Function in a Large Language
Model", Transformer Circuits Thread, 2026.
