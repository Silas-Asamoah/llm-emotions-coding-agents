from __future__ import annotations

import json
import platform
import random
import sys
from pathlib import Path
from importlib import metadata

import numpy as np
import pandas as pd

from emotion_coding_agents.config import load_config
from emotion_coding_agents.data import (
    NEUTRAL_SNIPPETS,
    coding_failure_prompts,
    emotion_training_snippets,
    generation_tasks,
)
from emotion_coding_agents.metrics import aggregate_marker_score, score_generation
from emotion_coding_agents.modeling import (
    activation_scores,
    build_direction_bundle,
    find_transformer_blocks,
    generate_text,
    load_model_and_tokenizer,
    mean_activations,
    select_layers,
)
from emotion_coding_agents.plots import write_plots


def run_experiment(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    force: bool = False,
) -> dict:
    config = load_config(config_path)
    output = Path(output_dir)
    _prepare_output_dir(output, force)
    seed = int(config.get("experiment", {}).get("seed", 0))
    random.seed(seed)
    np.random.seed(seed)
    _seed_torch(seed)

    model, tokenizer = load_model_and_tokenizer(config["model"])
    layers = select_layers(
        len(find_transformer_blocks(model)),
        config.get("experiment", {}).get("layers", "auto"),
    )
    _write_manifest(output, config_path, config, model, tokenizer, layers)

    emotions = list(config["emotions"])
    max_examples = int(
        config.get("experiment", {}).get("max_direction_examples_per_emotion", 5)
    )
    snippets = emotion_training_snippets(emotions, max_examples)
    bundle = build_direction_bundle(model, tokenizer, snippets, NEUTRAL_SNIPPETS, layers)
    bundle.save(str(output / "emotion_directions.npz"))

    activation_frame = _probe_failure_trajectory(model, tokenizer, bundle, output)
    generation_frame = _run_generations(model, tokenizer, bundle, config, output)
    plots = write_plots(output)

    summary = {
        "model": config["model"]["name"],
        "emotions": emotions,
        "layers": layers,
        "num_failure_prompts": len(coding_failure_prompts()),
        "num_generations": len(generation_frame),
        "plots": [str(path.relative_to(output)) for path in plots],
        "mean_activation_by_emotion": (
            activation_frame.groupby("emotion")["score"].mean().round(4).to_dict()
        ),
        "mean_marker_score_by_condition": (
            generation_frame.groupby("condition")["aggregate_marker_score"]
            .mean()
            .round(4)
            .to_dict()
        ),
    }
    with (output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def _seed_torch(seed: int) -> None:
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        return


def _probe_failure_trajectory(model, tokenizer, bundle, output: Path) -> pd.DataFrame:
    prompts = coding_failure_prompts()
    acts = mean_activations(model, tokenizer, [prompt.text for prompt in prompts], bundle.layers)
    scores = activation_scores(acts, bundle)
    rows = []
    for prompt_idx, prompt in enumerate(prompts):
        for emotion_idx, emotion in enumerate(bundle.emotions):
            for layer_idx, layer in enumerate(bundle.layers):
                rows.append(
                    {
                        "prompt_id": prompt.prompt_id,
                        "stage": prompt.stage,
                        "emotion": emotion,
                        "layer": layer,
                        "score": float(scores[prompt_idx, emotion_idx, layer_idx]),
                    }
                )
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "activation_scores.csv", index=False)
    return frame


def _prepare_output_dir(output: Path, force: bool) -> None:
    output.mkdir(parents=True, exist_ok=True)
    existing = [path for path in output.iterdir() if path.name != ".gitkeep"]
    if existing and not force:
        raise FileExistsError(
            f"output directory is not empty: {output}. Use --force to overwrite."
        )


def _write_manifest(
    output: Path,
    config_path: str | Path,
    config: dict,
    model,
    tokenizer,
    layers: list[int],
) -> None:
    manifest = {
        "config_path": str(config_path),
        "config": config,
        "layers": layers,
        "python": sys.version,
        "platform": platform.platform(),
        "packages": _package_versions(
            [
                "torch",
                "transformers",
                "accelerate",
                "mistral-common",
                "numpy",
                "pandas",
                "matplotlib",
            ]
        ),
        "model_commit": getattr(getattr(model, "config", None), "_commit_hash", None),
        "tokenizer_commit": getattr(tokenizer, "_commit_hash", None),
        "device": str(next(model.parameters()).device),
        "cuda_device": _cuda_device_name(),
    }
    with (output / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


def _cuda_device_name() -> str | None:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except ImportError:
        return None
    return None


def _package_versions(names: list[str]) -> dict[str, str | None]:
    versions = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _run_generations(model, tokenizer, bundle, config: dict, output: Path) -> pd.DataFrame:
    generation_config = config["model"]
    seed = int(config.get("experiment", {}).get("seed", 0))
    steering_emotions = config.get("experiment", {}).get("steering_emotions", [])
    strengths = [float(value) for value in config.get("experiment", {}).get("steering_strengths", [0.0])]
    generation_rows = []
    score_rows = []
    jsonl_path = output / "generations.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for task_index, task in enumerate(generation_tasks()):
            conditions = [(None, 0.0)]
            for emotion in steering_emotions:
                for strength in strengths:
                    if strength != 0:
                        conditions.append((emotion, strength))
            for emotion, strength in conditions:
                condition = "baseline" if emotion is None else f"{emotion}_{strength:+.1f}"
                generation_seed = seed + task_index
                text = generate_text(
                    model,
                    tokenizer,
                    task.text,
                    generation_config,
                    bundle=bundle,
                    emotion=emotion,
                    strength=strength,
                    seed=generation_seed,
                )
                generation_row = {
                    "task_id": task.task_id,
                    "condition": condition,
                    "emotion": emotion or "none",
                    "strength": strength,
                    "seed": generation_seed,
                    "prompt": task.text,
                    "generation": text,
                }
                handle.write(json.dumps(generation_row) + "\n")
                generation_rows.append(generation_row)
                scores = score_generation(text)
                scores["aggregate_marker_score"] = aggregate_marker_score(scores)
                score_rows.append(
                    {
                        "task_id": task.task_id,
                        "condition": condition,
                        "emotion": emotion or "none",
                        "strength": strength,
                        "seed": generation_seed,
                        **scores,
                    }
                )
    pd.DataFrame(generation_rows).to_csv(output / "generations.csv", index=False)
    frame = pd.DataFrame(score_rows)
    frame.to_csv(output / "generation_scores.csv", index=False)
    return frame
