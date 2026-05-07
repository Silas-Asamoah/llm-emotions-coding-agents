import json
from pathlib import Path

import pandas as pd

from emotion_coding_agents.compare import compare_runs


def test_compare_runs_writes_summary_and_plots(tmp_path: Path):
    run_a = _write_run(tmp_path, "run-a", "model-a", 0.2)
    run_b = _write_run(tmp_path, "run-b", "model-b", 0.4)

    result = compare_runs([run_a, run_b], tmp_path / "comparison")

    assert result["num_models"] == 2
    assert (tmp_path / "comparison" / "model_summary.csv").exists()
    assert (tmp_path / "comparison" / "activation_by_stage.csv").exists()
    assert (tmp_path / "comparison" / "markers_by_condition.csv").exists()
    assert all((tmp_path / "comparison" / plot).exists() for plot in result["plots"])


def _write_run(base: Path, name: str, model: str, activation: float) -> Path:
    run = base / name
    run.mkdir()
    (run / "summary.json").write_text(
        json.dumps({"model": model, "layers": [1], "num_generations": 1}),
        encoding="utf-8",
    )
    (run / "manifest.json").write_text(
        json.dumps({"cuda_device": "test-gpu"}),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {"stage": 0, "emotion": "calm", "score": activation},
            {"stage": 0, "emotion": "frustrated", "score": activation + 0.1},
            {"stage": 1, "emotion": "calm", "score": activation + 0.2},
            {"stage": 1, "emotion": "frustrated", "score": activation + 0.3},
        ]
    ).to_csv(run / "activation_scores.csv", index=False)
    pd.DataFrame(
        [
            {
                "condition": "baseline",
                "aggregate_marker_score": 1,
                "profanity_count": 0,
                "frustration_count": 1,
                "desperation_count": 0,
                "hardcode_count": 0,
                "give_up_count": 0,
                "all_caps_words": 0,
                "exclamation_count": 0,
            }
        ]
    ).to_csv(run / "generation_scores.csv", index=False)
    return run
