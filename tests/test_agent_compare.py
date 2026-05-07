import json
from pathlib import Path

import pandas as pd

from emotion_coding_agents.agent_compare import compare_agent_runs


def test_compare_agent_runs_writes_outputs(tmp_path: Path):
    run_a = _write_agent_run(tmp_path, "agent-a", "model-a", True)
    run_b = _write_agent_run(tmp_path, "agent-b", "model-b", False)

    result = compare_agent_runs([run_a, run_b], tmp_path / "comparison")

    assert result["num_models"] == 2
    assert (tmp_path / "comparison" / "agent_model_summary.csv").exists()
    assert (tmp_path / "comparison" / "agent_by_condition.csv").exists()
    assert all((tmp_path / "comparison" / plot).exists() for plot in result["plots"])


def _write_agent_run(base: Path, name: str, model: str, passed: bool) -> Path:
    run = base / name
    run.mkdir()
    (run / "summary.json").write_text(
        json.dumps({"model": model, "num_runs": 1}),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "task_id": "task",
                "condition": "baseline",
                "final_visible_pass": passed,
                "final_hidden_pass": passed,
                "final_task_pass": passed,
                "attempts_used": 1,
            }
        ]
    ).to_csv(run / "agent_runs.csv", index=False)
    pd.DataFrame(
        [
            {
                "task_id": "task",
                "condition": "baseline",
                "attempt": 0,
                "aggregate_marker_score": 1,
            }
        ]
    ).to_csv(run / "agent_attempts.csv", index=False)
    pd.DataFrame(
        [
            {
                "task_id": "task",
                "condition": "baseline",
                "attempt": 0,
                "emotion": "desperate",
                "score": 0.2,
            }
        ]
    ).to_csv(run / "agent_activation_scores.csv", index=False)
    return run
