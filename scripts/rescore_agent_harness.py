#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from emotion_coding_agents.agent_harness import AGENT_TASKS
from emotion_coding_agents.evaluation import FunctionTask, evaluate_generation
from emotion_coding_agents.metrics import aggregate_marker_score, score_generation
from emotion_coding_agents.plots import write_agent_plots


TASKS = {
    task.task_id: FunctionTask(
        task_id=task.task_id,
        function_name=task.function_name,
        visible_tests=task.visible_tests,
        hidden_tests=task.hidden_tests,
    )
    for task in AGENT_TASKS
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+")
    args = parser.parse_args()

    summary = {}
    for run_dir in args.run_dirs:
        run_path = Path(run_dir)
        attempt_frame = _rescore_attempts(run_path)
        run_frame = _summarize_runs(attempt_frame)
        attempt_frame.to_csv(run_path / "agent_attempts.csv", index=False)
        run_frame.to_csv(run_path / "agent_runs.csv", index=False)
        write_agent_plots(run_path)
        _update_summary(run_path, attempt_frame, run_frame)
        summary[str(run_path)] = {
            "num_attempts": len(attempt_frame),
            "num_runs": len(run_frame),
            "final_visible_pass_rate": round(float(run_frame["final_visible_pass"].mean()), 4),
            "final_hidden_pass_rate": round(float(run_frame["final_hidden_pass"].mean()), 4),
            "final_task_pass_rate": round(float(run_frame["final_task_pass"].mean()), 4),
        }
    print(json.dumps(summary, indent=2))


def _rescore_attempts(run_path: Path) -> pd.DataFrame:
    rows = []
    with (run_path / "agent_attempts.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            result = evaluate_generation(row["generation"], TASKS[row["task_id"]])
            scores = score_generation(row["generation"])
            scores["aggregate_marker_score"] = aggregate_marker_score(scores)
            metadata = {
                key: row[key]
                for key in [
                    "task_id",
                    "condition",
                    "emotion",
                    "strength",
                    "attempt",
                    "generation",
                ]
            }
            if "seed" in row:
                metadata["seed"] = row["seed"]
            rows.append(
                {
                    **metadata,
                    "visible_pass": result["visible_pass"],
                    "hidden_pass": result["hidden_pass"],
                    "task_pass": result["task_pass"],
                    "error": result["error"],
                    "extracted_source": result["extracted_source"],
                    **scores,
                }
            )
    return pd.DataFrame(rows)


def _summarize_runs(attempts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    groups = attempts.sort_values("attempt").groupby(
        ["task_id", "condition", "emotion", "strength"],
        sort=False,
    )
    for (task_id, condition, emotion, strength), group in groups:
        passing = group[group["visible_pass"]]
        final = passing.iloc[0] if not passing.empty else group.iloc[-1]
        rows.append(
            {
                "task_id": task_id,
                "condition": condition,
                "emotion": emotion,
                "strength": strength,
                "attempts_used": int(final["attempt"]) + 1,
                "final_visible_pass": bool(final["visible_pass"]),
                "final_hidden_pass": bool(final["hidden_pass"]),
                "final_task_pass": bool(final["task_pass"]),
                "final_error": final["error"],
            }
        )
    return pd.DataFrame(rows)


def _update_summary(run_path: Path, attempts: pd.DataFrame, runs: pd.DataFrame) -> None:
    with (run_path / "summary.json").open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    summary.update(
        {
            "final_visible_pass_rate": round(float(runs["final_visible_pass"].mean()), 4),
            "final_hidden_pass_rate": round(float(runs["final_hidden_pass"].mean()), 4),
            "final_task_pass_rate": round(float(runs["final_task_pass"].mean()), 4),
            "mean_attempts_used": round(float(runs["attempts_used"].mean()), 4),
            "mean_marker_score": round(float(attempts["aggregate_marker_score"].mean()), 4),
        }
    )
    with (run_path / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


if __name__ == "__main__":
    main()
