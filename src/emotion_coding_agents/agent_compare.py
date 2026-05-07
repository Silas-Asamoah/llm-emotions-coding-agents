from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


NEGATIVE_EMOTIONS = ["frustrated", "desperate", "stuck", "stressed"]


def compare_agent_runs(run_dirs: list[str | Path], output_dir: str | Path) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summaries = []
    run_frames = []
    attempt_frames = []
    activation_frames = []

    for run_dir in run_dirs:
        run_path = Path(run_dir)
        summary = _read_json(run_path / "summary.json")
        label = run_path.name
        runs = pd.read_csv(run_path / "agent_runs.csv")
        attempts = pd.read_csv(run_path / "agent_attempts.csv")
        activations = pd.read_csv(run_path / "agent_activation_scores.csv")
        for frame in [runs, attempts, activations]:
            frame["run"] = label
            frame["model"] = summary["model"]
        run_frames.append(runs)
        attempt_frames.append(attempts)
        activation_frames.append(activations)
        summaries.append(
            {
                "run": label,
                "model": summary["model"],
                "num_runs": summary["num_runs"],
                "final_visible_pass_rate": runs["final_visible_pass"].mean(),
                "final_hidden_pass_rate": runs["final_hidden_pass"].mean(),
                "final_task_pass_rate": runs["final_task_pass"].mean(),
                "mean_attempts_used": runs["attempts_used"].mean(),
                "mean_marker_score": attempts["aggregate_marker_score"].mean(),
                "mean_negative_activation": _negative_mean(activations),
            }
        )

    run_all = pd.concat(run_frames, ignore_index=True)
    attempt_all = pd.concat(attempt_frames, ignore_index=True)
    activation_all = pd.concat(activation_frames, ignore_index=True)
    summary_frame = pd.DataFrame(summaries)
    by_condition = (
        run_all.groupby(["run", "model", "condition"], as_index=False)[
            ["final_visible_pass", "final_hidden_pass", "final_task_pass", "attempts_used"]
        ]
        .mean()
        .sort_values(["run", "condition"])
    )
    markers = (
        attempt_all.groupby(["run", "model", "condition"], as_index=False)[
            "aggregate_marker_score"
        ]
        .mean()
        .sort_values(["run", "condition"])
    )
    negative_by_attempt = (
        activation_all[activation_all["emotion"].isin(NEGATIVE_EMOTIONS)]
        .groupby(["run", "model", "attempt", "condition"], as_index=False)["score"]
        .mean()
        .sort_values(["run", "attempt", "condition"])
    )

    summary_frame.to_csv(output / "agent_model_summary.csv", index=False)
    by_condition.to_csv(output / "agent_by_condition.csv", index=False)
    markers.to_csv(output / "agent_markers_by_condition.csv", index=False)
    negative_by_attempt.to_csv(output / "agent_negative_by_attempt.csv", index=False)
    _write_plots(output, by_condition, markers, negative_by_attempt)

    result = {
        "runs": [str(Path(run_dir)) for run_dir in run_dirs],
        "num_models": len(run_dirs),
        "summary": summary_frame.round(4).to_dict(orient="records"),
        "plots": [
            "plots/agent_task_pass_comparison.png",
            "plots/agent_marker_comparison.png",
            "plots/agent_negative_activation_comparison.png",
        ],
    }
    with (output / "agent_comparison_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    return result


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _negative_mean(frame: pd.DataFrame) -> float:
    return float(frame.loc[frame["emotion"].isin(NEGATIVE_EMOTIONS), "score"].mean())


def _write_plots(
    output: Path,
    by_condition: pd.DataFrame,
    markers: pd.DataFrame,
    negative_by_attempt: pd.DataFrame,
) -> None:
    import matplotlib.pyplot as plt

    plot_dir = output / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    pass_pivot = by_condition.pivot(
        index="condition",
        columns="run",
        values="final_task_pass",
    )
    fig, ax = plt.subplots(figsize=(10, 5.2))
    pass_pivot.plot(kind="bar", ax=ax)
    ax.set_title("Agent harness final task pass rate")
    ax.set_xlabel("Condition")
    ax.set_ylabel("Final task pass rate")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(plot_dir / "agent_task_pass_comparison.png", dpi=180)
    plt.close(fig)

    marker_pivot = markers.pivot(
        index="condition",
        columns="run",
        values="aggregate_marker_score",
    )
    fig, ax = plt.subplots(figsize=(10, 5.2))
    marker_pivot.plot(kind="bar", ax=ax)
    ax.set_title("Agent harness visible marker score")
    ax.set_xlabel("Condition")
    ax.set_ylabel("Mean marker count per attempt")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(plot_dir / "agent_marker_comparison.png", dpi=180)
    plt.close(fig)

    negative_pivot = negative_by_attempt.groupby(["run", "attempt"], as_index=False)[
        "score"
    ].mean().pivot(index="attempt", columns="run", values="score")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for run in negative_pivot.columns:
        ax.plot(negative_pivot.index, negative_pivot[run], marker="o", label=run)
    ax.set_title("Agent harness negative-emotion projection")
    ax.set_xlabel("Attempt")
    ax.set_ylabel("Mean cosine projection")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_dir / "agent_negative_activation_comparison.png", dpi=180)
    plt.close(fig)

