from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


NEGATIVE_EMOTIONS = ["frustrated", "desperate", "stuck", "stressed"]
POSITIVE_EMOTIONS = ["calm", "patient", "confident"]


def compare_runs(run_dirs: list[str | Path], output_dir: str | Path) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    activations = []
    generations = []
    executions = []
    summaries = []

    for run_dir in run_dirs:
        run_path = Path(run_dir)
        summary = _read_json(run_path / "summary.json")
        manifest = _read_json(run_path / "manifest.json")
        model = summary["model"]
        label = run_path.name
        activation_frame = pd.read_csv(run_path / "activation_scores.csv")
        activation_frame["model"] = model
        activation_frame["run"] = label
        activations.append(activation_frame)

        generation_frame = pd.read_csv(run_path / "generation_scores.csv")
        generation_frame["model"] = model
        generation_frame["run"] = label
        generations.append(generation_frame)

        execution_frame = _read_execution_scores(run_path)
        if execution_frame is not None:
            execution_frame["model"] = model
            execution_frame["run"] = label
            executions.append(execution_frame)

        summaries.append(
            {
                "run": label,
                "model": model,
                "layers": ",".join(str(layer) for layer in summary["layers"]),
                "cuda_device": manifest.get("cuda_device"),
                "num_generations": summary["num_generations"],
                "mean_marker_score": generation_frame["aggregate_marker_score"].mean(),
                **_execution_summary(execution_frame),
                "mean_negative_activation": _mean_activation(
                    activation_frame,
                    NEGATIVE_EMOTIONS,
                ),
                "mean_positive_activation": _mean_activation(
                    activation_frame,
                    POSITIVE_EMOTIONS,
                ),
            }
        )

    activation_all = pd.concat(activations, ignore_index=True)
    generation_all = pd.concat(generations, ignore_index=True)
    summary_frame = pd.DataFrame(summaries)

    activation_by_stage = (
        activation_all.groupby(["run", "model", "stage", "emotion"], as_index=False)["score"]
        .mean()
        .sort_values(["run", "stage", "emotion"])
    )
    markers_by_condition = (
        generation_all.groupby(["run", "model", "condition"], as_index=False)[
            [
                "aggregate_marker_score",
                "profanity_count",
                "frustration_count",
                "desperation_count",
                "hardcode_count",
                "give_up_count",
                "all_caps_words",
                "exclamation_count",
            ]
        ]
        .mean()
        .sort_values(["run", "condition"])
    )

    summary_frame.to_csv(output / "model_summary.csv", index=False)
    activation_by_stage.to_csv(output / "activation_by_stage.csv", index=False)
    markers_by_condition.to_csv(output / "markers_by_condition.csv", index=False)
    execution_by_condition = None
    if executions:
        execution_all = pd.concat(executions, ignore_index=True)
        execution_by_condition = (
            execution_all.groupby(["run", "model", "condition"], as_index=False)[
                ["valid_python", "visible_pass", "hidden_pass", "task_pass"]
            ]
            .mean()
            .sort_values(["run", "condition"])
        )
        execution_by_condition.to_csv(output / "execution_by_condition.csv", index=False)
    _write_plots(
        output,
        activation_by_stage,
        markers_by_condition,
        summary_frame,
        execution_by_condition,
    )

    result = {
        "runs": [str(Path(run_dir)) for run_dir in run_dirs],
        "num_models": len(run_dirs),
        "summary": summary_frame.round(4).to_dict(orient="records"),
        "plots": [
            "plots/negative_activation_by_stage.png",
            "plots/marker_score_by_condition.png",
            "plots/positive_vs_negative_activation.png",
            "plots/task_pass_by_condition.png",
        ],
    }
    with (output / "comparison_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    return result


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _mean_activation(frame: pd.DataFrame, emotions: list[str]) -> float:
    return float(frame.loc[frame["emotion"].isin(emotions), "score"].mean())


def _read_execution_scores(run_path: Path) -> pd.DataFrame | None:
    path = run_path / "execution_scores.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def _execution_summary(frame: pd.DataFrame | None) -> dict[str, float | None]:
    if frame is None:
        return {
            "valid_python_rate": None,
            "visible_pass_rate": None,
            "hidden_pass_rate": None,
            "task_pass_rate": None,
        }
    return {
        "valid_python_rate": frame["valid_python"].mean(),
        "visible_pass_rate": frame["visible_pass"].mean(),
        "hidden_pass_rate": frame["hidden_pass"].mean(),
        "task_pass_rate": frame["task_pass"].mean(),
    }


def _write_plots(
    output: Path,
    activation_by_stage: pd.DataFrame,
    markers_by_condition: pd.DataFrame,
    summary_frame: pd.DataFrame,
    execution_by_condition: pd.DataFrame | None,
) -> None:
    import matplotlib.pyplot as plt

    plot_dir = output / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    negative = activation_by_stage[
        activation_by_stage["emotion"].isin(NEGATIVE_EMOTIONS)
    ]
    negative_mean = (
        negative.groupby(["run", "stage"], as_index=False)["score"]
        .mean()
        .pivot(index="stage", columns="run", values="score")
    )
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for run in negative_mean.columns:
        ax.plot(negative_mean.index, negative_mean[run], marker="o", label=run)
    ax.set_title("Mean negative-emotion projection across failure prompts")
    ax.set_xlabel("Failure prompt stage")
    ax.set_ylabel("Mean cosine projection")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_dir / "negative_activation_by_stage.png", dpi=180)
    plt.close(fig)

    marker_pivot = markers_by_condition.pivot(
        index="condition",
        columns="run",
        values="aggregate_marker_score",
    )
    fig, ax = plt.subplots(figsize=(10, 5.2))
    marker_pivot.plot(kind="bar", ax=ax)
    ax.set_title("Aggregate visible marker score by condition")
    ax.set_xlabel("Condition")
    ax.set_ylabel("Mean count per generation")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(plot_dir / "marker_score_by_condition.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.scatter(
        summary_frame["mean_positive_activation"],
        summary_frame["mean_negative_activation"],
    )
    for row in summary_frame.itertuples():
        ax.annotate(row.run, (row.mean_positive_activation, row.mean_negative_activation))
    ax.set_title("Positive vs negative direction activation")
    ax.set_xlabel("Mean positive-emotion projection")
    ax.set_ylabel("Mean negative-emotion projection")
    fig.tight_layout()
    fig.savefig(plot_dir / "positive_vs_negative_activation.png", dpi=180)
    plt.close(fig)

    if execution_by_condition is not None:
        pass_pivot = execution_by_condition.pivot(
            index="condition",
            columns="run",
            values="task_pass",
        )
        fig, ax = plt.subplots(figsize=(10, 5.2))
        pass_pivot.plot(kind="bar", ax=ax)
        ax.set_title("Visible and hidden task pass rate by condition")
        ax.set_xlabel("Condition")
        ax.set_ylabel("Task pass rate")
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=35)
        fig.tight_layout()
        fig.savefig(plot_dir / "task_pass_by_condition.png", dpi=180)
        plt.close(fig)
