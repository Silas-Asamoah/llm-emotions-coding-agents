from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_plots(run_dir: str | Path) -> list[Path]:
    import matplotlib.pyplot as plt

    run_path = Path(run_dir)
    plot_dir = run_path / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    activation_path = run_path / "activation_scores.csv"
    if activation_path.exists():
        activations = pd.read_csv(activation_path)
        mean_scores = (
            activations.groupby(["stage", "emotion"], as_index=False)["score"]
            .mean()
            .pivot(index="stage", columns="emotion", values="score")
        )
        fig, ax = plt.subplots(figsize=(9, 4.8))
        for emotion in mean_scores.columns:
            ax.plot(mean_scores.index, mean_scores[emotion], marker="o", label=emotion)
        ax.set_title("Emotion-direction activation across coding failure stages")
        ax.set_xlabel("Failure stage")
        ax.set_ylabel("Mean cosine projection")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.legend(ncol=2, fontsize=8)
        fig.tight_layout()
        path = plot_dir / "activation_trajectory.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)

    score_path = run_path / "generation_scores.csv"
    if score_path.exists():
        scores = pd.read_csv(score_path)
        marker_cols = [
            "profanity_count",
            "frustration_count",
            "desperation_count",
            "hardcode_count",
            "give_up_count",
        ]
        marker_means = scores.groupby("condition")[marker_cols].mean()
        fig, ax = plt.subplots(figsize=(9, 4.8))
        marker_means.plot(kind="bar", ax=ax)
        ax.set_title("Visible behavioral markers by steering condition")
        ax.set_xlabel("Condition")
        ax.set_ylabel("Mean marker count per generation")
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        path = plot_dir / "behavior_markers.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)

        fig, ax = plt.subplots(figsize=(8, 4.5))
        totals = scores.groupby("condition")["aggregate_marker_score"].mean().sort_values()
        totals.plot(kind="barh", ax=ax, color="#5b7f95")
        ax.set_title("Aggregate visible marker score")
        ax.set_xlabel("Mean count per generation")
        fig.tight_layout()
        path = plot_dir / "aggregate_marker_score.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)

    return paths


def write_agent_plots(run_dir: str | Path) -> list[Path]:
    import matplotlib.pyplot as plt

    run_path = Path(run_dir)
    plot_dir = run_path / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    runs_path = run_path / "agent_runs.csv"
    if runs_path.exists():
        runs = pd.read_csv(runs_path)
        pass_rates = runs.groupby("condition")[
            ["final_visible_pass", "final_hidden_pass", "final_task_pass"]
        ].mean()
        fig, ax = plt.subplots(figsize=(9, 4.8))
        pass_rates.plot(kind="bar", ax=ax)
        ax.set_title("Agent harness pass rates by steering condition")
        ax.set_xlabel("Condition")
        ax.set_ylabel("Pass rate")
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        path = plot_dir / "agent_pass_rates.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)

        attempts = runs.groupby("condition")["attempts_used"].mean().sort_values()
        fig, ax = plt.subplots(figsize=(8, 4.5))
        attempts.plot(kind="barh", ax=ax, color="#7666b8")
        ax.set_title("Mean attempts used by condition")
        ax.set_xlabel("Attempts")
        fig.tight_layout()
        path = plot_dir / "agent_attempts_used.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)

    activation_path = run_path / "agent_activation_scores.csv"
    if activation_path.exists():
        activations = pd.read_csv(activation_path)
        negative = activations[
            activations["emotion"].isin(["frustrated", "desperate", "stuck", "stressed"])
        ]
        mean_scores = (
            negative.groupby(["attempt", "condition"], as_index=False)["score"]
            .mean()
            .pivot(index="attempt", columns="condition", values="score")
        )
        fig, ax = plt.subplots(figsize=(9, 4.8))
        for condition in mean_scores.columns:
            ax.plot(mean_scores.index, mean_scores[condition], marker="o", label=condition)
        ax.set_title("Negative-emotion projection across agent attempts")
        ax.set_xlabel("Attempt")
        ax.set_ylabel("Mean cosine projection")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = plot_dir / "agent_negative_activation.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)

    return paths
