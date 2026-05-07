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

