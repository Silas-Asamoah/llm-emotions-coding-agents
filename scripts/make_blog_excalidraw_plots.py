#!/usr/bin/env python3
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt


logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "blog" / "assets" / "plots"

NEGATIVE_EMOTIONS = ["frustrated", "desperate", "stuck", "stressed"]
SMOKE_RUN = ROOT / "results" / "runs" / "smoke-qwen-coder-0_5b"
MAIN_COMPARISON = ROOT / "results" / "comparisons" / "main-7b"
AGENT_COMPARISON = ROOT / "results" / "comparisons" / "agent-harness"
SERIOUS_AGENT_COMPARISON = ROOT / "results" / "comparisons" / "serious-agent-harness"

MODEL_LABELS = {
    "main-qwen-coder-7b": "Qwen 7B",
    "main-deepseek-coder-6_7b": "DeepSeek 6.7B",
    "qwen-coder-7b": "Qwen 7B",
    "qwen3-coder-30b-a3b": "Qwen3 30B-A3B",
    "devstral-small-2507": "Devstral 2507",
    "deepseek-coder-v2-lite": "DeepSeek V2 Lite",
}
MODEL_COLORS = {
    "Qwen 7B": "#228be6",
    "DeepSeek 6.7B": "#f08c00",
    "Qwen3 30B-A3B": "#228be6",
    "Devstral 2507": "#2f9e44",
    "DeepSeek V2 Lite": "#f08c00",
}
CONDITION_ORDER = [
    "baseline",
    "calm_+1.0",
    "calm_-1.0",
    "desperate_+1.0",
    "desperate_-1.0",
    "frustrated_+1.0",
    "frustrated_-1.0",
    "patient_+1.0",
    "patient_-1.0",
]
AGENT_CONDITION_ORDER = [
    "baseline",
    "calm_+1.0",
    "calm_-1.0",
    "desperate_+1.0",
    "desperate_-1.0",
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with plt.xkcd(scale=0.75, length=90, randomness=2):
        _plot_smoke_activation()
        _plot_smoke_behavior_markers()
        _plot_smoke_aggregate_markers()
        _plot_main_negative_activation()
        _plot_main_task_pass()
        _plot_main_markers()
        _plot_agent_task_pass()
        _plot_agent_markers()
        _plot_agent_negative_activation()
        _plot_serious_agent_task_pass()
        _plot_serious_agent_markers()
        _plot_serious_agent_negative_activation()


def _plot_smoke_activation() -> None:
    activations = pd.read_csv(SMOKE_RUN / "activation_scores.csv")
    mean_scores = (
        activations.groupby(["stage", "emotion"], as_index=False)["score"]
        .mean()
        .pivot(index="stage", columns="emotion", values="score")
    )
    negative = activations[activations["emotion"].isin(NEGATIVE_EMOTIONS)]
    negative_mean = negative.groupby("stage")["score"].mean()

    fig, ax = _figure("Smoke run: pressure rises across failure prompts")
    lines = {
        "negative mean": (negative_mean, "#e03131"),
        "calm": (mean_scores["calm"], "#228be6"),
        "patient": (mean_scores["patient"], "#2f9e44"),
        "confident": (mean_scores["confident"], "#7048e8"),
    }
    for label, (series, color) in lines.items():
        ax.plot(series.index, series.values, marker="o", linewidth=2.6, label=label, color=color)
    _style_axes(ax, "Failure stage", "Mean cosine projection")
    ax.set_xticks(mean_scores.index)
    ax.legend(frameon=False, ncol=2)
    _save(fig, "smoke-activation-excalidraw.png")


def _plot_smoke_behavior_markers() -> None:
    scores = pd.read_csv(SMOKE_RUN / "generation_scores.csv")
    marker_cols = [
        "frustration_count",
        "desperation_count",
        "hardcode_count",
        "give_up_count",
        "exclamation_count",
    ]
    grouped = scores.groupby("condition")[marker_cols].mean()
    conditions = _ordered_conditions(grouped.index, AGENT_CONDITION_ORDER)
    grouped = grouped.reindex(conditions)

    fig, ax = _figure("Smoke run: visible markers are sparse")
    bottom = np.zeros(len(grouped))
    colors = ["#e03131", "#f08c00", "#7048e8", "#868e96", "#fab005"]
    labels = ["frustration", "desperation", "hardcode", "give up", "exclaim"]
    for col, label, color in zip(marker_cols, labels, colors):
        ax.bar(np.arange(len(grouped)), grouped[col].values, bottom=bottom, label=label, color=color)
        bottom += grouped[col].values
    _style_axes(ax, "Steering condition", "Mean marker count")
    ax.set_xticks(np.arange(len(grouped)), [_condition_label(value) for value in grouped.index])
    ax.tick_params(axis="x", rotation=18)
    ax.legend(frameon=False, ncol=3)
    _save(fig, "smoke-behavior-markers-excalidraw.png")


def _plot_smoke_aggregate_markers() -> None:
    scores = pd.read_csv(SMOKE_RUN / "generation_scores.csv")
    totals = scores.groupby("condition")["aggregate_marker_score"].mean()
    conditions = _ordered_conditions(totals.index, AGENT_CONDITION_ORDER)

    fig, ax = _figure("Smoke run: aggregate visible marker score", figsize=(9, 5.2))
    values = totals.reindex(conditions)
    y = np.arange(len(values))
    ax.barh(y, values.values, color="#4dabf7")
    ax.set_yticks(y, [_condition_label(value) for value in values.index])
    _style_axes(ax, "Mean count per generation", "")
    _annotate_horizontal_bars(ax)
    _save(fig, "smoke-aggregate-markers-excalidraw.png")


def _plot_main_negative_activation() -> None:
    activations = pd.read_csv(MAIN_COMPARISON / "activation_by_stage.csv")
    negative = activations[activations["emotion"].isin(NEGATIVE_EMOTIONS)].copy()
    negative["label"] = negative["run"].map(MODEL_LABELS)
    pivot = (
        negative.groupby(["stage", "label"], as_index=False)["score"]
        .mean()
        .pivot(index="stage", columns="label", values="score")
    )

    fig, ax = _figure("7B comparison: negative projections by stage")
    for label in _ordered_model_labels(pivot.columns):
        ax.plot(
            pivot.index,
            pivot[label],
            marker="o",
            linewidth=2.6,
            color=MODEL_COLORS[label],
            label=label,
        )
    _style_axes(ax, "Failure stage", "Mean cosine projection")
    ax.set_xticks(pivot.index)
    ax.legend(frameon=False)
    _save(fig, "main-negative-activation-excalidraw.png")


def _plot_main_task_pass() -> None:
    execution = pd.read_csv(MAIN_COMPARISON / "execution_by_condition.csv")
    execution["label"] = execution["run"].map(MODEL_LABELS)
    conditions = _ordered_conditions(execution["condition"].unique(), CONDITION_ORDER)
    labels = _ordered_model_labels(execution["label"].unique())
    pivot = execution.pivot(index="condition", columns="label", values="task_pass").reindex(
        index=conditions, columns=labels
    )

    fig, ax = _figure("7B comparison: task pass by steering condition", figsize=(13, 6))
    _grouped_bars(ax, pivot, labels, annotate=False)
    _style_axes(ax, "Condition", "Task pass rate")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(np.arange(len(pivot)), [_condition_label(value) for value in pivot.index])
    ax.tick_params(axis="x", rotation=25)
    ax.legend(frameon=False, ncol=3)
    _save(fig, "main-task-pass-excalidraw.png")


def _plot_main_markers() -> None:
    markers = pd.read_csv(MAIN_COMPARISON / "markers_by_condition.csv")
    markers["label"] = markers["run"].map(MODEL_LABELS)
    conditions = _ordered_conditions(markers["condition"].unique(), CONDITION_ORDER)
    labels = _ordered_model_labels(markers["label"].unique())
    pivot = markers.pivot(
        index="condition",
        columns="label",
        values="aggregate_marker_score",
    ).reindex(index=conditions, columns=labels)

    fig, ax = _figure("7B comparison: visible markers by condition", figsize=(13, 6))
    _grouped_bars(ax, pivot, labels, annotate=False)
    _style_axes(ax, "Condition", "Mean marker count")
    ax.set_xticks(np.arange(len(pivot)), [_condition_label(value) for value in pivot.index])
    ax.tick_params(axis="x", rotation=25)
    ax.legend(frameon=False, ncol=3)
    _save(fig, "main-marker-score-excalidraw.png")


def _plot_agent_task_pass() -> None:
    by_condition = pd.read_csv(AGENT_COMPARISON / "agent_by_condition.csv")
    by_condition["label"] = by_condition["run"].map(MODEL_LABELS)
    conditions = _ordered_conditions(by_condition["condition"].unique(), AGENT_CONDITION_ORDER)
    labels = _ordered_model_labels(by_condition["label"].unique())
    pivot = by_condition.pivot(
        index="condition",
        columns="label",
        values="final_task_pass",
    ).reindex(index=conditions, columns=labels)

    fig, ax = _figure("Agent harness: Qwen sometimes recovers")
    _grouped_bars(ax, pivot, labels, annotate=True)
    _style_axes(ax, "Condition", "Final task pass rate")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(np.arange(len(pivot)), [_condition_label(value) for value in pivot.index])
    ax.tick_params(axis="x", rotation=16)
    ax.legend(frameon=False)
    _save(fig, "agent-task-pass-excalidraw.png")


def _plot_agent_markers() -> None:
    markers = pd.read_csv(AGENT_COMPARISON / "agent_markers_by_condition.csv")
    markers["label"] = markers["run"].map(MODEL_LABELS)
    conditions = _ordered_conditions(markers["condition"].unique(), AGENT_CONDITION_ORDER)
    labels = _ordered_model_labels(markers["label"].unique())
    pivot = markers.pivot(
        index="condition",
        columns="label",
        values="aggregate_marker_score",
    ).reindex(index=conditions, columns=labels)

    fig, ax = _figure("Agent harness: visible marker score")
    _grouped_bars(ax, pivot, labels, annotate=True, value_format="{:.1f}")
    _style_axes(ax, "Condition", "Mean markers per attempted generation")
    ax.set_xticks(np.arange(len(pivot)), [_condition_label(value) for value in pivot.index])
    ax.tick_params(axis="x", rotation=16)
    ax.legend(frameon=False)
    _save(fig, "agent-marker-score-excalidraw.png")


def _plot_agent_negative_activation() -> None:
    negative = pd.read_csv(AGENT_COMPARISON / "agent_negative_by_attempt.csv")
    negative["label"] = negative["run"].map(MODEL_LABELS)
    pivot = (
        negative.groupby(["attempt", "label"], as_index=False)["score"]
        .mean()
        .pivot(index="attempt", columns="label", values="score")
    )

    fig, ax = _figure("Agent harness: observed negative-emotion projection")
    for label in _ordered_model_labels(pivot.columns):
        ax.plot(
            pivot.index,
            pivot[label],
            marker="o",
            linewidth=2.8,
            color=MODEL_COLORS[label],
            label=label,
        )
    _style_axes(ax, "Attempt", "Mean cosine projection")
    ax.set_xticks(pivot.index)
    ax.legend(frameon=False)
    _save(fig, "agent-negative-activation-excalidraw.png")


def _plot_serious_agent_task_pass() -> None:
    by_condition = pd.read_csv(SERIOUS_AGENT_COMPARISON / "agent_by_condition.csv")
    by_condition["label"] = _labels_for_runs(by_condition["run"])
    conditions = _ordered_conditions(by_condition["condition"].unique(), AGENT_CONDITION_ORDER)
    labels = _ordered_model_labels(by_condition["label"].unique())
    pivot = by_condition.pivot(
        index="condition",
        columns="label",
        values="final_task_pass",
    ).reindex(index=conditions, columns=labels)

    fig, ax = _figure("Modern agent harness: final task pass", figsize=(13, 6))
    _grouped_bars(ax, pivot, labels, annotate=True)
    _style_axes(ax, "Condition", "Final task pass rate")
    ax.set_ylim(0, 1.08)
    ax.set_xticks(np.arange(len(pivot)), [_condition_label(value) for value in pivot.index])
    ax.tick_params(axis="x", rotation=16)
    ax.legend(frameon=False, ncol=3)
    _save(fig, "serious-agent-task-pass-excalidraw.png")


def _plot_serious_agent_markers() -> None:
    markers = pd.read_csv(SERIOUS_AGENT_COMPARISON / "agent_markers_by_condition.csv")
    markers["label"] = _labels_for_runs(markers["run"])
    conditions = _ordered_conditions(markers["condition"].unique(), AGENT_CONDITION_ORDER)
    labels = _ordered_model_labels(markers["label"].unique())
    pivot = markers.pivot(
        index="condition",
        columns="label",
        values="aggregate_marker_score",
    ).reindex(index=conditions, columns=labels)

    fig, ax = _figure("Modern agent harness: visible marker score", figsize=(13, 6))
    _grouped_bars(ax, pivot, labels, annotate=True, value_format="{:.1f}")
    _style_axes(ax, "Condition", "Mean markers per attempted generation")
    ax.set_xticks(np.arange(len(pivot)), [_condition_label(value) for value in pivot.index])
    ax.tick_params(axis="x", rotation=16)
    ax.legend(frameon=False, ncol=3)
    _save(fig, "serious-agent-marker-score-excalidraw.png")


def _plot_serious_agent_negative_activation() -> None:
    negative = pd.read_csv(SERIOUS_AGENT_COMPARISON / "agent_negative_by_attempt.csv")
    negative["label"] = _labels_for_runs(negative["run"])
    pivot = (
        negative.groupby(["attempt", "label"], as_index=False)["score"]
        .mean()
        .pivot(index="attempt", columns="label", values="score")
    )

    fig, ax = _figure("Modern agent harness: observed negative-emotion projection")
    for label in _ordered_model_labels(pivot.columns):
        ax.plot(
            pivot.index,
            pivot[label],
            marker="o",
            linewidth=2.8,
            color=MODEL_COLORS[label],
            label=label,
        )
    _style_axes(ax, "Attempt", "Mean cosine projection")
    ax.set_xticks(pivot.index)
    ax.legend(frameon=False, ncol=3)
    _save(fig, "serious-agent-negative-activation-excalidraw.png")


def _figure(title: str, figsize: tuple[float, float] = (10, 5.6)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#fffdf7")
    ax.set_facecolor("#fffdf7")
    ax.set_title(title, loc="left", fontsize=18, pad=14, weight="bold")
    return fig, ax


def _style_axes(ax, xlabel: str, ylabel: str) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle="--", linewidth=1.0, alpha=0.35, color="#495057")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.6)
    ax.spines["bottom"].set_linewidth(1.6)


def _grouped_bars(
    ax,
    pivot: pd.DataFrame,
    labels: list[str],
    *,
    annotate: bool,
    value_format: str = "{:.2f}",
) -> None:
    x = np.arange(len(pivot.index))
    width = 0.78 / max(len(labels), 1)
    for index, label in enumerate(labels):
        offset = (index - (len(labels) - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            pivot[label].fillna(0).values,
            width=width,
            label=label,
            color=MODEL_COLORS[label],
            alpha=0.88,
            linewidth=1.4,
            edgecolor="#212529",
        )
        if annotate:
            _annotate_vertical_bars(ax, bars, value_format=value_format)


def _annotate_vertical_bars(ax, bars, value_format: str) -> None:
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.02,
            value_format.format(height),
            ha="center",
            va="bottom",
            fontsize=10,
        )


def _annotate_horizontal_bars(ax) -> None:
    for patch in ax.patches:
        width = patch.get_width()
        ax.text(
            width + 0.03,
            patch.get_y() + patch.get_height() / 2,
            f"{width:.2f}",
            va="center",
            fontsize=10,
        )


def _save(fig, filename: str) -> None:
    path = OUT / filename
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def _ordered_conditions(values, order: list[str]) -> list[str]:
    value_set = set(values)
    known = [value for value in order if value in value_set]
    unknown = sorted(value_set - set(known))
    return known + unknown


def _ordered_model_labels(values) -> list[str]:
    order = [
        "Qwen 7B",
        "DeepSeek 6.7B",
        "Qwen3 30B-A3B",
        "Devstral 2507",
        "DeepSeek V2 Lite",
    ]
    value_set = set(values)
    return [value for value in order if value in value_set]


def _labels_for_runs(runs: pd.Series) -> pd.Series:
    return runs.map(MODEL_LABELS).fillna(runs)


def _condition_label(value: str) -> str:
    if value == "baseline":
        return "baseline"
    emotion, strength = value.rsplit("_", 1)
    return f"{emotion}\n{strength}"


if __name__ == "__main__":
    main()
