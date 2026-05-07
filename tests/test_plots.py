from pathlib import Path

import pandas as pd

from emotion_coding_agents.plots import write_plots


def test_write_plots_creates_expected_files(tmp_path: Path):
    pd.DataFrame(
        [
            {"stage": 0, "emotion": "calm", "score": 0.1},
            {"stage": 1, "emotion": "calm", "score": 0.2},
            {"stage": 0, "emotion": "desperate", "score": -0.1},
            {"stage": 1, "emotion": "desperate", "score": 0.3},
        ]
    ).to_csv(tmp_path / "activation_scores.csv", index=False)
    pd.DataFrame(
        [
            {
                "condition": "baseline",
                "profanity_count": 0,
                "frustration_count": 1,
                "desperation_count": 0,
                "hardcode_count": 0,
                "give_up_count": 0,
                "aggregate_marker_score": 1,
            },
            {
                "condition": "desperate_+1.0",
                "profanity_count": 0,
                "frustration_count": 1,
                "desperation_count": 1,
                "hardcode_count": 1,
                "give_up_count": 0,
                "aggregate_marker_score": 3,
            },
        ]
    ).to_csv(tmp_path / "generation_scores.csv", index=False)

    paths = write_plots(tmp_path)

    assert {path.name for path in paths} == {
        "activation_trajectory.png",
        "behavior_markers.png",
        "aggregate_marker_score.png",
    }
    assert all(path.exists() for path in paths)

