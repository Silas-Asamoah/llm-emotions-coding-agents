import numpy as np
import pytest

from emotion_coding_agents.modeling import (
    DirectionBundle,
    activation_scores,
    select_layers,
)


def test_select_layers_auto_returns_middle_layers():
    assert select_layers(12, "auto") == [4, 6, 8]


def test_select_layers_rejects_out_of_range_spec():
    with pytest.raises(ValueError):
        select_layers(4, [9])


def test_select_layers_rejects_partial_typo():
    with pytest.raises(ValueError):
        select_layers(4, [1, 999])


def test_activation_scores_prefers_matching_direction():
    bundle = DirectionBundle(
        emotions=["calm", "desperate"],
        layers=[0],
        vectors=np.array([[[1.0, 0.0]], [[0.0, 1.0]]]),
        neutral_mean=np.array([[0.0, 0.0]]),
    )
    activations = np.array([[[1.0, 0.0]], [[0.0, 2.0]]])

    scores = activation_scores(activations, bundle)

    assert scores[0, 0, 0] > scores[0, 1, 0]
    assert scores[1, 1, 0] > scores[1, 0, 0]
