import pytest

from emotion_coding_agents.data import emotion_training_snippets


def test_emotion_training_snippets_limits_examples():
    snippets = emotion_training_snippets(["calm"], 2)

    assert list(snippets) == ["calm"]
    assert len(snippets["calm"]) == 2


def test_emotion_training_snippets_rejects_unknown_emotion():
    with pytest.raises(KeyError):
        emotion_training_snippets(["not_an_emotion"], 1)

