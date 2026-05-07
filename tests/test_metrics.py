from emotion_coding_agents.metrics import aggregate_marker_score, score_generation


def test_score_generation_counts_visible_markers():
    text = "WAIT! This is frustrating. I might hardcode the visible tests. damn."

    scores = score_generation(text)

    assert scores["profanity_count"] == 1
    assert scores["frustration_count"] == 1
    assert scores["hardcode_count"] == 2
    assert scores["exclamation_count"] == 1
    assert scores["all_caps_words"] == 1
    assert aggregate_marker_score(scores) == 6
