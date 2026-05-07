from emotion_coding_agents.agent_harness import (
    AGENT_TASKS,
    _attempt_seed,
    _conditions,
    _format_tests,
)


def test_agent_tasks_have_visible_and_hidden_tests():
    assert AGENT_TASKS
    assert all(task.visible_tests and task.hidden_tests for task in AGENT_TASKS)


def test_conditions_include_baseline_and_nonzero_steering():
    config = {
        "experiment": {
            "steering_emotions": ["desperate"],
            "steering_strengths": [-1.0, 0.0, 1.0],
        }
    }

    conditions = _conditions(config)

    assert conditions == [(None, 0.0), ("desperate", -1.0), ("desperate", 1.0)]


def test_format_tests_includes_args_and_expected_values():
    formatted = _format_tests([(((1, 2),), 3)])

    assert "args=" in formatted
    assert "-> 3" in formatted


def test_attempt_seed_depends_on_logical_attempt_not_run_order():
    first = _attempt_seed(13, "parse_duration", "baseline", 0)
    same = _attempt_seed(13, "parse_duration", "baseline", 0)
    next_attempt = _attempt_seed(13, "parse_duration", "baseline", 1)
    other_condition = _attempt_seed(13, "parse_duration", "calm_+1.0", 0)

    assert first == same
    assert first != next_attempt
    assert first != other_condition
