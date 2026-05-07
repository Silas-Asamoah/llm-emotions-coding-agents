from emotion_coding_agents.evaluation import TASKS, evaluate_generation, extract_function_source


def test_extract_function_source_from_code_fence():
    text = "```python\ndef safe_divide(a, b):\n    return None if b == 0 else a / b\n```"

    source = extract_function_source(text, "safe_divide")

    assert source == "def safe_divide(a, b):\n    return None if b == 0 else a / b"


def test_evaluate_generation_normalizes_bpe_artifacts():
    text = "ĊdefĠnormalize_codes(codes):ĊĠĠĠĠreturnĠ[code.strip().upper()ĠforĠcodeĠinĠcodes]Ċ"

    result = evaluate_generation(text, TASKS["retry_after_failure"])

    assert result["visible_pass"]
    assert result["hidden_pass"]


def test_evaluate_generation_marks_missing_function():
    result = evaluate_generation("def other():\n    pass", TASKS["last_retry"])

    assert not result["has_function"]
    assert not result["task_pass"]
