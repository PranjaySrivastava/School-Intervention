import pytest

from env.graders import grade


def test_grade_easy_passes_for_high_attendance():
    result = grade("easy", {"attendance": 0.80})

    assert result["task"] == "easy"
    assert result["passed"] is True
    assert 0.0 <= result["score"] <= 1.0


def test_grade_medium_passes_for_balanced_state():
    state = {"performance": 0.75, "stress_level": 0.40}
    result = grade("medium", state)

    assert result["task"] == "medium"
    assert result["passed"] is True
    assert 0.0 <= result["score"] <= 1.0


def test_grade_hard_passes_for_low_risk_profile():
    state = {"risk_score": 0.25, "attendance": 0.80, "performance": 0.75}
    result = grade("hard", state)

    assert result["task"] == "hard"
    assert result["passed"] is True
    assert 0.0 <= result["score"] <= 1.0


def test_grade_rejects_unknown_task():
    with pytest.raises(ValueError):
        grade("unknown", {})