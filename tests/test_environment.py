import pytest

from env.environment import SchoolInterventionEnv


def test_reset_creates_valid_state():
    env = SchoolInterventionEnv(seed=42)
    obs = env.reset()

    assert set(obs.keys()) == {"student_id", "attendance", "performance", "stress_level", "risk_score", "week"}
    assert 0.0 <= obs["attendance"] <= 1.0
    assert 0.0 <= obs["performance"] <= 1.0
    assert 0.0 <= obs["stress_level"] <= 1.0
    assert 0.0 <= obs["risk_score"] <= 1.0
    assert obs["week"] == 1


def test_invalid_action_raises_value_error():
    env = SchoolInterventionEnv(seed=7)

    with pytest.raises(ValueError):
        env.step("invalid")


def test_episode_ends_at_max_steps():
    env = SchoolInterventionEnv(seed=1)
    env.reset()

    done = False
    for _ in range(SchoolInterventionEnv.MAX_STEPS):
        _, _, done, _ = env.step("no_action")

    assert done is True