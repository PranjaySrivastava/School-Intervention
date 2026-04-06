from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


def test_health_endpoint_returns_ok():
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["environment"] == "school-intervention-env"


def test_reset_endpoint_returns_observation_shape():
    response = client.post("/reset")

    assert response.status_code == 200
    payload = response.json()
    observation = payload["observation"]

    for key in ["student_id", "attendance", "performance", "stress_level", "risk_score", "week"]:
        assert key in observation

    assert 0.0 <= observation["attendance"] <= 1.0
    assert 0.0 <= observation["performance"] <= 1.0
    assert 0.0 <= observation["stress_level"] <= 1.0
    assert 0.0 <= observation["risk_score"] <= 1.0


def test_step_endpoint_rejects_invalid_action():
    response = client.post("/step", json={"action": "bad_action"})

    # Invalid enum value is rejected by request validation.
    assert response.status_code == 422


def test_grade_unknown_task_returns_400():
    response = client.post("/grade/unknown")

    assert response.status_code == 400
    assert "Unknown task" in response.json()["detail"]


def test_info_endpoint_contract():
    response = client.get("/info")

    assert response.status_code == 200
    payload = response.json()

    assert payload["name"] == "school-intervention-env"
    assert payload["max_steps"] == 20
    assert payload["tasks"] == ["easy", "medium", "hard"]