import random
from typing import Tuple, Dict, Any


class SchoolInterventionEnv:

    MAX_STEPS = 20

    ACTIONS = [
        "assign_tutor",
        "schedule_counseling",
        "notify_parents",
        "peer_study_group",
        "no_action",
    ]

    def __init__(self, seed: int = None):
        if seed is not None:
            random.seed(seed)
        self._state: Dict[str, Any] = {}
        self._hidden: Dict[str, Any] = {}
        self._step_count = 0
        self._prev_risk = 1.0
        self._action_history = []
        self.reset()

    def reset(self) -> Dict[str, Any]:
        self._step_count = 0
        self._action_history = []
        self._state = {
            "student_id": random.randint(1000, 9999),
            "attendance": round(random.uniform(0.3, 0.65), 2),
            "performance": round(random.uniform(0.25, 0.60), 2),
            "stress_level": round(random.uniform(0.55, 0.90), 2),
            "risk_score": round(random.uniform(0.60, 0.95), 2),
            "week": 1,
        }
        self._hidden = {
            "parent_engagement": round(random.uniform(0.1, 0.5), 2),
            "consecutive_same_action": 0,
            "last_action": None,
        }
        self._prev_risk = self._state["risk_score"]
        return self._get_observation()

    def step(self, action: str) -> Tuple[Dict, float, bool, Dict]:
        if action not in self.ACTIONS:
            raise ValueError(f"Invalid action '{action}'. Must be one of {self.ACTIONS}")

        reward = 0.0
        info = {}

        if action == self._hidden["last_action"] and action != "no_action":
            self._hidden["consecutive_same_action"] += 1
        else:
            self._hidden["consecutive_same_action"] = 0
        self._hidden["last_action"] = action
        self._action_history.append(action)

        if action == "assign_tutor":
            self._state["performance"] = min(0.99, self._state["performance"] + 0.08)
            self._state["stress_level"] = min(0.99, self._state["stress_level"] + 0.03)
            reward += 0.15

        elif action == "schedule_counseling":
            self._state["stress_level"] = max(0.01, self._state["stress_level"] - 0.18)
            self._state["attendance"] = min(0.99, self._state["attendance"] + 0.04)
            reward += 0.20

        elif action == "notify_parents":
            engagement_boost = 0.12 * (1 - self._hidden["parent_engagement"])
            self._hidden["parent_engagement"] = min(0.99, self._hidden["parent_engagement"] + 0.10)
            self._state["attendance"] = min(0.99, self._state["attendance"] + 0.06 + engagement_boost * 0.3)
            reward += 0.10

        elif action == "peer_study_group":
            self._state["performance"] = min(0.99, self._state["performance"] + 0.05)
            self._state["stress_level"] = max(0.01, self._state["stress_level"] - 0.06)
            reward += 0.12

        elif action == "no_action":
            reward -= 0.05

        if self._hidden["consecutive_same_action"] >= 2:
            reward -= 0.10 * self._hidden["consecutive_same_action"]
            info["warning"] = f"Action '{action}' repeated — diminishing returns"

        self._apply_stochastic_drift()
        self._update_risk_score()

        risk_delta = self._prev_risk - self._state["risk_score"]
        reward += risk_delta * 0.5
        self._prev_risk = self._state["risk_score"]

        self._clamp()
        self._step_count += 1
        self._state["week"] = self._step_count + 1

        done = self._check_done()
        info["week"] = self._state["week"]
        info["parent_engagement"] = round(self._hidden["parent_engagement"], 2)

        return self._get_observation(), round(reward, 4), done, info

    def state(self) -> Dict[str, Any]:
        return self._get_observation()

    def get_action_space(self):
        return self.ACTIONS

    def _get_observation(self) -> Dict[str, Any]:
        return {k: v for k, v in self._state.items()}

    def _apply_stochastic_drift(self):
        if random.random() < 0.20:
            self._state["attendance"] = max(0.01, self._state["attendance"] - random.uniform(0.03, 0.08))
            self._state["stress_level"] = min(0.99, self._state["stress_level"] + random.uniform(0.02, 0.06))
        if random.random() < 0.10:
            self._state["performance"] = min(0.99, self._state["performance"] + random.uniform(0.02, 0.05))

    def _update_risk_score(self):
        raw_risk = (
            (1 - self._state["attendance"]) * 0.30
            + (1 - self._state["performance"]) * 0.35
            + self._state["stress_level"] * 0.25
            + (1 - self._hidden["parent_engagement"]) * 0.10
        )
        self._state["risk_score"] = round(
            0.7 * self._state["risk_score"] + 0.3 * raw_risk, 3
        )

    def _clamp(self):
        for key in ["attendance", "performance", "stress_level", "risk_score"]:
            # Strictly within (0, 1) - never exactly 0.0 or 1.0
            self._state[key] = round(max(0.01, min(0.99, self._state[key])), 3)

    def _check_done(self) -> bool:
        if self._step_count >= self.MAX_STEPS:
            return True
        if (
            self._state["attendance"] >= 0.85
            and self._state["performance"] >= 0.80
            and self._state["stress_level"] <= 0.30
            and self._state["risk_score"] <= 0.25
        ):
            return True
        return False