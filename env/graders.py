from typing import Dict, Any


def _strict_open_unit_interval(score: float) -> float:
    eps = 1e-4
    if score <= 0.0:
        return eps
    if score >= 1.0:
        return 1.0 - eps
    return score


def grade_easy(state: Dict[str, Any]) -> Dict[str, Any]:
    attendance = state.get("attendance", 0.0)
    score = (attendance - 0.30) / (0.75 - 0.30)
    score = max(0.0, min(1.0, score))
    score = _strict_open_unit_interval(score)
    return {
        "score": round(score, 4),
        "passed": attendance >= 0.75,
        "breakdown": {"attendance": round(attendance, 3), "target": 0.75},
    }


def grade_medium(state: Dict[str, Any]) -> Dict[str, Any]:
    performance = state.get("performance", 0.0)
    stress = state.get("stress_level", 1.0)
    perf_score = max(0.0, min(1.0, (performance - 0.25) / (0.70 - 0.25)))
    stress_score = max(0.0, min(1.0, (0.90 - stress) / (0.90 - 0.45)))
    combined = (perf_score * stress_score) ** 0.5
    combined = _strict_open_unit_interval(combined)
    return {
        "score": round(combined, 4),
        "passed": performance >= 0.70 and stress <= 0.45,
        "breakdown": {
            "performance": round(performance, 3),
            "stress_level": round(stress, 3),
            "performance_component": round(perf_score, 4),
            "stress_component": round(stress_score, 4),
        },
    }


def grade_hard(state: Dict[str, Any]) -> Dict[str, Any]:
    risk = state.get("risk_score", 1.0)
    attendance = state.get("attendance", 0.0)
    performance = state.get("performance", 0.0)
    risk_score = max(0.0, min(1.0, (0.80 - risk) / (0.80 - 0.30)))
    att_score = max(0.0, min(1.0, (attendance - 0.30) / (0.70 - 0.30)))
    perf_score = max(0.0, min(1.0, (performance - 0.25) / (0.65 - 0.25)))
    combined = 0.50 * risk_score + 0.25 * att_score + 0.25 * perf_score
    conditions_met = sum([risk <= 0.30, attendance >= 0.70, performance >= 0.65])
    if conditions_met == 2 and combined > 0.6:
        combined *= 0.75
    combined = _strict_open_unit_interval(combined)
    return {
        "score": round(combined, 4),
        "passed": risk <= 0.30 and attendance >= 0.70 and performance >= 0.65,
        "breakdown": {
            "risk_score": round(risk, 3),
            "attendance": round(attendance, 3),
            "performance": round(performance, 3),
            "conditions_met": f"{conditions_met}/3",
        },
    }


GRADERS = {"easy": grade_easy, "medium": grade_medium, "hard": grade_hard}


def grade(task_name: str, state: Dict[str, Any]) -> Dict[str, Any]:
    if task_name not in GRADERS:
        raise ValueError(f"Unknown task '{task_name}'.")
    result = GRADERS[task_name](state)
    result["task"] = task_name
    return result