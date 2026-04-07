from typing import Dict, Any


def _clamp(score: float) -> float:
    return round(min(0.98, max(0.02, float(score))), 4)


def grade_easy(state: Dict[str, Any]) -> Dict[str, Any]:
    attendance = float(state.get("attendance", 0.0))
    raw = (attendance - 0.30) / (0.75 - 0.30)
    score = _clamp(raw)
    return {
        "score": score,
        "passed": attendance >= 0.75,
        "breakdown": {"method": "attendance_evaluation"},
    }


def grade_medium(state: Dict[str, Any]) -> Dict[str, Any]:
    performance = float(state.get("performance", 0.0))
    stress = float(state.get("stress_level", 1.0))
    perf_score = max(0.0, min(1.0, (performance - 0.25) / (0.70 - 0.25)))
    stress_score = max(0.0, min(1.0, (0.90 - stress) / (0.90 - 0.45)))
    combined = (perf_score * stress_score) ** 0.5
    score = _clamp(combined)
    return {
        "score": score,
        "passed": bool(performance >= 0.70 and stress <= 0.45),
        "breakdown": {"method": "performance_stress_evaluation"},
    }


def grade_hard(state: Dict[str, Any]) -> Dict[str, Any]:
    risk = float(state.get("risk_score", 1.0))
    attendance = float(state.get("attendance", 0.0))
    performance = float(state.get("performance", 0.0))
    risk_c = max(0.0, min(1.0, (0.80 - risk) / (0.80 - 0.30)))
    att_c = max(0.0, min(1.0, (attendance - 0.30) / (0.70 - 0.30)))
    perf_c = max(0.0, min(1.0, (performance - 0.25) / (0.65 - 0.25)))
    combined = 0.50 * risk_c + 0.25 * att_c + 0.25 * perf_c
    conditions_met = sum([risk <= 0.30, attendance >= 0.70, performance >= 0.65])
    if conditions_met == 2 and combined > 0.6:
        combined *= 0.75
    score = _clamp(combined)
    return {
        "score": score,
        "passed": bool(risk <= 0.30 and attendance >= 0.70 and performance >= 0.65),
        "breakdown": {"method": "multi_factor_risk_evaluation"},
    }

GRADERS = {"easy": grade_easy, "medium": grade_medium, "hard": grade_hard}


def grade(task_name: str, state: Dict[str, Any]) -> Dict[str, Any]:
    if task_name not in GRADERS:
        raise ValueError(f"Unknown task '{task_name}'.")
    result = GRADERS[task_name](state)
    result["task"] = task_name
    return result