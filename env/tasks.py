from typing import Dict, Any
from env.models import TaskDefinition, TaskDifficulty


TASKS = {
    "easy": TaskDefinition(
        name="Attendance Recovery",
        difficulty=TaskDifficulty.easy,
        description="Raise attendance above 0.75 within 20 steps.",
        success_criteria={"attendance": {"min": 0.75}},
        max_steps=20,
    ),
    "medium": TaskDefinition(
        name="Academic Stabilization",
        difficulty=TaskDifficulty.medium,
        description="Raise performance above 0.70 AND reduce stress below 0.45.",
        success_criteria={
            "performance": {"min": 0.70},
            "stress_level": {"max": 0.45},
        },
        max_steps=20,
    ),
    "hard": TaskDefinition(
        name="Full Risk Remediation",
        difficulty=TaskDifficulty.hard,
        description="Reduce risk_score below 0.30 while keeping attendance above 0.70 and performance above 0.65.",
        success_criteria={
            "risk_score": {"max": 0.30},
            "attendance": {"min": 0.70},
            "performance": {"min": 0.65},
        },
        max_steps=20,
    ),
}


def get_task(name: str) -> TaskDefinition:
    if name not in TASKS:
        raise ValueError(f"Unknown task '{name}'. Available: {list(TASKS.keys())}")
    return TASKS[name]


def list_tasks() -> Dict[str, TaskDefinition]:
    return TASKS


def check_task_success(task_name: str, state: Dict[str, Any]) -> bool:
    task = get_task(task_name)
    for metric, conditions in task.success_criteria.items():
        value = state.get(metric)
        if value is None:
            return False
        if "min" in conditions and value < conditions["min"]:
            return False
        if "max" in conditions and value > conditions["max"]:
            return False
    return True