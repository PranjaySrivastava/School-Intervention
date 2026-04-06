from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from enum import Enum


class Action(str, Enum):
    assign_tutor = "assign_tutor"
    schedule_counseling = "schedule_counseling"
    notify_parents = "notify_parents"
    peer_study_group = "peer_study_group"
    no_action = "no_action"


class Observation(BaseModel):
    student_id: int
    attendance: float = Field(..., ge=0.0, le=1.0)
    performance: float = Field(..., ge=0.0, le=1.0)
    stress_level: float = Field(..., ge=0.0, le=1.0)
    risk_score: float = Field(..., ge=0.0, le=1.0)
    week: int


class StepRequest(BaseModel):
    action: Action


class StepResponse(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: Dict[str, Any] = {}


class ResetResponse(BaseModel):
    observation: Observation
    message: str = "Environment reset. New student episode started."


class GradeResponse(BaseModel):
    task: str
    score: float = Field(..., ge=0.0, le=1.0)
    passed: bool
    breakdown: Dict[str, Any] = {}


class TaskDifficulty(str, Enum):
    easy = "easy"
    medium = "medium"
    hard = "hard"


class TaskDefinition(BaseModel):
    name: str
    difficulty: TaskDifficulty
    description: str
    success_criteria: Dict[str, Any]
    max_steps: int


class TaskListResponse(BaseModel):
    tasks: List[TaskDefinition]


class EnvInfo(BaseModel):
    name: str
    version: str
    description: str
    action_space: List[str]
    observation_keys: List[str]
    max_steps: int
    tasks: List[str]