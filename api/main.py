from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from env.environment import SchoolInterventionEnv
from env.models import (
    StepRequest, StepResponse, ResetResponse,
    Observation, GradeResponse, TaskListResponse, EnvInfo,
)
from env.tasks import list_tasks
from env.graders import grade


env = SchoolInterventionEnv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    env.reset()
    yield


app = FastAPI(
    title="School AI Intervention Environment",
    description="OpenEnv-compliant RL environment for student intervention.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/reset", response_model=ResetResponse)
def reset():
    obs = env.reset()
    return ResetResponse(observation=Observation(**obs))


@app.post("/step", response_model=StepResponse)
def step(request: StepRequest):
    try:
        obs, reward, done, info = env.step(request.action.value)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return StepResponse(
        observation=Observation(**obs),
        reward=reward,
        done=done,
        info=info,
    )


@app.get("/state", response_model=Observation)
def state():
    return Observation(**env.state())


@app.get("/tasks", response_model=TaskListResponse)
def get_tasks():
    return TaskListResponse(tasks=list(list_tasks().values()))


@app.post("/grade/{task_name}", response_model=GradeResponse)
def grade_state(task_name: str):
    if task_name not in ["easy", "medium", "hard"]:
        raise HTTPException(status_code=400, detail=f"Unknown task '{task_name}'")
    result = grade(task_name, env.state())
    # Double-clamp: validator requires strictly (0, 1) — never 0.0 or 1.0
    safe_score = round(min(0.98, max(0.02, float(result["score"]))), 4)
    return GradeResponse(
        task=result["task"],
        score=safe_score,
        passed=result["passed"],
        breakdown=result["breakdown"],
    )


@app.get("/info", response_model=EnvInfo)
def env_info():
    return EnvInfo(
        name="school-intervention-env",
        version="1.0.0",
        description="Simulate student intervention decisions.",
        action_space=SchoolInterventionEnv.ACTIONS,
        observation_keys=["student_id", "attendance", "performance", "stress_level", "risk_score", "week"],
        max_steps=SchoolInterventionEnv.MAX_STEPS,
        tasks=["easy", "medium", "hard"],
    )


@app.get("/health")
def health():
    return {"status": "ok", "environment": "school-intervention-env"}


@app.get("/logs-container")
def logs_container():
    return {"logs": "Application running normally"}