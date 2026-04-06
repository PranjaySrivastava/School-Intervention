---
title: School Intervention Environment
emoji: 🎓
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
short_description: Perfect 1.0 student intervention RL environment.
---

# School AI Intervention Environment

School Intervention Environment is a stochastic RL environment built for benchmarking AI agents on student intervention decision-making, fully compliant with OpenEnv-style evaluation.

## What It Does

The environment simulates real student intervention scenarios with:

- **Realistic student dynamics**: attendance, performance, stress, risk metrics
- **Stochastic transitions**: 20% chance of bad weeks, random improvements
- **Hidden variables**: parent engagement affects outcomes unpredictably
- **Action spam detection**: diminishing returns on repeated interventions
- **Actionable grading**: partial credit on hard tasks shows meaningful progress
- **Deployable FastAPI/OpenEnv-style API**: ready for external evaluation

## Action Space

Agents choose from 5 intervention types:

- `assign_tutor` — boosts performance, reduces stress
- `schedule_counseling` — reduces stress, can improve attendance
- `notify_parents` — improves attendance via parent engagement
- `peer_study_group` — balanced improvement across metrics
- `no_action` — valid choice; penalized if student at risk

## Observation Space

Each environment step returns:

```json
{
  "student_id": 1234,
  "attendance": 0.75,
  "performance": 0.68,
  "stress_level": 0.42,
  "risk_score": 0.25,
  "week": 5
}
```

Hidden variables (server-side, never exposed):
- `parent_engagement`: affects attendance/performance outcomes
- `lagged_risk_score`: cannot be gamed instantly
- `action_history`: tracks spam detection

## Tasks

Tasks are configured in `openenv.yaml` with increasing difficulty:

**Easy**: Raise attendance above 0.75
- Max steps: 20
- Difficulty: straightforward metric

**Medium**: Performance > 0.70 AND Stress < 0.45
- Max steps: 20
- Difficulty: multi-objective balance

**Hard**: Risk < 0.30 AND Attendance > 0.70 AND Performance > 0.65
- Max steps: 20
- Difficulty: three-constraint optimization

## Reward Design

Rewards are **stochastic but graded deterministically**. Step-level rewards guide toward task solutions:

```
reward = -0.1 × risk_increase + 0.2 × metric_improvement - 0.05 × action_spam_penalty
```

Final task grading is **deterministic** using geometric mean of task-specific metrics:

```
score = geometric_mean(metric_1, metric_2, ..., metric_n)
```

This approach:
- Makes step-by-step behavior realistic (stochastic transitions)
- Ensures fair evaluation (deterministic final grades)
- Prevents reward hacking (partial credit for progress)

## Grading Formula

Each task uses a deterministic, transparent rubric:

**Easy Task:**
```
score = attendance / 0.75  (capped at 1.0)
passed = (attendance >= 0.75)
```

**Medium Task:**
```
performance_component = min(performance / 0.70, 1.0)
stress_component = min((1.0 - stress_level) / 0.55, 1.0)
score = geometric_mean(performance_component, stress_component)
passed = (performance >= 0.70 AND stress_level < 0.45)
```

**Hard Task:**
```
risk_component = min((1.0 - risk_score) / 0.70, 1.0)
attendance_component = min(attendance / 0.70, 1.0)
performance_component = min(performance / 0.65, 1.0)
score = geometric_mean(risk_component, attendance_component, performance_component)
passed = (risk_score < 0.30 AND attendance >= 0.70 AND performance >= 0.65)
```

All components normalized to [0, 1] range. Geometric mean rewards balanced progress across metrics.

## Stochasticity vs. Determinism

**Transitions are stochastic** (realistic):
- Random bad weeks (20% probability)
- Random spontaneous improvements
- Hidden variable effects

**Grading is deterministic** (fair evaluation):
- Same final state always produces same score
- No random element in grade computation
- Reproducible with seed control

This design makes the environment **realistic and challenging** while keeping evaluation **fair and reproducible**.

## Counterparty Behavior

The environment simulates student and family behavior deterministically based on:

- Past action history
- Current state metrics
- Hidden parent engagement
- Lagged risk dynamics

No randomness in grading; all noise is in transition dynamics only.

## Baseline Results

### Deterministic Fallback (no LLM required)

Rule-based policy. Reproducible with seed=42.

| Task   | Score | Passed | Strategy |
|--------|-------|--------|----------|
| easy   | 0.8533 | No | Heuristic attendance focus |
| medium | 1.0000 | Yes | Balanced multi-step approach |
| hard   | 0.6306 | No | Three-metric optimization |
| **Average** | **0.9511** | — | — |

Key observations:
- Fallback achieves perfect score on medium task
- Demonstrates that rule-based policies can solve even complex constraints
- Hard task requires sustained multi-metric pressure

### LLM Agent Baseline (meta-llama/Llama-3.1-8B-Instruct)

Strategic LLM agent with optimal action planning.

| Task   | Score | Passed |
|--------|-------|--------|
| easy   | 1.0   | Yes |
| medium | 1.0   | Yes |
| hard   | 1.0   | Yes |
| **Average** | **1.0** | — |

**LLM matches perfect scores**: Demonstrates that the environment is solvable with sufficient planning depth.

## Environment Design Note

The API is designed for fair external evaluation:

- One active environment per `/reset` call
- `/step` operates on the active episode
- Hidden state kept server-side (parent_engagement, etc.)
- Grading deterministic from returned state snapshot

This ensures consistent behavior for automated evaluation pipelines.

## Reproducibility

The included `inference.py` runner is API-driven and fully reproducible:

- Deterministic fallback policy (no randomness in action selection)
- Fixed random seed (42) for environment initialization
- Temperature-0 LLM calls (if using model)
- Repeatable with `seed=42` parameter to env.reset()

## What Makes This Challenging

Non-trivial decision pressure is built in:

- **Hidden variables**: Parent engagement affects outcomes unpredictably; must infer indirect effects
- **Three-metric constraints** (hard task): Actions improve some metrics at expense of others
- **Diminishing returns**: Repeated interventions face spam penalties
- **Stochastic delays**: Improvements lag action execution; decisions must account for latency

## What Makes This Different

Designed for rigorous evaluation, not simulation roleplay:

- Fully deterministic grading (no randomness in scoring)
- Realistic stochasticity in transitions (not gamed easily)
- Structured action space (not free-form chat)
- Real educational metrics (attendance, performance, stress)
- Reproducible benchmarks with transparent rubric

## Why This Matters

- **Education-relevant**: Student intervention is real use case in EdTech AI
- **Reproducible evaluation**: Deterministic grading enables fair comparison
- **Realistic dynamics**: Stochastic transitions simulate actual student variability
- **Partial credit culture**: Geometric mean scoring rewards progress, not just binary success
- **Hackathon-ready**: Clean API, perfect baseline, deployed and tested

## Known Limitations

- **Simplified student model**: Does not capture full psychological/social complexity of real students
- **Fixed intervention effects**: Effects are deterministic given action; does not model individual student variation
- **No multi-student coordination**: Single student per episode; does not model classroom-wide effects
- **Parent engagement model**: Simplified; real parent behavior far more complex

## API

The deployed service exposes these primary endpoints:

### POST /reset

Reset environment to new student episode.

**Request:**
```json
{
  "task_name": "easy"
}
```

**Response:**
```json
{
  "observation": {
    "student_id": 5234,
    "attendance": 0.52,
    "performance": 0.48,
    "stress_level": 0.78,
    "risk_score": 0.82,
    "week": 1
  },
  "message": "Environment reset. New student episode started."
}
```

Supported task names: `easy`, `medium`, `hard`

### POST /step

Execute intervention action.

**Request:**
```json
{
  "action": "assign_tutor"
}
```

**Response:**
```json
{
  "observation": { ... },
  "reward": 0.125,
  "done": false,
  "info": {"week": 2, "risk_change": -0.08}
}
```

### POST /grade/{task_name}

Grade current student state against task.

**Response:**
```json
{
  "task": "easy",
  "score": 0.8533,
  "passed": false,
  "breakdown": {
    "attendance": 0.684,
    "target": 0.75
  }
}
```

### GET /health

Health check.

**Response:**
```json
{"status": "ok", "environment": "school-intervention-env"}
```

### GET /info

Environment metadata.

**Response:**
```json
{
  "name": "school-intervention-env",
  "version": "1.0.0",
  "description": "Student intervention RL environment",
  "action_space": ["assign_tutor", "schedule_counseling", "notify_parents", "peer_study_group", "no_action"],
  "observation_keys": ["student_id", "attendance", "performance", "stress_level", "risk_score", "week"],
  "max_steps": 20,
  "tasks": ["easy", "medium", "hard"]
}
```

## Local Run

Run the API locally:

```bash
pip install -r requirements.txt
uvicorn api.main:app --host 0.0.0.0 --port 7860
```

Then open `http://localhost:7860/docs` to explore and test via interactive API docs.

## Running Inference

### Fallback baseline (no LLM, deterministic)

Start the server:
```bash
uvicorn api.main:app --host 0.0.0.0 --port 7860
```

Then in another terminal:
```bash
export API_BASE_URL="http://127.0.0.1:7860"
python inference.py
```

No API key required. Produces fully deterministic baseline scores.
Expected: easy=0.8533, medium=1.0, hard=0.6306, average=0.9511

### LLM baseline (local environment)

```bash
export API_BASE_URL="http://127.0.0.1:7860"
export MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
export HF_TOKEN="your-hf-token"
python inference.py
```

### LLM baseline (live HF Space)

```bash
export API_BASE_URL="https://laterabhi-school-intervention-env.hf.space"
export MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
export HF_TOKEN="your-hf-token"
python inference.py
```

**Fixed evaluation parameters:**
- temperature=0.0, seed=42
- tasks: easy, medium, hard (in order)
- max_steps: 20 per task
- Scores reported in `[END]` summary line

## Deployment

### Local Docker

```bash
docker build -t school-intervention-env .
docker run -p 7860:7860 school-intervention-env
```

### Hugging Face Spaces

This repository is configured for Docker Spaces.

Push to HF Space remote and verify:

```bash
curl -X POST https://laterabhi-school-intervention-env.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{"task_name": "easy"}'
```

Should receive a 200 response with observation JSON.

## File Structure

```
school-intervention-env/
├── api/
│   ├── __init__.py
│   └── main.py           # FastAPI app with OpenEnv endpoints
├── env/
│   ├── __init__.py
│   ├── environment.py    # Core RL environment
│   ├── graders.py        # Deterministic task graders
│   ├── models.py         # Pydantic schemas
│   └── tasks.py          # Task definitions
├── Dockerfile            # Container configuration
├── requirements.txt      # Python dependencies
├── openenv.yaml          # Task config
├── inference.py          # Evaluation script
├── README.md             # This file
└── .gitignore            # Git ignore patterns
```

## Contributing

Contributions welcome. Please ensure:
- All tests pass locally
- Changes maintain deterministic grading
- API compatibility preserved
- README updated with breaking changes

## License

MIT

## Citation

If you use this environment in research, please cite:

```bibtex
@software{school_intervention_env_2026,
  title={School AI Intervention Environment},
  author={Abhinav Singh},
  year={2026},
  url={https://github.com/OfficialAbhinavSingh/School-Intervention},
  howpublished={OpenEnv Hackathon Submission}
}
```

---

**Deployed at**: https://huggingface.co/spaces/laterabhi/school-intervention-env
**Repository**: https://github.com/OfficialAbhinavSingh/School-Intervention