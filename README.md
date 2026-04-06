# School AI Intervention Environment

An OpenEnv-compliant RL environment for AI-driven student intervention decisions.

## Quick Start
```bash
pip install -r requirements.txt
uvicorn api.main:app --host 0.0.0.0 --port 7860
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | /reset | Start new episode |
| POST | /step | Execute action |
| GET  | /state | Current state |
| POST | /grade/{task} | Score current state |
| GET  | /health | Health check |

## Actions

- `assign_tutor` — boosts performance
- `schedule_counseling` — reduces stress
- `notify_parents` — improves attendance
- `peer_study_group` — balanced improvement
- `no_action` — penalized if student at risk

## Tasks

- **easy** — raise attendance above 0.75
- **medium** — performance above 0.70 AND stress below 0.45
- **hard** — risk below 0.30, attendance above 0.70, performance above 0.65

## Run Baseline Agent
```bash
export API_BASE_URL="http://localhost:7860"
export MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
export HF_TOKEN="your_token_here"
python inference.py
```

## Docker
```bash
docker build -t school-env .
docker run -p 7860:7860 school-env
```