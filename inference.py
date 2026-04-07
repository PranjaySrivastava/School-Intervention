import os
import json
import time
import requests
from openai import OpenAI

ENV_BASE_URL = os.getenv("ENV_BASE_URL", "https://laterabhi-school-intervention-env.hf.space")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
API_KEY = os.getenv("API_KEY")

ACTIONS = ["assign_tutor", "schedule_counseling", "notify_parents", "peer_study_group", "no_action"]

LLM_BASE_URL = os.getenv("API_BASE_URL")
if LLM_BASE_URL and API_KEY:
    client = OpenAI(base_url=LLM_BASE_URL, api_key=API_KEY)
elif HF_TOKEN:
    client = OpenAI(base_url="https://router.huggingface.co/v1", api_key=HF_TOKEN)
else:
    client = None

_last_actions = []


def _log(tag: str, message: str) -> None:
    print(f"[{tag}] {message}", flush=True)


def _safe_score(score: float) -> float:
    return round(min(0.98, max(0.02, float(score))), 4)


def _fallback(state: dict, task: str) -> str:
    att = state["attendance"]
    perf = state["performance"]
    stress = state["stress_level"]
    risk = state["risk_score"]

    if task == "easy":
        if att < 0.75:
            return "notify_parents"
        return "peer_study_group"
    elif task == "medium":
        if stress > 0.60:
            return "schedule_counseling"
        elif stress > 0.45:
            return "peer_study_group"
        elif perf < 0.70:
            return "assign_tutor"
        return "peer_study_group"
    else:
        if stress > 0.55:
            return "schedule_counseling"
        elif att < 0.70:
            return "notify_parents"
        elif perf < 0.65:
            return "assign_tutor"
        elif risk > 0.40:
            return "peer_study_group"
        return "schedule_counseling"


def choose_action(state: dict, task: str, step_num: int) -> str:
    global _last_actions

    if client is None:
        chosen = _fallback(state, task)
        _last_actions.append(chosen)
        return chosen

    prompt = f"""You are an AI school counselor. Help a struggling student.

Student state (week {step_num}/20):
- Attendance: {state['attendance']:.2f}
- Performance: {state['performance']:.2f}
- Stress Level: {state['stress_level']:.2f}
- Risk Score: {state['risk_score']:.2f}

Task: {task}
- easy: raise attendance above 0.75
- medium: raise performance above 0.70 AND stress below 0.45
- hard: risk_score below 0.30, attendance above 0.70, performance above 0.65

Available actions: assign_tutor, schedule_counseling, notify_parents, peer_study_group, no_action
IMPORTANT: Do NOT repeat the same action more than twice in a row.
Last 2 actions: {_last_actions[-2:]}

Reply with ONLY the action name, nothing else."""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=20,
            temperature=0.1,
        )
        raw = response.choices[0].message.content.strip().lower()
        for action in ACTIONS:
            if action in raw:
                chosen = action
                if len(_last_actions) >= 2 and _last_actions[-1] == chosen and _last_actions[-2] == chosen:
                    chosen = _fallback(state, task)
                _last_actions.append(chosen)
                return chosen
        chosen = _fallback(state, task)
        _last_actions.append(chosen)
        return chosen
    except Exception as e:
        _log("STEP", f"warn=llm_failed detail={str(e)[:80].replace(' ','_')} fallback=true")
        chosen = _fallback(state, task)
        _last_actions.append(chosen)
        return chosen


def env_reset():
    r = requests.post(f"{ENV_BASE_URL}/reset", timeout=30)
    r.raise_for_status()
    return r.json()["observation"]


def env_step(action: str):
    r = requests.post(f"{ENV_BASE_URL}/step", json={"action": action}, timeout=30)
    r.raise_for_status()
    d = r.json()
    return d["observation"], d["reward"], d["done"], d["info"]


def env_grade(task_name: str):
    r = requests.post(f"{ENV_BASE_URL}/grade/{task_name}", timeout=30)
    r.raise_for_status()
    return r.json()


def run_episode(task: str) -> dict:
    global _last_actions
    _last_actions = []

    state = env_reset()
    _log("START", f"task={task} week={state['week']} attendance={state['attendance']:.3f} performance={state['performance']:.3f} stress={state['stress_level']:.3f} risk={state['risk_score']:.3f}")
    total_reward = 0.0

    for step_num in range(1, 21):
        action = choose_action(state, task, step_num)
        state, reward, done, info = env_step(action)
        total_reward += reward
        _log("STEP", f"task={task} step={step_num} action={action} done={str(done).lower()} week={state['week']} risk={state['risk_score']:.3f}")
        if done:
            break

    result = env_grade(task)
    safe = _safe_score(result["score"])
    _log("END", f"task={task} score={safe:.4f} passed={str(result['passed']).lower()}")
    return {
        "task": task,
        "score": safe,
        "passed": result["passed"],
        "breakdown": result["breakdown"],
    }


def main():
    _log("START", f"run=all_tasks env_api={ENV_BASE_URL} model={MODEL_NAME}")

    try:
        r = requests.get(f"{ENV_BASE_URL}/health", timeout=10)
        r.raise_for_status()
        _log("STEP", f"health_status={r.json()['status']}")
    except Exception as e:
        _log("END", f"status=error reason=server_unreachable detail={str(e)[:60].replace(' ','_')}")
        return

    if client is not None:
        try:
            client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": "Reply with only: ok"}],
                max_tokens=5,
                temperature=0.0,
            )
            _log("STEP", "llm_proxy_check=ok")
        except Exception as e:
            _log("STEP", f"llm_proxy_check=failed detail={str(e)[:60].replace(' ','_')}")

    start = time.time()
    results = [run_episode(t) for t in ["easy", "medium", "hard"]]
    elapsed = time.time() - start

    avg = _safe_score(sum(r["score"] for r in results) / 3)
    _log("END", f"run=all_tasks average_score={avg:.4f} runtime_seconds={elapsed:.1f}")

    with open("inference_results.json", "w") as f:
        json.dump({
            "results": results,
            "average_score": round(avg, 4),
            "runtime_seconds": round(elapsed, 1),
            "model": MODEL_NAME,
            "api": ENV_BASE_URL,
        }, f, indent=2)
    _log("END", "status=results_saved file=inference_results.json")


if __name__ == "__main__":
    main()