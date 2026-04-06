import os
import json
import time
import requests
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:7860")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN", "")

ACTIONS = ["assign_tutor", "schedule_counseling", "notify_parents", "peer_study_group", "no_action"]

client = OpenAI(
    base_url=os.getenv("API_BASE_URL_LLM", "https://router.huggingface.co/v1"),
    api_key=HF_TOKEN if HF_TOKEN else "dummy",
)

_last_actions = []


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

    else:  # hard task
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
Last 2 actions taken: {_last_actions[-2:]}

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
        print(f"  [WARN] LLM failed: {e} — using fallback")
        chosen = _fallback(state, task)
        _last_actions.append(chosen)
        return chosen


def env_reset():
    r = requests.post(f"{API_BASE_URL}/reset")
    r.raise_for_status()
    return r.json()["observation"]


def env_step(action: str):
    r = requests.post(f"{API_BASE_URL}/step", json={"action": action})
    r.raise_for_status()
    d = r.json()
    return d["observation"], d["reward"], d["done"], d["info"]


def env_grade(task_name: str):
    r = requests.post(f"{API_BASE_URL}/grade/{task_name}")
    r.raise_for_status()
    return r.json()


def run_episode(task: str) -> dict:
    global _last_actions
    _last_actions = []

    print(f"\n{'='*50}\n  TASK: {task.upper()}\n{'='*50}")
    state = env_reset()
    print(f"  Start: att={state['attendance']:.2f} perf={state['performance']:.2f} stress={state['stress_level']:.2f} risk={state['risk_score']:.2f}")
    total_reward = 0.0

    for step_num in range(1, 21):
        action = choose_action(state, task, step_num)
        state, reward, done, info = env_step(action)
        total_reward += reward
        print(f"  Step {step_num:2d} | {action:<22} | reward={reward:+.3f} | risk={state['risk_score']:.2f}")
        if done:
            print(f"  Episode finished early at step {step_num}")
            break

    result = env_grade(task)
    print(f"  Score: {result['score']:.4f}  Passed: {result['passed']}")
    print(f"  Breakdown: {result['breakdown']}")
    return {
        "task": task,
        "score": result["score"],
        "passed": result["passed"],
        "total_reward": round(total_reward, 4),
        "breakdown": result["breakdown"],
    }


def main():
    print(f"\nSCHOOL AI INTERVENTION — BASELINE INFERENCE")
    print(f"API: {API_BASE_URL}  |  Model: {MODEL_NAME}")

    try:
        r = requests.get(f"{API_BASE_URL}/health", timeout=10)
        r.raise_for_status()
        print(f"Server: {r.json()['status']}")
    except Exception as e:
        print(f"[ERROR] Cannot reach server: {e}")
        return

    start = time.time()
    results = [run_episode(t) for t in ["easy", "medium", "hard"]]
    elapsed = time.time() - start

    print(f"\n{'='*50}\n  FINAL SCORES\n{'='*50}")
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"  {r['task']:<8} score={r['score']:.4f}  [{status}]  reward={r['total_reward']:.3f}")
    avg = sum(r["score"] for r in results) / 3
    print(f"  {'average':<8} score={avg:.4f}")
    print(f"\n  Runtime: {elapsed:.1f}s")

    with open("inference_results.json", "w") as f:
        json.dump({
            "results": results,
            "average_score": round(avg, 4),
            "runtime_seconds": round(elapsed, 1),
            "model": MODEL_NAME,
            "api": API_BASE_URL,
        }, f, indent=2)
    print("  Saved: inference_results.json")


if __name__ == "__main__":
    main()