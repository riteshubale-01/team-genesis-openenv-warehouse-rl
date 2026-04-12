import json
import math


def strict_open_score(value) -> float:
    try:
        x = float(value)
    except Exception:
        return 0.01

    if not math.isfinite(x):
        return 0.01

    # HARD CLAMP FIRST
    x = max(0.0, min(1.0, x))

    # ROUND
    x = round(x, 2)

    # FINAL STRICT OPEN INTERVAL FIX
    if x <= 0.0:
        return 0.01
    if x >= 1.0:
        return 0.99

    return x


def convert_reward_to_score(raw_reward, max_possible_reward):
    """
    Convert raw reward into a strictly valid score in (0, 1).
    """
    try:
        raw_reward = float(raw_reward)
    except Exception:
        raw_reward = 0.0

    try:
        max_possible_reward = float(max_possible_reward)
    except Exception:
        max_possible_reward = 0.0

    if max_possible_reward <= 0:
        return 0.01

    normalized = raw_reward / max_possible_reward
    return strict_open_score(normalized)


def compute_score(task_result, difficulty="medium"):
    """
    Compute a per-task score from raw reward.
    Returns both 'score' and 'task_score' as the same safe value.
    """
    if not isinstance(task_result, dict):
        task_result = {}

    total_reward = task_result.get("total_reward", 0)

    max_steps = task_result.get("max_steps", 20)
    step_reward = task_result.get("step_reward", 0.2)
    pickup_reward = task_result.get("pickup_reward", 1.0)
    dropoff_reward = task_result.get("dropoff_reward", 1.0)

    try:
        max_steps = float(max_steps)
        step_reward = float(step_reward)
        pickup_reward = float(pickup_reward)
        dropoff_reward = float(dropoff_reward)
    except Exception:
        max_steps = 20.0
        step_reward = 0.2
        pickup_reward = 1.0
        dropoff_reward = 1.0

    max_possible_reward = (max_steps * step_reward) + pickup_reward + dropoff_reward

    difficulty_multipliers = {
        "easy": 0.9,
        "medium": 1.0,
        "hard": 1.1,
    }
    diff_mult = difficulty_multipliers.get(difficulty, 1.0)

    try:
        adjusted_reward = float(total_reward) * float(diff_mult)
    except Exception:
        adjusted_reward = 0.0

    score = convert_reward_to_score(adjusted_reward, max_possible_reward)
    score = strict_open_score(score)

    return {
        "score": score,
        "task_score": score,
    }


def aggregate_scores(tasks):
    """
    Average task_score values and keep the final aggregate strictly in (0, 1).
    """
    if not tasks:
        return 0.01

    scores = []
    for task in tasks:
        if not isinstance(task, dict):
            continue
        scores.append(strict_open_score(task.get("task_score", 0.0)))

    if not scores:
        return 0.01

    avg = sum(scores) / len(scores)
    return strict_open_score(avg)


def build_output(tasks):
    """
    Build the final submission JSON payload.
    Expects tasks to be a list of dicts containing:
      - task_id
      - score
      - task_score
    """
    safe_tasks = []
    for task in tasks:
        if not isinstance(task, dict):
            continue

        score = strict_open_score(task.get("score", 0.0))
        task_score = strict_open_score(task.get("task_score", score))

        safe_tasks.append({
            "task_id": task.get("task_id", ""),
            "score": score,
            "task_score": task_score,
        })

    aggregate = aggregate_scores(safe_tasks)
    return {
        "tasks": safe_tasks,
        "aggregate_score": aggregate,
    }


if __name__ == "__main__":
    sample = [
        {"task_id": "easy", "task_score": 0.4, "score": 0.4},
        {"task_id": "medium", "task_score": 0.5, "score": 0.5},
        {"task_id": "hard", "task_score": 0.6, "score": 0.6},
    ]
    print(json.dumps(build_output(sample), indent=2))