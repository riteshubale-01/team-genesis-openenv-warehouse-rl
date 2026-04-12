import json


# ================================
# 🔒 FINAL SAFE CONVERSION
# ================================
def convert_reward_to_score(raw_reward, max_possible_reward):
    if max_possible_reward <= 0:
        return 0.01

    normalized = raw_reward / max_possible_reward

    # Clamp BEFORE rounding
    normalized = max(0.01, min(0.99, normalized))

    score = round(normalized, 2)

    # Clamp AFTER rounding
    score = max(0.01, min(0.99, score))

    return score


# ================================
# 🧠 SCORE COMPUTATION
# ================================
def compute_score(task_result, difficulty="medium"):
    total_reward = task_result.get("total_reward", 0)

    max_steps = task_result.get("max_steps", 20)
    step_reward = task_result.get("step_reward", 0.2)
    pickup_reward = task_result.get("pickup_reward", 1.0)
    dropoff_reward = task_result.get("dropoff_reward", 1.0)

    max_possible_reward = (
        max_steps * step_reward
        + pickup_reward
        + dropoff_reward
    )

    difficulty_multipliers = {
        "easy": 0.9,
        "medium": 1.0,
        "hard": 1.1
    }

    adjusted_reward = total_reward * difficulty_multipliers.get(difficulty, 1.0)

    score = convert_reward_to_score(adjusted_reward, max_possible_reward)

    return {
        "score": score,
        "task_score": score
    }


# ================================
# 📊 AGGREGATE SCORE
# ================================
def aggregate_scores(tasks):
    if not tasks:
        return 0.01

    scores = [t["task_score"] for t in tasks]

    avg = sum(scores) / len(scores)

    avg = max(0.01, min(0.99, avg))

    return round(avg, 2)


# ================================
# 📦 FINAL JSON
# ================================
def build_output(tasks):
    aggregate = aggregate_scores(tasks)
    return {
        "tasks": tasks,
        "aggregate_score": aggregate
    }


if __name__ == "__main__":
    sample = [
        {"task_id": "easy", "task_score": 0.4},
        {"task_id": "medium", "task_score": 0.5},
        {"task_id": "hard", "task_score": 0.6},
    ]
    print(json.dumps(build_output(sample), indent=2))