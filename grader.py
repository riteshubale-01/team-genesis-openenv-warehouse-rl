import json


# ================================
# 🔒 FINAL SAFE CONVERSION LAYER
# ================================
def convert_reward_to_score(raw_reward, max_possible_reward):
    """
    Converts raw reward into a strictly valid score:
    Ensures: 0 < score < 1
    """

    # Safety fallback
    if max_possible_reward <= 0:
        return 0.01

    try:
        normalized = raw_reward / max_possible_reward
    except Exception:
        return 0.01

    # HARD CLIPPING (validator-safe)
    if normalized <= 0:
        normalized = 0.01
    elif normalized >= 1:
        normalized = 0.99

    # Round to 2 decimal places
    score = round(normalized, 2)

    # FINAL GUARD (extra safety after rounding)
    if score <= 0:
        score = 0.01
    elif score >= 1:
        score = 0.99

    return score


# ================================
# 🧠 CORE GRADING FUNCTION
# ================================
def compute_score(task_result, difficulty="medium"):
    """
    Computes score for a single task.
    This is the ONLY place where score is generated.
    """

    # Extract raw reward safely
    total_reward = task_result.get("total_reward", 0)

    # -------------------------------
    # 🎯 DEFINE MAX POSSIBLE REWARD
    # -------------------------------
    # Adjust based on your environment
    # Example logic (safe default):

    max_steps = task_result.get("max_steps", 20)
    step_reward = task_result.get("step_reward", 0.2)
    pickup_reward = task_result.get("pickup_reward", 1.0)
    dropoff_reward = task_result.get("dropoff_reward", 1.0)

    max_possible_reward = (
        max_steps * step_reward
        + pickup_reward
        + dropoff_reward
    )

    # -------------------------------
    # 🧠 DIFFICULTY MULTIPLIER
    # -------------------------------
    difficulty_multipliers = {
        "easy": 0.9,
        "medium": 1.0,
        "hard": 1.1
    }

    diff_mult = difficulty_multipliers.get(difficulty, 1.0)

    # Apply multiplier ONLY to reward (not final score)
    adjusted_reward = total_reward * diff_mult

    # -------------------------------
    # 🔒 FINAL CONVERSION (MANDATORY)
    # -------------------------------
    score = convert_reward_to_score(
        adjusted_reward,
        max_possible_reward
    )

    return {
        "score": score,
        "task_score": score
    }


# ================================
# 📊 AGGREGATE SCORING
# ================================
def aggregate_scores(task_scores):
    """
    Aggregates multiple task scores into final score
    """

    if not task_scores:
        return 0.01

    total = 0
    count = 0

    for task in task_scores:
        s = task.get("task_score", 0)

        # SAFETY: ensure valid range
        if s <= 0:
            s = 0.01
        elif s >= 1:
            s = 0.99

        total += s
        count += 1

    avg_score = total / count if count > 0 else 0.01

    # FINAL CLIP
    if avg_score <= 0:
        avg_score = 0.01
    elif avg_score >= 1:
        avg_score = 0.99

    return round(avg_score, 2)


# ================================
# 📦 MAIN GRADING ENTRY POINT
# ================================
def grade_all(tasks):
    """
    Takes list of task results and returns final JSON
    """

    results = []
    scores = []

    for task in tasks:
        difficulty = task.get("difficulty", "medium")

        graded = compute_score(task, difficulty)

        results.append({
            "task_id": task.get("task_id", ""),
            "score": graded["score"],
            "task_score": graded["task_score"]
        })

        scores.append(graded)

    aggregate = aggregate_scores(results)

    return {
        "tasks": results,
        "aggregate_score": aggregate
    }


# ================================
# 🧪 LOCAL TEST (OPTIONAL)
# ================================
if __name__ == "__main__":
    sample_tasks = [
        {
            "task_id": "t1",
            "total_reward": 2.5,
            "difficulty": "easy"
        },
        {
            "task_id": "t2",
            "total_reward": 0,
            "difficulty": "hard"
        }
    ]

    output = grade_all(sample_tasks)
    print(json.dumps(output, indent=2))