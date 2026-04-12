"""Reward-first grading utilities for Warehouse OpenEnv.

This module follows a GridMind-style pipeline:
1) accumulate raw environment rewards during rollouts
2) convert raw totals to validator-safe scores only at the end
"""

from __future__ import annotations
import math
from typing import Dict, Any, List, Optional

from environment import MAX_STEPS_MAP, MAX_TOTAL_TASKS, R_PICKUP, R_STEP_CORRECT, R_TASK_COMPLETE


SCORE_EPSILON = 1e-6
SCORE_MIN = 0.000001    # must be > 0.0
SCORE_MAX = 0.999999    # must be < 1.0


def convert_reward_to_score(raw_reward: float, max_possible_reward: float) -> float:
    """Convert cumulative raw reward to an OpenEnv-safe score in (0, 1).

    Returns a value strictly within (SCORE_MIN, SCORE_MAX) so the
    OpenEnv validator never sees 0.0 or 1.0.
    """
    try:
        raw_reward = float(raw_reward)
        max_possible_reward = float(max_possible_reward)
    except Exception:
        return SCORE_MIN

    if not math.isfinite(raw_reward) or not math.isfinite(max_possible_reward):
        return SCORE_MIN
    if max_possible_reward <= 0:
        return SCORE_MIN

    # Safe normalization into [0, 1]
    normalized = raw_reward / max_possible_reward

    # Strict clamp to (SCORE_MIN, SCORE_MAX) — never 0.0 or 1.0
    clamped = max(SCORE_MIN, min(SCORE_MAX, normalized))

    # Round to 6 decimal places (matches openenv.yaml precision) and re-clamp
    final_score = round(clamped, 6)
    final_score = max(SCORE_MIN, min(SCORE_MAX, final_score))

    return final_score


def format_task_score(score: float) -> Dict[str, float]:
    """Format a single task score payload with validator-safe bounds."""
    safe = convert_reward_to_score(score, 1.0)
    return {
        "score": safe,
        "task_score": safe,
    }


def format_tasks_payload(tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Format full tasks payload, including aggregate_score, via one conversion path.

    Preferred input per task:
      - raw_reward: float
      - difficulty: easy|medium|hard
      - optional max_possible_reward override
    Fallback input per task:
      - score: pre-normalized value in [0,1]
    """
    clean_tasks: List[Dict[str, float]] = []
    for t in tasks:
        if "raw_reward" in t:
            raw_reward = float(t.get("raw_reward", 0.0))
            max_possible_reward = t.get("max_possible_reward")
            if max_possible_reward is None:
                difficulty = str(t.get("difficulty", "easy"))
                max_possible_reward = max_possible_reward_for_difficulty(difficulty)
            safe = convert_reward_to_score(raw_reward, float(max_possible_reward))
            clean_tasks.append({"score": safe, "task_score": safe})
        else:
            clean_tasks.append(format_task_score(float(t.get("score", SCORE_EPSILON))))

    if clean_tasks:
        aggregate_raw = sum(t["score"] for t in clean_tasks) / len(clean_tasks)
    else:
        aggregate_raw = SCORE_EPSILON
    aggregate_score = convert_reward_to_score(aggregate_raw, 1.0)
    return {
        "tasks": clean_tasks,
        "aggregate_score": aggregate_score,
    }


def max_possible_reward_for_difficulty(difficulty: str) -> float:
    """Conservative max-reward estimate used for final score conversion."""
    diff = difficulty if difficulty in MAX_STEPS_MAP else "easy"
    max_steps = MAX_STEPS_MAP[diff]
    max_tasks = MAX_TOTAL_TASKS.get(diff, 1)

    # Upper bound from always-correct movement and completing all possible tasks.
    return (max_steps * R_STEP_CORRECT) + (max_tasks * (R_PICKUP + R_TASK_COMPLETE))


def compute_score(
    completed_tasks: int,
    total_tasks_spawned: int,
    total_steps: int,
    max_steps: int,
    collision_count: int,
    battery_remaining: float,
    battery_depleted: bool,
    difficulty: str,
    total_reward: Optional[float] = None,
    max_possible_reward: Optional[float] = None,
) -> Dict[str, Any]:
    """Compute score by converting final raw reward at the grading boundary."""
    if total_reward is None:
        # Compatibility path for callers that still pass legacy metrics only.
        completion_ratio = (completed_tasks / total_tasks_spawned) if total_tasks_spawned > 0 else 0.0
        remaining_steps = (max_steps - total_steps) if max_steps > total_steps else 0
        battery_term = 0.0 if battery_depleted else (battery_remaining / 100.0)
        total_reward = (
            completion_ratio * 5.0
            + (remaining_steps * 0.01)
            - collision_count * 0.5
            + battery_term
        )

    if max_possible_reward is None:
        max_possible_reward = max_possible_reward_for_difficulty(difficulty)

    score = convert_reward_to_score(total_reward, max_possible_reward)

    return {
        "score": score,
        "task_score": score,
    }


def score_episode(episode_info_list: List[Dict[str, Any]], difficulty: str = "easy") -> Dict[str, Any]:
    """
    Convenience: compute score from a list of step info dicts.
    Each dict should be the `info` returned by env.step().
    """
    max_possible_reward = max_possible_reward_for_difficulty(difficulty)
    if not episode_info_list:
        score = convert_reward_to_score(0.0, max_possible_reward)
        return {"score": score, "task_score": score}

    last = episode_info_list[-1]
    # Prefer cumulative reward if present; otherwise reconstruct from per-step raw rewards.
    if "total_reward_raw" in last:
        raw_total = float(last.get("total_reward_raw", 0.0))
    else:
        raw_total = sum(float(i.get("reward_raw", 0.0)) for i in episode_info_list)

    score = convert_reward_to_score(raw_total, max_possible_reward)
    return {"score": score, "task_score": score}
