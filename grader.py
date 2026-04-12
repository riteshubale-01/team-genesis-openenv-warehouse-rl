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


def convert_reward_to_score(raw_reward: float, max_possible_reward: float) -> float:
    """Convert cumulative raw reward to an OpenEnv-safe score in (0, 1)."""
    try:
        raw_reward = float(raw_reward)
        max_possible_reward = float(max_possible_reward)
    except Exception:
        return 0.01

    if not math.isfinite(raw_reward) or not math.isfinite(max_possible_reward):
        return 0.01
    if max_possible_reward <= 0:
        return 0.01

    # 1) safe normalization
    normalized = raw_reward / max_possible_reward

    # 2) explicit edge handling
    if normalized <= 0:
        normalized = 0.01
    if normalized >= 1:
        normalized = 0.99

    # 3) strict clip
    normalized = min(0.99, max(0.01, normalized))

    # 4) round
    final_score = round(normalized, 2)

    # 5) re-guard after rounding
    if final_score <= 0:
        return 0.01
    if final_score >= 1:
        return 0.99

    return final_score


def max_possible_reward_for_difficulty(difficulty: str) -> float:
    """Conservative max-reward estimate used for final score conversion."""
    diff = difficulty if difficulty in MAX_STEPS_MAP else "easy"
    max_steps = MAX_STEPS_MAP[diff]
    max_tasks = MAX_TOTAL_TASKS.get(diff, 1)

    # Upper bound from always-correct movement and completing all possible tasks.
    return (max_steps * R_STEP_CORRECT) + (max_tasks * (R_PICKUP + R_TASK_COMPLETE))


def finalize_score(x: float) -> float:
    """Backward-compatible helper for already-normalized scores."""
    return convert_reward_to_score(x, 1.0)


def clamp_open01(x: float) -> float:
    return finalize_score(x)


def strict_open_score(x: float) -> float:
    return finalize_score(x)


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
        remaining_steps = max(0, max_steps - total_steps)
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
