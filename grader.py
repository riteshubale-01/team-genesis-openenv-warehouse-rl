"""
grader.py — Deterministic scoring for Warehouse OpenEnv.

Score is in (0, 1) based on:
  - Task completion ratio       (40%)
  - Step efficiency             (25%)
  - Safety (collision avoidance) (20%)
  - Battery management          (15%)
"""

from __future__ import annotations
import math
from typing import Dict, Any, List


SCORE_EPSILON = 1e-6


def finalize_score(x):
    try:
        x = float(x)
    except:
        return 0.01

    # Step 1: round to 2 decimals
    x = round(x, 2)

    # Step 2: enforce strict range
    if x <= 0:
        return 0.01
    if x >= 1:
        return 0.99

    return x


def clamp_open01(x):
    return finalize_score(x)


def strict_open_score(x: float) -> float:
    return clamp_open01(x)


def compute_score(
    completed_tasks: int,
    total_tasks_spawned: int,
    total_steps: int,
    max_steps: int,
    collision_count: int,
    battery_remaining: float,
    battery_depleted: bool,
    difficulty: str,
) -> Dict[str, Any]:
    """
    Compute a deterministic score in (0, 1).

    Parameters
    ----------
    completed_tasks    : number of tasks successfully delivered
    total_tasks_spawned: total tasks that appeared this episode
    total_steps        : steps taken
    max_steps          : episode step limit
    collision_count    : number of collision events
    battery_remaining  : battery % at end of episode
    battery_depleted   : whether battery hit 0
    difficulty         : easy | medium | hard

    Returns
    -------
    dict with 'score' (float strictly between 0 and 1) and component breakdown
    """
    # ── Task completion (40%) ──────────────────────
    if total_tasks_spawned == 0:
        completion_ratio = 0.0
    else:
        completion_ratio = completed_tasks / total_tasks_spawned
    completion_score = completion_ratio * 0.40

    # ── Efficiency (25%) ──────────────────────────
    # Reward finishing early; penalise using all steps
    if total_steps == 0:
        efficiency = 1.0
    else:
        efficiency = 1.0 - (total_steps / max_steps)
    # Boost efficiency score if tasks were completed
    if completion_ratio > 0:
        efficiency = 0.5 + 0.5 * efficiency  # at least 0.5 if tasks done
    else:
        efficiency = efficiency * 0.5
    efficiency_score = efficiency * 0.25

    # ── Safety (20%) ─────────────────────────────
    # Allow up to 3 collisions before penalizing heavily
    safety_score_raw = 1.0 - collision_count * 0.2
    safety_score = safety_score_raw * 0.20

    # ── Battery management (15%) ──────────────────
    if battery_depleted:
        battery_score_raw = 0.0
    else:
        battery_score_raw = battery_remaining / 100.0
    battery_score = battery_score_raw * 0.15

    # ── Difficulty multiplier ─────────────────────
    diff_mult = {"easy": 1.0, "medium": 1.1, "hard": 1.2}[difficulty]

    total_raw = completion_score + efficiency_score + safety_score + battery_score
    normalized_score = total_raw * diff_mult

    return {
        "score": normalized_score,
        "task_score": normalized_score,
    }


def score_episode(episode_info_list: List[Dict[str, Any]], difficulty: str = "easy") -> Dict[str, Any]:
    """
    Convenience: compute score from a list of step info dicts.
    Each dict should be the `info` returned by env.step().
    """
    if not episode_info_list:
        normalized_score = 0.5
        return {"score": normalized_score, "task_score": normalized_score}

    last = episode_info_list[-1]
    # Prefer cumulative reward if present; otherwise reconstruct from per-step raw rewards.
    if "total_reward_raw" in last:
        raw_total = float(last.get("total_reward_raw", 0.0))
    else:
        raw_total = sum(float(i.get("reward_raw", 0.0)) for i in episode_info_list)

    # Raw total reward -> smooth score in (0,1), centered near 0.5 for neutral episodes.
    normalized_score = 1.0 / (1.0 + math.exp(-raw_total / 5.0))
    return {"score": float(normalized_score), "task_score": float(normalized_score)}
