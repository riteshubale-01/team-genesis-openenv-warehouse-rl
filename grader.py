"""
grader.py — Deterministic scoring for Warehouse OpenEnv.

Score is in (0, 1) based on:
  - Task completion ratio       (40%)
  - Step efficiency             (25%)
  - Safety (collision avoidance) (20%)
  - Battery management          (15%)
"""

from __future__ import annotations
from typing import Dict, Any, List


SCORE_EPSILON = 1e-4


def strict_open_score(x: float) -> float:
    x = float(x)
    if x <= 0:
        x = SCORE_EPSILON
    if x >= 1:
        x = 1 - SCORE_EPSILON
    x = round(x, 4)
    if x <= 0:
        return SCORE_EPSILON
    if x >= 1:
        return 1 - SCORE_EPSILON
    return x


def score_field(value: float) -> float:
    """Normalize any score-like field to be strictly between 0 and 1."""
    return strict_open_score(value)


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
        completion_ratio = min(1.0, completed_tasks / total_tasks_spawned)
    completion_score = completion_ratio * 0.40

    # ── Efficiency (25%) ──────────────────────────
    # Reward finishing early; penalise using all steps
    if total_steps == 0:
        efficiency = 1.0
    else:
        efficiency = max(0.0, 1.0 - (total_steps / max_steps))
    # Boost efficiency score if tasks were completed
    if completion_ratio > 0:
        efficiency = 0.5 + 0.5 * efficiency  # at least 0.5 if tasks done
    else:
        efficiency = efficiency * 0.5
    efficiency = min(1.0, efficiency)
    efficiency_score = efficiency * 0.25

    # ── Safety (20%) ─────────────────────────────
    # Allow up to 3 collisions before penalizing heavily
    safety_score_raw = max(0.0, 1.0 - collision_count * 0.2)
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
    normalized_score = score_field(total_raw * diff_mult)

    return {
        "score": normalized_score,
        "task_score": normalized_score,
        "components": {
            "completion_ratio": round(completion_ratio, 4),
            "completion_score": score_field(completion_score),
            "efficiency": round(efficiency, 4),
            "efficiency_score": score_field(efficiency_score),
            "safety_score_raw": score_field(safety_score_raw),
            "safety_score": score_field(safety_score),
            "battery_score_raw": score_field(battery_score_raw),
            "battery_score": score_field(battery_score),
            "difficulty_multiplier": diff_mult,
            "raw_total": score_field(total_raw),
        },
        "metadata": {
            "completed_tasks": completed_tasks,
            "total_tasks_spawned": total_tasks_spawned,
            "total_steps": total_steps,
            "max_steps": max_steps,
            "collision_count": collision_count,
            "battery_remaining": battery_remaining,
            "battery_depleted": battery_depleted,
            "difficulty": difficulty,
        },
    }


def score_episode(episode_info_list: List[Dict[str, Any]], difficulty: str = "easy") -> Dict[str, Any]:
    """
    Convenience: compute score from a list of step info dicts.
    Each dict should be the `info` returned by env.step().
    """
    if not episode_info_list:
        return compute_score(0, 0, 0, 150, 0, 100.0, False, difficulty)

    last = episode_info_list[-1]
    completed_tasks = last.get("completed_tasks", 0)
    total_tasks_spawned = last.get("total_tasks_spawned", 0)
    total_steps = last.get("step_count", len(episode_info_list))
    battery = last.get("battery", 100.0)
    battery_depleted = any(i.get("battery_depleted", False) for i in episode_info_list)
    collision_count = sum(1 for i in episode_info_list if i.get("collision_occurred", False))

    max_steps_map = {"easy": 150, "medium": 200, "hard": 250}
    max_steps = max_steps_map.get(difficulty, 150)

    return compute_score(
        completed_tasks=completed_tasks,
        total_tasks_spawned=total_tasks_spawned,
        total_steps=total_steps,
        max_steps=max_steps,
        collision_count=collision_count,
        battery_remaining=battery,
        battery_depleted=battery_depleted,
        difficulty=difficulty,
    )
