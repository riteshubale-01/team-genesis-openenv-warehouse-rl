"""Reward-first grading utilities for Warehouse OpenEnv.

Follows the GridMind scoring pipeline:
  1) Per-step raw rewards are normalized via running min/max into (REWARD_MIN, REWARD_MAX)
  2) Episode score = mean(normalized_rewards) → clamp → round(4)
  3) All scores strictly in open interval (0, 1) — never 0.0 or 1.0
"""

from __future__ import annotations
import math
from typing import Dict, Any, List, Optional

from environment import MAX_STEPS_MAP, MAX_TOTAL_TASKS, R_PICKUP, R_STEP_CORRECT, R_TASK_COMPLETE


# ── Score Constants (aligned with GridMind) ──────────────────────────────────

# Reward range per step after normalization: (0.10, 0.90)
REWARD_MIN = 0.10
REWARD_MAX = 0.90

# Score clamp buffer — never output exactly 0.0 or 1.0
SCORE_EPSILON = 0.01


# ── GridMind-style scoring functions ─────────────────────────────────────────

def clamp_open_score(score: float) -> float:
    """Clamp score to strictly between 0 and 1 (never 0.0 or 1.0).

    Mirrors GridMind's clamp_open_score() exactly.
    """
    if score <= 0.0:
        return SCORE_EPSILON
    if score >= 1.0:
        return 1.0 - SCORE_EPSILON
    return score


def normalize_reward(raw_reward: float, raw_min: float, raw_max: float) -> float:
    """Normalize a single raw reward to (REWARD_MIN, REWARD_MAX) range.

    Uses running min/max from the episode to scale rewards, matching
    GridMind's normalize_reward().
    """
    if raw_max == raw_min:
        return (REWARD_MIN + REWARD_MAX) / 2
    normalized = (raw_reward - raw_min) / (raw_max - raw_min)
    normalized = normalized * (REWARD_MAX - REWARD_MIN) + REWARD_MIN
    return clamp_open_score(normalized)


def compute_score(rewards: List[float]) -> float:
    """Return mean reward clamped strictly to (0.01, 0.99).

    Mirrors GridMind's compute_score() exactly:
      mean → round(4) → clamp
    """
    if not rewards:
        return SCORE_EPSILON
    mean_reward = sum(rewards) / len(rewards)
    return clamp_open_score(round(mean_reward, 4))


# ── Payload formatting ──────────────────────────────────────────────────────

def format_task_score(score: float) -> Dict[str, float]:
    """Format a single task score payload with validator-safe bounds."""
    safe = clamp_open_score(round(float(score), 4))
    return {
        "score": safe,
        "task_score": safe,
    }


def format_tasks_payload(tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Format full tasks payload, including aggregate_score.

    Accepts per-task input as either:
      - score: pre-computed episode score (preferred — from compute_score)
      - raw_reward + difficulty: legacy path, normalizes via max_possible_reward
    """
    clean_tasks: List[Dict[str, float]] = []
    for t in tasks:
        if "score" in t:
            # Preferred path: score already computed GridMind-style
            safe = clamp_open_score(round(float(t["score"]), 4))
            clean_tasks.append({"score": safe, "task_score": safe})
        elif "raw_reward" in t:
            # Legacy path: normalize raw reward against theoretical max
            raw_reward = float(t.get("raw_reward", 0.0))
            max_possible_reward = t.get("max_possible_reward")
            if max_possible_reward is None:
                difficulty = str(t.get("difficulty", "easy"))
                max_possible_reward = max_possible_reward_for_difficulty(difficulty)
            max_possible_reward = float(max_possible_reward)
            if max_possible_reward <= 0:
                safe = SCORE_EPSILON
            else:
                normalized = raw_reward / max_possible_reward
                safe = clamp_open_score(round(normalized, 4))
            clean_tasks.append({"score": safe, "task_score": safe})
        else:
            clean_tasks.append(format_task_score(SCORE_EPSILON))

    if clean_tasks:
        aggregate_raw = sum(t["score"] for t in clean_tasks) / len(clean_tasks)
    else:
        aggregate_raw = SCORE_EPSILON
    aggregate_score = clamp_open_score(round(aggregate_raw, 4))
    return {
        "tasks": clean_tasks,
        "aggregate_score": aggregate_score,
    }


def max_possible_reward_for_difficulty(difficulty: str) -> float:
    """Conservative max-reward estimate used for fallback score conversion."""
    diff = difficulty if difficulty in MAX_STEPS_MAP else "easy"
    max_steps = MAX_STEPS_MAP[diff]
    max_tasks = MAX_TOTAL_TASKS.get(diff, 1)

    # Upper bound from always-correct movement and completing all possible tasks.
    return (max_steps * R_STEP_CORRECT) + (max_tasks * (R_PICKUP + R_TASK_COMPLETE))


# ── Legacy API (kept for backward compatibility with compute_score callers) ──

def compute_score_legacy(
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

    if max_possible_reward <= 0:
        score = SCORE_EPSILON
    else:
        normalized = total_reward / max_possible_reward
        score = clamp_open_score(round(normalized, 4))

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
        score = SCORE_EPSILON
        return {"score": score, "task_score": score}

    last = episode_info_list[-1]
    # Prefer cumulative reward if present; otherwise reconstruct from per-step raw rewards.
    if "total_reward_raw" in last:
        raw_total = float(last.get("total_reward_raw", 0.0))
    else:
        raw_total = sum(float(i.get("reward_raw", 0.0)) for i in episode_info_list)

    if max_possible_reward <= 0:
        score = SCORE_EPSILON
    else:
        normalized = raw_total / max_possible_reward
        score = clamp_open_score(round(normalized, 4))
    return {"score": score, "task_score": score}
