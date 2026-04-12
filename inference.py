"""
inference.py — Warehouse RL baseline inference script.
Compliant with OpenEnv hackathon output format requirements.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from openai import OpenAI

from grader import build_output, compute_score


# ─────────────────────────────────────────
# CONFIG — reads env vars per hackathon spec
# ─────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN     = os.environ.get("HF_TOKEN")          # mandatory, no default
SERVER_URL   = os.environ.get("ENV_SERVER_URL", "http://localhost:7860")

TASK_NAME  = "warehouse_delivery"
BENCHMARK  = "WarehouseRL-v1"

ACTION_NAMES = {
    0: "MOVE_UP", 1: "MOVE_DOWN", 2: "MOVE_LEFT", 3: "MOVE_RIGHT",
    4: "PICK",    5: "DROP",      6: "RECHARGE",  7: "WAIT",
}

MAX_STEPS_BY_DIFFICULTY = {"easy": 150, "medium": 200, "hard": 250}
QTABLE_PATH = Path("mnt") / "q_table.json"


# ─────────────────────────────────────────
# CLIENT
# ─────────────────────────────────────────
def create_client() -> OpenAI:
    if not HF_TOKEN:
        raise ValueError("HF_TOKEN environment variable is required")
    return OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)


# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────
def _manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _direction_action(from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> int:
    fr, fc = from_pos
    tr, tc = to_pos
    if tr < fr: return 0
    if tr > fr: return 1
    if tc < fc: return 2
    if tc > fc: return 3
    return 7


def parse_action(text: str) -> Optional[int]:
    for ch in text:
        if ch.isdigit():
            value = int(ch)
            if 0 <= value <= 7:
                return value
    return None


# ─────────────────────────────────────────
# ENV API
# ─────────────────────────────────────────
def env_reset(seed: int, difficulty: str) -> Dict[str, Any]:
    response = requests.post(
        f"{SERVER_URL}/reset",
        json={"seed": seed, "difficulty": difficulty},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def env_step(action: int) -> Dict[str, Any]:
    response = requests.post(
        f"{SERVER_URL}/step",
        json={"action": action},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


# ─────────────────────────────────────────
# Q-LEARNING AGENT
# ─────────────────────────────────────────
class QLearningAgent:
    def __init__(
        self,
        alpha: float = 0.15,
        gamma: float = 0.97,
        epsilon: float = 0.30,
        min_epsilon: float = 0.05,
        epsilon_decay: float = 0.995,
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.q_table: Dict[str, List[float]] = {}
        self._rng = random.Random(42)

    def load(self, path: Path) -> None:
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            table = data.get("q_table", {})
            if isinstance(table, dict):
                cleaned: Dict[str, List[float]] = {}
                for state, values in table.items():
                    if isinstance(values, list) and len(values) == 8:
                        cleaned[state] = [float(v) for v in values]
                self.q_table = cleaned
                self.epsilon = float(data.get("epsilon", self.epsilon))
        except Exception:
            self.q_table = {}

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "q_table": self.q_table,
            "epsilon": self.epsilon,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "min_epsilon": self.min_epsilon,
            "epsilon_decay": self.epsilon_decay,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def encode_state(self, obs: Dict[str, Any]) -> str:
        robot = obs.get("robot_pos", {})
        rr = int(robot.get("row", 0))
        rc = int(robot.get("col", 0))
        battery = float(obs.get("battery", 100.0))
        battery_bucket = max(0, min(9, int(battery // 10)))
        carrying = 1 if obs.get("carrying_item") else 0
        tasks = obs.get("active_tasks", [])
        active_count_bucket = min(3, len(tasks))
        nearest_dist_bucket = 4
        priority_bucket = 0
        dir_bucket = 4
        if tasks:
            best_task = self._select_reference_task(obs)
            if best_task:
                target = self._task_goal(best_task, carrying)
                if target:
                    dist = _manhattan((rr, rc), target)
                    nearest_dist_bucket = min(4, dist // 2)
                    priority_bucket = int(best_task.get("priority", 1))
                    dir_bucket = _direction_action((rr, rc), target)
        return (
            f"r{rr}_c{rc}_b{battery_bucket}_k{carrying}_"
            f"t{active_count_bucket}_d{nearest_dist_bucket}_p{priority_bucket}_dir{dir_bucket}"
        )

    def _select_reference_task(self, obs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        tasks = obs.get("active_tasks", [])
        if not tasks:
            return None
        robot = obs.get("robot_pos", {})
        rr = int(robot.get("row", 0))
        rc = int(robot.get("col", 0))
        def key_fn(task):
            priority = int(task.get("priority", 1))
            pickup = task.get("pickup_pos", {})
            dropoff = task.get("dropoff_pos", {})
            target = pickup if not task.get("picked_up", False) else dropoff
            dist = _manhattan((rr, rc), (int(target.get("row", rr)), int(target.get("col", rc))))
            return (-priority, dist)
        return sorted(tasks, key=key_fn)[0]

    def _task_goal(self, task: Dict[str, Any], carrying: int) -> Optional[Tuple[int, int]]:
        if carrying and task.get("picked_up", False):
            pos = task.get("dropoff_pos", {})
            return (int(pos.get("row", 0)), int(pos.get("col", 0)))
        if not task.get("picked_up", False):
            pos = task.get("pickup_pos", {})
            return (int(pos.get("row", 0)), int(pos.get("col", 0)))
        return None

    def _ensure_state(self, state_key: str) -> None:
        if state_key not in self.q_table:
            self.q_table[state_key] = [0.0] * 8

    def choose_action(self, state_key: str, llm_hint: Optional[int]) -> int:
        self._ensure_state(state_key)
        if self._rng.random() < self.epsilon:
            if llm_hint is not None and self._rng.random() < 0.5:
                return llm_hint
            return self._rng.randint(0, 7)
        values = self.q_table[state_key]
        max_q = max(values)
        best_actions = [idx for idx, v in enumerate(values) if v == max_q]
        if llm_hint in best_actions:
            return llm_hint
        return self._rng.choice(best_actions)

    def update(self, state_key: str, action: int, reward: float, next_state_key: str, done: bool) -> None:
        self._ensure_state(state_key)
        self._ensure_state(next_state_key)
        current_q = self.q_table[state_key][action]
        next_max = 0.0 if done else max(self.q_table[next_state_key])
        target = reward + self.gamma * next_max
        self.q_table[state_key][action] = current_q + self.alpha * (target - current_q)

    def decay_exploration(self) -> None:
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)


# ─────────────────────────────────────────
# OVERRIDE CONTROLLER
# ─────────────────────────────────────────
class OverrideController:
    def __init__(self, trigger_threshold: float = 0.65):
        self.trigger_threshold = trigger_threshold

    def should_override(self, obs: Dict[str, Any]) -> Tuple[bool, str, float]:
        battery = float(obs.get("battery", 100.0))
        tasks = obs.get("active_tasks", [])
        step_count = int(obs.get("step_count", 0))
        carrying = bool(obs.get("carrying_item", False))
        battery_risk = 1.0 if battery <= 10.0 else (0.6 if battery <= 18.0 else 0.0)
        urgent_risk = 1.0 if any(int(t.get("priority", 1)) >= 3 for t in tasks) else 0.0
        deadline_risk = 0.0
        if tasks:
            max_steps = MAX_STEPS_BY_DIFFICULTY.get(obs.get("difficulty", "medium"), 200)
            progress = step_count / max(1, max_steps)
            if progress > 0.85:
                deadline_risk = 0.8
            elif progress > 0.7:
                deadline_risk = 0.4
        carrying_risk = 0.2 if carrying and battery < 20 else 0.0
        score = (0.45 * battery_risk + 0.35 * urgent_risk +
                 0.15 * deadline_risk + 0.05 * carrying_risk)
        reason = "none"
        if battery_risk >= 0.6:       reason = "low_battery"
        elif urgent_risk > 0:         reason = "urgent_task"
        elif deadline_risk > 0:       reason = "deadline_pressure"
        return score >= self.trigger_threshold, reason, score

    def override_policy(self, obs: Dict[str, Any]) -> int:
        robot = obs.get("robot_pos", {})
        rr = int(robot.get("row", 0))
        rc = int(robot.get("col", 0))
        battery = float(obs.get("battery", 100.0))
        carrying = bool(obs.get("carrying_item", False))
        if battery <= 18.0:
            charger_targets = [(0, 0), (0, 9), (9, 0), (9, 9)]
            nearest = min(charger_targets, key=lambda p: _manhattan((rr, rc), p))
            if nearest == (rr, rc):
                return 6
            return _direction_action((rr, rc), nearest)
        tasks = obs.get("active_tasks", [])
        if not tasks:
            return 7
        prioritized = sorted(
            tasks,
            key=lambda t: (-int(t.get("priority", 1)),
                           _manhattan((rr, rc), self._task_target(t, carrying))),
        )
        task = prioritized[0]
        target = self._task_target(task, carrying)
        if carrying and task.get("picked_up", False) and (rr, rc) == target:
            return 5
        if (not carrying) and (not task.get("picked_up", False)) and (rr, rc) == target:
            return 4
        return _direction_action((rr, rc), target)

    @staticmethod
    def _task_target(task: Dict[str, Any], carrying: bool) -> Tuple[int, int]:
        if carrying and task.get("picked_up", False):
            pos = task.get("dropoff_pos", {})
            return (int(pos.get("row", 0)), int(pos.get("col", 0)))
        pos = task.get("pickup_pos", {})
        return (int(pos.get("row", 0)), int(pos.get("col", 0)))


# ─────────────────────────────────────────
# LLM HINT
# ─────────────────────────────────────────
def call_llm_action_hint(client: OpenAI, obs: Dict[str, Any], step_n: int) -> Optional[int]:
    robot = obs.get("robot_pos", {})
    tasks = obs.get("active_tasks", [])
    battery = float(obs.get("battery", 100.0))
    carrying = bool(obs.get("carrying_item", False))
    prompt = (
        "Choose one action id for warehouse robot.\n"
        "Actions: 0 up, 1 down, 2 left, 3 right, 4 pick, 5 drop, 6 recharge, 7 wait.\n"
        f"Step={step_n}, battery={battery:.1f}, carrying={carrying}, "
        f"robot=({robot.get('row')},{robot.get('col')}), tasks={len(tasks)}.\n"
        "Respond with only one digit 0-7."
    )
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=6,
        )
        text = response.choices[0].message.content or ""
        return parse_action(text)
    except Exception:
        return None


# ─────────────────────────────────────────
# EPISODE RUNNER
# ─────────────────────────────────────────
def run_episode(
    client: OpenAI,
    q_agent: QLearningAgent,
    override_ctrl: OverrideController,
    difficulty: str,
    seed: int,
) -> Dict[str, Any]:
    task_name = f"{TASK_NAME}_{difficulty}"

    # [START] required format: task= env= model=
    print(f"[START] task={task_name} env={BENCHMARK} model={MODEL_NAME}")

    reset_data = env_reset(seed=seed, difficulty=difficulty)
    obs = reset_data.get("observation", {})
    obs["difficulty"] = difficulty

    done = False
    step_n = 0
    total_reward = 0.0
    step_rewards: List[float] = []
    last_error: Optional[str] = None

    # FIX: use the difficulty-based maximum as both the cap AND what we report
    max_steps = MAX_STEPS_BY_DIFFICULTY.get(difficulty, 200)

    while not done and step_n < max_steps:
        step_n += 1
        state_key = q_agent.encode_state(obs)

        override_active, override_reason, _ = override_ctrl.should_override(obs)
        llm_hint = call_llm_action_hint(client, obs, step_n)

        if override_active:
            action = override_ctrl.override_policy(obs)
        else:
            action = q_agent.choose_action(state_key, llm_hint)

        try:
            step_data = env_step(action)
            last_error = step_data.get("info", {}).get("last_action_error") or None
        except Exception as exc:
            last_error = str(exc)
            step_data = {}

        next_obs = step_data.get("observation", {})
        next_obs["difficulty"] = difficulty
        reward = float(step_data.get("reward", 0.0))
        done = bool(step_data.get("done", False))
        total_reward += reward
        step_rewards.append(reward)

        next_state_key = q_agent.encode_state(next_obs)
        if not override_active:
            q_agent.update(state_key, action, reward, next_state_key, done)

        action_name = ACTION_NAMES.get(action, "WAIT")
        error_str = last_error if last_error else "null"

        # [STEP] required format: step= action= reward= done= error=
        print(
            f"[STEP] step={step_n} action={action_name} "
            f"reward={reward:.2f} done={'true' if done else 'false'} error={error_str}"
        )

        obs = next_obs

    q_agent.decay_exploration()

    rewards_str = ",".join(f"{r:.2f}" for r in step_rewards)

    # [END] required format: success= steps= rewards=
    print(
        f"[END] success={'true' if done else 'false'} "
        f"steps={step_n} rewards={rewards_str}"
    )

    return {
        "total_reward": total_reward,
        # FIX: always the difficulty cap, never the actual step count
        "max_steps": max_steps,
        "step_reward": 0.2,
        "pickup_reward": 0.3,
        "dropoff_reward": 0.3,
        "difficulty": difficulty,
    }


# ─────────────────────────────────────────
# BASELINE RUNNER
# ─────────────────────────────────────────
def run_baseline(
    difficulties: List[str],
    seed: int,
    output_json: str,
) -> Dict[str, Any]:
    client = create_client()

    q_agent = QLearningAgent()
    q_agent.load(QTABLE_PATH)
    override_ctrl = OverrideController()

    tasks: List[Dict[str, Any]] = []

    for idx, diff in enumerate(difficulties):
        result = run_episode(
            client=client,
            q_agent=q_agent,
            override_ctrl=override_ctrl,
            difficulty=diff,
            seed=seed + idx,
        )
        graded = compute_score(result, diff)
        tasks.append({
            "task_id": diff,
            "score": graded["score"],
            "task_score": graded["task_score"],
        })

    q_agent.save(QTABLE_PATH)

    output = build_output(tasks)

    with open(output_json, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)

    print(json.dumps(output, indent=2))
    return output


# ─────────────────────────────────────────
# ENTRY POINT — exactly ONE block
# ─────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Warehouse RL baseline inference")
    parser.add_argument("--difficulties", default="easy,medium,hard")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-json", default="baseline_scores.json")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    difficulties = [d.strip() for d in args.difficulties.split(",") if d.strip()]
    allowed = {"easy", "medium", "hard"}
    invalid = [d for d in difficulties if d not in allowed]
    if not difficulties or invalid:
        raise ValueError(f"Invalid difficulties: {invalid}. Allowed: easy,medium,hard")

    run_baseline(difficulties=difficulties, seed=args.seed, output_json=args.output_json)