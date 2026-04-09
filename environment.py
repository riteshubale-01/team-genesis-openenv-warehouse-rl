"""
Warehouse RL Environment — server/environment.py

Implements:
  - Deterministic seed-based grid world
  - Robot movement, battery, item pick/drop
  - Dynamic obstacle (human) movement
  - Task system with priorities and interruptions
  - Partial observability (local grid view)
  - Three difficulty modes: easy / medium / hard
  - reset(), step(), get_state(), get_observation()
"""

from __future__ import annotations

import random
import math
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from warehouse_env.models import (
    ActionType, CellType, Position, Task, ObstacleState,
    PartialObservation, StepReward, StepInfo, FullState,
    Action,
)


# ─────────────────────────────────────────
# Constants
# ─────────────────────────────────────────

GRID_SIZE = 10          # 10x10 grid
VIEW_RADIUS_MAP = {"easy": 4, "medium": 3, "hard": 2}
MAX_STEPS_MAP   = {"easy": 150, "medium": 200, "hard": 250}
BATTERY_DRAIN   = {"easy": 0.3, "medium": 0.6, "hard": 0.9}   # % per step
BATTERY_MAX     = 100.0
RECHARGE_RATE   = 20.0   # % per recharge action

# Reward constants
R_TASK_COMPLETE    = 10.0
R_PICKUP           = 2.0
R_STEP_PENALTY     = -0.1
R_COLLISION        = -2.0
R_BATTERY_LOW      = -0.5   # when battery < 20%
R_DEAD_BATTERY     = -5.0
R_INVALID_ACTION   = -0.3
R_PRIORITY_2       = 5.0    # bonus for high-priority task
R_PRIORITY_3       = 10.0   # bonus for urgent task
R_EFFICIENCY_BONUS = 2.0    # completing in fewer steps

# The raw step reward is mapped to [0, 1] for API-visible reward outputs.
RAW_REWARD_MIN = R_STEP_PENALTY + R_BATTERY_LOW + R_DEAD_BATTERY + R_COLLISION + R_INVALID_ACTION
RAW_REWARD_MAX = R_STEP_PENALTY + R_TASK_COMPLETE + R_PRIORITY_3 + R_EFFICIENCY_BONUS


# ─────────────────────────────────────────
# Direction helpers
# ─────────────────────────────────────────

DIRECTION_DELTAS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}  # up/down/left/right


def manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


# ─────────────────────────────────────────
# WarehouseEnvironment
# ─────────────────────────────────────────

class WarehouseEnvironment:
    """Deterministic warehouse grid RL environment."""

    def __init__(self):
        self._seed: int = 42
        self._difficulty: str = "easy"
        self._rng: random.Random = random.Random(42)
        self._initialized: bool = False

        # State
        self._grid: List[List[int]] = []   # base layout (no robot/obstacles)
        self._robot_pos: Tuple[int, int] = (0, 0)
        self._battery: float = BATTERY_MAX
        self._carrying_item: bool = False
        self._carrying_task_id: Optional[int] = None
        self._tasks: List[Task] = []
        self._obstacles: List[ObstacleState] = []
        self._step_count: int = 0
        self._done: bool = False
        self._total_reward: float = 0.0
        self._total_reward_raw: float = 0.0
        self._completed_tasks: int = 0
        self._total_tasks_spawned: int = 0
        self._charger_positions: List[Tuple[int, int]] = []
        self._shelf_positions: List[Tuple[int, int]] = []
        self._target_positions: List[Tuple[int, int]] = []

    # ─────────────────────────────────────
    # Public API
    # ─────────────────────────────────────

    def reset(self, seed: int = 42, difficulty: str = "easy") -> PartialObservation:
        self._seed = seed
        self._difficulty = difficulty
        self._rng = random.Random(seed)
        self._step_count = 0
        self._done = False
        self._total_reward = 0.0
        self._total_reward_raw = 0.0
        self._completed_tasks = 0
        self._total_tasks_spawned = 0
        self._carrying_item = False
        self._carrying_task_id = None
        self._battery = BATTERY_MAX
        self._tasks = []
        self._obstacles = []

        self._build_grid()
        self._place_robot()
        self._spawn_initial_tasks()
        self._spawn_obstacles()
        self._initialized = True
        return self.get_observation()

    def step(self, action_int: int) -> Tuple[PartialObservation, float, bool, Dict[str, Any]]:
        """Execute one step. Returns (obs, reward, done, info)."""
        if not self._initialized:
            raise RuntimeError("Call reset() before step()")
        if self._done:
            obs = self.get_observation()
            return obs, 0.0, True, {"reason": "episode_already_done"}

        reward_obj = StepReward(total=0.0)
        info_obj   = StepInfo()

        # Step penalty always
        reward_obj.step_penalty = R_STEP_PENALTY
        reward_obj.total += R_STEP_PENALTY

        # Battery low penalty
        if self._battery < 20.0:
            reward_obj.battery_penalty = R_BATTERY_LOW
            reward_obj.total += R_BATTERY_LOW

        # Validate action
        if action_int not in [a.value for a in ActionType]:
            action_int = ActionType.WAIT.value
            reward_obj.invalid_action_penalty = R_INVALID_ACTION
            reward_obj.total += R_INVALID_ACTION
            info_obj.action_valid = False

        action = ActionType(action_int)

        # Execute action
        if action in (ActionType.MOVE_UP, ActionType.MOVE_DOWN,
                      ActionType.MOVE_LEFT, ActionType.MOVE_RIGHT):
            self._execute_move(action, reward_obj, info_obj)
        elif action == ActionType.PICK:
            self._execute_pick(reward_obj, info_obj)
        elif action == ActionType.DROP:
            self._execute_drop(reward_obj, info_obj)
        elif action == ActionType.RECHARGE:
            self._execute_recharge(reward_obj, info_obj)
        elif action == ActionType.WAIT:
            pass  # no-op

        # Battery drain (except during recharge which adds battery)
        if action != ActionType.RECHARGE:
            drain = BATTERY_DRAIN[self._difficulty]
            self._battery = max(0.0, self._battery - drain)

        if self._battery <= 0.0:
            reward_obj.battery_penalty += R_DEAD_BATTERY
            reward_obj.total += R_DEAD_BATTERY
            info_obj.battery_depleted = True
            self._done = True
            info_obj.reason = "battery_depleted"

        # Move obstacles
        self._move_obstacles()

        # Possibly spawn new task (medium/hard)
        new_task = self._maybe_spawn_task()
        if new_task:
            info_obj.new_task_spawned = new_task.task_id

        self._step_count += 1

        # Check max steps
        if self._step_count >= MAX_STEPS_MAP[self._difficulty]:
            self._done = True
            if not info_obj.reason:
                info_obj.reason = "max_steps_reached"

        # Check all tasks done (easy mode termination)
        if self._difficulty == "easy" and self._all_tasks_done():
            self._done = True
            if not info_obj.reason:
                info_obj.reason = "all_tasks_completed"

        step_reward_raw = reward_obj.total
        step_reward = self._normalize_reward(step_reward_raw)
        self._total_reward_raw += step_reward_raw
        self._total_reward += step_reward

        avg_total_reward = self._total_reward / max(1, self._step_count)

        info_dict: Dict[str, Any] = info_obj.model_dump()
        info_dict["reward_breakdown"] = reward_obj.model_dump()
        info_dict["reward_raw"] = round(step_reward_raw, 4)
        info_dict["total_reward"] = round(avg_total_reward, 4)
        info_dict["total_reward_raw"] = round(self._total_reward_raw, 4)
        info_dict["step_count"] = self._step_count
        info_dict["battery"] = round(self._battery, 2)
        info_dict["completed_tasks"] = self._completed_tasks
        info_dict["total_tasks_spawned"] = self._total_tasks_spawned

        return self.get_observation(), round(step_reward, 4), self._done, info_dict

    def _normalize_reward(self, raw_reward: float) -> float:
        """Scale raw reward to [0, 1] for external API compatibility."""
        if RAW_REWARD_MAX <= RAW_REWARD_MIN:
            return 0.0
        normalized = (raw_reward - RAW_REWARD_MIN) / (RAW_REWARD_MAX - RAW_REWARD_MIN)
        return max(0.0, min(1.0, normalized))

    def get_observation(self) -> PartialObservation:
        """Return partial observation centered on robot."""
        radius = VIEW_RADIUS_MAP[self._difficulty]
        size = 2 * radius + 1
        r, c = self._robot_pos
        local_grid: List[List[int]] = []

        # Build current full grid snapshot
        full = self._get_full_grid_snapshot()

        for dr in range(-radius, radius + 1):
            row_cells = []
            for dc in range(-radius, radius + 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE:
                    row_cells.append(full[nr][nc])
                else:
                    row_cells.append(CellType.WALL.value)
            local_grid.append(row_cells)

        # Visible tasks (pickup in view)
        visible_tasks = []
        for t in self._tasks:
            pr, pc = t.pickup_pos.row, t.pickup_pos.col
            dr_ = abs(pr - r) <= radius
            dc_ = abs(pc - c) <= radius
            if (dr_ and dc_) or not t.completed:
                visible_tasks.append(t)

        return PartialObservation(
            local_grid=local_grid,
            view_radius=radius,
            robot_pos=Position(row=r, col=c),
            battery=round(self._battery, 2),
            carrying_item=self._carrying_item,
            carrying_task_id=self._carrying_task_id,
            active_tasks=[t for t in visible_tasks if not t.completed],
            step_count=self._step_count,
            difficulty=self._difficulty,
        )

    def get_state(self) -> FullState:
        """Full internal state (for /state endpoint)."""
        return FullState(
            grid_size=GRID_SIZE,
            difficulty=self._difficulty,
            robot_pos=Position(row=self._robot_pos[0], col=self._robot_pos[1]),
            battery=round(self._battery, 2),
            carrying_item=self._carrying_item,
            carrying_task_id=self._carrying_task_id,
            tasks=deepcopy(self._tasks),
            obstacles=deepcopy(self._obstacles),
            step_count=self._step_count,
            max_steps=MAX_STEPS_MAP[self._difficulty],
            done=self._done,
            seed=self._seed,
            total_reward=round(self._total_reward / max(1, self._step_count), 4),
            completed_tasks=self._completed_tasks,
            total_tasks_spawned=self._total_tasks_spawned,
        )

    # ─────────────────────────────────────
    # Grid Construction
    # ─────────────────────────────────────

    def _build_grid(self):
        """Build deterministic grid layout seeded by self._rng."""
        G = GRID_SIZE
        self._grid = [[CellType.EMPTY.value] * G for _ in range(G)]
        self._shelf_positions = []
        self._target_positions = []
        self._charger_positions = []

        # Fixed charger at corners
        charger_locs = [(0, 0), (0, G-1), (G-1, 0), (G-1, G-1)]
        for cr, cc in charger_locs:
            self._grid[cr][cc] = CellType.CHARGER.value
            self._charger_positions.append((cr, cc))

        # Place shelves in aisle pattern (rows 2,4,6 cols 2-7)
        for row in [2, 4, 6]:
            for col in range(2, 8):
                self._grid[row][col] = CellType.SHELF.value
                self._shelf_positions.append((row, col))

        # Place target zones (rows 1, 8 — delivery areas)
        for col in range(2, 8, 2):
            self._grid[1][col] = CellType.TARGET.value
            self._target_positions.append((1, col))
            self._grid[8][col] = CellType.TARGET.value
            self._target_positions.append((8, col))

        # Shuffle for variety per seed
        self._rng.shuffle(self._shelf_positions)
        self._rng.shuffle(self._target_positions)

    def _place_robot(self):
        """Place robot at a deterministic free cell."""
        # Robot starts at row 5, col 0 (left aisle)
        self._robot_pos = (5, 0)

    def _spawn_initial_tasks(self):
        """Create initial tasks based on difficulty."""
        n_tasks = {"easy": 1, "medium": 3, "hard": 2}[self._difficulty]
        for i in range(n_tasks):
            self._create_task(task_id=i)
        self._total_tasks_spawned = n_tasks

    def _create_task(self, task_id: int, priority: Optional[int] = None) -> Task:
        if priority is None:
            if self._difficulty == "easy":
                priority = 1
            else:
                priority = self._rng.randint(1, 3)

        shelves = [s for s in self._shelf_positions
                   if s != self._robot_pos]
        targets = [t for t in self._target_positions]

        pickup = self._rng.choice(shelves)
        dropoff = self._rng.choice(targets)

        task = Task(
            task_id=task_id,
            pickup_pos=Position(row=pickup[0], col=pickup[1]),
            dropoff_pos=Position(row=dropoff[0], col=dropoff[1]),
            priority=priority,
        )
        self._tasks.append(task)
        return task

    def _spawn_obstacles(self):
        """Spawn moving obstacles (humans) for medium/hard."""
        n = {"easy": 0, "medium": 1, "hard": 3}[self._difficulty]
        occupied = {self._robot_pos}
        for cr, cc in self._charger_positions:
            occupied.add((cr, cc))

        obstacle_id = 0
        attempts = 0
        while len(self._obstacles) < n and attempts < 100:
            attempts += 1
            r = self._rng.randint(1, GRID_SIZE - 2)
            c = self._rng.randint(1, GRID_SIZE - 2)
            if (r, c) in occupied:
                continue
            if self._grid[r][c] in (CellType.CHARGER.value, CellType.SHELF.value,
                                     CellType.TARGET.value):
                continue
            occupied.add((r, c))
            self._obstacles.append(ObstacleState(
                obstacle_id=obstacle_id,
                position=Position(row=r, col=c),
                direction=self._rng.randint(0, 3),
            ))
            obstacle_id += 1

    # ─────────────────────────────────────
    # Action Execution
    # ─────────────────────────────────────

    def _execute_move(self, action: ActionType,
                      reward: StepReward, info: StepInfo):
        dr, dc = DIRECTION_DELTAS[action.value]
        nr = self._robot_pos[0] + dr
        nc = self._robot_pos[1] + dc

        # Boundary check
        if not (0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE):
            reward.invalid_action_penalty += R_INVALID_ACTION
            reward.total += R_INVALID_ACTION
            info.action_valid = False
            return

        # Wall/Shelf blocking (shelves are physical barriers)
        cell = self._grid[nr][nc]
        if cell == CellType.WALL.value:
            reward.invalid_action_penalty += R_INVALID_ACTION
            reward.total += R_INVALID_ACTION
            info.action_valid = False
            return

        # Shelf blocks movement
        if cell == CellType.SHELF.value:
            reward.invalid_action_penalty += R_INVALID_ACTION
            reward.total += R_INVALID_ACTION
            info.action_valid = False
            return

        # Obstacle collision check
        obs_positions = {(o.position.row, o.position.col) for o in self._obstacles}
        if (nr, nc) in obs_positions:
            reward.collision_penalty = R_COLLISION
            reward.total += R_COLLISION
            info.collision_occurred = True
            # Don't move
            return

        self._robot_pos = (nr, nc)

    def _execute_pick(self, reward: StepReward, info: StepInfo):
        if self._carrying_item:
            reward.invalid_action_penalty += R_INVALID_ACTION
            reward.total += R_INVALID_ACTION
            info.action_valid = False
            return

        r, c = self._robot_pos
        # Find task whose pickup is adjacent or at current pos
        target_task = None
        for t in self._tasks:
            if t.completed or t.picked_up:
                continue
            pr, pc = t.pickup_pos.row, t.pickup_pos.col
            if manhattan((r, c), (pr, pc)) <= 1:
                target_task = t
                break

        if target_task is None:
            reward.invalid_action_penalty += R_INVALID_ACTION
            reward.total += R_INVALID_ACTION
            info.action_valid = False
            return

        target_task.picked_up = True
        self._carrying_item = True
        self._carrying_task_id = target_task.task_id
        reward.task_completion = R_PICKUP
        reward.total += R_PICKUP

    def _execute_drop(self, reward: StepReward, info: StepInfo):
        if not self._carrying_item:
            reward.invalid_action_penalty += R_INVALID_ACTION
            reward.total += R_INVALID_ACTION
            info.action_valid = False
            return

        r, c = self._robot_pos
        task = next((t for t in self._tasks if t.task_id == self._carrying_task_id), None)
        if task is None:
            # Shouldn't happen
            self._carrying_item = False
            self._carrying_task_id = None
            return

        dr, dc = task.dropoff_pos.row, task.dropoff_pos.col
        if manhattan((r, c), (dr, dc)) <= 1:
            # Successful delivery
            task.completed = True
            self._carrying_item = False
            self._carrying_task_id = None
            self._completed_tasks += 1

            completion_reward = R_TASK_COMPLETE
            # Priority bonus
            if task.priority == 2:
                completion_reward += R_PRIORITY_2
                reward.priority_bonus = R_PRIORITY_2
            elif task.priority == 3:
                completion_reward += R_PRIORITY_3
                reward.priority_bonus = R_PRIORITY_3

            # Efficiency bonus (completed quickly)
            steps_used = self._step_count
            max_steps = MAX_STEPS_MAP[self._difficulty]
            if steps_used < max_steps * 0.5:
                completion_reward += R_EFFICIENCY_BONUS
                reward.efficiency_bonus = R_EFFICIENCY_BONUS

            reward.task_completion = completion_reward
            reward.total += completion_reward
            info.task_completed = task.task_id
        else:
            # Wrong location — penalise
            reward.invalid_action_penalty += R_INVALID_ACTION
            reward.total += R_INVALID_ACTION
            info.action_valid = False

    def _execute_recharge(self, reward: StepReward, info: StepInfo):
        r, c = self._robot_pos
        if (r, c) in self._charger_positions:
            self._battery = min(BATTERY_MAX, self._battery + RECHARGE_RATE)
        else:
            reward.invalid_action_penalty += R_INVALID_ACTION
            reward.total += R_INVALID_ACTION
            info.action_valid = False

    # ─────────────────────────────────────
    # Dynamic Obstacles
    # ─────────────────────────────────────

    def _move_obstacles(self):
        """Deterministically move obstacles each step."""
        occupied = {self._robot_pos}
        new_obs_positions: set = set()

        for obs in self._obstacles:
            r, c = obs.position.row, obs.position.col
            dr, dc = DIRECTION_DELTAS[obs.direction]
            nr, nc = r + dr, c + dc

            # Bounce off walls and shelves
            can_move = (
                0 <= nr < GRID_SIZE and
                0 <= nc < GRID_SIZE and
                self._grid[nr][nc] not in (CellType.SHELF.value,) and
                (nr, nc) not in occupied and
                (nr, nc) not in new_obs_positions
            )

            if can_move:
                obs.position = Position(row=nr, col=nc)
                new_obs_positions.add((nr, nc))
            else:
                # Reverse direction
                obs.direction = (obs.direction + 2) % 4
                new_obs_positions.add((r, c))

    def _maybe_spawn_task(self) -> Optional[Task]:
        """Randomly spawn a new task in medium/hard."""
        if self._difficulty == "easy":
            return None
        # Spawn probability: 5% per step in medium, 10% in hard
        prob = {"medium": 0.05, "hard": 0.10}.get(self._difficulty, 0.0)
        if self._rng.random() < prob:
            task_id = self._total_tasks_spawned
            self._total_tasks_spawned += 1
            task = self._create_task(task_id=task_id)
            return task
        return None

    # ─────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────

    def _get_full_grid_snapshot(self) -> List[List[int]]:
        """Overlay robot and obstacles on base grid."""
        snapshot = [row[:] for row in self._grid]
        # Obstacles
        for obs in self._obstacles:
            snapshot[obs.position.row][obs.position.col] = CellType.OBSTACLE.value
        # Robot
        r, c = self._robot_pos
        snapshot[r][c] = CellType.ROBOT.value
        return snapshot

    def _all_tasks_done(self) -> bool:
        return all(t.completed for t in self._tasks)
