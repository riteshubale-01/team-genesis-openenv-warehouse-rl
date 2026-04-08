"""
Pydantic models for Warehouse OpenEnv environment.
Supports partial observability, battery, tasks, and dynamic obstacles.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
from enum import IntEnum
from pydantic import BaseModel, Field


# ─────────────────────────────────────────
# Enumerations
# ─────────────────────────────────────────

class ActionType(IntEnum):
    MOVE_UP    = 0
    MOVE_DOWN  = 1
    MOVE_LEFT  = 2
    MOVE_RIGHT = 3
    PICK       = 4
    DROP       = 5
    RECHARGE   = 6
    WAIT       = 7


class CellType(IntEnum):
    EMPTY    = 0
    SHELF    = 1
    TARGET   = 2
    CHARGER  = 3
    OBSTACLE = 4  # moving human
    ROBOT    = 5
    WALL     = 6


class Difficulty(str):
    EASY   = "easy"
    MEDIUM = "medium"
    HARD   = "hard"


# ─────────────────────────────────────────
# Core Models
# ─────────────────────────────────────────

class Position(BaseModel):
    row: int = Field(..., ge=0, description="Row index (0-based)")
    col: int = Field(..., ge=0, description="Column index (0-based)")

    def as_tuple(self) -> Tuple[int, int]:
        return (self.row, self.col)


class Task(BaseModel):
    task_id: int = Field(..., description="Unique task identifier")
    pickup_pos: Position = Field(..., description="Where to pick up the item")
    dropoff_pos: Position = Field(..., description="Where to deliver the item")
    priority: int = Field(default=1, ge=1, le=3, description="1=normal, 2=high, 3=urgent")
    completed: bool = Field(default=False)
    picked_up: bool = Field(default=False)


class ObstacleState(BaseModel):
    obstacle_id: int
    position: Position
    direction: int = Field(..., description="0=up,1=down,2=left,3=right")


class LocalCell(BaseModel):
    cell_type: int = Field(..., description="CellType value")
    has_item: bool = False
    is_target: bool = False
    is_charger: bool = False


class PartialObservation(BaseModel):
    """
    What the robot sees: a (2*view_radius+1) x (2*view_radius+1) local grid.
    Cells outside the warehouse boundary are marked as WALL.
    """
    local_grid: List[List[int]] = Field(..., description="2D grid of CellType values")
    view_radius: int = Field(..., description="Radius of visibility")
    robot_pos: Position = Field(..., description="Robot's absolute position")
    battery: float = Field(..., ge=0.0, le=100.0, description="Battery percentage")
    carrying_item: bool = Field(default=False, description="Whether robot holds an item")
    carrying_task_id: Optional[int] = Field(default=None)
    active_tasks: List[Task] = Field(default_factory=list, description="Known/visible tasks")
    step_count: int = Field(default=0)
    difficulty: str = Field(default="easy")


class Action(BaseModel):
    action_type: int = Field(
        ...,
        ge=0,
        le=7,
        description="0=up,1=down,2=left,3=right,4=pick,5=drop,6=recharge,7=wait"
    )

    @property
    def name(self) -> str:
        return ActionType(self.action_type).name


class StepReward(BaseModel):
    total: float = Field(..., description="Total step reward")
    task_completion: float = Field(default=0.0)
    step_penalty: float = Field(default=0.0)
    collision_penalty: float = Field(default=0.0)
    battery_penalty: float = Field(default=0.0)
    efficiency_bonus: float = Field(default=0.0)
    priority_bonus: float = Field(default=0.0)
    invalid_action_penalty: float = Field(default=0.0)


class StepInfo(BaseModel):
    action_valid: bool = Field(default=True)
    collision_occurred: bool = Field(default=False)
    task_completed: Optional[int] = Field(default=None, description="Task ID if completed")
    task_interrupted: Optional[int] = Field(default=None, description="Task ID if interrupted")
    new_task_spawned: Optional[int] = Field(default=None)
    battery_depleted: bool = Field(default=False)
    reason: Optional[str] = Field(default=None, description="Reason for termination")


class FullState(BaseModel):
    """Complete internal state (used by /state endpoint)."""
    grid_size: int
    difficulty: str
    robot_pos: Position
    battery: float
    carrying_item: bool
    carrying_task_id: Optional[int]
    tasks: List[Task]
    obstacles: List[ObstacleState]
    step_count: int
    max_steps: int
    done: bool
    seed: int
    total_reward: float
    completed_tasks: int
    total_tasks_spawned: int


class ResetRequest(BaseModel):
    seed: int = Field(default=42, description="Random seed for determinism")
    difficulty: str = Field(default="easy", description="easy | medium | hard")


class StepRequest(BaseModel):
    action: int = Field(..., ge=0, le=7, description="Action integer 0-7")


class StepResponse(BaseModel):
    observation: PartialObservation
    reward: float
    done: bool
    info: Dict[str, Any]


class ResetResponse(BaseModel):
    observation: PartialObservation
    info: Dict[str, Any]


class HealthResponse(BaseModel):
    status: str = "ok"
    environment: str = "WarehouseRL-v1"
    version: str = "1.0.0"
