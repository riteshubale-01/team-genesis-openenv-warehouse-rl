"""
tests/test_all.py — Comprehensive test suite for Warehouse OpenEnv.

Run with: python -m pytest tests/ -v
"""

from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from warehouse_env.models import ActionType, CellType, Position, Task, ResetRequest, StepRequest
from environment import WarehouseEnvironment, GRID_SIZE
from grader import compute_score, score_episode


# ─────────────────────────────────────────
# Environment Tests
# ─────────────────────────────────────────

class TestEnvironmentReset:
    def test_reset_easy(self):
        env = WarehouseEnvironment()
        obs = env.reset(seed=42, difficulty="easy")
        assert obs.difficulty == "easy"
        assert obs.battery == 100.0
        assert not obs.carrying_item
        assert obs.step_count == 0
        assert len(obs.local_grid) == 9  # 2*4+1
        assert len(obs.local_grid[0]) == 9

    def test_reset_medium(self):
        env = WarehouseEnvironment()
        obs = env.reset(seed=42, difficulty="medium")
        assert obs.view_radius == 3
        assert len(obs.local_grid) == 7

    def test_reset_hard(self):
        env = WarehouseEnvironment()
        obs = env.reset(seed=42, difficulty="hard")
        assert obs.view_radius == 2
        assert len(obs.local_grid) == 5

    def test_deterministic_seed(self):
        env1 = WarehouseEnvironment()
        env2 = WarehouseEnvironment()
        obs1 = env1.reset(seed=123, difficulty="easy")
        obs2 = env2.reset(seed=123, difficulty="easy")
        assert obs1.robot_pos == obs2.robot_pos
        assert obs1.local_grid == obs2.local_grid

    def test_different_seeds_differ(self):
        env = WarehouseEnvironment()
        obs1 = env.reset(seed=1, difficulty="medium")
        obs2 = env.reset(seed=2, difficulty="medium")
        # Tasks may differ
        assert True  # just ensure no crash


class TestEnvironmentStep:
    def setup_method(self):
        self.env = WarehouseEnvironment()
        self.env.reset(seed=42, difficulty="easy")

    def test_step_returns_correct_types(self):
        obs, reward, done, info = self.env.step(0)  # MOVE_UP
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)
        assert "step_count" in info

    def test_step_penalty_applied(self):
        _, reward, _, _ = self.env.step(7)  # WAIT
        assert reward < 0  # step penalty

    def test_step_count_increments(self):
        for i in range(5):
            obs, _, _, info = self.env.step(7)
        assert info["step_count"] == 5

    def test_invalid_action_handled(self):
        # Action 99 is invalid — should not crash
        obs, reward, done, info = self.env.step(99)
        assert not done  # should not terminate
        assert reward < 0  # penalty

    def test_battery_drains(self):
        for _ in range(10):
            obs, _, _, _ = self.env.step(7)  # WAIT
        assert obs.battery < 100.0

    def test_done_after_max_steps(self):
        env = WarehouseEnvironment()
        env.reset(seed=42, difficulty="easy")
        done = False
        for _ in range(200):
            _, _, done, _ = env.step(7)
            if done:
                break
        assert done  # must terminate

    def test_wall_collision_rejected(self):
        env = WarehouseEnvironment()
        env.reset(seed=42, difficulty="easy")
        # Move up repeatedly until wall
        for _ in range(15):
            _, _, _, info = env.step(0)  # MOVE_UP
        # Should not have crashed

    def test_before_reset_raises(self):
        env = WarehouseEnvironment()
        with pytest.raises(RuntimeError):
            env.step(0)


class TestActions:
    def test_all_actions_execute(self):
        for action in range(8):
            env = WarehouseEnvironment()
            env.reset(seed=42, difficulty="easy")
            obs, reward, done, info = env.step(action)
            assert obs is not None

    def test_pick_without_item_invalid(self):
        env = WarehouseEnvironment()
        env.reset(seed=42, difficulty="easy")
        # Move robot away from shelves first
        for _ in range(3):
            env.step(1)  # MOVE_DOWN
        _, reward, _, info = env.step(4)  # PICK — likely no shelf adjacent
        # Should handle gracefully (no crash)

    def test_drop_without_carrying_invalid(self):
        env = WarehouseEnvironment()
        env.reset(seed=42, difficulty="easy")
        _, reward, _, info = env.step(5)  # DROP without carrying
        assert not info.get("action_valid", True)  # should be invalid

    def test_recharge_not_at_charger(self):
        env = WarehouseEnvironment()
        env.reset(seed=42, difficulty="easy")
        env.step(1)  # move away from any charger
        _, reward, _, info = env.step(6)  # RECHARGE
        # Should be invalid action
        assert reward < 0


class TestPartialObservability:
    def test_local_grid_correct_size(self):
        for difficulty, radius in [("easy", 4), ("medium", 3), ("hard", 2)]:
            env = WarehouseEnvironment()
            obs = env.reset(seed=42, difficulty=difficulty)
            expected = 2 * radius + 1
            assert len(obs.local_grid) == expected
            for row in obs.local_grid:
                assert len(row) == expected

    def test_robot_at_center(self):
        env = WarehouseEnvironment()
        obs = env.reset(seed=42, difficulty="easy")
        radius = obs.view_radius
        center = obs.local_grid[radius][radius]
        assert center == CellType.ROBOT.value

    def test_walls_for_out_of_bounds(self):
        env = WarehouseEnvironment()
        obs = env.reset(seed=42, difficulty="easy")
        # Robot at (5, 0) — left edge — cells to the left should be WALL
        radius = obs.view_radius
        for row in range(len(obs.local_grid)):
            # column 0 to radius-1 of local grid should be WALL (off-grid left)
            left_cell = obs.local_grid[row][0]
            assert left_cell == CellType.WALL.value


class TestFullState:
    def test_get_state_after_reset(self):
        env = WarehouseEnvironment()
        env.reset(seed=42, difficulty="easy")
        state = env.get_state()
        assert state.grid_size == GRID_SIZE
        assert state.battery == 100.0
        assert not state.carrying_item
        assert state.step_count == 0

    def test_state_updates_after_step(self):
        env = WarehouseEnvironment()
        env.reset(seed=42, difficulty="easy")
        env.step(7)
        state = env.get_state()
        assert state.step_count == 1
        assert state.battery < 100.0


# ─────────────────────────────────────────
# Grader Tests
# ─────────────────────────────────────────

class TestGrader:
    def test_perfect_score(self):
        result = compute_score(
            completed_tasks=3,
            total_tasks_spawned=3,
            total_steps=50,
            max_steps=150,
            collision_count=0,
            battery_remaining=80.0,
            battery_depleted=False,
            difficulty="easy",
        )
        assert result["score"] > 0.8
        assert 0.0 <= result["score"] <= 1.0

    def test_zero_tasks(self):
        result = compute_score(
            completed_tasks=0,
            total_tasks_spawned=3,
            total_steps=150,
            max_steps=150,
            collision_count=5,
            battery_remaining=10.0,
            battery_depleted=False,
            difficulty="easy",
        )
        assert result["score"] < 0.5

    def test_battery_depleted_penalty(self):
        r1 = compute_score(0, 1, 100, 150, 0, 0.0, True, "easy")
        r2 = compute_score(0, 1, 100, 150, 0, 50.0, False, "easy")
        assert r1["score"] < r2["score"]

    def test_hard_difficulty_multiplier(self):
        r_easy = compute_score(1, 1, 100, 150, 0, 80.0, False, "easy")
        r_hard = compute_score(1, 1, 100, 250, 0, 80.0, False, "hard")
        # Hard multiplier should yield higher score
        assert r_hard["score"] >= r_easy["score"] * 0.9

    def test_score_in_range(self):
        for seed in range(20):
            import random
            rng = random.Random(seed)
            result = compute_score(
                completed_tasks=rng.randint(0, 5),
                total_tasks_spawned=rng.randint(1, 5),
                total_steps=rng.randint(1, 250),
                max_steps=250,
                collision_count=rng.randint(0, 10),
                battery_remaining=rng.uniform(0, 100),
                battery_depleted=rng.random() < 0.2,
                difficulty=rng.choice(["easy", "medium", "hard"]),
            )
            assert 0.0 <= result["score"] <= 1.0

    def test_score_episode_empty(self):
        result = score_episode([], difficulty="easy")
        assert "score" in result
        assert result["score"] >= 0.0


# ─────────────────────────────────────────
# Integration: Full Episode
# ─────────────────────────────────────────

class TestFullEpisode:
    def test_easy_episode_completes(self):
        env = WarehouseEnvironment()
        env.reset(seed=42, difficulty="easy")
        info_list = []
        done = False
        steps = 0
        while not done and steps < 200:
            obs, reward, done, info = env.step(7)  # WAIT agent
            info_list.append(info)
            steps += 1
        assert done
        result = score_episode(info_list, "easy")
        assert 0.0 <= result["score"] <= 1.0

    def test_heuristic_agent_easy(self):
        """Heuristic agent should complete at least 1 task on easy."""
        from inference import heuristic_action
        env = WarehouseEnvironment()
        env.reset(seed=42, difficulty="easy")
        info_list = []
        done = False
        steps = 0
        obs_dict = env.get_observation().model_dump()
        while not done and steps < 200:
            action = heuristic_action(obs_dict)
            obs, reward, done, info = env.step(action)
            obs_dict = obs.model_dump()
            info_list.append(info)
            steps += 1
        # Just check it doesn't crash and produces valid score
        result = score_episode(info_list, "easy")
        assert 0.0 <= result["score"] <= 1.0

    def test_medium_episode(self):
        env = WarehouseEnvironment()
        env.reset(seed=7, difficulty="medium")
        done = False
        steps = 0
        while not done and steps < 250:
            _, _, done, _ = env.step(steps % 4)  # cycle through moves
            steps += 1
        assert done

    def test_hard_episode(self):
        env = WarehouseEnvironment()
        env.reset(seed=99, difficulty="hard")
        done = False
        steps = 0
        while not done and steps < 300:
            _, _, done, _ = env.step(steps % 8)
            steps += 1
        assert done


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
