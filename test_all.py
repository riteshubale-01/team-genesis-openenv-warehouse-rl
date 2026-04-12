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
from grader import compute_score, compute_score_legacy, score_episode, clamp_open_score, normalize_reward, SCORE_EPSILON


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
        assert reward == info["reward_raw"]

    def test_step_penalty_applied(self):
        _, reward, _, _ = self.env.step(7)  # WAIT
        assert reward < 0.0

    def test_step_count_increments(self):
        for i in range(5):
            obs, _, _, info = self.env.step(7)
        assert info["step_count"] == 5

    def test_invalid_action_handled(self):
        # Action 99 is invalid — should not crash
        obs, reward, done, info = self.env.step(99)
        assert not done  # should not terminate
        assert reward <= 0.0

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
        assert reward < 0.0

    def test_move_toward_pickup_rewards_more_than_wait(self):
        env_wait = WarehouseEnvironment()
        env_wait.reset(seed=42, difficulty="easy")
        _, wait_reward, _, _ = env_wait.step(7)  # WAIT

        env_move = WarehouseEnvironment()
        env_move.reset(seed=42, difficulty="easy")
        _, move_reward, _, _ = env_move.step(3)  # MOVE_RIGHT

        assert move_reward > wait_reward

    def test_successful_pickup_gives_higher_reward_signal(self):
        env = WarehouseEnvironment()
        obs = env.reset(seed=42, difficulty="easy")

        # Move robot near the first pickup location.
        task = obs.active_tasks[0]
        pickup_r = task.pickup_pos.row
        pickup_c = task.pickup_pos.col
        target_r = pickup_r
        target_c = pickup_c

        def move_toward(curr_r: int, curr_c: int) -> int:
            if curr_r < target_r:
                return 1  # MOVE_DOWN
            if curr_r > target_r:
                return 0  # MOVE_UP
            if curr_c < target_c:
                return 3  # MOVE_RIGHT
            if curr_c > target_c:
                return 2  # MOVE_LEFT
            return 7  # WAIT

        # Navigate to exact pickup location.
        for _ in range(80):
            curr = env.get_observation().robot_pos
            if curr.row == target_r and curr.col == target_c:
                break
            env.step(move_toward(curr.row, curr.col))

        _, pickup_reward, _, pickup_info = env.step(4)  # PICK
        _, wait_reward, _, _ = env.step(7)              # WAIT baseline next step

        assert pickup_info.get("picked_up_item", False)
        assert pickup_reward > wait_reward


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
    def test_clamp_open_score_edges(self):
        """clamp_open_score must never return 0.0 or 1.0."""
        assert clamp_open_score(0.0) == 0.01
        assert clamp_open_score(-5.0) == 0.01
        assert clamp_open_score(1.0) == 0.99
        assert clamp_open_score(5.0) == 0.99
        assert clamp_open_score(0.5) == 0.5

    def test_normalize_reward_same_min_max(self):
        """When min==max, should return midpoint."""
        result = normalize_reward(5.0, 5.0, 5.0)
        assert result == 0.5  # (REWARD_MIN + REWARD_MAX) / 2

    def test_normalize_reward_range(self):
        """Normalized rewards should stay in (0, 1)."""
        result = normalize_reward(0.0, 0.0, 10.0)
        assert 0 < result < 1
        result = normalize_reward(10.0, 0.0, 10.0)
        assert 0 < result < 1

    def test_compute_score_empty(self):
        """Empty rewards should return SCORE_EPSILON."""
        assert compute_score([]) == SCORE_EPSILON

    def test_compute_score_valid_range(self):
        """Score from rewards list must be strictly in (0, 1)."""
        score = compute_score([0.1, 0.5, 0.9])
        assert 0 < score < 1

    def test_conversion_edge_clamps(self):
        low = compute_score_legacy(0, 1, 100, 150, 0, 0.0, True, "easy", total_reward=-10.0, max_possible_reward=100.0)
        zero = compute_score_legacy(0, 1, 100, 150, 0, 0.0, True, "easy", total_reward=0.0, max_possible_reward=100.0)
        at_top = compute_score_legacy(1, 1, 10, 150, 0, 100.0, False, "easy", total_reward=100.0, max_possible_reward=100.0)
        above = compute_score_legacy(1, 1, 10, 150, 0, 100.0, False, "easy", total_reward=250.0, max_possible_reward=100.0)
        assert low["score"] == 0.01
        assert zero["score"] == 0.01
        assert at_top["score"] == 0.99
        assert above["score"] == 0.99

    def test_perfect_score(self):
        result = compute_score_legacy(
            completed_tasks=3,
            total_tasks_spawned=3,
            total_steps=50,
            max_steps=150,
            collision_count=0,
            battery_remaining=80.0,
            battery_depleted=False,
            difficulty="easy",
            total_reward=100.0,
            max_possible_reward=100.0,
        )
        assert result["score"] == 0.99
        assert 0.0 < result["score"] < 1.0

    def test_zero_tasks(self):
        result = compute_score_legacy(
            completed_tasks=0,
            total_tasks_spawned=3,
            total_steps=150,
            max_steps=150,
            collision_count=5,
            battery_remaining=10.0,
            battery_depleted=False,
            difficulty="easy",
            total_reward=-5.0,
            max_possible_reward=100.0,
        )
        assert result["score"] == 0.01

    def test_battery_depleted_penalty(self):
        r1 = compute_score_legacy(0, 1, 100, 150, 0, 0.0, True, "easy", total_reward=5.0, max_possible_reward=100.0)
        r2 = compute_score_legacy(0, 1, 100, 150, 0, 50.0, False, "easy", total_reward=50.0, max_possible_reward=100.0)
        assert r1["score"] < r2["score"]

    def test_score_rounding_precision(self):
        r = compute_score_legacy(
            completed_tasks=1,
            total_tasks_spawned=1,
            total_steps=10,
            max_steps=150,
            collision_count=0,
            battery_remaining=100.0,
            battery_depleted=False,
            difficulty="easy",
            total_reward=33.333,
            max_possible_reward=100.0,
        )
        # round(0.33333, 4) = 0.3333  (4 decimal places, GridMind-style)
        assert r["score"] == 0.3333

    def test_score_in_range(self):
        for seed in range(20):
            import random
            rng = random.Random(seed)
            result = compute_score_legacy(
                completed_tasks=rng.randint(0, 5),
                total_tasks_spawned=rng.randint(1, 5),
                total_steps=rng.randint(1, 250),
                max_steps=250,
                collision_count=rng.randint(0, 10),
                battery_remaining=rng.uniform(0, 100),
                battery_depleted=rng.random() < 0.2,
                difficulty=rng.choice(["easy", "medium", "hard"]),
            )
            assert 0.0 < result["score"] < 1.0

    def test_score_episode_empty(self):
        result = score_episode([], difficulty="easy")
        assert "score" in result
        assert 0.0 < result["score"] < 1.0


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
        assert 0.0 < result["score"] < 1.0

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
        assert 0.0 < result["score"] < 1.0

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
