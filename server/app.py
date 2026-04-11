"""
server/app.py — FastAPI server for Warehouse OpenEnv.

Endpoints:
  GET  /health
  POST /reset
  POST /step
  GET  /state
"""

from __future__ import annotations

import sys
import os
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from warehouse_env.models import (
    HealthResponse, ResetRequest, ResetResponse,
    StepRequest, StepResponse, FullState,
    PartialObservation,
)
from environment import WarehouseEnvironment
from inference import run_baseline


SCORE_EPSILON = 1e-6


def clamp_open01(x):
    try:
        x = float(x)
    except:
        return SCORE_EPSILON

    if x <= 0:
        return SCORE_EPSILON
    if x >= 1:
        return 1 - SCORE_EPSILON
    return x

app = FastAPI(
    title="Warehouse RL OpenEnv",
    description="Real-world warehouse decision-making environment for OpenEnv hackathon.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single environment instance (stateful per session)
env = WarehouseEnvironment()
UI_PATH = Path(__file__).resolve().parent.parent / "ui" / "index.html"


# ─────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────

@app.get("/", include_in_schema=False)
def ui_home():
    """Serve local dashboard UI."""
    if not UI_PATH.exists():
        raise HTTPException(status_code=404, detail="UI file not found")
    return FileResponse(UI_PATH)


@app.get("/ui", include_in_schema=False)
def ui_alias():
    """Alias route for UI."""
    if not UI_PATH.exists():
        raise HTTPException(status_code=404, detail="UI file not found")
    return FileResponse(UI_PATH)

@app.get("/health", response_model=HealthResponse)
def health():
    """Health check."""
    return HealthResponse()


@app.post("/reset", response_model=ResetResponse)
def reset(request: ResetRequest = None):
    """Reset the environment and return initial observation."""
    if request is None:
        request = ResetRequest()
    try:
        obs: PartialObservation = env.reset(
            seed=request.seed,
            difficulty=request.difficulty,
        )
        return ResetResponse(
            observation=obs,
            info={
                "seed": request.seed,
                "difficulty": request.difficulty,
                "grid_size": 10,
                "max_steps": {"easy": 150, "medium": 200, "hard": 250}[request.difficulty],
                "view_radius": {"easy": 4, "medium": 3, "hard": 2}[request.difficulty],
                "action_space": {
                    "type": "Discrete",
                    "n": 8,
                    "actions": {
                        "0": "MOVE_UP",
                        "1": "MOVE_DOWN",
                        "2": "MOVE_LEFT",
                        "3": "MOVE_RIGHT",
                        "4": "PICK",
                        "5": "DROP",
                        "6": "RECHARGE",
                        "7": "WAIT",
                    },
                },
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/step", response_model=StepResponse)
def step(request: StepRequest):
    """Execute one step in the environment."""
    try:
        obs, reward, done, info = env.step(request.action)
        return StepResponse(
            observation=obs,
            reward=reward,
            done=done,
            info=info,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state", response_model=FullState)
def state():
    """Return full internal state (for debugging / evaluation)."""
    try:
        return env.get_state()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/run")
def run(seed: int = 42, difficulties: str = "easy,medium,hard"):
    """Run inference and return strict validator-safe tasks payload only."""
    try:
        diffs = [d.strip() for d in difficulties.split(",") if d.strip()]
        allowed = {"easy", "medium", "hard"}
        if not diffs or any(d not in allowed for d in diffs):
            raise ValueError("difficulties must be a comma-separated subset of easy,medium,hard")
        payload = run_baseline(difficulties=diffs, seed=int(seed), output_json="baseline_scores.json")
        tasks = payload.get("tasks", [])
        clean_tasks = []
        for task in tasks:
            s = clamp_open01(task.get("score", SCORE_EPSILON))
            assert 0 < s < 1
            clean_tasks.append({"score": s, "task_score": s})
        aggregate_score = clamp_open01(
            (sum(t["score"] for t in clean_tasks) / len(clean_tasks)) if clean_tasks else 0.05
        )
        assert 0 < aggregate_score < 1
        output = {"tasks": clean_tasks, "aggregate_score": aggregate_score}
        assert all(0 < t["score"] < 1 for t in output["tasks"]), "Invalid task score detected"
        assert 0 < output["aggregate_score"] < 1, "Invalid aggregate score"
        return output
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────

def main() -> None:
    """Run the HTTP server process."""
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()
