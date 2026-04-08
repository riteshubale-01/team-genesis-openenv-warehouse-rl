---
title: Team Genesis Openenv Warehouse RL
emoji: 🤖
colorFrom: indigo
colorTo: blue
sdk: docker
pinned: false
license: mit
---


# 🏭 WarehouseRL-v1 — OpenEnv Hackathon Submission

A production-grade reinforcement learning environment simulating real-world warehouse robot decision-making under uncertainty — inspired by systems at Amazon and Flipkart.

---

## 🎯 Overview

A robot navigates a 10×10 warehouse grid to:
- **Pick** items from shelves
- **Deliver** them to target zones
- **Manage battery** (recharge at corner stations)
- **Avoid moving obstacles** (human workers)
- **Handle dynamic task interruptions**

All under **partial observability** (limited local grid view).

---

## 🏗️ Architecture

```
warehouse-openenv/
├── models.py               # Pydantic data models
├── environment.py          # Core RL environment logic
├── app.py                  # FastAPI HTTP server
├── grader.py               # Deterministic scoring (0–1)
├── inference.py            # OpenEnv inference script (OpenAI SDK)
├── openenv.yaml            # OpenEnv compliance config
├── Dockerfile              # HF Spaces compatible
├── requirements.txt
├── test_all.py             # Full test suite
└── README.md
```

---

## ⚙️ Difficulty Levels

| Level  | Tasks | Obstacles | View Radius | Max Steps | Battery Drain/Step |
|--------|-------|-----------|-------------|-----------|-------------------|
| Easy   | 1     | 0         | 4           | 150       | 0.3%              |
| Medium | 3     | 1         | 3           | 200       | 0.6%              |
| Hard   | 2     | 3         | 2           | 250       | 0.9%              |

---

## 🎮 Action Space

| Action | Value | Description                          |
|--------|-------|--------------------------------------|
| MOVE_UP    | 0 | Move robot north                 |
| MOVE_DOWN  | 1 | Move robot south                 |
| MOVE_LEFT  | 2 | Move robot west                  |
| MOVE_RIGHT | 3 | Move robot east                  |
| PICK       | 4 | Pick item from adjacent shelf    |
| DROP       | 5 | Deliver item at adjacent target  |
| RECHARGE   | 6 | Recharge at corner charger       |
| WAIT       | 7 | Do nothing                       |

---

## 🏆 Reward System

| Event               | Reward  |
|---------------------|---------|
| Step penalty        | −0.10   |
| Pickup item         | +2.00   |
| Task complete       | +10.00  |
| High-priority bonus | +5.00   |
| Urgent priority     | +10.00  |
| Efficiency bonus    | +2.00   |
| Collision           | −2.00   |
| Battery low (<20%)  | −0.50   |
| Battery dead        | −5.00   |
| Invalid action      | −0.30   |

---

## 📊 Scoring (0–1)

| Component          | Weight |
|--------------------|--------|
| Task completion    | 40%    |
| Step efficiency    | 25%    |
| Safety             | 20%    |
| Battery management | 15%    |

Hard difficulty applies a 1.2× multiplier.

---

## 🚀 Quick Start

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the environment server
uvicorn app:app --host 0.0.0.0 --port 7860

# Open dashboard UI
# http://localhost:7860  (or /ui)

# Run tests
python -m pytest -v test_all.py

# Run inference baseline (requires HF_TOKEN)
python inference.py --difficulties easy,medium,hard --seed 42 --output-json baseline_scores.json
```

### Docker

```bash
docker build -t warehouse-openenv .
docker run -p 7860:7860 warehouse-openenv
```

---

## 🌐 API Reference

## 🖥️ Dashboard UI

A built-in dashboard is available for manual testing and demos:

- URL: `http://localhost:7860` (alias: `http://localhost:7860/ui`)
- Controls: reset with seed+difficulty, execute any action (0-7), refresh state
- Automation: auto-play with `heuristic`, `random`, or `sweep` mode and configurable delay
- Keyboard: Arrow keys for movement, `P` pick, `D` drop, `R` recharge, `Space` wait
- Visuals: full 10x10 warehouse map, robot/obstacle/task overlays, battery meter, reward trend chart, task status table

This UI is intended for explainability and debugging; the OpenEnv API remains unchanged.

### `GET /health`
```json
{"status": "ok", "environment": "WarehouseRL-v1", "version": "1.0.0"}
```

### `POST /reset`
```json
// Request
{"seed": 42, "difficulty": "easy"}

// Response
{"observation": {...}, "info": {...}}
```

### `POST /step`
```json
// Request
{"action": 3}

// Response
{"observation": {...}, "reward": -0.1, "done": false, "info": {...}}
```

### `GET /state`
Returns full internal state including all tasks, obstacles, and grid layout.

---

## 🤖 Inference

This project uses the OpenAI Python SDK (`openai`) for all LLM calls in `inference.py`.

### Environment Variables

| Variable      | Description                        |
|---------------|------------------------------------|
| `API_BASE_URL`| OpenAI-compatible API base URL (default: `https://api.openai.com/v1`) |
| `MODEL_NAME`  | Model identifier (default: `gpt-4o-mini`) |
| `HF_TOKEN`    | Bearer token for authentication (required) |
| `ENV_SERVER_URL` | Environment server URL          |
| `SEED`        | Random seed for determinism        |

### Output Format

```
[START] task=warehouse_delivery env=WarehouseRL-v1 model=gpt-4o-mini
[STEP]  step=1 action=MOVE_RIGHT reward=-0.10 done=false error=null
[STEP]  step=2 action=PICK reward=1.90 done=false error=null
...
[END]   success=true steps=42 rewards=-0.10,1.90,...
```

### Reproducible Baseline Run

```bash
set HF_TOKEN=your_token_here
python inference.py --difficulties easy,medium,hard --seed 42 --output-json baseline_scores.json
```

The script saves grader-based scores for each difficulty and an aggregate score to `baseline_scores.json`.

### Baseline Performance Scores

Record your submission baseline from `baseline_scores.json`:

| Difficulty | Seed | Score (0.0-1.0) |
|------------|------|-----------------|
| easy       | 42   | 0.2375 |
| medium     | 42   | 0.2888 |
| hard       | 42   | 0.3498 |
| aggregate  | 42   | 0.2920 |

---

## 🧠 Environment Details

### Grid Layout (10×10)

- **Chargers** at all 4 corners
- **Shelves** in rows 2, 4, 6 (columns 2–7) — act as physical barriers
- **Target zones** at rows 1 and 8 (columns 2, 4, 6) — these are the **drop-off points** where carried items must be delivered using `DROP`
- **Robot** starts at row 5, col 0

### Partial Observability

The robot only sees a `(2r+1)×(2r+1)` local grid where `r` is the view radius. Cells outside the warehouse boundary appear as WALL (type 6).

### Dynamic Obstacles

Obstacles move deterministically each step, reversing direction when hitting walls, shelves, or other obstacles. Their initial positions and directions are seeded.

### Task System

Tasks have a pickup location (shelf) and dropoff location (target). In medium/hard difficulty, new tasks can spawn mid-episode with a small probability per step.

---

## 🔬 Determinism

All randomness is seeded via `random.Random(seed)`. Given the same seed and difficulty, every episode produces identical results.

---

## 📦 Hugging Face Spaces

This project is ready to deploy on HF Spaces (Docker SDK):

1. Create a Space with **Docker** SDK
2. Push this repository
3. The Space will automatically run on port 7860

---

## 🏅 Hackathon Compliance

- ✅ OpenEnv HTTP interface (`/health`, `/reset`, `/step`, `/state`)
- ✅ Deterministic + reproducible (seed-based)
- ✅ Container deployable (Dockerfile included)
- ✅ OpenEnv inference output format
- ✅ Pydantic models for all data structures
- ✅ Graceful invalid action handling (never crashes)
- ✅ Lightweight (≤2 vCPU / 8GB RAM)
- ✅ HF Spaces compatible

