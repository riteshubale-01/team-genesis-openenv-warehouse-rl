import argparse
import json
import os
import sys
from typing import Any, Dict, List

import requests
from openai import OpenAI

from grader import compute_score, build_output


# ================================
# 🔑 CONFIG (STRICT)
# ================================
API_BASE_URL = os.environ["API_BASE_URL"]
API_KEY = os.environ["API_KEY"]

SERVER_URL = os.environ.get("ENV_SERVER_URL", "http://localhost:7860")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")

ACTION_NAMES = {
    0: "MOVE_UP",
    1: "MOVE_DOWN",
    2: "MOVE_LEFT",
    3: "MOVE_RIGHT",
    4: "PICK",
    5: "DROP",
    6: "RECHARGE",
    7: "WAIT",
}


# ================================
# 🤖 OPENAI CLIENT (MANDATORY)
# ================================
def create_client():
    return OpenAI(
        base_url=API_BASE_URL,
        api_key=API_KEY
    )


def call_llm(client, prompt):
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=10,
    )
    return resp.choices[0].message.content.strip()


def parse_action(text):
    for ch in text:
        if ch.isdigit():
            return int(ch)
    return 7


# ================================
# 🌍 ENV API
# ================================
def env_reset(seed, difficulty):
    return requests.post(
        f"{SERVER_URL}/reset",
        json={"seed": seed, "difficulty": difficulty}
    ).json()


def env_step(action):
    return requests.post(
        f"{SERVER_URL}/step",
        json={"action": action}
    ).json()


# ================================
# 🔁 RUN ONE EPISODE
# ================================
def run_episode(client, difficulty, seed):
    print(f"[START] task={difficulty}")

    data = env_reset(seed, difficulty)
    obs = data.get("observation", {})

    done = False
    total_reward = 0
    steps = 0

    while not done:
        steps += 1

        prompt = f"Choose action (0-7). Step {steps}"
        
        try:
            raw = call_llm(client, prompt)
            action = parse_action(raw)
        except:
            action = 7  # fallback but AFTER API attempt

        step = env_step(action)

        reward = float(step.get("reward", 0))
        done = step.get("done", False)

        total_reward += reward

        print(f"[STEP] step={steps} action={action} reward={reward:.2f} done={done}")

    print(f"[END] steps={steps}")

    return {
        "total_reward": total_reward,
        "max_steps": steps,
        "difficulty": difficulty
    }


# ================================
# 🚀 MAIN
# ================================
def run_all(difficulties, seed, output_file):
    client = create_client()

    tasks = []

    for diff in difficulties:
        result = run_episode(client, diff, seed)

        graded = compute_score(result, diff)

        tasks.append({
            "task_id": diff,
            "score": graded["score"],
            "task_score": graded["task_score"]
        })

    output = build_output(tasks)

    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    print(json.dumps(output, indent=2))


# ================================
# 🧪 ENTRY
# ================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--difficulties", default="easy,medium,hard")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-json", default="baseline_scores.json")

    args = parser.parse_args()

    diffs = [d.strip() for d in args.difficulties.split(",")]

    run_all(diffs, args.seed, args.output_json)