from fastapi import FastAPI
from server.environment import WarehouseEnv

app = FastAPI()

env = WarehouseEnv()

@app.get("/")
def root():
    return {"status": "running"}

@app.post("/reset")
def reset():
    return env.reset()

@app.post("/step")
def step(action: dict):
    return env.step(action)
