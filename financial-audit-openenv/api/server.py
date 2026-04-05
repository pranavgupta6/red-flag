from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from env.environment import AuditEnv
from env.models import AuditAction

app = FastAPI(
    title="Financial Audit Sampling Environment",
    description="OpenEnv-compliant financial audit sampling environment for AI agents.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

env = AuditEnv()


@app.get("/")
async def root():
    return {
        "name": "financial-audit-sampling",
        "version": "1.0.0",
        "description": "AI agent acts as a financial auditor to detect fraud under budget constraints.",
        "endpoints": ["/reset", "/step", "/state", "/tasks", "/baseline", "/grader"],
    }


@app.post("/reset")
async def reset(task: str = "rule_based_audit"):
    try:
        obs = env.reset(task=task)
        return obs.model_dump()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step")
async def step(action: AuditAction):
    try:
        obs, reward, done, info = env.step(action)
        return {
            "observation": obs.model_dump(),
            "reward": reward,
            "done": done,
            "info": info,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state")
async def state():
    return env.state()


@app.get("/tasks")
async def tasks():
    return {
        "tasks": [
            {
                "id": "rule_based_audit",
                "name": "Rule-Based Anomaly Detection",
                "difficulty": "easy",
                "description": "Flag transactions violating explicit rules: round amounts >$10k, duplicates, late-night timestamps.",
                "budget": 15,
                "action_schema": AuditAction.model_json_schema(),
            },
            {
                "id": "statistical_audit",
                "name": "Statistical Anomaly Detection",
                "difficulty": "medium",
                "description": "Flag transactions that are statistical outliers relative to vendor historical behavior.",
                "budget": 20,
                "action_schema": AuditAction.model_json_schema(),
            },
            {
                "id": "structuring_audit",
                "name": "Pattern-Based Fraud (Structuring)",
                "difficulty": "hard",
                "description": "Identify coordinated clusters of transactions from shell companies designed to evade reporting thresholds.",
                "budget": 20,
                "action_schema": AuditAction.model_json_schema(),
            },
        ]
    }


@app.post("/grader")
async def grader():
    score = env.grade()
    return {"score": score, "task": env._task, "flagged_count": len(env._flagged)}


@app.post("/baseline")
async def baseline():
    import subprocess
    import sys
    result = subprocess.run(
        [sys.executable, "inference.py"],
        capture_output=True,
        text=True,
        cwd="."
    )
    return {
        "stdout": result.stdout,
        "stderr": result.stderr,
        "returncode": result.returncode,
    }
