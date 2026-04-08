---
title: Red Flag
emoji: 🚩
colorFrom: red
colorTo: red
sdk: docker
pinned: false
tags:
  - openenv
---

<div align="center">

# 🚩 Red Flag — Financial Audit Sampling Environment

**An OpenEnv-compliant AI environment where an agent acts as a financial auditor,  
intelligently sampling transactions from a seeded ledger to maximize fraud detection under budget constraints.**

[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compliant-brightgreen?style=for-the-badge)](https://huggingface.co/spaces/pranavgupta6/red-flag)
[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-Space-yellow?style=for-the-badge)](https://huggingface.co/spaces/pranavgupta6/red-flag)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue?style=for-the-badge&logo=docker)](https://huggingface.co/spaces/pranavgupta6/red-flag)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://pranavgupta6-red-flag.hf.space/docs)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)

**[🚀 Live Demo](https://pranavgupta6-red-flag.hf.space/docs) · [📋 API Docs](https://pranavgupta6-red-flag.hf.space/docs) · [🏆 Hackathon Submission](https://huggingface.co/spaces/pranavgupta6/red-flag)**

</div>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Why This Problem Matters](#-why-this-problem-matters)
- [Environment Design](#-environment-design)
- [The Three Tasks](#-the-three-tasks)
- [Reward Function](#-reward-function)
- [Baseline Results](#-baseline-results)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [API Reference](#-api-reference)
- [Inference Script](#-inference-script)
- [Reproducibility](#-reproducibility)
- [OpenEnv Compliance](#-openenv-compliance)
- [Deployment](#-deployment)
- [Author](#-author)

---

## 🧠 Overview

**Red Flag** is a fully OpenEnv-compliant reinforcement learning environment that simulates real-world financial audit workflows. An AI agent is placed in the role of a professional auditor — given a large transaction ledger seeded with hidden fraud — and must intelligently select which transactions to inspect under strict budget constraints.

The agent never sees ground truth labels. It must reason from patterns, statistical anomalies, and behavioral signals to flag suspicious transactions — exactly like a real auditor would.

> **Hackathon:** Meta + HuggingFace OpenEnv Hackathon  
> **Category:** Finance / Fraud Detection / Agentic Decision-Making

---

## 💡 Why This Problem Matters

Financial fraud costs the global economy **over $5 trillion annually**. Traditional rule-based audit tools miss sophisticated schemes. AI agents that can reason about transaction patterns, statistical outliers, and structuring behaviors represent the next frontier in automated financial compliance.

This environment models **three real audit scenarios** that professional auditors face:

| Scenario | Real-World Analog |
|---|---|
| Rule-Based Anomaly Detection | Basic compliance checks (AML screening, round-trip detection) |
| Statistical Anomaly Detection | Benford's Law analysis, vendor baseline deviation |
| Structuring Detection | Bank Secrecy Act structuring violations (smurfing) |

By framing these as RL environments, **Red Flag** enables training and benchmarking AI agents on audit tasks that have real financial, legal, and societal impact.

---

## 🏗️ Environment Design

### Core Mechanics

The environment follows a **budget-constrained sampling** paradigm:

```
Agent receives ledger (N transactions, M hidden fraud)
         ↓
Agent selects transactions to inspect (costs 1 budget unit each)
         ↓
Agent flags selected transactions as fraud or legitimate
         ↓
Partial reward issued at every step (not just terminal)
         ↓
Episode ends when budget exhausted or agent signals done
         ↓
Final score = F1 or cluster-aware grade (0.0 – 1.0)
```

### Key Design Principles

- **Partial observability** — agent sees transaction metadata but never `is_fraud` labels
- **Budget constraints** — forces intelligent prioritization over brute-force sampling
- **Partial rewards** — step-level feedback enables RL training, not just bandit evaluation
- **Deterministic seeding** — `random.seed(42)` ensures fully reproducible episodes
- **Pure Python graders** — zero LLM calls in evaluation, fast and fair

### Observation Space (`AuditObservation`)

```python
class AuditObservation(BaseModel):
    transactions: List[Transaction]   # visible ledger (no fraud labels)
    budget_remaining: int             # budget units left
    step: int                         # current step number
    flagged_ids: List[str]            # transactions flagged so far
    sampled_ids: List[str]            # transactions inspected so far
    done: bool                        # episode terminal flag
```

### Action Space (`AuditAction`)

```python
class AuditAction(BaseModel):
    sample_ids: List[str]    # transaction IDs to inspect this step
    flag_ids: List[str]      # transaction IDs to flag as fraudulent
    done: bool               # agent signals episode complete
```

---

## 🎯 The Three Tasks

### Task 1 — `rule_based_audit` (Easy)

> **The Setup:** A 97-transaction ledger with 12 hidden fraudulent transactions. Budget: 15 inspections.

**Fraud Patterns Hidden in the Ledger:**
- 💰 Round-dollar amounts above $10,000 (structuring avoidance)
- 🔁 Exact duplicate transactions (double-billing fraud)
- 🌙 Transactions timestamped between 2:00 AM – 4:00 AM (off-hours anomaly)

**What a Good Agent Does:** Scan for obvious signals — large round numbers, duplicates, and suspicious timestamps — and prioritize those first.

**Grader:** F1 score with 0.2 penalty if budget exceeded.

---

### Task 2 — `statistical_audit` (Medium)

> **The Setup:** A 180-transaction ledger across 10 vendors, with 10 hidden fraud transactions (one per vendor). Budget: 20 inspections.

**Fraud Patterns Hidden in the Ledger:**
- 📊 Each fraudulent transaction is >3.5 standard deviations above that vendor's historical transaction mean
- Fraud IDs are deterministic: `TXN_0170` through `TXN_0179`

**What a Good Agent Does:** Compute per-vendor baselines, identify statistical outliers, and prioritize extreme deviations.

**Grader:** F1 score with 0.2 penalty if budget exceeded.

---

### Task 3 — `structuring_audit` (Hard)

> **The Setup:** A 294-transaction ledger with 24 hidden fraudulent transactions in 3 structured clusters. Budget: 20 inspections.

**Fraud Patterns Hidden in the Ledger:**
- 🏦 3 clusters of 8 shell company transactions each
- Each transaction just below the $10,000 federal reporting threshold (classic structuring / smurfing)
- All 8 transactions in a cluster sent to the same recipient within a 72-hour window

**What a Good Agent Does:** Identify recipient concentration, detect sub-$10k clustering, and reason about timing windows.

**Grader:** Cluster-aware with partial credit — catching >50% of any cluster earns partial score.

---

## 📐 Reward Function

Rewards are issued **at every step**, not just at episode end, enabling proper RL training:

```
step_reward = (0.5 × step_precision)
            + (0.3 × coverage_improvement)
            - fp_penalty
            - budget_penalty
            - inaction_penalty
```

| Component | Description |
|---|---|
| `step_precision` | Fraction of flagged transactions that are actually fraud |
| `coverage_improvement` | New fraud covered this step vs total fraud |
| `fp_penalty` | Penalty for flagging legitimate transactions |
| `budget_penalty` | Penalty for exceeding budget |
| `inaction_penalty` | Penalty for taking no action when budget remains |

**Reward range:** `[-1.0, 1.0]`

---

## 📊 Baseline Results

These scores were achieved using `gpt-4o-mini` via OpenAI-compatible inference at `temperature=0`, `seed=42`:

| Task | Score | Difficulty |
|---|---|---|
| `rule_based_audit` | **0.53** | Easy |
| `statistical_audit` | **0.91** | Medium |
| `structuring_audit` | **0.62** | Hard |

The default inference model for hackathon judges is `Qwen/Qwen2.5-72B-Instruct` via the HuggingFace router.

> 💡 The statistical audit scores highest because vendor-level outlier detection is a well-structured signal that LLMs reason about effectively. The hard structuring task requires multi-transaction pattern matching across time windows — a harder reasoning challenge.

---

## 📁 Project Structure

```
red-flag/
│
├── inference.py              ← Baseline agent script (mandatory at root)
├── openenv.yaml              ← OpenEnv metadata (mandatory at root)
├── Dockerfile                ← Docker container config (port 7860)
├── requirements.txt          ← Python dependencies
├── validate.sh               ← Pre-submission validation script
├── README.md                 ← You are here
├── .env.example              ← Environment variable template
│
├── env/
│   ├── environment.py        ← AuditEnv class (reset/step/grade/state)
│   ├── ledger.py             ← Deterministic ledger generators (seed=42)
│   ├── models.py             ← Pydantic models (Transaction, AuditObservation, AuditAction)
│   ├── graders.py            ← grade_easy, grade_medium, grade_hard (no LLM calls)
│   └── reward.py             ← compute_step_reward (partial rewards)
│
├── tasks/
│   ├── task_easy.py          ← rule_based_audit task config
│   ├── task_medium.py        ← statistical_audit task config
│   └── task_hard.py          ← structuring_audit task config
│
├── api/
│   └── server.py             ← FastAPI server, all 6 OpenEnv endpoints
│
└── tests/
    ├── test_graders.py
    ├── test_environment.py
    └── test_reproducibility.py
```

---

## ⚡ Quick Start

### Prerequisites

- Python 3.11+
- An OpenAI-compatible API key (HuggingFace token, OpenAI key, or AIPipe proxy)

### 1. Clone and Install

```bash
git clone https://github.com/pranavgupta6/red-flag.git
cd red-flag
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your API credentials
```

Your `.env` should look like:

```env
HF_TOKEN=hf_your_token_here
API_BASE_URL=https://router.huggingface.co/v1
MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
ENV_URL=http://localhost:7860
```

### 3. Start the Server

```bash
uvicorn api.server:app --host 0.0.0.0 --port 7860 --reload
```

### 4. Run the Baseline Agent

```bash
python inference.py
```

You'll see structured logs like:

```
[START] task=rule_based_audit model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 sampled=3 flagged=2 reward=0.41 budget_remaining=12
[STEP] step=2 sampled=4 flagged=3 reward=0.28 budget_remaining=8
[STEP] step=3 sampled=2 flagged=1 reward=0.15 budget_remaining=6
[END] score=0.53 steps=3
```

### 5. Explore the API

Visit `http://localhost:7860/docs` for the interactive Swagger UI.

---

## 🔌 API Reference

All endpoints are live at: **`https://pranavgupta6-red-flag.hf.space`**

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check and environment metadata |
| `POST` | `/reset?task=<task_id>` | Start a new episode, returns initial observation |
| `POST` | `/step` | Submit an `AuditAction`, returns observation + reward |
| `GET` | `/state` | Get current environment state |
| `GET` | `/tasks` | List all available tasks |
| `POST` | `/grader` | Get final episode score (0.0 – 1.0) |
| `POST` | `/baseline` | Run the baseline inference script |

### Example: Start an Episode

```bash
curl -X POST "https://pranavgupta6-red-flag.hf.space/reset?task=rule_based_audit"
```

**Response:**
```json
{
  "transactions": [...],
  "budget_remaining": 15,
  "step": 0,
  "flagged_ids": [],
  "sampled_ids": [],
  "done": false
}
```

### Example: Submit an Action

```bash
curl -X POST "https://pranavgupta6-red-flag.hf.space/step" \
  -H "Content-Type: application/json" \
  -d '{"sample_ids": ["TXN_0001", "TXN_0042"], "flag_ids": ["TXN_0042"], "done": false}'
```

### Available Task IDs

| Task ID | Difficulty |
|---|---|
| `rule_based_audit` | Easy |
| `statistical_audit` | Medium |
| `structuring_audit` | Hard |

---

## 🤖 Inference Script

`inference.py` implements the baseline agent following the mandatory OpenEnv stdout format.

### Key Design Decisions

- Uses `from openai import OpenAI` — mandatory per submission spec
- Default model: `Qwen/Qwen2.5-72B-Instruct` via HuggingFace router
- `HF_TOKEN` has **no default** — must be set explicitly
- `temperature=0`, `seed=42` for reproducibility
- Always emits `[END]` even on exception (via `try/finally`)
- Score clamped to `[0.0, 1.0]`

### Agent Strategy

The baseline agent uses a chain-of-thought prompting approach:

1. **Observe** — receives the full transaction ledger
2. **Analyze** — LLM reasons about patterns (round amounts, duplicates, outliers, clustering)
3. **Prioritize** — selects highest-suspicion transactions within budget
4. **Flag** — marks transactions as fraudulent with justification
5. **Repeat** — continues until budget exhausted or agent signals done

---

## 🔬 Reproducibility

All environments are fully deterministic:

```python
random.seed(42)
numpy.random.seed(42)
```

- Ledger structure is identical on every `reset()` call for the same task
- Fraud transaction IDs are fixed and predictable
- Grader outputs are pure Python — no LLM calls, no randomness
- Medium task fraud IDs are always `TXN_0170` – `TXN_0179`

This ensures fair comparison across different models and agents.

---

## ✅ OpenEnv Compliance

| Requirement | Status |
|---|---|
| `openenv.yaml` at root with all required fields | ✅ |
| `inference.py` at root with `[START]/[STEP]/[END]` format | ✅ |
| `Dockerfile` builds cleanly on port 7860 | ✅ |
| 3 tasks with deterministic graders | ✅ |
| Graders return different scores for different inputs | ✅ |
| `reset()` produces clean seeded state | ✅ |
| `step()` returns partial reward at every step | ✅ |
| `/tasks`, `/baseline`, `/grader` endpoints respond correctly | ✅ |
| `is_fraud` field never exposed to agent in observations | ✅ |
| No LLM calls inside grader functions | ✅ |
| Baseline uses `temperature=0`, `seed=42` | ✅ |
| `HF_TOKEN` has no hardcoded default | ✅ |
| All LLM calls use OpenAI client | ✅ |

---

## 🚀 Deployment

### HuggingFace Spaces (Live)

The environment is deployed as a Docker Space on HuggingFace:

**Space URL:** `https://huggingface.co/spaces/pranavgupta6/red-flag`  
**API URL:** `https://pranavgupta6-red-flag.hf.space`  
**Swagger UI:** `https://pranavgupta6-red-flag.hf.space/docs`

### Docker (Local)

```bash
docker build -t red-flag .
docker run -p 7860:7860 red-flag
```

The Dockerfile uses `python:3.11-slim`, exposes port `7860`, and runs uvicorn directly:

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 7860
CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "7860"]
```

---

## 👤 Author

**Pranav Gupta**  
B.Tech Information Technology  
📧 Open to collaboration and feedback

---

<div align="center">

Built for the **Meta + HuggingFace OpenEnv Hackathon** 🏆

*Modeling real-world financial audit intelligence as an AI environment*

</div>