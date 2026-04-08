---
title: Red Flag
emoji: 🚩
colorFrom: red
colorTo: orange
sdk: docker
pinned: false
tags:
  - openenv
---


# Financial Audit Sampling Environment

An OpenEnv-compliant AI environment where an agent acts as a financial auditor,
intelligently sampling transactions from a seeded ledger to maximize fraud detection
under budget constraints. Models real audit sampling workflows used by professional accountants.

## Quick Start

```bash
pip install -r requirements.txt
uvicorn api.server:app --host 0.0.0.0 --port 7860
```

## Tasks

| Task ID | Difficulty | Description | Budget |
|---|---|---|---|
| rule_based_audit | Easy | Flag transactions violating explicit rules: round amounts >$10k, duplicates, 2am–4am timestamps | 15 flags / 97 txns |
| statistical_audit | Medium | Flag statistical outliers relative to vendor historical behavior (>3σ above mean) | 20 flags / 173 txns |
| structuring_audit | Hard | Identify coordinated clusters of shell company transactions just below $10k reporting threshold | 20 flags / 294 txns |

## Observation Space

```json
{
  "task": "rule_based_audit",
  "step": 1,
  "budget_remaining": 13,
  "transactions": [{"id": "TXN_0001", "vendor_id": "V001", "vendor_name": "Acme Corp", "amount": 412.50, "timestamp": "2023-06-15T14:22:00", "category": "Consulting", "department": "Finance", "approver_id": "APR_03"}],
  "flagged_so_far": ["TXN_0040"],
  "vendor_history": {"V001": {"mean": 1200.0, "std": 300.0, "normal_monthly_count": 8}},
  "message": "Step 1: flagged 2 new transaction(s)."
}
```

## Action Space

```json
{"flag_ids": ["TXN_0001", "TXN_0042"], "reasoning": "Round amount over $10k and late-night timestamp"}
```

## Reward Function

Partial reward is given at every step (not just at episode end):
- `+0.5 × step_precision` — reward for flagging true fraud
- `+0.3 × coverage` — reward for overall fraud coverage so far
- `-0.05 per false positive` — penalty for wrong flags
- `-0.2` if over budget
- `-0.1` for inaction after step 1

Reward range: `[-1.0, 1.0]`

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/reset?task=<task_id>` | POST | Start new episode |
| `/step` | POST | Submit action, get observation + reward |
| `/state` | GET | Current episode state |
| `/tasks` | GET | List all tasks and schemas |
| `/grader` | POST | Get final episode score |
| `/baseline` | POST | Run baseline inference script |

## Baseline Scores

| Task | Difficulty | Expected Score |
|---|---|---|
| rule_based_audit | Easy | ~0.72 |
| statistical_audit | Medium | ~0.48 |
| structuring_audit | Hard | ~0.18 |

## Running the Baseline

```bash
export HF_TOKEN=your_token_here
python inference.py
```

## Docker

```bash
docker build -t financial-audit-env .
docker run -p 7860:7860 financial-audit-env
```
