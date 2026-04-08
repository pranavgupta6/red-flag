import os
import json
import sys
from pathlib import Path
from openai import OpenAI
import urllib.request
import urllib.error

# Load .env file if present (local development only)
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())

# Exactly as required by submission checklist
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

# Fallback for local testing via aipipe or openai
API_KEY = HF_TOKEN or os.getenv("OPENAI_API_KEY")

HF_SPACE_URL = "https://pranavgupta6-red-flag.hf.space"

def resolve_env_url():
    local_url = os.getenv("ENV_URL", "http://localhost:7860")
    try:
        req = urllib.request.Request(f"{local_url}/tasks", method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            if resp.status == 200:
                print(f"[INFO]  Using local environment: {local_url}")
                return local_url
    except Exception:
        pass
    print(f"[INFO]  Local server not reachable, falling back to HF Space: {HF_SPACE_URL}")
    return HF_SPACE_URL

ENV_URL = resolve_env_url()

TASKS = ["rule_based_audit", "statistical_audit", "structuring_audit"]

SYSTEM_PROMPT = """You are a financial auditor reviewing a transaction ledger for potential fraud.
You will see a list of transactions with their details.
Your job is to flag suspicious transaction IDs for audit.
You have a limited budget — only flag transactions you have strong reason to suspect.

Look for these patterns:
- Round dollar amounts over $10,000 (e.g. exactly $15,000.00)
- Duplicate transactions (same vendor, same amount, same timestamp)
- Transactions timestamped between 2am and 4am
- Amounts more than 3 standard deviations above a vendor's historical mean (use vendor_history)
- Clusters of transactions just below $10,000 from similar vendor names within 72 hours

Respond ONLY with a JSON object, no markdown, no explanation outside the JSON:
{"flag_ids": ["TXN_0001", "TXN_0042"], "reasoning": "brief explanation"}"""


def run_task(client: OpenAI, task: str) -> float:
    score = 0.0
    rewards = []
    step = 0
    done = False
    error_msg = None

    try:
        # Reset environment
        req = urllib.request.Request(
            f"{ENV_URL}/reset?task={task}",
            method="POST",
            headers={"Content-Type": "application/json"},
            data=b""
        )
        with urllib.request.urlopen(req) as resp:
            obs = json.loads(resp.read().decode())

        print(f"[START] task={task} env=financial-audit-sampling model={MODEL_NAME}")

        while not done and step < 10:
            step += 1

            # Build prompt from observation
            txn_list = []
            for t in obs["transactions"]:
                txn_list.append(
                    f"ID:{t['id']} vendor:{t['vendor_name']}({t['vendor_id']}) "
                    f"amount:${t['amount']} time:{t['timestamp']} "
                    f"cat:{t['category']} dept:{t['department']}"
                )

            vendor_info = ""
            if obs.get("vendor_history"):
                vh_lines = []
                for vid, stats in obs["vendor_history"].items():
                    vh_lines.append(f"{vid}: mean=${stats['mean']} std=${stats['std']}")
                vendor_info = "\nVendor History:\n" + "\n".join(vh_lines)

            user_content = (
                f"Task: {task}\n"
                f"Step: {obs['step']} | Budget remaining: {obs['budget_remaining']}\n"
                f"Already flagged: {obs['flagged_so_far']}\n"
                f"{vendor_info}\n\n"
                f"Transactions:\n" + "\n".join(txn_list)
            )

            # Call LLM
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_content},
                    ],
                    temperature=0,
                    seed=42,
                    max_tokens=1000,
                )
                raw = response.choices[0].message.content.strip()
                # Strip markdown fences if present
                if raw.startswith("```"):
                    raw = raw.split("```")[1]
                    if raw.startswith("json"):
                        raw = raw[4:]
                action = json.loads(raw)
                flag_ids = action.get("flag_ids", [])
            except Exception as e:
                flag_ids = []
                error_msg = str(e)

            # Send action to environment
            action_payload = json.dumps({"flag_ids": flag_ids}).encode()
            req = urllib.request.Request(
                f"{ENV_URL}/step",
                method="POST",
                headers={"Content-Type": "application/json"},
                data=action_payload
            )
            with urllib.request.urlopen(req) as resp:
                result = json.loads(resp.read().decode())

            obs = result["observation"]
            reward = result["reward"]
            done = result["done"]
            rewards.append(reward)

            print(f"[STEP]  step={step} action={flag_ids} reward={reward:.2f} done={str(done).lower()} error={error_msg or 'null'}")

        # Get final grade
        req = urllib.request.Request(
            f"{ENV_URL}/grader",
            method="POST",
            headers={"Content-Type": "application/json"},
            data=b""
        )
        with urllib.request.urlopen(req) as resp:
            grade_result = json.loads(resp.read().decode())
        score = max(0.0, min(1.0, grade_result["score"]))

    except Exception as e:
        error_msg = str(e)
        score = 0.0

    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END]   success={str(score > 0).lower()} steps={step} score={score:.2f} rewards={rewards_str}")
    return score


def main():
    if not API_KEY:
        print("ERROR: Set HF_TOKEN environment variable", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
    all_scores = {}

    for task in TASKS:
        score = run_task(client, task)
        all_scores[task] = score

    print("\n=== FINAL SCORES ===")
    for task, score in all_scores.items():
        print(f"{task}: {score:.2f}")


if __name__ == "__main__":
    main()