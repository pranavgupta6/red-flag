from typing import Tuple, Dict, Any, List
from env.models import AuditObservation, AuditAction, AuditReward, TransactionView, Transaction
from env.ledger import generate_easy_ledger, generate_medium_ledger, generate_hard_ledger
from env.graders import grade_easy, grade_medium, grade_hard
from env.reward import compute_step_reward


TASK_CONFIG = {
    "rule_based_audit": {
        "generator": generate_easy_ledger,
        "budget": 15,
        "max_steps": 10,
    },
    "statistical_audit": {
        "generator": generate_medium_ledger,
        "budget": 20,
        "max_steps": 10,
    },
    "structuring_audit": {
        "generator": generate_hard_ledger,
        "budget": 20,
        "max_steps": 10,
    },
}


class AuditEnv:
    def __init__(self):
        self._task: str = "rule_based_audit"
        self._transactions: List[Transaction] = []
        self._ground_truth: set = set()
        self._metadata: Dict[str, Any] = {}
        self._flagged: set = set()
        self._step: int = 0
        self._done: bool = False
        self._cumulative_reward: float = 0.0
        self._clusters: List[set] = []
        self._budget: int = 15
        self._max_steps: int = 10
        self._budget_used: int = 0

    def reset(self, task: str = "rule_based_audit") -> AuditObservation:
        if task not in TASK_CONFIG:
            raise ValueError(f"Unknown task: {task}. Choose from {list(TASK_CONFIG.keys())}")

        self._task = task
        config = TASK_CONFIG[task]
        self._budget = config["budget"]
        self._max_steps = config["max_steps"]
        self._flagged = set()
        self._step = 0
        self._done = False
        self._cumulative_reward = 0.0
        self._budget_used = 0

        self._transactions, self._ground_truth, self._metadata = config["generator"](seed=42)

        # Extract clusters for hard task
        if task == "structuring_audit":
            raw_clusters = self._metadata.get("clusters", [])
            self._clusters = [set(c) for c in raw_clusters]
        else:
            self._clusters = []

        return self._make_observation("Episode started. Review transactions and flag suspicious ones.")

    def step(self, action: AuditAction) -> Tuple[AuditObservation, float, bool, Dict[str, Any]]:
        if self._done:
            obs = self._make_observation("Episode is already done. Call reset() to start a new episode.")
            return obs, 0.0, True, {"warning": "episode_already_done"}

        self._step += 1

        # Validate and filter flag_ids
        valid_ids = {t.id for t in self._transactions}
        newly_flagged = set()
        invalid_ids = []

        for fid in action.flag_ids:
            if fid in valid_ids and fid not in self._flagged:
                newly_flagged.add(fid)
            elif fid not in valid_ids:
                invalid_ids.append(fid)

        self._flagged.update(newly_flagged)
        self._budget_used = len(self._flagged)
        budget_remaining = self._budget - self._budget_used

        # Compute step reward
        reward = compute_step_reward(
            newly_flagged=newly_flagged,
            all_flagged=self._flagged,
            ground_truth=self._ground_truth,
            budget_remaining=budget_remaining,
            budget_total=self._budget,
            step=self._step,
            max_steps=self._max_steps,
        )
        self._cumulative_reward += reward

        # Check done conditions
        self._done = (
            self._step >= self._max_steps
            or self._budget_used >= self._budget
            or getattr(action, "done", False)
        )

        # Build message
        msg_parts = [f"Step {self._step}: flagged {len(newly_flagged)} new transaction(s)."]
        if invalid_ids:
            msg_parts.append(f"Invalid IDs ignored: {invalid_ids}")
        if self._done:
            msg_parts.append("Episode complete.")
        message = " ".join(msg_parts)

        info = {
            "newly_flagged": list(newly_flagged),
            "invalid_ids": invalid_ids,
            "budget_used": self._budget_used,
            "budget_remaining": budget_remaining,
            "cumulative_reward": round(self._cumulative_reward, 4),
            "step": self._step,
        }

        obs = self._make_observation(message)
        return obs, reward, self._done, info

    def grade(self) -> float:
        if self._task == "rule_based_audit":
            return grade_easy(self._flagged, self._ground_truth, self._budget)
        elif self._task == "statistical_audit":
            return grade_medium(self._flagged, self._ground_truth, self._budget)
        elif self._task == "structuring_audit":
            return grade_hard(self._flagged, self._clusters, self._budget)
        return 0.0

    def state(self) -> Dict[str, Any]:
        return {
            "task": self._task,
            "step": self._step,
            "done": self._done,
            "flagged": list(self._flagged),
            "budget_used": self._budget_used,
            "budget_remaining": self._budget - self._budget_used,
            "cumulative_reward": round(self._cumulative_reward, 4),
            "total_transactions": len(self._transactions),
            "fraud_count_hidden": len(self._ground_truth),
        }

    def _make_observation(self, message: str = None) -> AuditObservation:
        transaction_views = [
            TransactionView(
                id=t.id, vendor_id=t.vendor_id, vendor_name=t.vendor_name,
                amount=t.amount, timestamp=t.timestamp, category=t.category,
                department=t.department, approver_id=t.approver_id,
            )
            for t in self._transactions
        ]

        vendor_history = {}
        if "vendor_stats" in self._metadata:
            for vid, stats in self._metadata["vendor_stats"].items():
                vendor_history[vid] = {
                    "mean": stats["mean"],
                    "std": stats["std"],
                    "normal_monthly_count": stats["normal_monthly_count"],
                }

        return AuditObservation(
            task=self._task,
            step=self._step,
            budget_remaining=self._budget - self._budget_used,
            transactions=transaction_views,
            flagged_so_far=list(self._flagged),
            vendor_history=vendor_history,
            message=message,
        )
