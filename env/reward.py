def compute_step_reward(
    newly_flagged: set,
    all_flagged: set,
    ground_truth: set,
    budget_remaining: int,
    budget_total: int,
    step: int,
    max_steps: int
) -> float:
    new_tp = newly_flagged & ground_truth
    new_fp = newly_flagged - ground_truth

    step_precision = len(new_tp) / len(newly_flagged) if newly_flagged else 0.0
    coverage = len(all_flagged & ground_truth) / len(ground_truth) if ground_truth else 0.0

    fp_penalty = len(new_fp) * 0.05
    budget_penalty = 0.2 if budget_remaining < 0 else 0.0
    inaction_penalty = 0.1 if (len(newly_flagged) == 0 and step > 1) else 0.0

    step_reward = (0.5 * step_precision) + (0.3 * coverage) - fp_penalty - budget_penalty - inaction_penalty
    return round(max(-1.0, min(1.0, step_reward)), 4)
