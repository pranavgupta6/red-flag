from typing import List


def grade_easy(agent_flags: set, ground_truth: set, budget: int) -> float:
    over_budget_penalty = 0.2 if len(agent_flags) > budget else 0.0
    tp = agent_flags & ground_truth
    fp = agent_flags - ground_truth
    recall = len(tp) / len(ground_truth) if ground_truth else 0.0
    precision = len(tp) / len(agent_flags) if agent_flags else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return round(max(0.0, f1 - over_budget_penalty), 4)


def grade_medium(agent_flags: set, ground_truth: set, budget: int) -> float:
    over_budget_penalty = 0.2 if len(agent_flags) > budget else 0.0
    tp = agent_flags & ground_truth
    recall = len(tp) / len(ground_truth) if ground_truth else 0.0
    precision = len(tp) / len(agent_flags) if agent_flags else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return round(max(0.0, f1 - over_budget_penalty), 4)


def grade_hard(agent_flags: set, clusters: List[set], budget: int) -> float:
    cluster_scores = []
    for cluster in clusters:
        overlap = len(agent_flags & cluster) / len(cluster)
        cluster_scores.append(overlap)
    avg_cluster_score = sum(cluster_scores) / len(cluster_scores) if cluster_scores else 0.0
    budget_penalty = max(0.0, (len(agent_flags) - budget) * 0.05)
    return round(max(0.0, min(1.0, avg_cluster_score - budget_penalty)), 4)
