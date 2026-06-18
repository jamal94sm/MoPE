"""
utils/metrics.py — Evaluation metrics for CTTA.
"""

import numpy as np
from collections import defaultdict


def compute_ctta_metrics(results_dict, num_rounds=None):
    """
    Compute standard CTTA metrics from results.

    results_dict: {domain_name: {"error": float, "accuracy": float, ...}}

    Returns dict with:
      - mean_error: average error across all domains
      - mean_accuracy: average accuracy
      - per_corruption_error: dict of corruption → error
      - rf: Repeat Forget metric (for CRS only)
    """
    errors = [r["error"] for r in results_dict.values()]
    accs = [r["accuracy"] for r in results_dict.values()]

    metrics = {
        "mean_error": np.mean(errors),
        "mean_accuracy": np.mean(accs),
        "per_domain_error": {k: v["error"] for k, v in results_dict.items()},
    }

    # compute RF if there are repeated domains
    domain_rounds = defaultdict(list)
    for name, r in results_dict.items():
        base = name.rsplit("_R", 1)[0] if "_R" in name else name
        domain_rounds[base].append(r["error"])

    rf_values = []
    for base, errs in domain_rounds.items():
        if len(errs) >= 2:
            rf_values.append(errs[-1] - errs[0])

    if rf_values:
        metrics["rf"] = np.mean(rf_values)
    else:
        metrics["rf"] = 0.0

    # per-round mean (for CRS)
    if num_rounds:
        domains_per_round = len(results_dict) // num_rounds
        keys = list(results_dict.keys())
        for r in range(num_rounds):
            start = r * domains_per_round
            end = start + domains_per_round
            round_keys = keys[start:end]
            round_errors = [results_dict[k]["error"] for k in round_keys]
            metrics[f"round_{r+1}_mean_error"] = np.mean(round_errors)

    return metrics
