"""
RVAA Evaluation Module

This module provides evaluation scripts, metrics, and ablations
for academic benchmarking.
"""

from rvaa.eval.metrics import (
    compute_accuracy,
    compute_f1,
    compute_map,
    compute_recall_at_k,
)
from rvaa.eval.cost_accounting import CostTracker

__all__ = [
    "compute_accuracy",
    "compute_f1",
    "compute_map",
    "compute_recall_at_k",
    "CostTracker",
]
