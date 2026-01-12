"""
Evaluation Metrics

This module provides metrics for evaluating video understanding models,
matching academic evaluation standards.

Metrics:
- Accuracy: For single-answer QA
- F1: For token-overlap evaluation
- mAP: For retrieval ranking
- Recall@K: For retrieval coverage
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Sequence

logger = logging.getLogger(__name__)


def normalize_answer(text: str) -> str:
    """Normalize text for comparison.
    
    Lowercases, strips whitespace, removes punctuation.
    """
    import re
    import string
    
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    text = ' '.join(text.split())
    return text


def compute_accuracy(
    predictions: Sequence[str],
    references: Sequence[str],
    normalize: bool = True,
) -> float:
    """Compute exact match accuracy.
    
    Args:
        predictions: Predicted answers
        references: Ground truth answers
        normalize: Whether to normalize text before comparison
        
    Returns:
        Accuracy score (0-1)
    """
    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have same length")
    
    if len(predictions) == 0:
        return 0.0
    
    correct = 0
    for pred, ref in zip(predictions, references):
        if normalize:
            pred = normalize_answer(pred)
            ref = normalize_answer(ref)
        if pred == ref:
            correct += 1
    
    return correct / len(predictions)


def compute_f1(
    prediction: str,
    reference: str,
    normalize: bool = True,
) -> float:
    """Compute token-level F1 score between two strings.
    
    This is useful for evaluating partial matches in QA.
    
    Args:
        prediction: Predicted answer
        reference: Ground truth answer
        normalize: Whether to normalize text
        
    Returns:
        F1 score (0-1)
    """
    if normalize:
        prediction = normalize_answer(prediction)
        reference = normalize_answer(reference)
    
    pred_tokens = set(prediction.split())
    ref_tokens = set(reference.split())
    
    if len(pred_tokens) == 0 and len(ref_tokens) == 0:
        return 1.0
    if len(pred_tokens) == 0 or len(ref_tokens) == 0:
        return 0.0
    
    common = pred_tokens & ref_tokens
    
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)
    
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def compute_f1_batch(
    predictions: Sequence[str],
    references: Sequence[str],
    normalize: bool = True,
) -> float:
    """Compute average F1 over multiple examples."""
    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have same length")
    
    if len(predictions) == 0:
        return 0.0
    
    scores = [
        compute_f1(pred, ref, normalize)
        for pred, ref in zip(predictions, references)
    ]
    
    return sum(scores) / len(scores)


def compute_map(
    retrieved: Sequence[Sequence[str]],
    relevant: Sequence[set[str]],
) -> float:
    """Compute Mean Average Precision (mAP) for retrieval.
    
    Args:
        retrieved: List of retrieved segment IDs per query (ranked)
        relevant: List of relevant segment ID sets per query
        
    Returns:
        mAP score (0-1)
    """
    if len(retrieved) != len(relevant):
        raise ValueError("Retrieved and relevant must have same length")
    
    if len(retrieved) == 0:
        return 0.0
    
    aps = []
    for ret, rel in zip(retrieved, relevant):
        if len(rel) == 0:
            continue
        
        ap = 0.0
        num_correct = 0
        
        for i, segment_id in enumerate(ret):
            if segment_id in rel:
                num_correct += 1
                precision_at_i = num_correct / (i + 1)
                ap += precision_at_i
        
        ap = ap / len(rel) if len(rel) > 0 else 0.0
        aps.append(ap)
    
    return sum(aps) / len(aps) if aps else 0.0


def compute_recall_at_k(
    retrieved: Sequence[Sequence[str]],
    relevant: Sequence[set[str]],
    k: int,
) -> float:
    """Compute Recall@K for retrieval.
    
    Args:
        retrieved: List of retrieved segment IDs per query (ranked)
        relevant: List of relevant segment ID sets per query
        k: Number of top results to consider
        
    Returns:
        Recall@K score (0-1)
    """
    if len(retrieved) != len(relevant):
        raise ValueError("Retrieved and relevant must have same length")
    
    if len(retrieved) == 0:
        return 0.0
    
    recalls = []
    for ret, rel in zip(retrieved, relevant):
        if len(rel) == 0:
            continue
        
        top_k = set(ret[:k])
        hits = len(top_k & rel)
        recall = hits / len(rel)
        recalls.append(recall)
    
    return sum(recalls) / len(recalls) if recalls else 0.0


def compute_precision_at_k(
    retrieved: Sequence[Sequence[str]],
    relevant: Sequence[set[str]],
    k: int,
) -> float:
    """Compute Precision@K for retrieval.
    
    Args:
        retrieved: List of retrieved segment IDs per query (ranked)
        relevant: List of relevant segment ID sets per query
        k: Number of top results to consider
        
    Returns:
        Precision@K score (0-1)
    """
    if len(retrieved) != len(relevant):
        raise ValueError("Retrieved and relevant must have same length")
    
    if len(retrieved) == 0:
        return 0.0
    
    precisions = []
    for ret, rel in zip(retrieved, relevant):
        top_k = ret[:k]
        if len(top_k) == 0:
            continue
        
        hits = sum(1 for r in top_k if r in rel)
        precision = hits / len(top_k)
        precisions.append(precision)
    
    return sum(precisions) / len(precisions) if precisions else 0.0


def compute_mrr(
    retrieved: Sequence[Sequence[str]],
    relevant: Sequence[set[str]],
) -> float:
    """Compute Mean Reciprocal Rank (MRR).
    
    Args:
        retrieved: List of retrieved segment IDs per query (ranked)
        relevant: List of relevant segment ID sets per query
        
    Returns:
        MRR score (0-1)
    """
    if len(retrieved) != len(relevant):
        raise ValueError("Retrieved and relevant must have same length")
    
    if len(retrieved) == 0:
        return 0.0
    
    rrs = []
    for ret, rel in zip(retrieved, relevant):
        rr = 0.0
        for i, segment_id in enumerate(ret):
            if segment_id in rel:
                rr = 1.0 / (i + 1)
                break
        rrs.append(rr)
    
    return sum(rrs) / len(rrs)
