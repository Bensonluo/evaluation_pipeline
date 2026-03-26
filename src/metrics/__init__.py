"""
RAG Chatbot Evaluation Pipeline - Metrics Module

This module provides comprehensive metrics for evaluating RAG chatbots across
multiple dimensions: retrieval, generation, RAGAS, and statistical analysis.

Exported metrics functions:
    - Retrieval: mrr_at_k, ndcg_at_k, hitrate_at_k, precision_at_k
    - Generation: rouge_scores, bleu_score, bertscore
    - RAGAS: evaluate_faithfulness, evaluate_answer_relevance,
             evaluate_context_precision, evaluate_context_recall
    - Statistics: bootstrap_confidence_interval, compute_p_value
"""

from .retrieval import (
    mrr_at_k,
    ndcg_at_k,
    hitrate_at_k,
    precision_at_k,
)

from .generation import (
    rouge_scores,
    bleu_score,
    bertscore,
)

from .ragas import (
    evaluate_faithfulness,
    evaluate_answer_relevance,
    evaluate_context_precision,
    evaluate_context_recall,
)

from .statistics import (
    bootstrap_confidence_interval,
    compute_p_value,
)

__all__ = [
    # Retrieval metrics
    "mrr_at_k",
    "ndcg_at_k",
    "hitrate_at_k",
    "precision_at_k",
    # Generation metrics
    "rouge_scores",
    "bleu_score",
    "bertscore",
    # RAGAS metrics
    "evaluate_faithfulness",
    "evaluate_answer_relevance",
    "evaluate_context_precision",
    "evaluate_context_recall",
    # Statistics
    "bootstrap_confidence_interval",
    "compute_p_value",
]
