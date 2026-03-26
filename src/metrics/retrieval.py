"""
Retrieval Metrics for RAG Evaluation

This module implements standard information retrieval metrics for evaluating
the quality of retrieved documents in a RAG system.

Metrics implemented:
    - MRR@K (Mean Reciprocal Rank): Measures how highly the first relevant
      document appears in the ranked results
    - NDCG@K (Normalized Discounted Cumulative Gain): Evaluates ranking
      quality considering position-based discounting
    - HitRate@K: Binary measure of whether at least one relevant document
      appears in the top-K results
    - Precision@K: Proportion of retrieved documents that are relevant

Reference: DESIGN.md Chapter 2.1 - Retrieval Stage Metrics
"""

import math
from typing import List, Dict, Union, Any


def mrr_at_k(results: List[Dict[str, Any]], k: int = 5) -> float:
    """
    Calculate Mean Reciprocal Rank at K.

    MRR = Σ(1/rank_i) / N
    where rank_i is the position of the first relevant document for query i

    Args:
        results: List of retrieval results, each containing:
            - "ranked_doc_ids": List[str] - Ranked document IDs
            - "relevant_docs": Set[str] or List[str] - Relevant document IDs
        k: Cut-off rank (default: 5)

    Returns:
        MRR score between 0.0 and 1.0

    Example:
        >>> results = [
        ...     {"ranked_doc_ids": ["doc3", "doc1", "doc5"], "relevant_docs": {"doc1", "doc5"}},
        ...     {"ranked_doc_ids": ["doc2", "doc7", "doc1"], "relevant_docs": {"doc3"}}
        ... ]
        >>> mrr_at_k(results, k=3)
        0.3333  # (1/2 + 0) / 2
    """
    if not results:
        return 0.0

    reciprocal_ranks = []

    for result in results:
        ranked_docs = result.get("ranked_doc_ids", [])[:k]
        relevant_docs = set(result.get("relevant_docs", []))

        if not relevant_docs:
            reciprocal_ranks.append(0.0)
            continue

        # Find the rank of the first relevant document
        for rank, doc_id in enumerate(ranked_docs, start=1):
            if doc_id in relevant_docs:
                reciprocal_ranks.append(1.0 / rank)
                break
        else:
            reciprocal_ranks.append(0.0)

    return sum(reciprocal_ranks) / len(reciprocal_ranks)


def ndcg_at_k(
    results: List[Dict[str, Any]],
    k: int,
    relevances: Dict[str, Dict[str, float]] = None
) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain at K.

    NDCG@K = DCG@K / IDCG@K

    DCG = Σ(relevance_i / log2(i + 1))
    IDCG = Σ(sorted_relevance_i / log2(i + 1))

    Args:
        results: List of retrieval results, each containing:
            - "ranked_doc_ids": List[str] - Ranked document IDs
            - "relevant_docs": Set[str] or List[str] - Relevant document IDs
            - "relevances": Dict[str, float] - Optional per-document relevance scores
        k: Cut-off rank
        relevances: Optional global relevance mapping {doc_id: {query_idx: relevance}}

    Returns:
        NDCG score between 0.0 and 1.0

    Example:
        >>> results = [
        ...     {
        ...         "ranked_doc_ids": ["doc1", "doc2", "doc3"],
        ...         "relevant_docs": {"doc1", "doc3"},
        ...         "relevances": {"doc1": 2.0, "doc2": 1.0, "doc3": 2.0}
        ...     }
        ... ]
        >>> ndcg_at_k(results, k=3)
        0.89  # Normalized score
    """
    if not results:
        return 0.0

    ndcg_scores = []

    for idx, result in enumerate(results):
        ranked_docs = result.get("ranked_doc_ids", [])[:k]
        relevant_docs = set(result.get("relevant_docs", []))

        # Build relevance scores
        result_relevances = result.get("relevances", {})
        if not result_relevances:
            # Binary relevance: 1 for relevant, 0 for non-relevant
            result_relevances = {
                doc_id: 1.0 if doc_id in relevant_docs else 0.0
                for doc_id in ranked_docs
            }

        # Override with global relevances if provided
        if relevances and idx in relevances:
            for doc_id, rel_score in relevances[idx].items():
                result_relevances[doc_id] = rel_score

        # Calculate DCG
        dcg = 0.0
        for i, doc_id in enumerate(ranked_docs, start=1):
            relevance = result_relevances.get(doc_id, 0.0)
            dcg += relevance / math.log2(i + 1)

        # Calculate IDCG (ideal DCG with sorted relevances)
        ideal_relevances = sorted(
            [result_relevances.get(doc_id, 0.0) for doc_id in ranked_docs],
            reverse=True
        )
        idcg = 0.0
        for i, relevance in enumerate(ideal_relevances, start=1):
            idcg += relevance / math.log2(i + 1)

        # Calculate NDCG
        if idcg > 0:
            ndcg_scores.append(dcg / idcg)
        else:
            ndcg_scores.append(0.0)

    return sum(ndcg_scores) / len(ndcg_scores)


def hitrate_at_k(results: List[Dict[str, Any]], k: int = 3) -> float:
    """
    Calculate Hit Rate at K.

    HitRate = (Number of queries with at least one relevant doc in top-K) / Total queries

    Args:
        results: List of retrieval results, each containing:
            - "ranked_doc_ids": List[str] - Ranked document IDs
            - "relevant_docs": Set[str] or List[str] - Relevant document IDs
        k: Cut-off rank (default: 3)

    Returns:
        Hit rate between 0.0 and 1.0

    Example:
        >>> results = [
        ...     {"ranked_doc_ids": ["doc3", "doc1", "doc5"], "relevant_docs": {"doc1"}},
        ...     {"ranked_doc_ids": ["doc2", "doc7", "doc1"], "relevant_docs": {"doc3"}}
        ... ]
        >>> hitrate_at_k(results, k=3)
        0.5  # Only first query has a hit
    """
    if not results:
        return 0.0

    hits = 0

    for result in results:
        ranked_docs = result.get("ranked_doc_ids", [])[:k]
        relevant_docs = set(result.get("relevant_docs", []))

        # Check if any relevant document is in top-K
        if any(doc_id in relevant_docs for doc_id in ranked_docs):
            hits += 1

    return hits / len(results)


def precision_at_k(results: List[Dict[str, Any]], k: int = 5) -> float:
    """
    Calculate Precision at K.

    P@K = |Relevant ∩ Retrieved| / K
    Measures the proportion of retrieved documents that are relevant.

    Args:
        results: List of retrieval results, each containing:
            - "ranked_doc_ids": List[str] - Ranked document IDs
            - "relevant_docs": Set[str] or List[str] - Relevant document IDs
        k: Cut-off rank (default: 5)

    Returns:
        Average precision at K across all queries

    Example:
        >>> results = [
        ...     {"ranked_doc_ids": ["doc1", "doc2", "doc3"], "relevant_docs": {"doc1", "doc3"}},
        ...     {"ranked_doc_ids": ["doc4", "doc5", "doc6"], "relevant_docs": {"doc4"}}
        ... ]
        >>> precision_at_k(results, k=3)
        0.5  # (2/3 + 1/3) / 2
    """
    if not results:
        return 0.0

    precisions = []

    for result in results:
        ranked_docs = result.get("ranked_doc_ids", [])[:k]
        relevant_docs = set(result.get("relevant_docs", []))

        if not ranked_docs:
            precisions.append(0.0)
            continue

        # Count relevant documents in top-K
        relevant_count = sum(1 for doc_id in ranked_docs if doc_id in relevant_docs)
        precisions.append(relevant_count / len(ranked_docs))

    return sum(precisions) / len(precisions)


def recall_at_k(results: List[Dict[str, Any]], k: int = 10) -> float:
    """
    Calculate Recall at K.

    Recall@K = |Relevant ∩ Retrieved| / |Relevant|
    Measures the proportion of relevant documents that are retrieved.

    Args:
        results: List of retrieval results, each containing:
            - "ranked_doc_ids": List[str] - Ranked document IDs
            - "relevant_docs": Set[str] or List[str] - Relevant document IDs
        k: Cut-off rank (default: 10)

    Returns:
        Average recall at K across all queries
    """
    if not results:
        return 0.0

    recalls = []

    for result in results:
        ranked_docs = result.get("ranked_doc_ids", [])[:k]
        relevant_docs = set(result.get("relevant_docs", []))

        if not relevant_docs:
            recalls.append(1.0)  # No relevant docs = perfect recall
            continue

        # Count relevant documents in top-K
        relevant_count = sum(1 for doc_id in ranked_docs if doc_id in relevant_docs)
        recalls.append(relevant_count / len(relevant_docs))

    return sum(recalls) / len(recalls)


def compute_retrieval_metrics(
    results: List[Dict[str, Any]],
    k_values: Dict[str, int] = None
) -> Dict[str, float]:
    """
    Compute all retrieval metrics at once.

    Args:
        results: List of retrieval results
        k_values: Custom K values for each metric (optional)
            Default: {"mrr": 5, "ndcg": 5, "hitrate": 3, "precision": 5, "recall": 10}

    Returns:
        Dictionary with all computed metrics

    Example:
        >>> results = [...]
        >>> compute_retrieval_metrics(results)
        {
            "mrr@5": 0.789,
            "ndcg@5": 0.712,
            "hitrate@3": 0.856,
            "precision@5": 0.623,
            "recall@10": 0.845
        }
    """
    if k_values is None:
        k_values = {"mrr": 5, "ndcg": 5, "hitrate": 3, "precision": 5, "recall": 10}

    return {
        f"mrr@{k_values['mrr']}": mrr_at_k(results, k=k_values["mrr"]),
        f"ndcg@{k_values['ndcg']}": ndcg_at_k(results, k=k_values["ndcg"]),
        f"hitrate@{k_values['hitrate']}": hitrate_at_k(results, k=k_values["hitrate"]),
        f"precision@{k_values['precision']}": precision_at_k(results, k=k_values["precision"]),
        f"recall@{k_values['recall']}": recall_at_k(results, k=k_values["recall"]),
    }
