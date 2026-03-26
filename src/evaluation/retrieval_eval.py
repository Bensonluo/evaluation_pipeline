"""
Retrieval evaluation module.

Computes standard retrieval metrics:
- MRR (Mean Reciprocal Rank)
- NDCG (Normalized Discounted Cumulative Gain)
- Hit Rate
- Precision@K
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Set
import numpy as np


@dataclass
class RetrievalResult:
    """Results from retrieval evaluation."""
    mrr_at_5: float
    ndcg_at_5: float
    hitrate_at_3: float
    precision_at_5: float
    per_query_results: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mrr_at_5": self.mrr_at_5,
            "ndcg_at_5": self.ndcg_at_5,
            "hitrate_at_3": self.hitrate_at_3,
            "precision_at_5": self.precision_at_5,
            "per_query_results": self.per_query_results,
        }


class RetrievalEvaluator:
    """Evaluates retrieval quality using standard IR metrics."""

    def __init__(self, config: Dict[str, Any], qdrant_client=None):
        """
        Initialize retrieval evaluator.

        Args:
            config: Evaluation configuration including k values for metrics
            qdrant_client: Optional Qdrant client for vector operations
        """
        self.config = config
        self.qdrant_client = qdrant_client
        self.k_mrr = config.get("mrr_k", 5)
        self.k_ndcg = config.get("ndcg_k", 5)
        self.k_hitrate = config.get("hitrate_k", 3)
        self.k_precision = config.get("precision_k", 5)

    async def evaluate(self, dataset: List[Dict[str, Any]]) -> RetrievalResult:
        """
        Evaluate retrieval on a dataset.

        Args:
            dataset: List of evaluation items with:
                - query: str
                - retrieved_ids: List[str] - retrieved document IDs
                - relevant_ids: Set[str] - ground truth relevant document IDs

        Returns:
            RetrievalResult with computed metrics
        """
        mrr_scores = []
        ndcg_scores = []
        hitrate_scores = []
        precision_scores = []
        per_query_results = []

        for item in dataset:
            retrieved_ids = item.get("retrieved_ids", [])
            relevant_ids = set(item.get("relevant_ids", []))
            query = item.get("query", "")

            # Compute metrics for this query
            mrr = self._compute_mrr(retrieved_ids, relevant_ids)
            ndcg = self._compute_ndcg(retrieved_ids, relevant_ids)
            hitrate = self._compute_hitrate(retrieved_ids, relevant_ids)
            precision = self._compute_precision(retrieved_ids, relevant_ids)

            mrr_scores.append(mrr)
            ndcg_scores.append(ndcg)
            hitrate_scores.append(hitrate)
            precision_scores.append(precision)

            # Track per-query results for error analysis
            per_query_results.append({
                "query": query,
                "mrr": mrr,
                "ndcg": ndcg,
                "hitrate": hitrate,
                "precision": precision,
                "retrieved_count": len(retrieved_ids),
                "relevant_count": len(relevant_ids),
            })

        return RetrievalResult(
            mrr_at_5=float(np.mean(mrr_scores)) if mrr_scores else 0.0,
            ndcg_at_5=float(np.mean(ndcg_scores)) if ndcg_scores else 0.0,
            hitrate_at_3=float(np.mean(hitrate_scores)) if hitrate_scores else 0.0,
            precision_at_5=float(np.mean(precision_scores)) if precision_scores else 0.0,
            per_query_results=per_query_results,
        )

    def _compute_mrr(self, retrieved_ids: List[str], relevant_ids: Set[str]) -> float:
        """Compute Mean Reciprocal Rank at K."""
        for i, doc_id in enumerate(retrieved_ids[:self.k_mrr]):
            if doc_id in relevant_ids:
                return 1.0 / (i + 1)
        return 0.0

    def _compute_ndcg(self, retrieved_ids: List[str], relevant_ids: Set[str]) -> float:
        """Compute Normalized Discounted Cumulative Gain at K."""
        dcg = 0.0
        for i, doc_id in enumerate(retrieved_ids[:self.k_ndcg]):
            if doc_id in relevant_ids:
                dcg += 1.0 / np.log2(i + 2)  # i+2 because log2(1)=0

        # Ideal DCG: all top K are relevant
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(self.k_ndcg, len(relevant_ids))))

        return dcg / idcg if idcg > 0 else 0.0

    def _compute_hitrate(self, retrieved_ids: List[str], relevant_ids: Set[str]) -> float:
        """Compute Hit Rate at K."""
        top_k = retrieved_ids[:self.k_hitrate]
        return 1.0 if any(doc_id in relevant_ids for doc_id in top_k) else 0.0

    def _compute_precision(self, retrieved_ids: List[str], relevant_ids: Set[str]) -> float:
        """Compute Precision at K."""
        if not retrieved_ids:
            return 0.0
        top_k = retrieved_ids[:self.k_precision]
        hits = sum(1 for doc_id in top_k if doc_id in relevant_ids)
        return hits / len(top_k)
