"""
Retrieval evaluation module (black-box).

Computes standard retrieval metrics against the documents the chatbot actually
returns (`sources` from POST /api/v1/chat), not a private vector-search probe:
- MRR (Mean Reciprocal Rank)
- NDCG (Normalized Discounted Cumulative Gain)
- Hit Rate
- Precision@K

The black-box approach measures the real retrieval path the user sees (hybrid
search + rerank + optional GraphRAG fusion), which a direct Qdrant probe cannot
reproduce. Use `evaluate_via_api()` to call the chatbot and collect sources,
then compute metrics; `evaluate()` handles the pure-metric step when sources
are already in the dataset (e.g. cached runs).
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Set
import numpy as np


@dataclass
class RetrievalResult:
    """Results from retrieval evaluation."""
    mrr_at_5: float
    ndcg_at_5: float
    hitrate_at_3: float
    precision_at_5: float
    per_query_results: List[Dict[str, Any]] = field(default_factory=list)
    # Number of queries where the API returned no sources (non-RAG intents or
    # errors). Useful for diagnosing intent-misroute issues.
    empty_source_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mrr_at_5": self.mrr_at_5,
            "ndcg_at_5": self.ndcg_at_5,
            "hitrate_at_3": self.hitrate_at_3,
            "precision_at_5": self.precision_at_5,
            "per_query_results": self.per_query_results,
            "empty_source_count": self.empty_source_count,
        }


class RetrievalEvaluator:
    """Evaluates retrieval quality using standard IR metrics."""

    def __init__(self, config: Dict[str, Any], qdrant_client=None):
        """Initialize retrieval evaluator.

        Args:
            config: Evaluation configuration including k values for metrics.
            qdrant_client: Deprecated. Kept only for backward compatibility;
                the black-box path does not use it. Pass None.
        """
        self.config = config
        self.qdrant_client = qdrant_client
        self.k_mrr = config.get("mrr_k", 5)
        self.k_ndcg = config.get("ndcg_k", 5)
        self.k_hitrate = config.get("hitrate_k", 3)
        self.k_precision = config.get("precision_k", 5)

    async def evaluate_via_api(
        self,
        dataset: List[Dict[str, Any]],
        chatbot_client,
    ) -> RetrievalResult:
        """Call the chatbot for each query, then score its `sources`.

        This is the black-box retrieval evaluation. For each sample we POST the
        query to /api/v1/chat and treat the returned `sources` as the ranked
        retrieved-doc list, scored against `relevant_docs`.

        Args:
            dataset: Samples with at least `query` and `relevant_docs`
                (list of doc-id strings). `session_id` optional; a fresh thread
                is allocated per query to avoid memory bleed.
            chatbot_client: A ChatbotClient instance (see src/api/chatbot.py).

        Returns:
            RetrievalResult over the chatbot's real retrieval output.
        """
        queries = [item.get("query", "") for item in dataset]
        session_ids = [
            item.get("session_id") or chatbot_client.allocate_session_id()
            for item in dataset
        ]

        batch = await chatbot_client.batch_generate(
            messages=queries,
            session_ids=session_ids,
        )

        # Attach the chatbot's sources back onto each item, then run metrics.
        enriched: List[Dict[str, Any]] = []
        empty_source_count = 0
        for item, resp in zip(dataset, batch.responses):
            sources = list(resp.sources) if resp.status == "success" else []
            if not sources:
                empty_source_count += 1
            enriched.append({
                **item,
                "retrieved_ids": sources,
                "_intent": resp.intent,
                "_error": resp.error,
                "_latency_ms": resp.latency_ms,
            })

        result = await self.evaluate(enriched)
        result.empty_source_count = empty_source_count
        return result

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
            # Accept both `relevant_ids` (legacy) and `relevant_docs`
            # (the key the dataset loader / retrieval_labels.json emit).
            relevant_ids = set(
                item.get("relevant_ids")
                or item.get("relevant_docs")
                or []
            )
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
