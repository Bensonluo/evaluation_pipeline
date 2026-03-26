"""
Unit tests for retrieval metrics (MRR, NDCG, HitRate, Precision, Recall).

Tests the metrics.retrieval module which implements standard information
retrieval metrics for evaluating RAG system retrieval quality.
"""

import pytest
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from metrics.retrieval import (
    mrr_at_k,
    ndcg_at_k,
    hitrate_at_k,
    precision_at_k,
    recall_at_k,
    compute_retrieval_metrics
)


class TestMRR:
    """Test Mean Reciprocal Rank metric."""

    def test_mrr_perfect_ranking(self):
        """Test MRR when first document is always relevant."""
        results = [
            {"ranked_doc_ids": ["doc1", "doc2", "doc3"], "relevant_docs": {"doc1"}},
            {"ranked_doc_ids": ["doc5", "doc6", "doc7"], "relevant_docs": {"doc5"}},
        ]
        score = mrr_at_k(results, k=3)
        assert score == pytest.approx(1.0)

    def test_mrr_partial_ranking(self):
        """Test MRR with varying ranks."""
        results = [
            {"ranked_doc_ids": ["doc3", "doc1", "doc5"], "relevant_docs": {"doc1", "doc5"}},
            {"ranked_doc_ids": ["doc2", "doc7", "doc1"], "relevant_docs": {"doc3"}},
        ]
        score = mrr_at_k(results, k=3)
        # First query: first relevant at rank 2 -> 1/2 = 0.5
        # Second query: no relevant in top-3 -> 0
        # Average: 0.25
        assert score == pytest.approx(0.25)

    def test_mrr_empty_results(self):
        """Test MRR with empty results list."""
        assert mrr_at_k([], k=5) == 0.0

    def test_mrr_no_relevant_docs(self):
        """Test MRR when no relevant docs specified."""
        results = [
            {"ranked_doc_ids": ["doc1", "doc2", "doc3"], "relevant_docs": set()},
        ]
        score = mrr_at_k(results, k=3)
        assert score == 0.0

    def test_mrr_cutoff(self):
        """Test MRR with K cutoff."""
        results = [
            {"ranked_doc_ids": ["doc5", "doc4", "doc3", "doc2", "doc1"],
             "relevant_docs": {"doc1"}},
        ]
        # Relevant at rank 5, but k=3, so not counted
        score = mrr_at_k(results, k=3)
        assert score == 0.0

    def test_mrr_list_relevant_docs(self):
        """Test MRR with relevant docs as list instead of set."""
        results = [
            {"ranked_doc_ids": ["doc3", "doc1", "doc5"], "relevant_docs": ["doc1", "doc5"]},
        ]
        score = mrr_at_k(results, k=3)
        assert score == pytest.approx(0.5)  # 1/2

    def test_mrr_missing_keys(self):
        """Test MRR with missing keys in result dict."""
        results = [
            {"ranked_doc_ids": ["doc1", "doc2"]},
            {"relevant_docs": {"doc1"}},
        ]
        # Should handle gracefully
        score = mrr_at_k(results, k=3)
        assert score >= 0.0


class TestNDCG:
    """Test Normalized Discounted Cumulative Gain metric."""

    def test_ndcg_binary_relevance(self):
        """Test NDCG with binary relevance (0 or 1)."""
        results = [
            {
                "ranked_doc_ids": ["doc1", "doc2", "doc3"],
                "relevant_docs": {"doc1", "doc3"},
            }
        ]
        score = ndcg_at_k(results, k=3)
        # DCG: 1/log2(2) + 0 + 1/log2(4) = 1 + 0.5 = 1.5
        # IDCG: 1/log2(2) + 1/log2(3) + 1/log2(4) = 1 + 0.63 + 0.5 = 2.13
        # NDCG: 1.5 / 2.13 ≈ 0.70
        assert 0.65 < score < 0.75

    def test_ndcg_graded_relevance(self):
        """Test NDCG with graded relevance scores."""
        results = [
            {
                "ranked_doc_ids": ["doc1", "doc2", "doc3"],
                "relevant_docs": {"doc1", "doc2", "doc3"},
                "relevances": {"doc1": 3.0, "doc2": 2.0, "doc3": 1.0},
            }
        ]
        score = ndcg_at_k(results, k=3)
        # Perfect ranking
        assert score == pytest.approx(1.0)

    def test_ndcg_imperfect_ranking(self):
        """Test NDCG with imperfect ranking."""
        results = [
            {
                "ranked_doc_ids": ["doc3", "doc2", "doc1"],
                "relevant_docs": {"doc1", "doc2", "doc3"},
                "relevances": {"doc1": 3.0, "doc2": 2.0, "doc3": 1.0},
            }
        ]
        score = ndcg_at_k(results, k=3)
        # Reverse ranking should be < 1.0
        assert 0.0 < score < 1.0

    def test_ndcg_empty_results(self):
        """Test NDCG with empty results."""
        assert ndcg_at_k([], k=5) == 0.0

    def test_ndcg_no_relevant_docs(self):
        """Test NDCG with no relevant documents."""
        results = [
            {
                "ranked_doc_ids": ["doc1", "doc2", "doc3"],
                "relevant_docs": set(),
            }
        ]
        # When no relevant docs, DCG=0, IDCG=0 -> NDCG=0
        score = ndcg_at_k(results, k=3)
        assert score == 0.0

    def test_ndcg_global_relevances(self):
        """Test NDCG with global relevance mapping."""
        results = [
            {"ranked_doc_ids": ["doc1", "doc2"], "relevant_docs": {"doc1"}},
            {"ranked_doc_ids": ["doc3", "doc4"], "relevant_docs": {"doc3"}},
        ]
        relevances = {
            0: {"doc1": 2.0, "doc2": 1.0},
            1: {"doc3": 3.0, "doc4": 0.5},
        }
        score = ndcg_at_k(results, k=2, relevances=relevances)
        assert 0.0 < score <= 1.0


class TestHitRate:
    """Test Hit Rate metric."""

    def test_hitrate_all_hits(self):
        """Test HitRate when all queries have hits."""
        results = [
            {"ranked_doc_ids": ["doc1", "doc2", "doc3"], "relevant_docs": {"doc1"}},
            {"ranked_doc_ids": ["doc4", "doc5", "doc6"], "relevant_docs": {"doc6"}},
        ]
        score = hitrate_at_k(results, k=3)
        assert score == 1.0

    def test_hitrate_partial_hits(self):
        """Test HitRate with some queries missing."""
        results = [
            {"ranked_doc_ids": ["doc1", "doc2", "doc3"], "relevant_docs": {"doc1"}},
            {"ranked_doc_ids": ["doc4", "doc5", "doc6"], "relevant_docs": {"doc7"}},
        ]
        score = hitrate_at_k(results, k=3)
        assert score == 0.5

    def test_hitrate_no_hits(self):
        """Test HitRate with no hits."""
        results = [
            {"ranked_doc_ids": ["doc1", "doc2", "doc3"], "relevant_docs": {"doc7"}},
            {"ranked_doc_ids": ["doc4", "doc5", "doc6"], "relevant_docs": {"doc8"}},
        ]
        score = hitrate_at_k(results, k=3)
        assert score == 0.0

    def test_hitrate_cutoff(self):
        """Test HitRate with K cutoff."""
        results = [
            {"ranked_doc_ids": ["doc1", "doc2", "doc3", "doc4", "doc5"],
             "relevant_docs": {"doc5"}},
        ]
        # Relevant at rank 5, but k=3
        score = hitrate_at_k(results, k=3)
        assert score == 0.0

    def test_hitrate_empty_results(self):
        """Test HitRate with empty results."""
        assert hitrate_at_k([], k=3) == 0.0


class TestPrecision:
    """Test Precision at K metric."""

    def test_precision_perfect(self):
        """Test Precision when all retrieved are relevant."""
        results = [
            {"ranked_doc_ids": ["doc1", "doc2", "doc3"], "relevant_docs": {"doc1", "doc2", "doc3"}},
        ]
        score = precision_at_k(results, k=3)
        assert score == 1.0

    def test_precision_partial(self):
        """Test Precision with some irrelevant docs."""
        results = [
            {"ranked_doc_ids": ["doc1", "doc2", "doc3"], "relevant_docs": {"doc1", "doc3"}},
            {"ranked_doc_ids": ["doc4", "doc5", "doc6"], "relevant_docs": {"doc4"}},
        ]
        score = precision_at_k(results, k=3)
        # (2/3 + 1/3) / 2 = 0.5
        assert score == pytest.approx(0.5)

    def test_precision_empty_retrieved(self):
        """Test Precision with empty retrieved list."""
        results = [
            {"ranked_doc_ids": [], "relevant_docs": {"doc1"}},
        ]
        score = precision_at_k(results, k=3)
        assert score == 0.0

    def test_precision_no_relevant(self):
        """Test Precision with no relevant docs found."""
        results = [
            {"ranked_doc_ids": ["doc1", "doc2", "doc3"], "relevant_docs": set()},
        ]
        score = precision_at_k(results, k=3)
        assert score == 0.0


class TestRecall:
    """Test Recall at K metric."""

    def test_recall_perfect(self):
        """Test Recall when all relevant docs retrieved."""
        results = [
            {"ranked_doc_ids": ["doc1", "doc2", "doc3"], "relevant_docs": {"doc1", "doc2", "doc3"}},
        ]
        score = recall_at_k(results, k=10)
        assert score == 1.0

    def test_recall_partial(self):
        """Test Recall with partial retrieval."""
        results = [
            {"ranked_doc_ids": ["doc1", "doc2", "doc3"], "relevant_docs": {"doc1", "doc2", "doc3", "doc4"}},
        ]
        score = recall_at_k(results, k=10)
        # Retrieved 3 out of 4 relevant
        assert score == pytest.approx(0.75)

    def test_recall_no_relevant_docs(self):
        """Test Recall when no relevant docs defined."""
        results = [
            {"ranked_doc_ids": ["doc1", "doc2", "doc3"], "relevant_docs": set()},
        ]
        # No relevant docs = perfect recall
        score = recall_at_k(results, k=10)
        assert score == 1.0

    def test_recall_cutoff(self):
        """Test Recall with K cutoff limiting results."""
        results = [
            {"ranked_doc_ids": ["doc1", "doc2", "doc3", "doc4", "doc5"],
             "relevant_docs": {"doc1", "doc3", "doc5"}},
        ]
        score = recall_at_k(results, k=3)
        # Only first 3 considered, retrieved doc1 and doc3 = 2/3
        assert score == pytest.approx(2.0 / 3.0)


class TestComputeRetrievalMetrics:
    """Test the combined retrieval metrics computation."""

    def test_compute_all_metrics_default_k(self):
        """Test computing all metrics with default K values."""
        results = [
            {
                "ranked_doc_ids": ["doc1", "doc2", "doc3", "doc4", "doc5"],
                "relevant_docs": {"doc1", "doc3", "doc5"},
            },
            {
                "ranked_doc_ids": ["doc6", "doc7", "doc8", "doc9", "doc10"],
                "relevant_docs": {"doc6", "doc9"},
            },
        ]

        metrics = compute_retrieval_metrics(results)

        # Check all expected keys exist
        expected_keys = ["mrr@5", "ndcg@5", "hitrate@3", "precision@5", "recall@10"]
        for key in expected_keys:
            assert key in metrics
            assert 0.0 <= metrics[key] <= 1.0

    def test_compute_all_metrics_custom_k(self):
        """Test computing all metrics with custom K values."""
        results = [
            {
                "ranked_doc_ids": ["doc1", "doc2", "doc3"],
                "relevant_docs": {"doc1"},
            },
        ]

        k_values = {"mrr": 10, "ndcg": 10, "hitrate": 5, "precision": 10, "recall": 20}
        metrics = compute_retrieval_metrics(results, k_values=k_values)

        # Check custom K values are reflected in keys
        assert "mrr@10" in metrics
        assert "ndcg@10" in metrics
        assert "hitrate@5" in metrics
        assert "precision@10" in metrics
        assert "recall@20" in metrics

    def test_compute_metrics_single_result(self):
        """Test metrics with single result."""
        results = [
            {
                "ranked_doc_ids": ["doc1", "doc2", "doc3"],
                "relevant_docs": {"doc1", "doc2"},
            },
        ]

        metrics = compute_retrieval_metrics(results)

        # MRR: first relevant at rank 1 -> 1.0
        assert metrics["mrr@5"] == 1.0
        # HitRate: has relevant in top-3 -> 1.0
        assert metrics["hitrate@3"] == 1.0


@pytest.fixture
def sample_retrieval_results():
    """Fixture providing sample retrieval results for testing."""
    return [
        {
            "ranked_doc_ids": ["doc1", "doc2", "doc3", "doc4", "doc5"],
            "relevant_docs": {"doc1", "doc3", "doc5"},
            "relevances": {"doc1": 3.0, "doc2": 1.0, "doc3": 2.0, "doc4": 0.0, "doc5": 2.0},
        },
        {
            "ranked_doc_ids": ["doc6", "doc7", "doc8", "doc9", "doc10"],
            "relevant_docs": {"doc6", "doc9"},
            "relevances": {"doc6": 2.0, "doc7": 0.0, "doc8": 1.0, "doc9": 3.0, "doc10": 0.0},
        },
        {
            "ranked_doc_ids": ["doc11", "doc12", "doc13", "doc14", "doc15"],
            "relevant_docs": {"doc12"},
            "relevances": {"doc11": 0.0, "doc12": 2.0, "doc13": 0.0, "doc14": 0.0, "doc15": 0.0},
        },
    ]


class TestRetrievalMetricsIntegration:
    """Integration tests for retrieval metrics."""

    def test_realistic_evaluation(self, sample_retrieval_results):
        """Test metrics on a realistic retrieval scenario."""
        metrics = compute_retrieval_metrics(sample_retrieval_results)

        # MRR should be between 0 and 1
        assert 0.0 < metrics["mrr@5"] <= 1.0

        # NDCG should be between 0 and 1
        assert 0.0 < metrics["ndcg@5"] <= 1.0

        # HitRate at K=3: check how many queries have hit in top 3
        # Query 1: doc1, doc3 in top 3 -> hit
        # Query 2: doc6 in top 3 -> hit
        # Query 3: doc12 in top 3 -> hit
        # Should be 1.0
        assert metrics["hitrate@3"] == 1.0

        # All metrics should be valid
        for key, value in metrics.items():
            assert isinstance(key, str)
            assert isinstance(value, (int, float))
            assert 0.0 <= value <= 1.0

    def test_metrics_consistency(self, sample_retrieval_results):
        """Test that metrics are internally consistent."""
        # Higher K should never decrease recall
        recall_5 = recall_at_k(sample_retrieval_results, k=5)
        recall_10 = recall_at_k(sample_retrieval_results, k=10)
        assert recall_10 >= recall_5

        # Higher K should never decrease hit rate
        hitrate_3 = hitrate_at_k(sample_retrieval_results, k=3)
        hitrate_5 = hitrate_at_k(sample_retrieval_results, k=5)
        assert hitrate_5 >= hitrate_3
