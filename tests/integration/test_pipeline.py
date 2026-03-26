"""
Integration tests for the complete evaluation pipeline.

Tests the end-to-end workflow including:
- Dataset loading and sampling
- Retrieval evaluation
- Generation evaluation
- Statistical analysis
- Reporting
"""

import pytest
import sys
from pathlib import Path
from typing import Dict, List, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from metrics.retrieval import compute_retrieval_metrics
from metrics.generation import compute_generation_metrics
from metrics.statistics import (
    compute_significance_for_all_metrics,
    compute_delta_status
)


class TestRetrievalPipeline:
    """Test retrieval evaluation pipeline."""

    @pytest.fixture
    def sample_retrieval_data(self):
        """Sample retrieval test data."""
        return [
            {
                "query": "What is machine learning?",
                "ranked_doc_ids": ["doc1", "doc5", "doc3", "doc2", "doc4"],
                "relevant_docs": {"doc1", "doc3"},
                "relevances": {"doc1": 2.0, "doc2": 0.0, "doc3": 1.0, "doc4": 0.0, "doc5": 0.0},
            },
            {
                "query": "How does neural network work?",
                "ranked_doc_ids": ["doc7", "doc2", "doc8", "doc1", "doc9"],
                "relevant_docs": {"doc7", "doc8"},
                "relevances": {"doc1": 0.0, "doc2": 1.0, "doc7": 2.0, "doc8": 1.0, "doc9": 0.0},
            },
            {
                "query": "What is deep learning?",
                "ranked_doc_ids": ["doc3", "doc6", "doc1", "doc4", "doc5"],
                "relevant_docs": {"doc3"},
                "relevances": {"doc1": 0.0, "doc3": 2.0, "doc4": 0.0, "doc5": 0.0, "doc6": 0.0},
            },
            {
                "query": "Explain gradient descent",
                "ranked_doc_ids": ["doc10", "doc11", "doc12", "doc13", "doc14"],
                "relevant_docs": {"doc11", "doc13"},
                "relevances": {"doc10": 0.0, "doc11": 1.0, "doc12": 0.0, "doc13": 1.0, "doc14": 0.0},
            },
            {
                "query": "What is backpropagation?",
                "ranked_doc_ids": ["doc15", "doc16", "doc17", "doc18", "doc19"],
                "relevant_docs": set(),
                "relevances": {"doc15": 0.0, "doc16": 0.0, "doc17": 0.0, "doc18": 0.0, "doc19": 0.0},
            },
        ]

    def test_retrieval_metrics_computation(self, sample_retrieval_data):
        """Test computing all retrieval metrics."""
        metrics = compute_retrieval_metrics(sample_retrieval_data)

        # Check all metrics are computed
        expected_metrics = ["mrr@5", "ndcg@5", "hitrate@3", "precision@5", "recall@10"]
        for metric in expected_metrics:
            assert metric in metrics
            assert 0.0 <= metrics[metric] <= 1.0

        # MRR should be moderate (some hits, some misses)
        assert 0.2 <= metrics["mrr@5"] <= 0.8

        # HitRate should reflect hits in top-3
        assert metrics["hitrate@3"] >= 0.0

    def test_retrieval_per_query_analysis(self, sample_retrieval_data):
        """Test per-query retrieval analysis."""
        from metrics.retrieval import mrr_at_k, hitrate_at_k

        # Analyze first query specifically
        first_query = [sample_retrieval_data[0]]
        mrr = mrr_at_k(first_query, k=5)
        hitrate = hitrate_at_k(first_query, k=3)

        # First query has relevant docs at ranks 1 and 3
        assert mrr == 1.0  # First relevant at rank 1
        assert hitrate == 1.0  # Hit in top-3


class TestGenerationPipeline:
    """Test generation evaluation pipeline."""

    @pytest.fixture
    def sample_generation_data(self):
        """Sample generation test data."""
        return {
            "predictions": [
                "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
                "Neural networks are computing systems inspired by biological neural networks in the human brain.",
                "Deep learning is a type of machine learning that uses neural networks with multiple layers.",
                "Gradient descent is an optimization algorithm used to minimize the loss function in machine learning.",
                "Backpropagation is an algorithm for training neural networks by calculating gradients.",
            ],
            "references": [
                "Machine learning enables computers to learn and improve from experience without being explicitly programmed.",
                "Neural networks are computing systems vaguely inspired by the biological neural networks in animal brains.",
                "Deep learning is a machine learning technique that teaches computers to do what comes naturally to humans.",
                "Gradient descent is an iterative optimization algorithm for finding the local minimum of a differentiable function.",
                "Backpropagation is a method for calculating the gradient of the loss function with respect to each weight.",
            ],
        }

    def test_generation_metrics_computation(self, sample_generation_data):
        """Test computing all generation metrics."""
        metrics = compute_generation_metrics(
            sample_generation_data["predictions"],
            sample_generation_data["references"]
        )

        # Check ROUGE metrics
        assert "rouge_l_precision" in metrics
        assert "rouge_l_recall" in metrics
        assert "rouge_l_fmeasure" in metrics

        # Check BLEU metrics
        assert "bleu" in metrics
        assert "bleu_1" in metrics
        assert "bleu_2" in metrics
        assert "bleu_3" in metrics
        assert "bleu_4" in metrics

        # All values should be valid
        for key, value in metrics.items():
            assert 0.0 <= value <= 1.0

        # ROUGE-L should have decent overlap
        assert metrics["rouge_l_fmeasure"] > 0.3

    def test_generation_with_bertscore(self, sample_generation_data):
        """Test generation metrics including BERTScore."""
        metrics = compute_generation_metrics(
            sample_generation_data["predictions"],
            sample_generation_data["references"],
            use_bertscore=True
        )

        # Should include BERTScore metrics (even if placeholder)
        assert "bertscore_precision" in metrics
        assert "bertscore_recall" in metrics
        assert "bertscore_f1" in metrics


class TestStatisticalAnalysis:
    """Test statistical analysis in pipeline."""

    @pytest.fixture
    def sample_baseline_scores(self):
        """Sample baseline scores for multiple queries."""
        return {
            "mrr": [0.65, 0.70, 0.68, 0.72, 0.66, 0.69, 0.71, 0.67, 0.73, 0.68],
            "ndcg": [0.60, 0.63, 0.61, 0.65, 0.62, 0.64, 0.66, 0.63, 0.67, 0.64],
            "precision": [0.55, 0.58, 0.56, 0.60, 0.57, 0.59, 0.61, 0.58, 0.62, 0.59],
        }

    @pytest.fixture
    def sample_current_scores(self):
        """Sample current scores for multiple queries."""
        return {
            "mrr": [0.70, 0.75, 0.73, 0.77, 0.71, 0.74, 0.76, 0.72, 0.78, 0.73],
            "ndcg": [0.65, 0.68, 0.66, 0.70, 0.67, 0.69, 0.71, 0.68, 0.72, 0.69],
            "precision": [0.60, 0.63, 0.61, 0.65, 0.62, 0.64, 0.66, 0.63, 0.67, 0.64],
        }

    def test_significance_computation(self, sample_baseline_scores, sample_current_scores):
        """Test statistical significance computation."""
        results = compute_significance_for_all_metrics(
            sample_baseline_scores,
            sample_current_scores,
            n_bootstrap=1000,
            correction_method="fdr",
            random_seed=42
        )

        # Should have results for all metrics
        assert len(results) == 3

        # Each result should have required fields
        for metric_name, result in results.items():
            assert "delta_mean" in result
            assert "p_value" in result
            assert "is_significant" in result
            assert "baseline_mean" in result
            assert "current_mean" in result

            # Current should be higher than baseline
            assert result["current_mean"] > result["baseline_mean"]
            assert result["delta_mean"] > 0

    def test_delta_status_computation(self, sample_baseline_scores, sample_current_scores):
        """Test delta status computation."""
        # Get significance result for MRR
        from metrics.statistics import bootstrap_confidence_interval

        sig_result = bootstrap_confidence_interval(
            sample_baseline_scores["mrr"],
            sample_current_scores["mrr"],
            n_bootstrap=1000,
            random_seed=42
        )

        # Compute delta status
        status = compute_delta_status(
            sig_result["baseline_mean"],
            sig_result["current_mean"],
            sig_result
        )

        # Should show improvement
        assert status["is_improved"] is True
        assert status["is_degraded"] is False
        assert status["delta"] > 0

        # Status should indicate significance
        assert "IMPROVED" in status["status"] or "SIGNIFICANT" in status["status"]


class TestEndToEndPipeline:
    """Test complete end-to-end pipeline."""

    @pytest.fixture
    def pipeline_data(self):
        """Complete pipeline test data."""
        return {
            "baseline": {
                "retrieval": [
                    {"ranked_doc_ids": ["doc1", "doc2", "doc3"], "relevant_docs": {"doc1"}},
                    {"ranked_doc_ids": ["doc4", "doc5", "doc6"], "relevant_docs": {"doc5"}},
                    {"ranked_doc_ids": ["doc7", "doc8", "doc9"], "relevant_docs": set()},
                ],
                "generation": {
                    "predictions": ["the cat sat", "the dog ran", "the bird flew"],
                    "references": ["a cat sat", "a dog ran", "a bird flew"],
                },
                "per_query_mrr": [1.0, 0.5, 0.0],
            },
            "current": {
                "retrieval": [
                    {"ranked_doc_ids": ["doc1", "doc2", "doc3"], "relevant_docs": {"doc1"}},
                    {"ranked_doc_ids": ["doc4", "doc5", "doc6"], "relevant_docs": {"doc5", "doc6"}},
                    {"ranked_doc_ids": ["doc7", "doc8", "doc9"], "relevant_docs": {"doc9"}},
                ],
                "generation": {
                    "predictions": ["the cat sat", "the dog ran quickly", "the bird flew high"],
                    "references": ["a cat sat", "a dog ran", "a bird flew"],
                },
                "per_query_mrr": [1.0, 0.5, 0.33],
            },
        }

    def test_complete_pipeline(self, pipeline_data):
        """Test running complete evaluation pipeline."""
        # 1. Compute retrieval metrics
        baseline_retrieval = compute_retrieval_metrics(pipeline_data["baseline"]["retrieval"])
        current_retrieval = compute_retrieval_metrics(pipeline_data["current"]["retrieval"])

        # 2. Compute generation metrics
        baseline_generation = compute_generation_metrics(
            pipeline_data["baseline"]["generation"]["predictions"],
            pipeline_data["baseline"]["generation"]["references"]
        )
        current_generation = compute_generation_metrics(
            pipeline_data["current"]["generation"]["predictions"],
            pipeline_data["current"]["generation"]["references"]
        )

        # 3. Prepare per-query scores for significance testing
        baseline_scores = {
            "mrr": pipeline_data["baseline"]["per_query_mrr"],
        }
        current_scores = {
            "mrr": pipeline_data["current"]["per_query_mrr"],
        }

        # 4. Compute statistical significance
        significance = compute_significance_for_all_metrics(
            baseline_scores,
            current_scores,
            n_bootstrap=1000,
            random_seed=42
        )

        # 5. Verify all steps completed
        # Retrieval metrics should be computed
        assert "mrr@5" in baseline_retrieval
        assert "mrr@5" in current_retrieval

        # Generation metrics should be computed
        assert "rouge_l_fmeasure" in baseline_generation
        assert "rouge_l_fmeasure" in current_generation

        # Significance should be computed
        assert "mrr" in significance
        assert "delta_mean" in significance["mrr"]

        # Current should perform better or equal
        assert current_retrieval["mrr@5"] >= baseline_retrieval["mrr@5"]

    def test_pipeline_report_generation(self, pipeline_data):
        """Test generating evaluation report from pipeline results."""
        # Compute metrics
        baseline_metrics = compute_retrieval_metrics(pipeline_data["baseline"]["retrieval"])
        current_metrics = compute_retrieval_metrics(pipeline_data["current"]["retrieval"])

        # Create a simple report structure
        report = {
            "baseline": baseline_metrics,
            "current": current_metrics,
            "deltas": {},
            "summary": {}
        }

        # Compute deltas
        for key in baseline_metrics:
            delta = current_metrics[key] - baseline_metrics[key]
            report["deltas"][key] = delta

        # Verify report structure
        assert "baseline" in report
        assert "current" in report
        assert "deltas" in report
        assert len(report["deltas"]) == len(baseline_metrics)


class TestPipelineEdgeCases:
    """Test pipeline edge cases and error handling."""

    def test_empty_dataset(self):
        """Test pipeline with empty dataset."""
        retrieval_results = []
        metrics = compute_retrieval_metrics(retrieval_results)

        # Should handle gracefully
        assert metrics["mrr@5"] == 0.0
        assert metrics["ndcg@5"] == 0.0
        assert metrics["hitrate@3"] == 0.0

    def test_single_query(self):
        """Test pipeline with single query."""
        retrieval_results = [
            {"ranked_doc_ids": ["doc1", "doc2", "doc3"], "relevant_docs": {"doc1"}},
        ]
        metrics = compute_retrieval_metrics(retrieval_results)

        # Should compute metrics
        assert metrics["mrr@5"] == 1.0
        assert metrics["hitrate@3"] == 1.0

    def test_no_relevant_documents(self):
        """Test pipeline when no relevant documents exist."""
        retrieval_results = [
            {"ranked_doc_ids": ["doc1", "doc2", "doc3"], "relevant_docs": set()},
            {"ranked_doc_ids": ["doc4", "doc5", "doc6"], "relevant_docs": set()},
        ]
        metrics = compute_retrieval_metrics(retrieval_results)

        # MRR should be 0 (no relevant docs)
        assert metrics["mrr@5"] == 0.0

    def test_all_relevant_retrieved(self):
        """Test pipeline when all relevant docs are retrieved."""
        retrieval_results = [
            {
                "ranked_doc_ids": ["doc1", "doc2", "doc3"],
                "relevant_docs": {"doc1", "doc2", "doc3"},
            },
        ]
        metrics = compute_retrieval_metrics(retrieval_results)

        # Should have perfect metrics
        assert metrics["mrr@5"] == 1.0
        assert metrics["precision@5"] == 1.0
        assert metrics["recall@10"] == 1.0

    def test_identical_generations(self):
        """Test pipeline with identical predictions and references."""
        predictions = ["the cat sat on the mat"]
        references = ["the cat sat on the mat"]

        metrics = compute_generation_metrics(predictions, references)

        # Should have perfect scores
        assert metrics["bleu"] > 0.95
        assert metrics["rouge_l_fmeasure"] > 0.95


@pytest.fixture
def multi_model_comparison_data():
    """Data for comparing multiple models."""
    return {
        "baseline": {
            "retrieval": [
                {"ranked_doc_ids": ["doc1", "doc2", "doc3"], "relevant_docs": {"doc1"}},
                {"ranked_doc_ids": ["doc4", "doc5", "doc6"], "relevant_docs": {"doc5"}},
            ],
            "per_query_scores": [0.5, 0.6, 0.7, 0.8, 0.9],
        },
        "model_a": {
            "retrieval": [
                {"ranked_doc_ids": ["doc1", "doc2", "doc3"], "relevant_docs": {"doc1"}},
                {"ranked_doc_ids": ["doc4", "doc5", "doc6"], "relevant_docs": {"doc4", "doc5"}},
            ],
            "per_query_scores": [0.55, 0.65, 0.75, 0.85, 0.95],
        },
        "model_b": {
            "retrieval": [
                {"ranked_doc_ids": ["doc1", "doc2", "doc3"], "relevant_docs": {"doc1", "doc2"}},
                {"ranked_doc_ids": ["doc4", "doc5", "doc6"], "relevant_docs": {"doc4", "doc5", "doc6"}},
            ],
            "per_query_scores": [0.6, 0.7, 0.8, 0.9, 1.0],
        },
    }


class TestMultiModelComparison:
    """Test comparing multiple models."""

    def test_three_way_comparison(self, multi_model_comparison_data):
        """Test comparing three models."""
        baseline_metrics = compute_retrieval_metrics(multi_model_comparison_data["baseline"]["retrieval"])
        model_a_metrics = compute_retrieval_metrics(multi_model_comparison_data["model_a"]["retrieval"])
        model_b_metrics = compute_retrieval_metrics(multi_model_comparison_data["model_b"]["retrieval"])

        # Model B should be best
        assert model_b_metrics["mrr@5"] >= model_a_metrics["mrr@5"] >= baseline_metrics["mrr@5"]

    def test_statistical_comparison(self, multi_model_comparison_data):
        """Test statistical comparison between models."""
        baseline_scores = {"score": multi_model_comparison_data["baseline"]["per_query_scores"]}
        model_a_scores = {"score": multi_model_comparison_data["model_a"]["per_query_scores"]}
        model_b_scores = {"score": multi_model_comparison_data["model_b"]["per_query_scores"]}

        # Compare baseline vs model A
        sig_a = compute_significance_for_all_metrics(
            baseline_scores, model_a_scores,
            n_bootstrap=1000,
            random_seed=42
        )

        # Compare baseline vs model B
        sig_b = compute_significance_for_all_metrics(
            baseline_scores, model_b_scores,
            n_bootstrap=1000,
            random_seed=42
        )

        # Both should show improvement
        assert sig_a["score"]["delta_mean"] > 0
        assert sig_b["score"]["delta_mean"] > 0

        # Model B should have larger improvement
        assert sig_b["score"]["delta_mean"] > sig_a["score"]["delta_mean"]
