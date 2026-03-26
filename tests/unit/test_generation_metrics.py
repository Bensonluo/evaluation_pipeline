"""
Unit tests for generation metrics (ROUGE, BLEU, BERTScore).

Tests the metrics.generation module which implements lexical and semantic
similarity metrics for evaluating RAG system generation quality.
"""

import pytest
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from metrics.generation import (
    rouge_scores,
    bleu_score,
    bertscore,
    compute_generation_metrics,
    _simple_rouge_l,
    _simple_bleu
)


class TestROUGE:
    """Test ROUGE score calculation."""

    def test_rouge_perfect_match(self):
        """Test ROUGE with identical strings."""
        score = rouge_scores("the cat sat on the mat", "the cat sat on the mat")
        assert score["precision"] == pytest.approx(1.0, abs=0.01)
        assert score["recall"] == pytest.approx(1.0, abs=0.01)
        assert score["fmeasure"] == pytest.approx(1.0, abs=0.01)

    def test_rouge_partial_match(self):
        """Test ROUGE with partial overlap."""
        prediction = "the cat sat on the mat"
        reference = "a cat sat on the mat"
        score = rouge_scores(prediction, reference)
        # Should have high overlap
        assert score["fmeasure"] > 0.7

    def test_rouge_no_match(self):
        """Test ROUGE with no overlap."""
        score = rouge_scores("hello world", "goodbye universe")
        # Should have very low score
        assert score["fmeasure"] < 0.3

    def test_rouge_empty_strings(self):
        """Test ROUGE with empty strings."""
        score = rouge_scores("", "")
        # Empty strings should handle gracefully
        assert "fmeasure" in score

    def test_rouge_list_input(self):
        """Test ROUGE with list of strings."""
        predictions = ["the cat", "the dog"]
        references = ["a cat", "a dog"]
        scores = rouge_scores(predictions, references)
        # Should return averaged scores
        assert "precision" in scores
        assert "recall" in scores
        assert "fmeasure" in scores

    def test_rouge_l_type(self):
        """Test ROUGE-L type specifically."""
        score = rouge_scores(
            "the quick brown fox jumps",
            "the quick brown cat jumps",
            rouge_type="L"
        )
        assert "precision" in score
        assert "recall" in score
        assert "fmeasure" in score

    def test_simple_rouge_l_fallback(self):
        """Test simplified ROUGE-L implementation."""
        prediction = "the cat sat on the mat"
        reference = "the cat is on the mat"
        score = _simple_rouge_l(prediction, reference)
        # Should have decent overlap
        assert score["fmeasure"] > 0.5

    def test_rouge_rouge1_type(self):
        """Test ROUGE-1 type."""
        score = rouge_scores(
            "the cat sat",
            "the cat sat",
            rouge_type="1"
        )
        assert score["fmeasure"] > 0.9

    def test_rouge_rouge2_type(self):
        """Test ROUGE-2 type."""
        score = rouge_scores(
            "the cat sat on the mat",
            "the cat sat on the mat",
            rouge_type="2"
        )
        assert score["fmeasure"] > 0.8


class TestBLEU:
    """Test BLEU score calculation."""

    def test_bleu_perfect_match(self):
        """Test BLEU with identical strings."""
        score = bleu_score("the cat is on the mat", "the cat is on the mat")
        assert score["bleu"] == pytest.approx(1.0, abs=0.05)

    def test_bleu_no_match(self):
        """Test BLEU with no overlap."""
        score = bleu_score("hello world", "goodbye universe")
        # Should have very low score
        assert score["bleu"] < 0.3

    def test_bleu_partial_match(self):
        """Test BLEU with partial overlap."""
        score = bleu_score(
            "the cat is on the mat",
            "the cat was on the mat"
        )
        # Should have decent score due to overlapping n-grams
        assert score["bleu"] > 0.5

    def test_bleu_short_prediction(self):
        """Test BLEU with shorter prediction (brevity penalty)."""
        score = bleu_score("the cat", "the cat is on the mat")
        # Should have brevity penalty
        assert score["brevity_penalty"] < 1.0
        assert score["bleu"] < 1.0

    def test_bleu_longer_prediction(self):
        """Test BLEU with longer prediction."""
        score = bleu_score(
            "the cat is definitely sitting on the mat",
            "the cat is on the mat"
        )
        # No brevity penalty
        assert score["brevity_penalty"] == 1.0

    def test_bleu_max_order(self):
        """Test BLEU with different max order."""
        score = bleu_score(
            "the cat is on the mat",
            "the cat is on the mat",
            max_order=2
        )
        assert "bleu" in score
        assert len(score["precisions"]) == 2

    def test_bleu_list_input(self):
        """Test BLEU with list of strings."""
        predictions = ["the cat", "the dog"]
        references = ["a cat", "a dog"]
        score = bleu_score(predictions, references)
        assert "bleu" in score
        assert 0.0 <= score["bleu"] <= 1.0

    def test_bleu_smooth(self):
        """Test BLEU with smoothing."""
        score = bleu_score(
            "the cat",
            "the cat is on the mat",
            smooth=True
        )
        assert "bleu" in score

    def test_simple_bleu_fallback(self):
        """Test simplified BLEU implementation."""
        prediction = "the cat is on the mat"
        reference = "the cat was on the mat"
        score = _simple_bleu(prediction, reference, max_order=4)
        assert "bleu" in score
        assert "precisions" in score
        assert len(score["precisions"]) == 4

    def test_bleu_individual_ngrams(self):
        """Test individual n-gram precisions."""
        score = bleu_score(
            "the cat is on the mat",
            "the cat is on the mat"
        )
        # All n-grams should match perfectly
        for p in score["precisions"]:
            assert p > 0.9


class TestBERTScore:
    """Test BERTScore calculation."""

    def test_bertscore_without_package(self):
        """Test BERTScore when package is not installed."""
        # Should return placeholder values
        score = bertscore("the cat sat", "the cat sat")
        assert "precision" in score
        assert "recall" in score
        assert "f1" in score
        # Without package, should return 0.0 or error
        assert score["precision"] == 0.0 or "error" in score

    def test_bertscore_list_input(self):
        """Test BERTScore with list input."""
        score = bertscore(
            ["the cat sat", "the dog ran"],
            ["a cat sat", "a dog ran"]
        )
        # Should handle gracefully
        assert "precision" in score
        assert "recall" in score
        assert "f1" in score

    def test_bertscore_model_type(self):
        """Test BERTScore with different model type."""
        score = bertscore(
            "the cat sat",
            "the cat sat",
            model_type="bert-base-uncased"
        )
        # Should return structure even without package
        assert "f1" in score


class TestComputeGenerationMetrics:
    """Test combined generation metrics computation."""

    def test_compute_all_metrics(self):
        """Test computing all generation metrics."""
        predictions = ["the cat is on the mat", "hello world"]
        references = ["the cat was on the mat", "hi world"]

        metrics = compute_generation_metrics(predictions, references)

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

        # Check values are valid
        for key, value in metrics.items():
            assert isinstance(value, (int, float))
            assert 0.0 <= value <= 1.0

    def test_compute_metrics_single_string(self):
        """Test computing metrics with single strings."""
        metrics = compute_generation_metrics(
            "the cat is on the mat",
            "the cat was on the mat"
        )
        assert "rouge_l_fmeasure" in metrics
        assert "bleu" in metrics

    def test_compute_metrics_with_bertscore(self):
        """Test computing metrics including BERTScore."""
        metrics = compute_generation_metrics(
            "the cat is on the mat",
            "the cat was on the mat",
            use_bertscore=True
        )
        # Should include BERTScore (even if placeholder)
        assert "bertscore_precision" in metrics
        assert "bertscore_recall" in metrics
        assert "bertscore_f1" in metrics

    def test_compute_metrics_empty_input(self):
        """Test computing metrics with empty input."""
        metrics = compute_generation_metrics("", "")
        # Should handle gracefully
        assert "rouge_l_fmeasure" in metrics
        assert "bleu" in metrics


class TestGenerationMetricsEdgeCases:
    """Test edge cases for generation metrics."""

    def test_very_long_strings(self):
        """Test metrics with very long strings."""
        long_pred = "word " * 100
        long_ref = "word " * 100
        score = rouge_scores(long_pred, long_ref)
        assert "fmeasure" in score

    def test_unicode_characters(self):
        """Test metrics with unicode characters."""
        score = rouge_scores(
            "hello 世界",
            "hello 世界"
        )
        assert "fmeasure" in score

    def test_special_characters(self):
        """Test metrics with special characters."""
        score = rouge_scores(
            "hello! @#$ %^&* world",
            "hello! @#$ %^&* world"
        )
        assert "fmeasure" in score

    def test_single_word(self):
        """Test metrics with single word."""
        score = bleu_score("hello", "hello")
        assert score["bleu"] > 0.9

    def test_different_cases(self):
        """Test metrics with different casing."""
        score = rouge_scores(
            "The Cat Sat On The Mat",
            "the cat sat on the mat"
        )
        # Case-sensitive, so should have lower score
        assert score["fmeasure"] < 1.0


@pytest.fixture
def sample_generation_data():
    """Fixture providing sample generation data for testing."""
    return {
        "predictions": [
            "The cat is sitting on the mat",
            "I love eating pizza and pasta",
            "Python is a great programming language",
            "The weather is beautiful today",
            "Machine learning is fascinating"
        ],
        "references": [
            "A cat is sitting on the mat",
            "I enjoy pizza and pasta",
            "Python is an excellent programming language",
            "The weather is nice today",
            "Machine learning is interesting"
        ]
    }


class TestGenerationMetricsIntegration:
    """Integration tests for generation metrics."""

    def test_realistic_evaluation(self, sample_generation_data):
        """Test metrics on realistic generation scenario."""
        metrics = compute_generation_metrics(
            sample_generation_data["predictions"],
            sample_generation_data["references"]
        )

        # ROUGE should have decent scores for similar meanings
        assert metrics["rouge_l_fmeasure"] > 0.5

        # BLEU should have reasonable scores
        assert metrics["bleu"] > 0.3

        # All individual BLEU n-grams should be valid
        assert 0.0 <= metrics["bleu_1"] <= 1.0
        assert 0.0 <= metrics["bleu_2"] <= 1.0
        assert 0.0 <= metrics["bleu_3"] <= 1.0
        assert 0.0 <= metrics["bleu_4"] <= 1.0

    def test_metrics_correlation(self, sample_generation_data):
        """Test that different metrics correlate reasonably."""
        metrics = compute_generation_metrics(
            sample_generation_data["predictions"],
            sample_generation_data["references"]
        )

        # Higher-order n-grams should have lower scores
        assert metrics["bleu_1"] >= metrics["bleu_2"]
        assert metrics["bleu_2"] >= metrics["bleu_3"]
        assert metrics["bleu_3"] >= metrics["bleu_4"]

    def test_consistency_across_calls(self, sample_generation_data):
        """Test that metrics are consistent across multiple calls."""
        metrics1 = compute_generation_metrics(
            sample_generation_data["predictions"],
            sample_generation_data["references"]
        )
        metrics2 = compute_generation_metrics(
            sample_generation_data["predictions"],
            sample_generation_data["references"]
        )

        # Results should be identical
        for key in metrics1:
            assert metrics1[key] == pytest.approx(metrics2[key])
