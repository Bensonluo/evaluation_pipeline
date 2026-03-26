"""
Unit tests for statistical analysis module.

Tests the metrics.statistics module which implements bootstrap confidence
intervals, p-value calculation, and significance testing.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from metrics.statistics import (
    bootstrap_confidence_interval,
    compute_p_value,
    paired_bootstrap_confidence_interval,
    compute_delta_status,
    multiple_comparison_correction,
    compute_significance_for_all_metrics
)


class TestBootstrapConfidenceInterval:
    """Test bootstrap confidence interval calculation."""

    def test_bootstrap_significant_improvement(self):
        """Test CI with significant improvement."""
        baseline = [0.70, 0.72, 0.68, 0.71, 0.69]
        current = [0.78, 0.80, 0.76, 0.79, 0.77]

        result = bootstrap_confidence_interval(
            baseline,
            current,
            confidence_level=0.95,
            n_bootstrap=1000,
            random_seed=42
        )

        # Check structure
        assert "delta_mean" in result
        assert "ci_lower" in result
        assert "ci_upper" in result
        assert "p_value" in result
        assert "is_significant" in result
        assert "baseline_mean" in result
        assert "current_mean" in result

        # Current should be higher than baseline
        assert result["current_mean"] > result["baseline_mean"]
        assert result["delta_mean"] > 0

        # Should be significant (clear separation)
        assert result["is_significant"] is True
        assert result["p_value"] < 0.05

    def test_bootstrap_no_significant_difference(self):
        """Test CI with no significant difference."""
        np.random.seed(42)
        baseline = np.random.normal(0.75, 0.02, 50).tolist()
        current = np.random.normal(0.755, 0.02, 50).tolist()

        result = bootstrap_confidence_interval(
            baseline,
            current,
            confidence_level=0.95,
            n_bootstrap=1000,
            random_seed=42
        )

        # Small difference, likely not significant
        # CI should include 0
        assert result["ci_lower"] <= 0 <= result["ci_upper"]

    def test_bootstrap_different_confidence_levels(self):
        """Test bootstrap with different confidence levels."""
        baseline = [0.70, 0.72, 0.68, 0.71, 0.69]
        current = [0.78, 0.80, 0.76, 0.79, 0.77]

        result_90 = bootstrap_confidence_interval(
            baseline, current,
            confidence_level=0.90,
            n_bootstrap=1000,
            random_seed=42
        )
        result_99 = bootstrap_confidence_interval(
            baseline, current,
            confidence_level=0.99,
            n_bootstrap=1000,
            random_seed=42
        )

        # 99% CI should be wider than 90% CI
        ci_width_90 = result_99["ci_upper"] - result_99["ci_lower"]
        ci_width_99 = result_90["ci_upper"] - result_90["ci_lower"]
        assert ci_width_99 >= ci_width_90

    def test_bootstrap_reproducibility(self):
        """Test that bootstrap is reproducible with same seed."""
        baseline = [0.70, 0.72, 0.68, 0.71, 0.69]
        current = [0.78, 0.80, 0.76, 0.79, 0.77]

        result1 = bootstrap_confidence_interval(
            baseline, current,
            n_bootstrap=1000,
            random_seed=42
        )
        result2 = bootstrap_confidence_interval(
            baseline, current,
            n_bootstrap=1000,
            random_seed=42
        )

        # Results should be identical
        assert result1["delta_mean"] == pytest.approx(result2["delta_mean"])
        assert result1["p_value"] == pytest.approx(result2["p_value"])
        assert result1["ci_lower"] == pytest.approx(result2["ci_lower"])
        assert result1["ci_upper"] == pytest.approx(result2["ci_upper"])

    def test_bootstrap_delta_percentage(self):
        """Test delta percentage calculation."""
        baseline = [0.50, 0.52, 0.48]
        current = [0.60, 0.62, 0.58]

        result = bootstrap_confidence_interval(
            baseline, current,
            n_bootstrap=1000,
            random_seed=42
        )

        # Delta should be about 20% (0.10 / 0.50)
        assert result["delta_pct"] > 15  # Approximately

    def test_bootstrap_zero_baseline(self):
        """Test bootstrap when baseline mean is zero."""
        baseline = [0.0, 0.0, 0.0]
        current = [0.1, 0.1, 0.1]

        result = bootstrap_confidence_interval(
            baseline, current,
            n_bootstrap=1000,
            random_seed=42
        )

        # Should handle gracefully
        assert "delta_pct" in result
        # Delta percentage should be 0 when baseline is 0
        assert result["delta_pct"] == 0.0


class TestComputePValue:
    """Test p-value calculation for different statistical tests."""

    def test_ttest_significant(self):
        """Test t-test with significant difference."""
        baseline = [0.70, 0.72, 0.68, 0.71, 0.69]
        current = [0.78, 0.80, 0.76, 0.79, 0.77]

        result = compute_p_value(baseline, current, test_type="ttest")

        assert "p_value" in result
        assert "statistic" in result
        assert result["test_type"] == "ttest"
        assert result["is_significant"] is True
        assert result["p_value"] < 0.05

    def test_ttest_two_sided(self):
        """Test t-test with two-sided alternative."""
        result = compute_p_value(
            [0.5, 0.6, 0.7],
            [0.7, 0.8, 0.9],
            test_type="ttest",
            alternative="two-sided"
        )
        assert result["alternative"] == "two-sided"

    def test_ttest_greater(self):
        """Test t-test with greater alternative."""
        result = compute_p_value(
            [0.5, 0.6, 0.7],
            [0.7, 0.8, 0.9],
            test_type="ttest",
            alternative="greater"
        )
        assert result["alternative"] == "greater"
        assert result["is_significant"] is True

    def test_wilcoxon_test(self):
        """Test Wilcoxon rank-sum test."""
        baseline = [0.70, 0.72, 0.68, 0.71, 0.69]
        current = [0.78, 0.80, 0.76, 0.79, 0.77]

        result = compute_p_value(baseline, current, test_type="wilcoxon")

        assert result["test_type"] in ["wilcoxon", "mannwhitney"]
        assert result["p_value"] < 0.05
        assert result["is_significant"] is True

    def test_mannwhitney_test(self):
        """Test Mann-Whitney U test."""
        result = compute_p_value(
            [0.5, 0.6, 0.7],
            [0.8, 0.9, 1.0],
            test_type="mannwhitney"
        )
        assert result["test_type"] == "mannwhitney"
        assert result["p_value"] < 0.05

    def test_unknown_test_type(self):
        """Test with unknown test type."""
        with pytest.raises(ValueError):
            compute_p_value([0.5, 0.6], [0.7, 0.8], test_type="unknown")

    def test_p_value_no_difference(self):
        """Test p-value when distributions are similar."""
        np.random.seed(42)
        baseline = np.random.normal(0.75, 0.01, 30).tolist()
        current = np.random.normal(0.751, 0.01, 30).tolist()

        result = compute_p_value(baseline, current, test_type="ttest")
        # Should not be significant
        assert result["p_value"] > 0.05


class TestPairedBootstrap:
    """Test paired bootstrap confidence interval."""

    def test_paired_bootstrap_significant(self):
        """Test paired bootstrap with significant difference."""
        baseline = [0.70, 0.72, 0.68, 0.71, 0.69]
        current = [0.78, 0.80, 0.76, 0.79, 0.77]

        result = paired_bootstrap_confidence_interval(
            baseline, current,
            n_bootstrap=1000,
            random_seed=42
        )

        assert "delta_mean" in result
        assert "ci_lower" in result
        assert "ci_upper" in result
        assert "p_value" in result
        assert result["delta_mean"] > 0
        assert result["is_significant"] is True

    def test_paired_bootstrap_unequal_length(self):
        """Test paired bootstrap with unequal length arrays."""
        baseline = [0.70, 0.72, 0.68]
        current = [0.78, 0.80]

        with pytest.raises(ValueError):
            paired_bootstrap_confidence_interval(baseline, current)

    def test_paired_vs_pooled(self):
        """Compare paired vs pooled bootstrap."""
        baseline = [0.70, 0.72, 0.68, 0.71, 0.69]
        current = [0.75, 0.77, 0.73, 0.76, 0.74]

        paired = paired_bootstrap_confidence_interval(
            baseline, current,
            n_bootstrap=1000,
            random_seed=42
        )
        pooled = bootstrap_confidence_interval(
            baseline, current,
            n_bootstrap=1000,
            random_seed=42
        )

        # Both should detect significance
        assert paired["is_significant"] is True
        assert pooled["is_significant"] is True

        # Paired should have tighter CI (more sensitive)
        paired_width = paired["ci_upper"] - paired["ci_lower"]
        pooled_width = pooled["ci_upper"] - pooled["ci_lower"]
        assert paired_width <= pooled_width


class TestComputeDeltaStatus:
    """Test delta status computation."""

    def test_significant_improved(self):
        """Test SIGNIFICANT_IMPROVED status."""
        sig_result = {"is_significant": True, "delta_pct": 9.1}
        result = compute_delta_status(0.723, 0.789, sig_result)

        assert result["status"] == "SIGNIFICANT_IMPROVED"
        assert result["is_improved"] is True
        assert result["is_degraded"] is False
        assert result["is_significant"] is True
        assert result["delta"] == pytest.approx(0.066)
        assert result["delta_pct"] == pytest.approx(9.1, abs=0.1)

    def test_marginal_improved(self):
        """Test MARGINAL_IMPROVED status."""
        sig_result = {"is_significant": True}
        result = compute_delta_status(0.800, 0.802, sig_result, delta_pct_threshold=1.0)

        assert result["status"] == "MARGINAL_IMPROVED"
        assert result["is_improved"] is True

    def test_significant_degraded(self):
        """Test SIGNIFICANT_DEGRADED status."""
        sig_result = {"is_significant": True}
        result = compute_delta_status(0.800, 0.720, sig_result, delta_pct_threshold=1.0)

        assert result["status"] == "SIGNIFICANT_DEGRADED"
        assert result["is_degraded"] is True
        assert result["is_improved"] is False

    def test_marginal_degraded(self):
        """Test MARGINAL_DEGRADED status."""
        sig_result = {"is_significant": True}
        result = compute_delta_status(0.800, 0.795, sig_result, delta_pct_threshold=1.0)

        assert result["status"] == "MARGINAL_DEGRADED"
        assert result["is_degraded"] is False  # Not below degrade_threshold

    def test_not_significant(self):
        """Test NOT_SIGNIFICANT status."""
        sig_result = {"is_significant": False}
        result = compute_delta_status(0.750, 0.760, sig_result)

        assert result["status"] == "NOT_SIGNIFICANT"
        assert result["is_significant"] is False

    def test_without_significance_testing(self):
        """Test delta status without significance testing."""
        result = compute_delta_status(0.750, 0.760)

        assert "status" in result
        assert "delta" in result
        assert "is_significant" in result

    def test_degraded_without_significance(self):
        """Test DEGRADED status for large drop without significance."""
        result = compute_delta_status(
            0.800, 0.700,
            significance_result=None,
            degrade_threshold_pct=5.0
        )

        # Delta is -12.5%, which exceeds degrade_threshold
        assert result["is_degraded"] is True

    def test_zero_baseline(self):
        """Test delta status with zero baseline."""
        result = compute_delta_status(0.0, 0.1)

        assert result["delta_pct"] == 0.0
        assert result["status"] in ["IMPROVED", "NOT_SIGNIFICANT"]


class TestMultipleComparisonCorrection:
    """Test multiple comparison correction methods."""

    def test_bonferroni_correction(self):
        """Test Bonferroni correction."""
        p_values = [0.01, 0.04, 0.03, 0.20]
        corrected = multiple_comparison_correction(p_values, method="bonferroni")

        # All p-values should be multiplied by n=4
        assert corrected[0] == pytest.approx(0.04)
        assert corrected[1] == pytest.approx(0.16)
        assert corrected[2] == pytest.approx(0.12)
        assert corrected[3] == pytest.approx(0.80)

    def test_bonferroni_cap_at_one(self):
        """Test that corrected p-values cap at 1.0."""
        p_values = [0.3, 0.4, 0.5]
        corrected = multiple_comparison_correction(p_values, method="bonferroni")

        # Should not exceed 1.0
        for p in corrected:
            assert p <= 1.0

    def test_holm_correction(self):
        """Test Holm-Bonferroni correction."""
        p_values = [0.01, 0.04, 0.03, 0.20]
        corrected = multiple_comparison_correction(p_values, method="holm")

        # Holm should be less conservative than Bonferroni
        bonferroni = multiple_comparison_correction(p_values, method="bonferroni")

        # At least some should be lower than Bonferroni
        assert sum(c < b for c, b in zip(corrected, bonferroni)) >= 1

    def test_fdr_correction(self):
        """Test Benjamini-Hochberg FDR correction."""
        p_values = [0.01, 0.04, 0.03, 0.20]
        corrected = multiple_comparison_correction(p_values, method="fdr")

        # FDR should be less conservative than Bonferroni
        bonferroni = multiple_comparison_correction(p_values, method="bonferroni")

        # FDR should give lower values
        for c, b in zip(corrected, bonferroni):
            assert c <= b

    def test_unknown_correction_method(self):
        """Test with unknown correction method."""
        with pytest.raises(ValueError):
            multiple_comparison_correction([0.01, 0.02], method="unknown")

    def test_single_p_value(self):
        """Test correction with single p-value."""
        corrected = multiple_comparison_correction([0.05], method="bonferroni")
        assert corrected[0] == 0.05


class TestComputeSignificanceForAllMetrics:
    """Test computing significance for multiple metrics."""

    def test_multiple_metrics(self):
        """Test significance computation for multiple metrics."""
        baseline = {
            "mrr": [0.70, 0.72, 0.68, 0.71, 0.69],
            "ndcg": [0.60, 0.62, 0.58, 0.61, 0.59],
            "hitrate": [0.80, 0.82, 0.78, 0.81, 0.79],
        }
        current = {
            "mrr": [0.75, 0.77, 0.73, 0.76, 0.74],
            "ndcg": [0.63, 0.65, 0.61, 0.64, 0.62],
            "hitrate": [0.85, 0.87, 0.83, 0.86, 0.84],
        }

        results = compute_significance_for_all_metrics(
            baseline, current,
            n_bootstrap=1000,
            correction_method="fdr",
            random_seed=42
        )

        # Should have results for all metrics
        assert len(results) == 3
        for metric in ["mrr", "ndcg", "hitrate"]:
            assert metric in results
            assert "p_value" in results[metric]
            assert "is_significant" in results[metric]
            assert "p_value_corrected" in results[metric]
            assert "is_significant_corrected" in results[metric]

    def test_fdr_correction_applied(self):
        """Test that FDR correction is properly applied."""
        baseline = {
            "metric1": [0.5, 0.6, 0.7],
            "metric2": [0.5, 0.6, 0.7],
        }
        current = {
            "metric1": [0.6, 0.7, 0.8],
            "metric2": [0.55, 0.65, 0.75],
        }

        results = compute_significance_for_all_metrics(
            baseline, current,
            n_bootstrap=500,
            correction_method="fdr",
            random_seed=42
        )

        # Corrected p-values should be >= original p-values
        for metric_result in results.values():
            if "p_value_corrected" in metric_result:
                assert metric_result["p_value_corrected"] >= metric_result["p_value"]

    def test_no_correction_for_single_metric(self):
        """Test that no correction is applied for single metric."""
        baseline = {"metric": [0.5, 0.6, 0.7]}
        current = {"metric": [0.6, 0.7, 0.8]}

        results = compute_significance_for_all_metrics(
            baseline, current,
            n_bootstrap=500,
            correction_method="fdr",
            random_seed=42
        )

        # Should not have corrected values for single metric
        assert "p_value_corrected" not in results["metric"]
        assert "is_significant_corrected" not in results["metric"]

    def test_missing_metric_in_current(self):
        """Test handling of missing metric in current."""
        baseline = {
            "metric1": [0.5, 0.6, 0.7],
            "metric2": [0.5, 0.6, 0.7],
        }
        current = {
            "metric1": [0.6, 0.7, 0.8],
            # metric2 is missing
        }

        results = compute_significance_for_all_metrics(
            baseline, current,
            n_bootstrap=500,
            random_seed=42
        )

        # Should only have results for metric1
        assert "metric1" in results
        assert "metric2" not in results


@pytest.fixture
def sample_metric_scores():
    """Fixture providing sample metric scores for testing."""
    return {
        "baseline": {
            "mrr": [0.72, 0.75, 0.68, 0.80, 0.71, 0.74, 0.69, 0.77, 0.73, 0.70],
            "ndcg": [0.65, 0.68, 0.62, 0.70, 0.66, 0.69, 0.63, 0.67, 0.64, 0.61],
            "precision": [0.60, 0.63, 0.58, 0.65, 0.61, 0.64, 0.59, 0.62, 0.60, 0.57],
        },
        "current": {
            "mrr": [0.78, 0.82, 0.75, 0.85, 0.79, 0.81, 0.76, 0.83, 0.80, 0.77],
            "ndcg": [0.70, 0.73, 0.68, 0.75, 0.71, 0.74, 0.69, 0.72, 0.71, 0.68],
            "precision": [0.65, 0.68, 0.63, 0.70, 0.66, 0.69, 0.64, 0.67, 0.66, 0.63],
        }
    }


class TestStatisticsIntegration:
    """Integration tests for statistical analysis."""

    def test_full_significance_workflow(self, sample_metric_scores):
        """Test complete workflow: CI, p-value, status."""
        baseline_scores = sample_metric_scores["baseline"]["mrr"]
        current_scores = sample_metric_scores["current"]["mrr"]

        # Compute CI
        ci_result = bootstrap_confidence_interval(
            baseline_scores, current_scores,
            n_bootstrap=1000,
            random_seed=42
        )

        # Compute delta status
        status_result = compute_delta_status(
            ci_result["baseline_mean"],
            ci_result["current_mean"],
            ci_result
        )

        # Should show significant improvement
        assert status_result["status"] == "SIGNIFICANT_IMPROVED"
        assert status_result["is_significant"] is True

    def test_multiple_metrics_with_correction(self, sample_metric_scores):
        """Test significance testing with multiple comparison correction."""
        results = compute_significance_for_all_metrics(
            sample_metric_scores["baseline"],
            sample_metric_scores["current"],
            n_bootstrap=1000,
            correction_method="bonferroni",
            random_seed=42
        )

        # All metrics should show improvement
        for metric_name, result in results.items():
            assert result["delta_mean"] > 0
            assert result["current_mean"] > result["baseline_mean"]

    def test_reproducibility_across_runs(self, sample_metric_scores):
        """Test that results are reproducible with same seed."""
        baseline_scores = sample_metric_scores["baseline"]["mrr"]
        current_scores = sample_metric_scores["current"]["mrr"]

        result1 = bootstrap_confidence_interval(
            baseline_scores, current_scores,
            n_bootstrap=1000,
            random_seed=42
        )
        result2 = bootstrap_confidence_interval(
            baseline_scores, current_scores,
            n_bootstrap=1000,
            random_seed=42
        )

        # Should be identical
        assert result1["delta_mean"] == pytest.approx(result2["delta_mean"])
        assert result1["p_value"] == pytest.approx(result2["p_value"])
