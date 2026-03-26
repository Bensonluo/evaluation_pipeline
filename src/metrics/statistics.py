"""
Statistical Analysis for Evaluation Metrics

This module implements statistical methods for analyzing evaluation metrics,
particularly for comparing baseline vs. current model performance.

Functions implemented:
    - bootstrap_confidence_interval: Bootstrap CI for metric differences
    - compute_p_value: P-value calculation for significance testing

Reference: DESIGN.md Chapter 14 - Statistical Significance Testing
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from scipy import stats


def bootstrap_confidence_interval(
    baseline_scores: Union[List[float], np.ndarray],
    current_scores: Union[List[float], np.ndarray],
    confidence_level: float = 0.95,
    n_bootstrap: int = 1000,
    random_seed: Optional[int] = None
) -> Dict[str, float]:
    """
    Calculate bootstrap confidence interval for metric differences.

    Uses pooled bootstrap sampling to determine if the difference between
    baseline and current scores is statistically significant.

    Args:
        baseline_scores: List/array of baseline metric scores (per-sample)
        current_scores: List/array of current metric scores (per-sample)
        confidence_level: Confidence level for CI (default: 0.95)
        n_bootstrap: Number of bootstrap iterations (default: 1000)
        random_seed: Optional random seed for reproducibility

    Returns:
        Dictionary with:
            - delta_mean: Observed mean difference (current - baseline)
            - ci_lower: Lower bound of confidence interval
            - ci_upper: Upper bound of confidence interval
            - p_value: Two-tailed p-value
            - is_significant: Whether difference is statistically significant
            - baseline_mean: Mean of baseline scores
            - current_mean: Mean of current scores

    Example:
        >>> baseline = [0.72, 0.75, 0.68, 0.80, 0.71]
        >>> current = [0.78, 0.82, 0.75, 0.85, 0.79]
        >>> bootstrap_confidence_interval(baseline, current)
        {
            "delta_mean": 0.06,
            "ci_lower": 0.012,
            "ci_upper": 0.108,
            "p_value": 0.023,
            "is_significant": True,
            "baseline_mean": 0.732,
            "current_mean": 0.792
        }

    Reference:
        DESIGN.md Chapter 14.2 - Bootstrap Confidence Interval Method
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    baseline = np.asarray(baseline_scores)
    current = np.asarray(current_scores)

    # Calculate observed delta
    baseline_mean = np.mean(baseline)
    current_mean = np.mean(current)
    observed_delta = current_mean - baseline_mean

    # Pooled bootstrap sampling
    pooled = np.concatenate([baseline, current])
    n_baseline = len(baseline)
    n_current = len(current)

    bootstrap_deltas = []
    for _ in range(n_bootstrap):
        # Resample with replacement
        baseline_sample = np.random.choice(pooled, size=n_baseline, replace=True)
        current_sample = np.random.choice(pooled, size=n_current, replace=True)
        bootstrap_deltas.append(np.mean(current_sample) - np.mean(baseline_sample))

    bootstrap_deltas = np.array(bootstrap_deltas)

    # Calculate confidence interval
    alpha = 1 - confidence_level
    ci_lower = np.percentile(bootstrap_deltas, alpha / 2 * 100)
    ci_upper = np.percentile(bootstrap_deltas, (1 - alpha / 2) * 100)

    # Calculate p-value (proportion of bootstrap deltas >= observed delta)
    p_value = np.mean(np.abs(bootstrap_deltas) >= np.abs(observed_delta))

    # Determine significance: p < 0.05 AND CI doesn't include 0
    is_significant = p_value < 0.05 and (ci_lower > 0 or ci_upper < 0)

    return {
        "delta_mean": float(observed_delta),
        "delta_pct": float(observed_delta / baseline_mean * 100) if baseline_mean > 0 else 0.0,
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "p_value": float(p_value),
        "is_significant": bool(is_significant),
        "baseline_mean": float(baseline_mean),
        "current_mean": float(current_mean),
        "confidence_level": confidence_level
    }


def compute_p_value(
    baseline_scores: Union[List[float], np.ndarray],
    current_scores: Union[List[float], np.ndarray],
    test_type: str = "two-sided",
    alternative: str = "greater"
) -> Dict[str, float]:
    """
    Compute p-value for statistical significance testing.

    Supports multiple statistical tests for comparing two distributions.

    Args:
        baseline_scores: Baseline metric scores
        current_scores: Current metric scores
        test_type: Type of statistical test
            - "ttest": Student's t-test (assumes normal distribution)
            - "wilcoxon": Wilcoxon rank-sum test (non-parametric)
            - "mannwhitney": Mann-Whitney U test (non-parametric)
        alternative: Alternative hypothesis
            - "two-sided": current != baseline
            - "greater": current > baseline
            - "less": current < baseline

    Returns:
        Dictionary with p-value and test statistic

    Example:
        >>> baseline = [0.72, 0.75, 0.68, 0.80, 0.71]
        >>> current = [0.78, 0.82, 0.75, 0.85, 0.79]
        >>> compute_p_value(baseline, current, test_type="ttest")
        {"p_value": 0.023, "statistic": 3.45, "test_type": "ttest"}

    Note:
        For small samples (n < 30), prefer non-parametric tests.
    """
    baseline = np.asarray(baseline_scores)
    current = np.asarray(current_scores)

    if test_type == "ttest":
        # Student's independent t-test
        statistic, p_value = stats.ttest_ind(current, baseline, alternative=alternative)
    elif test_type == "wilcoxon":
        # Wilcoxon rank-sum test
        if alternative == "two-sided":
            statistic, p_value = stats.mannwhitneyu(current, baseline, alternative="two-sided")
        elif alternative == "greater":
            statistic, p_value = stats.mannwhitneyu(current, baseline, alternative="greater")
        else:
            statistic, p_value = stats.mannwhitneyu(baseline, current, alternative="greater")
    elif test_type == "mannwhitney":
        # Mann-Whitney U test (same as Wilcoxon rank-sum in scipy)
        statistic, p_value = stats.mannwhitneyu(
            current, baseline,
            alternative="two-sided" if alternative == "two-sided" else alternative
        )
    else:
        raise ValueError(f"Unknown test_type: {test_type}")

    return {
        "p_value": float(p_value),
        "statistic": float(statistic),
        "test_type": test_type,
        "alternative": alternative,
        "is_significant": p_value < 0.05
    }


def paired_bootstrap_confidence_interval(
    baseline_scores: Union[List[float], np.ndarray],
    current_scores: Union[List[float], np.ndarray],
    confidence_level: float = 0.95,
    n_bootstrap: int = 1000,
    random_seed: Optional[int] = None
) -> Dict[str, float]:
    """
    Calculate paired bootstrap confidence interval.

    Use this when baseline and current scores are from the same samples
    (e.g., same test set evaluated on different models).

    Args:
        baseline_scores: Baseline metric scores
        current_scores: Current metric scores (same samples as baseline)
        confidence_level: Confidence level for CI
        n_bootstrap: Number of bootstrap iterations
        random_seed: Optional random seed

    Returns:
        Dictionary with confidence interval and p-value

    Example:
        >>> baseline = [0.72, 0.75, 0.68, 0.80, 0.71]
        >>> current = [0.78, 0.79, 0.70, 0.82, 0.75]
        >>> paired_bootstrap_confidence_interval(baseline, current)
        {
            "delta_mean": 0.04,
            "ci_lower": 0.01,
            "ci_upper": 0.07,
            "p_value": 0.015,
            "is_significant": True
        }
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    baseline = np.asarray(baseline_scores)
    current = np.asarray(current_scores)

    if len(baseline) != len(current):
        raise ValueError("Paired test requires equal length arrays")

    # Calculate observed delta
    observed_deltas = current - baseline
    observed_mean = np.mean(observed_deltas)

    # Bootstrap the differences
    n = len(observed_deltas)
    bootstrap_means = []

    for _ in range(n_bootstrap):
        # Resample differences with replacement
        sample = np.random.choice(observed_deltas, size=n, replace=True)
        bootstrap_means.append(np.mean(sample))

    bootstrap_means = np.array(bootstrap_means)

    # Calculate confidence interval
    alpha = 1 - confidence_level
    ci_lower = np.percentile(bootstrap_means, alpha / 2 * 100)
    ci_upper = np.percentile(bootstrap_means, (1 - alpha / 2) * 100)

    # Calculate p-value
    p_value = np.mean(np.abs(bootstrap_means) >= np.abs(observed_mean))

    is_significant = p_value < 0.05 and (ci_lower > 0 or ci_upper < 0)

    return {
        "delta_mean": float(observed_mean),
        "delta_pct": float(observed_mean / np.mean(baseline) * 100) if np.mean(baseline) > 0 else 0.0,
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "p_value": float(p_value),
        "is_significant": bool(is_significant),
        "baseline_mean": float(np.mean(baseline)),
        "current_mean": float(np.mean(current)),
        "confidence_level": confidence_level
    }


def compute_delta_status(
    baseline_value: float,
    current_value: float,
    significance_result: Optional[Dict[str, float]] = None,
    delta_pct_threshold: float = 1.0,
    degrade_threshold_pct: float = 5.0
) -> Dict[str, Union[str, float, bool]]:
    """
    Compute delta status with optional significance testing.

    Determines the status of metric change following DESIGN.md Chapter 14.3:
    - SIGNIFICANT_IMPROVED: Upward, significant, |delta| > 1%
    - MARGINAL_IMPROVED: Upward, significant, |delta| <= 1%
    - SIGNIFICANT_DEGRADED: Downward, significant, |delta| > 1%
    - MARGINAL_DEGRADED: Downward, significant, |delta| <= 1%
    - NOT_SIGNIFICANT: Not statistically significant
    - UNKNOWN: Without significance testing

    Args:
        baseline_value: Baseline metric value
        current_value: Current metric value
        significance_result: Optional result from bootstrap_confidence_interval
        delta_pct_threshold: Minimum delta % to be considered meaningful (default: 1%)
        degrade_threshold_pct: Delta % threshold for DEGRADED status (default: 5%)

    Returns:
        Dictionary with status, delta values, and flags

    Example:
        >>> compute_delta_status(0.723, 0.789, {"is_significant": True, "delta_pct": 9.1})
        {
            "status": "SIGNIFICANT_IMPROVED",
            "delta": 0.066,
            "delta_pct": 9.1,
            "is_improved": True,
            "is_degraded": False,
            "is_significant": True
        }
    """
    delta = current_value - baseline_value
    delta_pct = delta / baseline_value * 100 if baseline_value > 0 else 0.0

    is_significant = significance_result.get("is_significant", False) if significance_result else False
    abs_delta_pct = abs(delta_pct)

    if significance_result and is_significant:
        if delta > 0:
            if abs_delta_pct > delta_pct_threshold:
                status = "SIGNIFICANT_IMPROVED"
            else:
                status = "MARGINAL_IMPROVED"
        else:
            if abs_delta_pct > delta_pct_threshold:
                status = "SIGNIFICANT_DEGRADED"
            else:
                status = "MARGINAL_DEGRADED"
    else:
        # Without significance or not significant
        if delta < -degrade_threshold_pct:
            status = "DEGRADED"
        elif delta > 0 and abs_delta_pct > delta_pct_threshold:
            status = "IMPROVED"
        else:
            status = "NOT_SIGNIFICANT"

    return {
        "status": status,
        "delta": delta,
        "delta_pct": delta_pct,
        "is_improved": delta > 0,
        "is_degraded": delta < -degrade_threshold_pct,
        "is_significant": is_significant,
        "baseline": baseline_value,
        "current": current_value
    }


def multiple_comparison_correction(
    p_values: List[float],
    method: str = "bonferroni"
) -> List[float]:
    """
    Apply multiple comparison correction to p-values.

    When testing multiple metrics, correct for family-wise error rate.

    Args:
        p_values: List of raw p-values
        method: Correction method
            - "bonferroni": Divide alpha by number of tests (conservative)
            - "holm": Holm-Bonferroni step-down procedure
            - "fdr": Benjamini-Hochberg FDR control

    Returns:
        List of corrected p-values

    Example:
        >>> p_values = [0.01, 0.04, 0.03, 0.20]
        >>> multiple_comparison_correction(p_values, method="bonferroni")
        [0.04, 0.16, 0.12, 0.80]
    """
    p_values = np.array(p_values)
    n = len(p_values)

    if method == "bonferroni":
        corrected = np.minimum(p_values * n, 1.0)

    elif method == "holm":
        # Sort p-values
        sorted_indices = np.argsort(p_values)
        sorted_p = p_values[sorted_indices]

        # Step-down correction
        corrected = np.zeros_like(sorted_p)
        for i, p in enumerate(sorted_p):
            corrected[i] = min(p * (n - i), 1.0)

        # Enforce monotonicity
        for i in range(len(corrected) - 2, -1, -1):
            corrected[i] = min(corrected[i], corrected[i + 1])

        # Unsort
        unsorted_corrected = np.zeros_like(corrected)
        unsorted_corrected[sorted_indices] = corrected
        corrected = unsorted_corrected

    elif method == "fdr":
        # Benjamini-Hochberg procedure
        sorted_indices = np.argsort(p_values)
        sorted_p = p_values[sorted_indices]

        corrected = np.zeros_like(sorted_p)
        for i, p in enumerate(sorted_p):
            corrected[i] = p * n / (i + 1)

        # Enforce monotonicity from right
        for i in range(len(corrected) - 2, -1, -1):
            corrected[i] = min(corrected[i], corrected[i + 1])

        corrected = np.minimum(corrected, 1.0)

        # Unsort
        unsorted_corrected = np.zeros_like(corrected)
        unsorted_corrected[sorted_indices] = corrected
        corrected = unsorted_corrected

    else:
        raise ValueError(f"Unknown correction method: {method}")

    return corrected.tolist()


def compute_significance_for_all_metrics(
    baseline_metrics: Dict[str, List[float]],
    current_metrics: Dict[str, List[float]],
    confidence_level: float = 0.95,
    n_bootstrap: int = 1000,
    correction_method: str = "fdr"
) -> Dict[str, Dict[str, float]]:
    """
    Compute statistical significance for all metrics at once.

    Args:
        baseline_metrics: Dict mapping metric name to list of baseline scores
        current_metrics: Dict mapping metric name to list of current scores
        confidence_level: Confidence level for CI
        n_bootstrap: Number of bootstrap iterations
        correction_method: Multiple comparison correction method

    Returns:
        Dict mapping metric name to significance results

    Example:
        >>> baseline = {"mrr": [0.7, 0.8, 0.75], "ndcg": [0.6, 0.7, 0.65]}
        >>> current = {"mrr": [0.75, 0.85, 0.8], "ndcg": [0.65, 0.72, 0.68]}
        >>> compute_significance_for_all_metrics(baseline, current)
        {
            "mrr": {"p_value": 0.023, "is_significant": True, ...},
            "ndcg": {"p_value": 0.15, "is_significant": False, ...}
        }
    """
    results = {}
    p_values = []

    # First pass: compute uncorrected significance
    for metric_name in baseline_metrics:
        if metric_name not in current_metrics:
            continue

        baseline_scores = baseline_metrics[metric_name]
        current_scores = current_metrics[metric_name]

        result = bootstrap_confidence_interval(
            baseline_scores,
            current_scores,
            confidence_level=confidence_level,
            n_bootstrap=n_bootstrap
        )
        results[metric_name] = result
        p_values.append(result["p_value"])

    # Apply multiple comparison correction
    if len(p_values) > 1 and correction_method:
        corrected_p_values = multiple_comparison_correction(p_values, method=correction_method)

        # Update results with corrected p-values
        for i, metric_name in enumerate(results.keys()):
            results[metric_name]["p_value_corrected"] = corrected_p_values[i]
            results[metric_name]["is_significant_corrected"] = corrected_p_values[i] < 0.05

    return results
