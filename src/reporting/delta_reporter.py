"""
Delta Metrics Reporter for RAG Evaluation

This module implements delta metrics calculation and baseline comparison functionality.
It computes the relative changes between current evaluation results and baseline metrics,
performing statistical significance testing using bootstrap confidence intervals.

Reference: DESIGN.md Chapter 14 - Statistical Significance Testing
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class DeltaStatus(Enum):
    """Delta status enumeration"""
    SIGNIFICANT_IMPROVED = "SIGNIFICANT_IMPROVED"
    MARGINAL_IMPROVED = "MARGINAL_IMPROVED"
    SIGNIFICANT_DEGRADED = "SIGNIFICANT_DEGRADED"
    MARGINAL_DEGRADED = "MARGINAL_DEGRADED"
    NOT_SIGNIFICANT = "NOT_SIGNIFICANT"
    NO_BASELINE = "NO_BASELINE"


@dataclass
class SignificanceResult:
    """Statistical significance test result"""
    p_value: float
    ci_lower: float
    ci_upper: float
    is_significant: bool
    delta_mean: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "p_value": round(self.p_value, 4),
            "ci_lower": round(self.ci_lower, 4),
            "ci_upper": round(self.ci_upper, 4),
            "is_significant": self.is_significant,
            "delta_mean": round(self.delta_mean, 4)
        }


@dataclass
class DeltaMetric:
    """Single delta metric result"""
    metric_name: str
    baseline: Optional[float]
    current: float
    delta: float
    delta_pct: float
    significance: Optional[SignificanceResult]
    status: DeltaStatus

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "metric_name": self.metric_name,
            "baseline": self.baseline,
            "current": round(self.current, 4),
            "delta": round(self.delta, 4),
            "delta_pct": f"{self.delta_pct:+.1f}%",
            "status": self.status.value
        }
        if self.significance:
            result["significance"] = self.significance.to_dict()
        return result


def calculate_significance(
    baseline_scores: List[float],
    current_scores: List[float],
    confidence_level: float = 0.95,
    n_bootstrap: int = 1000
) -> SignificanceResult:
    """
    Calculate statistical significance using bootstrap confidence intervals.

    Uses pooled bootstrap method to test if the difference between baseline
    and current scores is statistically significant.

    Args:
        baseline_scores: List of baseline scores for each sample
        current_scores: List of current scores for each sample
        confidence_level: Confidence level for CI (default: 0.95)
        n_bootstrap: Number of bootstrap iterations (default: 1000)

    Returns:
        SignificanceResult containing p-value, confidence interval, and significance flag

    Example:
        >>> baseline = [0.8, 0.7, 0.9, 0.75, 0.85]
        >>> current = [0.85, 0.78, 0.92, 0.8, 0.88]
        >>> result = calculate_significance(baseline, current)
        >>> result.is_significant
        True
    """
    if len(baseline_scores) < 2 or len(current_scores) < 2:
        logger.warning("Insufficient samples for significance testing")
        return SignificanceResult(
            p_value=1.0,
            ci_lower=0.0,
            ci_upper=0.0,
            is_significant=False,
            delta_mean=0.0
        )

    original_delta = np.mean(current_scores) - np.mean(baseline_scores)

    # Pooled bootstrap sampling
    pooled = np.concatenate([baseline_scores, current_scores])
    deltas = []

    for _ in range(n_bootstrap):
        baseline_sample = np.random.choice(
            pooled,
            size=len(baseline_scores),
            replace=True
        )
        current_sample = np.random.choice(
            pooled,
            size=len(current_scores),
            replace=True
        )
        deltas.append(np.mean(current_sample) - np.mean(baseline_sample))

    deltas = np.array(deltas)

    # Calculate confidence interval
    alpha = 1 - confidence_level
    ci_lower = np.percentile(deltas, alpha / 2 * 100)
    ci_upper = np.percentile(deltas, (1 - alpha / 2) * 100)

    # Calculate p-value (two-tailed)
    p_value = np.mean(np.abs(deltas) >= np.abs(original_delta))

    # Determine significance: p < 0.05 and CI doesn't include 0
    is_significant = p_value < 0.05 and (ci_lower > 0 or ci_upper < 0)

    return SignificanceResult(
        p_value=p_value,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        is_significant=is_significant,
        delta_mean=original_delta
    )


def determine_status(
    delta: float,
    is_significant: bool,
    delta_threshold: float = 0.01
) -> DeltaStatus:
    """
    Determine delta status based on direction, significance, and magnitude.

    Status logic:
        - SIGNIFICANT_IMPROVED: delta > 0, significant, |delta| > threshold
        - MARGINAL_IMPROVED: delta > 0, significant, |delta| <= threshold
        - SIGNIFICANT_DEGRADED: delta < 0, significant, |delta| > threshold
        - MARGINAL_DEGRADED: delta < 0, significant, |delta| <= threshold
        - NOT_SIGNIFICANT: not significant

    Args:
        delta: Absolute delta value (current - baseline)
        is_significant: Whether the change is statistically significant
        delta_threshold: Minimum delta magnitude to be considered meaningful (default: 1%)

    Returns:
        DeltaStatus enum value

    Example:
        >>> determine_status(0.05, True)
        <DeltaStatus.SIGNIFICANT_IMPROVED: 'SIGNIFICANT_IMPROVED'>
    """
    if not is_significant:
        return DeltaStatus.NOT_SIGNIFICANT

    if delta > 0:
        if abs(delta) > delta_threshold:
            return DeltaStatus.SIGNIFICANT_IMPROVED
        else:
            return DeltaStatus.MARGINAL_IMPROVED
    else:
        if abs(delta) > delta_threshold:
            return DeltaStatus.SIGNIFICANT_DEGRADED
        else:
            return DeltaStatus.MARGINAL_DEGRADED


def compute_delta_metrics(
    baseline_metrics: Dict[str, float],
    current_metrics: Dict[str, float],
    baseline_scores: Optional[Dict[str, List[float]]] = None,
    current_scores: Optional[Dict[str, List[float]]] = None,
    delta_threshold: float = 0.01
) -> Dict[str, DeltaMetric]:
    """
    Compute delta metrics for all evaluation dimensions.

    Args:
        baseline_metrics: Dictionary of baseline metric values
        current_metrics: Dictionary of current metric values
        baseline_scores: Optional per-sample baseline scores for significance testing
        current_scores: Optional per-sample current scores for significance testing
        delta_threshold: Minimum delta magnitude for meaningful change (default: 1%)

    Returns:
        Dictionary mapping metric names to DeltaMetric objects

    Example:
        >>> baseline = {"mrr@5": 0.723, "ndcg@5": 0.681}
        >>> current = {"mrr@5": 0.789, "ndcg@5": 0.712}
        >>> baseline_scores = {"mrr@5": [0.8, 0.7, 0.9]}
        >>> current_scores = {"mrr@5": [0.85, 0.78, 0.92]}
        >>> deltas = compute_delta_metrics(baseline, current, baseline_scores, current_scores)
    """
    delta_metrics = {}

    for metric_name, current_value in current_metrics.items():
        baseline_value = baseline_metrics.get(metric_name)

        # Calculate delta
        if baseline_value is None:
            delta_metrics[metric_name] = DeltaMetric(
                metric_name=metric_name,
                baseline=None,
                current=current_value,
                delta=0.0,
                delta_pct=0.0,
                significance=None,
                status=DeltaStatus.NO_BASELINE
            )
            continue

        delta = current_value - baseline_value
        delta_pct = (delta / baseline_value * 100) if baseline_value != 0 else 0.0

        # Calculate significance if per-sample scores available
        significance = None
        is_significant = False

        if baseline_scores and current_scores:
            if metric_name in baseline_scores and metric_name in current_scores:
                baseline_vals = baseline_scores[metric_name]
                current_vals = current_scores[metric_name]
                significance = calculate_significance(baseline_vals, current_vals)
                is_significant = significance.is_significant

        # Determine status
        status = determine_status(delta, is_significant, delta_threshold)

        delta_metrics[metric_name] = DeltaMetric(
            metric_name=metric_name,
            baseline=baseline_value,
            current=current_value,
            delta=delta,
            delta_pct=delta_pct,
            significance=significance,
            status=status
        )

    return delta_metrics


class DeltaReporter:
    """
    Delta metrics reporter for baseline comparison.

    Handles the computation and reporting of delta metrics between
    current evaluation results and a baseline reference.

    Attributes:
        baseline_metrics: Dictionary of baseline metric values
        baseline_scores: Optional per-sample baseline scores
        delta_threshold: Minimum delta magnitude for meaningful change

    Example:
        >>> reporter = DeltaReporter(
        ...     baseline_metrics={"mrr@5": 0.723, "rouge_l": 0.452},
        ...     baseline_scores={"mrr@5": [0.8, 0.7, 0.9]}
        ... )
        >>> current = {"mrr@5": 0.789, "rouge_l": 0.438}
        >>> current_scores = {"mrr@5": [0.85, 0.78, 0.92]}
        >>> deltas = reporter.compute_deltas(current, current_scores)
    """

    def __init__(
        self,
        baseline_metrics: Dict[str, float],
        baseline_scores: Optional[Dict[str, List[float]]] = None,
        delta_threshold: float = 0.01
    ):
        """
        Initialize DeltaReporter.

        Args:
            baseline_metrics: Dictionary of baseline metric values
            baseline_scores: Optional per-sample baseline scores for significance testing
            delta_threshold: Minimum delta magnitude for meaningful change (default: 1%)
        """
        self.baseline_metrics = baseline_metrics
        self.baseline_scores = baseline_scores or {}
        self.delta_threshold = delta_threshold

    def compute_deltas(
        self,
        current_metrics: Dict[str, float],
        current_scores: Optional[Dict[str, List[float]]] = None
    ) -> Dict[str, DeltaMetric]:
        """
        Compute delta metrics against baseline.

        Args:
            current_metrics: Dictionary of current metric values
            current_scores: Optional per-sample current scores for significance testing

        Returns:
            Dictionary mapping metric names to DeltaMetric objects
        """
        return compute_delta_metrics(
            baseline_metrics=self.baseline_metrics,
            current_metrics=current_metrics,
            baseline_scores=self.baseline_scores if self.baseline_scores else None,
            current_scores=current_scores,
            delta_threshold=self.delta_threshold
        )

    def get_summary(self, delta_metrics: Dict[str, DeltaMetric]) -> Dict[str, Any]:
        """
        Generate a summary of delta metrics.

        Args:
            delta_metrics: Dictionary of DeltaMetric objects

        Returns:
            Summary dictionary with counts by status and overall assessment
        """
        status_counts = {status.value: 0 for status in DeltaStatus}
        overall_assessment = "NO_BASELINE"

        for metric in delta_metrics.values():
            status_counts[metric.status.value] += 1

        # Determine overall assessment
        if status_counts[DeltaStatus.SIGNIFICANT_DEGRADED.value] > 0:
            overall_assessment = "DEGRADED"
        elif status_counts[DeltaStatus.SIGNIFICANT_IMPROVED.value] > 0:
            if status_counts[DeltaStatus.SIGNIFICANT_DEGRADED.value] == 0:
                overall_assessment = "IMPROVED"
            else:
                overall_assessment = "MIXED"
        elif status_counts[DeltaStatus.NOT_SIGNIFICANT.value] == len(delta_metrics):
            overall_assessment = "STABLE"
        elif status_counts[DeltaStatus.NO_BASELINE.value] == len(delta_metrics):
            overall_assessment = "NO_BASELINE"

        return {
            "status_counts": status_counts,
            "overall_assessment": overall_assessment,
            "total_metrics": len(delta_metrics)
        }

    def to_dict(self, delta_metrics: Dict[str, DeltaMetric]) -> Dict[str, Any]:
        """
        Convert delta metrics to dictionary format for JSON serialization.

        Args:
            delta_metrics: Dictionary of DeltaMetric objects

        Returns:
            Dictionary suitable for JSON serialization
        """
        return {
            metric_name: metric.to_dict()
            for metric_name, metric in delta_metrics.items()
        }
