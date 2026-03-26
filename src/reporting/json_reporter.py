"""
JSON Report Generator for RAG Evaluation

This module generates structured JSON reports for evaluation results,
including delta metrics, significance tests, and comprehensive metadata.

Reference: DESIGN.md Chapter 19 - Complete Evaluation Report Template
"""

import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
import logging

from .delta_reporter import DeltaMetric, DeltaStatus

logger = logging.getLogger(__name__)


class JSONReporter:
    """
    JSON report generator for evaluation results.

    Generates comprehensive JSON reports following the schema defined in
    DESIGN.md Chapter 19.

    Attributes:
        eval_name: Name of the evaluation run
        dataset_info: Information about the test dataset

    Example:
        >>> reporter = JSONReporter(
        ...     eval_name="finetune_v2_compare",
        ...     dataset_info={"name": "customer_service_qa_v2", "total_samples": 500}
        ... )
        >>> report = reporter.generate_report(
        ...     retrieval_metrics={"mrr@5": 0.789},
        ...     generation_metrics={"rouge_l": 0.452},
        ...     delta_metrics={"mrr@5": delta_metric_obj}
        ... )
    """

    def __init__(
        self,
        eval_name: str,
        dataset_info: Optional[Dict[str, Any]] = None,
        eval_version: str = "2.0"
    ):
        """
        Initialize JSONReporter.

        Args:
            eval_name: Name of the evaluation run
            dataset_info: Optional dataset metadata
            eval_version: Report schema version (default: "2.0")
        """
        self.eval_name = eval_name
        self.dataset_info = dataset_info or {}
        self.eval_version = eval_version
        self.generated_at = datetime.utcnow().isoformat() + "Z"

    def _build_report_metadata(self) -> Dict[str, Any]:
        """Build report metadata section."""
        return {
            "eval_name": self.eval_name,
            "eval_version": self.eval_version,
            "generated_at": self.generated_at,
            "dataset": self.dataset_info
        }

    def _build_scorecard(
        self,
        delta_metrics: Optional[Dict[str, DeltaMetric]]
    ) -> Dict[str, Any]:
        """
        Build scorecard section with overall status.

        Args:
            delta_metrics: Dictionary of DeltaMetric objects

        Returns:
            Scorecard dictionary
        """
        if not delta_metrics:
            return {
                "overall_status": "NO_BASELINE",
                "summary": {
                    "retrieval_quality": "UNKNOWN",
                    "generation_quality": "UNKNOWN",
                    "latency": "UNKNOWN",
                    "business_impact": "UNKNOWN"
                }
            }

        # Count statuses
        improved = sum(1 for m in delta_metrics.values()
                      if m.status == DeltaStatus.SIGNIFICANT_IMPROVED)
        degraded = sum(1 for m in delta_metrics.values()
                      if m.status == DeltaStatus.SIGNIFICANT_DEGRADED)

        # Determine overall status
        if degraded > 0:
            overall_status = "DEGRADED"
        elif improved > 0:
            overall_status = "IMPROVED"
        else:
            overall_status = "STABLE"

        # Build dimension summaries
        summary = {
            "retrieval_quality": "GOOD",
            "generation_quality": "GOOD",
            "latency": "ACCEPTABLE",
            "business_impact": "NEUTRAL"
        }

        # Update summaries based on retrieval metrics
        retrieval_metrics = [k for k in delta_metrics.keys()
                            if any(x in k for x in ['mrr', 'ndcg', 'hitrate', 'precision'])]
        if retrieval_metrics:
            retrieval_status = [delta_metrics[k].status for k in retrieval_metrics]
            if DeltaStatus.SIGNIFICANT_IMPROVED in retrieval_status:
                summary["retrieval_quality"] = "EXCELLENT"
            elif DeltaStatus.SIGNIFICANT_DEGRADED in retrieval_status:
                summary["retrieval_quality"] = "POOR"

        return {
            "overall_status": overall_status,
            "summary": summary
        }

    def _format_metric_section(
        self,
        metrics: Dict[str, float],
        delta_metrics: Optional[Dict[str, DeltaMetric]],
        targets: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Format a metric section with values and deltas.

        Args:
            metrics: Dictionary of metric values
            delta_metrics: Optional DeltaMetric objects
            targets: Optional target thresholds

        Returns:
            Formatted metric section dictionary
        """
        section = {}

        for metric_name, value in metrics.items():
            metric_entry = {
                "value": round(value, 4)
            }

            # Add target if available
            if targets and metric_name in targets:
                metric_entry["target"] = targets[metric_name]
                metric_entry["status"] = "PASS" if value >= targets[metric_name] else "FAIL"

            # Add delta information
            if delta_metrics and metric_name in delta_metrics:
                delta = delta_metrics[metric_name]
                metric_entry["delta"] = round(delta.delta, 4)
                metric_entry["delta_pct"] = delta.delta_pct
                metric_entry["status"] = delta.status.value

                if delta.significance:
                    metric_entry["significance"] = delta.significance.to_dict()

            section[metric_name] = metric_entry

        return section

    def generate_report(
        self,
        retrieval_metrics: Optional[Dict[str, float]] = None,
        generation_metrics: Optional[Dict[str, float]] = None,
        ragas_metrics: Optional[Dict[str, float]] = None,
        intent_metrics: Optional[Dict[str, float]] = None,
        business_metrics: Optional[Dict[str, float]] = None,
        latency_metrics: Optional[Dict[str, int]] = None,
        delta_metrics: Optional[Dict[str, DeltaMetric]] = None,
        error_analysis: Optional[Dict[str, Any]] = None,
        stratified_results: Optional[Dict[str, Any]] = None,
        significance_tests: Optional[Dict[str, Any]] = None,
        recommendations: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive JSON evaluation report.

        Args:
            retrieval_metrics: Retrieval quality metrics
            generation_metrics: Generation quality metrics
            ragas_metrics: RAGAS framework metrics
            intent_metrics: Intent classification metrics
            business_metrics: Business-level metrics
            latency_metrics: Latency metrics (P50, P95, P99)
            delta_metrics: DeltaMetric objects for baseline comparison
            error_analysis: Error analysis results
            stratified_results: Stratified results by intent/complexity
            significance_tests: Statistical significance test results
            recommendations: List of recommendation strings
            metadata: Additional metadata

        Returns:
            Complete report dictionary matching DESIGN.md Chapter 19 schema

        Example:
            >>> report = reporter.generate_report(
            ...     retrieval_metrics={"mrr@5": 0.789, "ndcg@5": 0.712},
            ...     generation_metrics={"rouge_l": 0.452},
            ...     delta_metrics={"mrr@5": delta_obj}
            ... )
        """
        report = {
            "report_metadata": self._build_report_metadata(),
            "scorecard": self._build_scorecard(delta_metrics),
            "metrics": {}
        }

        # Add metric sections
        if retrieval_metrics:
            report["metrics"]["retrieval"] = self._format_metric_section(
                retrieval_metrics, delta_metrics
            )

        if generation_metrics:
            report["metrics"]["generation"] = self._format_metric_section(
                generation_metrics, delta_metrics
            )

        if ragas_metrics:
            report["metrics"]["ragas"] = self._format_metric_section(
                ragas_metrics, delta_metrics
            )

        if intent_metrics:
            report["metrics"]["intent"] = self._format_metric_section(
                intent_metrics, delta_metrics
            )

        if business_metrics:
            report["metrics"]["business"] = self._format_metric_section(
                business_metrics, delta_metrics
            )

        if latency_metrics:
            report["metrics"]["latency"] = self._format_metric_section(
                {k: float(v) for k, v in latency_metrics.items()},
                None  # No delta for latency typically
            )

        # Add optional sections
        if stratified_results:
            report["stratified_results"] = stratified_results

        if error_analysis:
            report["error_analysis"] = error_analysis

        if significance_tests:
            report["significance_tests"] = significance_tests

        if recommendations:
            report["recommendations"] = recommendations

        if metadata:
            report["metadata"] = metadata

        return report

    def save_report(
        self,
        report: Dict[str, Any],
        output_path: Path,
        pretty: bool = True
    ) -> None:
        """
        Save report to JSON file.

        Args:
            report: Report dictionary
            output_path: Output file path
            pretty: Whether to pretty-print JSON (default: True)

        Raises:
            IOError: If file cannot be written
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            if pretty:
                json.dump(report, f, indent=2, ensure_ascii=False)
            else:
                json.dump(report, f, ensure_ascii=False)

        logger.info(f"Report saved to {output_path}")

    def load_report(self, input_path: Path) -> Dict[str, Any]:
        """
        Load report from JSON file.

        Args:
            input_path: Input file path

        Returns:
            Report dictionary

        Raises:
            FileNotFoundError: If file doesn't exist
            json.JSONDecodeError: If file is not valid JSON
        """
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Report file not found: {input_path}")

        with open(input_path, 'r', encoding='utf-8') as f:
            report = json.load(f)

        logger.info(f"Report loaded from {input_path}")
        return report


def generate_json_report(
    eval_name: str,
    metrics: Dict[str, Dict[str, float]],
    delta_metrics: Optional[Dict[str, DeltaMetric]] = None,
    output_path: Optional[Path] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function to generate a JSON report.

    Args:
        eval_name: Name of the evaluation
        metrics: Dictionary of metric sections (retrieval, generation, etc.)
        delta_metrics: Optional DeltaMetric objects
        output_path: Optional path to save the report
        **kwargs: Additional arguments passed to JSONReporter

    Returns:
        Report dictionary

    Example:
        >>> report = generate_json_report(
        ...     eval_name="my_eval",
        ...     metrics={
        ...         "retrieval": {"mrr@5": 0.789},
        ...         "generation": {"rouge_l": 0.452}
        ...     },
        ...     output_path="reports/eval.json"
        ... )
    """
    reporter = JSONReporter(eval_name=eval_name, **kwargs)

    report = reporter.generate_report(
        retrieval_metrics=metrics.get("retrieval"),
        generation_metrics=metrics.get("generation"),
        ragas_metrics=metrics.get("ragas"),
        intent_metrics=metrics.get("intent"),
        business_metrics=metrics.get("business"),
        latency_metrics=metrics.get("latency"),
        delta_metrics=delta_metrics
    )

    if output_path:
        reporter.save_report(report, output_path)

    return report
