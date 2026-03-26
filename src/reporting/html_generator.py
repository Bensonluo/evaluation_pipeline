"""
HTML Report Generator for RAG Evaluation

This module generates visually appealing HTML reports for evaluation results,
including delta metrics, significance tests, and interactive visualizations.

Reference: DESIGN.md Chapter 9 - Report Examples, Chapter 19 - Complete Report Template
"""

from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
from string import Template
import logging

from .delta_reporter import DeltaMetric, DeltaStatus

logger = logging.getLogger(__name__)


# HTML template with embedded CSS for self-contained reports
HTML_TEMPLATE = Template("""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>$title - RAG Evaluation Report</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            background: #f5f7fa;
            padding: 20px;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            overflow: hidden;
        }

        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }

        .header h1 {
            font-size: 2rem;
            margin-bottom: 10px;
        }

        .header .subtitle {
            opacity: 0.9;
            font-size: 0.95rem;
        }

        .metadata {
            display: flex;
            justify-content: center;
            gap: 30px;
            margin-top: 20px;
            flex-wrap: wrap;
        }

        .metadata-item {
            background: rgba(255,255,255,0.1);
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 0.9rem;
        }

        .scorecard {
            padding: 30px;
            text-align: center;
            border-bottom: 1px solid #e5e7eb;
        }

        .overall-status {
            display: inline-block;
            padding: 12px 30px;
            border-radius: 30px;
            font-weight: bold;
            font-size: 1.2rem;
            margin-bottom: 20px;
        }

        .status-PASS, .status-IMPROVED, .status-STABLE {
            background: #d1fae5;
            color: #065f46;
        }

        .status-FAIL, .status-DEGRADED {
            background: #fee2e2;
            color: #991b1b;
        }

        .status-MIXED {
            background: #fef3c7;
            color: #92400e;
        }

        .summary-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }

        .summary-item {
            background: #f9fafb;
            padding: 15px;
            border-radius: 8px;
        }

        .summary-label {
            font-size: 0.85rem;
            color: #6b7280;
            margin-bottom: 5px;
        }

        .summary-value {
            font-weight: 600;
            font-size: 1.1rem;
        }

        .section {
            padding: 30px;
            border-bottom: 1px solid #e5e7eb;
        }

        .section:last-child {
            border-bottom: none;
        }

        .section-title {
            font-size: 1.3rem;
            font-weight: 600;
            margin-bottom: 20px;
            color: #1f2937;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .section-icon {
            font-size: 1.5rem;
        }

        .metrics-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }

        .metrics-table th,
        .metrics-table td {
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #e5e7eb;
        }

        .metrics-table th {
            background: #f9fafb;
            font-weight: 600;
            color: #374151;
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .metrics-table tr:hover {
            background: #f9fafb;
        }

        .metric-value {
            font-weight: 600;
            font-family: 'Courier New', monospace;
        }

        .delta-positive {
            color: #059669;
        }

        .delta-negative {
            color: #dc2626;
        }

        .delta-neutral {
            color: #6b7280;
        }

        .status-badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.8rem;
            font-weight: 600;
        }

        .status-badge.IMPROVED {
            background: #d1fae5;
            color: #065f46;
        }

        .status-badge.DEGRADED {
            background: #fee2e2;
            color: #991b1b;
        }

        .status-badge.PASS {
            background: #d1fae5;
            color: #065f46;
        }

        .status-badge.FAIL {
            background: #fee2e2;
            color: #991b1b;
        }

        .status-badge.STABLE {
            background: #e5e7eb;
            color: #374151;
        }

        .recommendations {
            background: #fef3c7;
            border-left: 4px solid #f59e0b;
            padding: 20px;
            border-radius: 8px;
            margin-top: 20px;
        }

        .recommendations h4 {
            color: #92400e;
            margin-bottom: 15px;
        }

        .recommendations ul {
            list-style: none;
            padding: 0;
        }

        .recommendations li {
            padding: 8px 0;
            padding-left: 25px;
            position: relative;
        }

        .recommendations li:before {
            content: "→";
            position: absolute;
            left: 0;
            color: #f59e0b;
            font-weight: bold;
        }

        .stratified-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }

        .stratified-card {
            background: #f9fafb;
            padding: 15px;
            border-radius: 8px;
            border-left: 3px solid #667eea;
        }

        .stratified-card .label {
            font-size: 0.85rem;
            color: #6b7280;
            margin-bottom: 5px;
        }

        .stratified-card .value {
            font-weight: 600;
            font-size: 1.1rem;
        }

        .stratified-card .count {
            font-size: 0.8rem;
            color: #9ca3af;
        }

        .error-list {
            margin-top: 15px;
        }

        .error-item {
            background: #fef2f2;
            padding: 12px;
            border-radius: 6px;
            margin-bottom: 10px;
            border-left: 3px solid #ef4444;
        }

        .error-item .type {
            font-weight: 600;
            color: #991b1b;
        }

        .error-item .details {
            font-size: 0.9rem;
            color: #7f1d1d;
            margin-top: 5px;
        }

        .footer {
            text-align: center;
            padding: 20px;
            background: #f9fafb;
            color: #6b7280;
            font-size: 0.85rem;
        }

        @media (max-width: 768px) {
            .metrics-table {
                font-size: 0.85rem;
            }

            .metrics-table th,
            .metrics-table td {
                padding: 8px 10px;
            }

            .summary-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>$title</h1>
            <div class="subtitle">RAG Chatbot Evaluation Report</div>
            <div class="metadata">
                <div class="metadata-item">📅 $eval_date</div>
                <div class="metadata-item">📊 $dataset_info</div>
                <div class="metadata-item">🎯 Baseline: $baseline</div>
            </div>
        </div>

        $content

        <div class="footer">
            Generated by RAG Evaluation Pipeline v2.0 | $generated_at
        </div>
    </div>
</body>
</html>
""")


class HTMLGenerator:
    """
    HTML report generator for evaluation results.

    Generates visually appealing, self-contained HTML reports with
    embedded CSS and responsive design.

    Attributes:
        eval_name: Name of the evaluation run
        dataset_info: Information about the test dataset
        baseline_name: Optional baseline name for comparison

    Example:
        >>> generator = HTMLGenerator(
        ...     eval_name="finetune_v2_compare",
        ...     dataset_info={"name": "customer_service_qa_v2", "total_samples": 500},
        ...     baseline_name="v1.0"
        ... )
        >>> html = generator.generate_report(
        ...     retrieval_metrics={"mrr@5": 0.789},
        ...     delta_metrics={"mrr@5": delta_obj}
        ... )
    """

    def __init__(
        self,
        eval_name: str,
        dataset_info: Optional[Dict[str, Any]] = None,
        baseline_name: Optional[str] = None
    ):
        """
        Initialize HTMLGenerator.

        Args:
            eval_name: Name of the evaluation run
            dataset_info: Optional dataset metadata
            baseline_name: Optional baseline name for comparison
        """
        self.eval_name = eval_name
        self.dataset_info = dataset_info or {}
        self.baseline_name = baseline_name
        self.generated_at = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    def _format_metric_value(self, value: float) -> str:
        """Format metric value for display."""
        return f"{value:.4f}"

    def _format_delta(self, delta: float, delta_pct: float) -> str:
        """Format delta value with color class."""
        if delta > 0:
            return f'<span class="delta-positive">+{delta_pct:.1f}%</span>'
        elif delta < 0:
            return f'<span class="delta-negative">{delta_pct:.1f}%</span>'
        else:
            return f'<span class="delta-neutral">0.0%</span>'

    def _get_status_emoji(self, status: DeltaStatus) -> str:
        """Get emoji for status."""
        emoji_map = {
            DeltaStatus.SIGNIFICANT_IMPROVED: "✅",
            DeltaStatus.MARGINAL_IMPROVED: "📈",
            DeltaStatus.SIGNIFICANT_DEGRADED: "❌",
            DeltaStatus.MARGINAL_DEGRADED: "⚠️",
            DeltaStatus.NOT_SIGNIFICANT: "➡️",
            DeltaStatus.NO_BASELINE: "➖"
        }
        return emoji_map.get(status, "➖")

    def _build_metrics_table(
        self,
        title: str,
        icon: str,
        metrics: Dict[str, float],
        delta_metrics: Optional[Dict[str, DeltaMetric]] = None,
        target_prefix: str = ""
    ) -> str:
        """
        Build HTML table for a metric section.

        Args:
            title: Section title
            icon: Section emoji icon
            metrics: Dictionary of metric values
            delta_metrics: Optional DeltaMetric objects
            target_prefix: Prefix for target lookup

        Returns:
            HTML string
        """
        if not metrics:
            return ""

        rows = []
        for metric_name, value in sorted(metrics.items()):
            delta_html = ""
            status_html = ""

            if delta_metrics and metric_name in delta_metrics:
                delta = delta_metrics[metric_name]
                delta_html = self._format_delta(delta.delta, delta.delta_pct)
                status_html = f'<span class="status-badge {delta.status.value}">{delta.status.value}</span>'

            rows.append(f"""
                <tr>
                    <td><strong>{metric_name}</strong></td>
                    <td class="metric-value">{self._format_metric_value(value)}</td>
                    <td>{delta_html}</td>
                    <td>{status_html}</td>
                </tr>
            """)

        return f"""
            <div class="section">
                <div class="section-title">
                    <span class="section-icon">{icon}</span>
                    {title}
                </div>
                <table class="metrics-table">
                    <thead>
                        <tr>
                            <th>Metric</th>
                            <th>Value</th>
                            <th>Delta</th>
                            <th>Status</th>
                        </tr>
                    </thead>
                    <tbody>
                        {''.join(rows)}
                    </tbody>
                </table>
            </div>
        """

    def _build_scorecard(
        self,
        overall_status: str,
        summary: Dict[str, str]
    ) -> str:
        """Build scorecard section."""
        status_class = f"status-{overall_status}"

        summary_items = []
        for key, value in summary.items():
            label = key.replace("_", " ").title()
            summary_items.append(f"""
                <div class="summary-item">
                    <div class="summary-label">{label}</div>
                    <div class="summary-value">{value}</div>
                </div>
            """)

        return f"""
            <div class="scorecard">
                <div class="overall-status {status_class}">{overall_status}</div>
                <div class="summary-grid">
                    {''.join(summary_items)}
                </div>
            </div>
        """

    def _build_recommendations(self, recommendations: List[str]) -> str:
        """Build recommendations section."""
        if not recommendations:
            return ""

        items = "\n".join(f"<li>{rec}</li>" for rec in recommendations)

        return f"""
            <div class="section">
                <div class="section-title">
                    <span class="section-icon">💡</span>
                    Recommendations
                </div>
                <div class="recommendations">
                    <h4>Key Takeaways</h4>
                    <ul>{items}</ul>
                </div>
            </div>
        """

    def _build_stratified_results(self, stratified: Dict[str, Dict[str, Any]]) -> str:
        """Build stratified results section."""
        if not stratified:
            return ""

        cards = []
        for category, data in sorted(stratified.items()):
            for key, stats in data.items():
                cards.append(f"""
                    <div class="stratified-card">
                        <div class="label">{category} - {key}</div>
                        <div class="value">{stats.get('value', stats.get('accuracy', 0)):.2%}</div>
                        <div class="count">n={stats.get('count', 0)}</div>
                    </div>
                """)

        return f"""
            <div class="section">
                <div class="section-title">
                    <span class="section-icon">📊</span>
                    Stratified Results
                </div>
                <div class="stratified-grid">
                    {''.join(cards)}
                </div>
            </div>
        """

    def _build_error_analysis(self, error_analysis: Dict[str, Any]) -> str:
        """Build error analysis section."""
        if not error_analysis:
            return ""

        top_errors = error_analysis.get("top_errors", [])
        error_items = []

        for error in top_errors[:5]:  # Top 5 errors
            error_items.append(f"""
                <div class="error-item">
                    <div class="type">{error.get('type', 'Unknown')}</div>
                    <div class="details">
                        Count: {error.get('count', 0)} | Rate: {error.get('rate', 0):.2%}
                    </div>
                </div>
            """)

        return f"""
            <div class="section">
                <div class="section-title">
                    <span class="section-icon">🔍</span>
                    Error Analysis
                </div>
                <div class="error-list">
                    {''.join(error_items)}
                </div>
            </div>
        """

    def generate_report(
        self,
        retrieval_metrics: Optional[Dict[str, float]] = None,
        generation_metrics: Optional[Dict[str, float]] = None,
        ragas_metrics: Optional[Dict[str, float]] = None,
        intent_metrics: Optional[Dict[str, float]] = None,
        business_metrics: Optional[Dict[str, float]] = None,
        latency_metrics: Optional[Dict[str, int]] = None,
        delta_metrics: Optional[Dict[str, DeltaMetric]] = None,
        overall_status: str = "STABLE",
        summary: Optional[Dict[str, str]] = None,
        stratified_results: Optional[Dict[str, Any]] = None,
        error_analysis: Optional[Dict[str, Any]] = None,
        recommendations: Optional[List[str]] = None
    ) -> str:
        """
        Generate complete HTML report.

        Args:
            retrieval_metrics: Retrieval quality metrics
            generation_metrics: Generation quality metrics
            ragas_metrics: RAGAS framework metrics
            intent_metrics: Intent classification metrics
            business_metrics: Business-level metrics
            latency_metrics: Latency metrics
            delta_metrics: DeltaMetric objects for baseline comparison
            overall_status: Overall evaluation status
            summary: Summary dictionary for scorecard
            stratified_results: Stratified results by category
            error_analysis: Error analysis results
            recommendations: List of recommendation strings

        Returns:
            Complete HTML report as string

        Example:
            >>> html = generator.generate_report(
            ...     retrieval_metrics={"mrr@5": 0.789},
            ...     generation_metrics={"rouge_l": 0.452},
            ...     delta_metrics={"mrr@5": delta_obj},
            ...     overall_status="IMPROVED"
            ... )
        """
        # Build content sections
        content_parts = []

        # Scorecard
        if summary:
            content_parts.append(self._build_scorecard(overall_status, summary))

        # Metric sections
        if retrieval_metrics:
            content_parts.append(
                self._build_metrics_table(
                    "Retrieval Quality",
                    "🔍",
                    retrieval_metrics,
                    delta_metrics
                )
            )

        if generation_metrics:
            content_parts.append(
                self._build_metrics_table(
                    "Generation Quality",
                    "✍️",
                    generation_metrics,
                    delta_metrics
                )
            )

        if ragas_metrics:
            content_parts.append(
                self._build_metrics_table(
                    "RAGAS Metrics",
                    "🎯",
                    ragas_metrics,
                    delta_metrics
                )
            )

        if intent_metrics:
            content_parts.append(
                self._build_metrics_table(
                    "Intent Classification",
                    "🏷️",
                    intent_metrics,
                    delta_metrics
                )
            )

        if business_metrics:
            content_parts.append(
                self._build_metrics_table(
                    "Business Metrics",
                    "💼",
                    business_metrics,
                    delta_metrics
                )
            )

        if latency_metrics:
            content_parts.append(
                self._build_metrics_table(
                    "Latency Metrics",
                    "⏱️",
                    {k: float(v) for k, v in latency_metrics.items()},
                    None
                )
            )

        # Stratified results
        if stratified_results:
            content_parts.append(self._build_stratified_results(stratified_results))

        # Error analysis
        if error_analysis:
            content_parts.append(self._build_error_analysis(error_analysis))

        # Recommendations
        if recommendations:
            content_parts.append(self._build_recommendations(recommendations))

        # Build dataset info string
        dataset_str = self.dataset_info.get("name", "Unknown")
        if "total_samples" in self.dataset_info:
            dataset_str += f" (n={self.dataset_info['total_samples']})"

        # Fill template
        html = HTML_TEMPLATE.substitute(
            title=self.eval_name,
            eval_date=datetime.utcnow().strftime("%Y-%m-%d"),
            dataset_info=dataset_str,
            baseline=self.baseline_name or "None",
            content="".join(content_parts),
            generated_at=self.generated_at
        )

        return html

    def save_report(
        self,
        html: str,
        output_path: Path
    ) -> None:
        """
        Save HTML report to file.

        Args:
            html: HTML content string
            output_path: Output file path

        Raises:
            IOError: If file cannot be written
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)

        logger.info(f"HTML report saved to {output_path}")


def generate_html_report(
    eval_name: str,
    metrics: Dict[str, Dict[str, float]],
    delta_metrics: Optional[Dict[str, DeltaMetric]] = None,
    output_path: Optional[Path] = None,
    baseline_name: Optional[str] = None,
    dataset_info: Optional[Dict[str, Any]] = None,
    overall_status: str = "STABLE",
    **kwargs
) -> str:
    """
    Convenience function to generate an HTML report.

    Args:
        eval_name: Name of the evaluation
        metrics: Dictionary of metric sections (retrieval, generation, etc.)
        delta_metrics: Optional DeltaMetric objects
        output_path: Optional path to save the report
        baseline_name: Optional baseline name
        dataset_info: Optional dataset metadata
        overall_status: Overall evaluation status
        **kwargs: Additional arguments passed to HTMLGenerator

    Returns:
        HTML string

    Example:
        >>> html = generate_html_report(
        ...     eval_name="my_eval",
        ...     metrics={
        ...         "retrieval": {"mrr@5": 0.789},
        ...         "generation": {"rouge_l": 0.452}
        ...     },
        ...     output_path="reports/eval.html"
        ... )
    """
    generator = HTMLGenerator(
        eval_name=eval_name,
        dataset_info=dataset_info,
        baseline_name=baseline_name
    )

    html = generator.generate_report(
        retrieval_metrics=metrics.get("retrieval"),
        generation_metrics=metrics.get("generation"),
        ragas_metrics=metrics.get("ragas"),
        intent_metrics=metrics.get("intent"),
        business_metrics=metrics.get("business"),
        latency_metrics=metrics.get("latency"),
        delta_metrics=delta_metrics,
        overall_status=overall_status,
        **kwargs
    )

    if output_path:
        generator.save_report(html, output_path)

    return html
