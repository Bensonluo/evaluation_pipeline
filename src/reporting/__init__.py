"""
Reporting Module for RAG Evaluation Pipeline

This module provides functionality for generating evaluation reports in multiple formats,
including delta metrics calculation for baseline comparisons.

Components:
    - delta_reporter: Calculate delta metrics and compare against baselines
    - html_generator: Generate HTML reports
    - json_reporter: Generate JSON reports

Reference: DESIGN.md Chapters 14, 19
"""

from .delta_reporter import (
    DeltaReporter,
    compute_delta_metrics,
    calculate_significance,
    determine_status,
    DeltaStatus,
    DeltaMetric,
    SignificanceResult
)

from .json_reporter import (
    JSONReporter,
    generate_json_report
)

from .html_generator import (
    HTMLGenerator,
    generate_html_report
)

__all__ = [
    'DeltaReporter',
    'compute_delta_metrics',
    'calculate_significance',
    'determine_status',
    'DeltaStatus',
    'DeltaMetric',
    'SignificanceResult',
    'JSONReporter',
    'generate_json_report',
    'HTMLGenerator',
    'generate_html_report',
]
