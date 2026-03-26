"""
RAG Chatbot Evaluation Pipeline - Evaluation Module

This module provides comprehensive evaluation capabilities for RAG chatbots
across multiple dimensions: retrieval, generation, intent classification, and
error analysis.

Core evaluators:
    - RetrievalEvaluator: Evaluates retrieval quality (MRR, NDCG, HitRate, Precision)
    - GenerationEvaluator: Evaluates generation quality (Relevance, Fluency, Completeness)
    - IntentEvaluator: Evaluates intent classification accuracy
    - ErrorAnalyzer: Analyzes errors and provides insights

Reference: DESIGN.md - Complete Pipeline Architecture
"""

from .retrieval_eval import RetrievalEvaluator, RetrievalResult
from .generation_eval import GenerationEvaluator, GenerationResult, GenerationMetric
from .intent_eval import IntentEvaluator, IntentResult
from .error_analyzer import ErrorAnalyzer, ErrorAnalysis

__all__ = [
    # Retrieval evaluation
    "RetrievalEvaluator",
    "RetrievalResult",
    # Generation evaluation
    "GenerationEvaluator",
    "GenerationResult",
    "GenerationMetric",
    # Intent evaluation
    "IntentEvaluator",
    "IntentResult",
    # Error analysis
    "ErrorAnalyzer",
    "ErrorAnalysis",
]
