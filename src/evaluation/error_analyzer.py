"""
Error Analyzer for RAG Evaluation

Analyzes errors from retrieval, generation, and intent evaluation
to provide actionable insights for model improvement.

Reference: DESIGN.md Chapter 17 - Error Analysis
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set, Tuple
from collections import Counter, defaultdict
from enum import Enum


class ErrorType(Enum):
    """Categories of errors in RAG systems."""
    # Retrieval errors
    RETRIEVAL_COMPLETE_MISS = "retrieval_complete_miss"  # No relevant docs retrieved
    RETRIEVAL_PARTIAL_MISS = "retrieval_partial_miss"  # Some relevant docs missed
    RETRIEVAL_LOW_RANK = "retrieval_low_rank"  # Relevant docs ranked low

    # Generation errors
    GENERATION_HALLUCINATION = "generation_hallucination"  # Made up information
    GENERATION_INCOMPLETE = "generation_incomplete"  # Incomplete answer
    GENERATION_IRRELEVANT = "generation_irrelevant"  # Off-topic response
    GENERATION_LOW_QUALITY = "generation_low_quality"  # Poor quality overall

    # Intent errors
    INTENT_MISCLASSIFICATION = "intent_misclassification"  # Wrong intent
    INTENT_LOW_CONFIDENCE = "intent_low_confidence"  # Low prediction confidence

    # System errors
    SYSTEM_TIMEOUT = "system_timeout"
    SYSTEM_ERROR = "system_error"


@dataclass
class ErrorAnalysis:
    """Result of error analysis."""
    # Error counts by type
    error_counts: Dict[str, int] = field(default_factory=dict)

    # Error rates by type
    error_rates: Dict[str, float] = field(default_factory=dict)

    # Top errors (most frequent)
    top_errors: List[Dict[str, Any]] = field(default_factory=list)

    # Worst cases (individual samples with errors)
    worst_cases: List[Dict[str, Any]] = field(default_factory=list)

    # Error patterns by intent
    errors_by_intent: Dict[str, Dict[str, int]] = field(default_factory=dict)

    # Recommendations
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "error_counts": self.error_counts,
            "error_rates": self.error_rates,
            "top_errors": self.top_errors,
            "worst_cases_count": len(self.worst_cases),
            "errors_by_intent": self.errors_by_intent,
            "recommendations": self.recommendations,
        }


@dataclass
class ErrorSample:
    """A single error sample for analysis."""
    sample_id: str
    error_type: ErrorType
    query: str
    true_label: Optional[str] = None
    predicted_label: Optional[str] = None
    context: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class ErrorAnalyzer:
    """
    Error analyzer for RAG evaluation results.

    Analyzes errors from retrieval, generation, and intent evaluation
    to provide actionable insights.

    Example:
        >>> analyzer = ErrorAnalyzer()
        >>> analysis = analyzer.analyze_retrieval(retrieval_result)
        >>> print(analysis.recommendations)
    """

    def __init__(
        self,
        error_thresholds: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize error analyzer.

        Args:
            error_thresholds: Custom thresholds for error detection
                Default: {
                    "low_rank_threshold": 5,  # Relevant doc at position > 5
                    "low_f1_threshold": 0.3,   # F1 score below 0.3
                    "low_rouge_threshold": 0.2,  # ROUGE-L below 0.2
                }
        """
        self.error_thresholds = error_thresholds or {
            "low_rank_threshold": 5,
            "low_f1_threshold": 0.3,
            "low_rouge_threshold": 0.2,
            "hallucination_threshold": 0.7,  # Faithfulness below 0.7
        }

    def analyze_retrieval(
        self,
        retrieval_result,
        total_queries: int,
    ) -> ErrorAnalysis:
        """
        Analyze retrieval errors.

        Args:
            retrieval_result: RetrievalResult from RetrievalEval
            total_queries: Total number of queries evaluated

        Returns:
            ErrorAnalysis with retrieval-specific insights
        """
        error_samples = []
        errors_by_intent = defaultdict(lambda: defaultdict(int))

        # Analyze per-query results
        for query_result in retrieval_result.per_query_results:
            query_id = query_result["query_id"]
            query_text = query_result.get("query_text", "")
            intent = query_result.get("intent", "UNKNOWN")

            # Check for complete miss (no relevant docs in top-k)
            if not query_result.get("hit", False):
                error_samples.append(ErrorSample(
                    sample_id=query_id,
                    error_type=ErrorType.RETRIEVAL_COMPLETE_MISS,
                    query=query_text,
                    context=f"First relevant rank: {query_result.get('first_relevant_rank', 0)}",
                ))
                errors_by_intent[intent][ErrorType.RETRIEVAL_COMPLETE_MISS.value] += 1

            # Check for low rank (first relevant > threshold)
            first_rank = query_result.get("first_relevant_rank", 0)
            if first_rank > self.error_thresholds["low_rank_threshold"]:
                error_samples.append(ErrorSample(
                    sample_id=query_id,
                    error_type=ErrorType.RETRIEVAL_LOW_RANK,
                    query=query_text,
                    context=f"First relevant at position: {first_rank}",
                ))
                errors_by_intent[intent][ErrorType.RETRIEVAL_LOW_RANK.value] += 1

        # Compile error counts
        error_counts = Counter([e.error_type.value for e in error_samples])
        error_rates = {
            k: v / total_queries for k, v in error_counts.items()
        }

        # Get top errors
        top_errors = [
            {"type": k, "count": v, "rate": error_rates[k]}
            for k, v in error_counts.most_common(5)
        ]

        # Generate recommendations
        recommendations = self._generate_retrieval_recommendations(
            error_counts, total_queries
        )

        return ErrorAnalysis(
            error_counts=dict(error_counts),
            error_rates=error_rates,
            top_errors=top_errors,
            worst_cases=[{
                "sample_id": e.sample_id,
                "error_type": e.error_type.value,
                "query": e.query,
                "context": e.context,
            } for e in error_samples[:10]],
            errors_by_intent=dict(errors_by_intent),
            recommendations=recommendations,
        )

    def analyze_generation(
        self,
        generation_result,
        total_queries: int,
    ) -> ErrorAnalysis:
        """
        Analyze generation errors.

        Args:
            generation_result: GenerationResult from GenerationEval
            total_queries: Total number of queries evaluated

        Returns:
            ErrorAnalysis with generation-specific insights
        """
        error_samples = []
        errors_by_intent = defaultdict(lambda: defaultdict(int))

        # Analyze per-response results
        for response_result in generation_result.per_response_results:
            query_id = response_result["query_id"]
            query_text = response_result.get("query_text", "")
            intent = response_result.get("intent", "UNKNOWN")

            # Check for low ROUGE (poor quality)
            # Note: This would require per-sample ROUGE scores
            # For now, use overall metrics

            # Check for hallucination (if RAGAS metrics available)
            if generation_result.faithfulness is not None:
                if generation_result.faithfulness < self.error_thresholds["hallucination_threshold"]:
                    error_samples.append(ErrorSample(
                        sample_id=query_id,
                        error_type=ErrorType.GENERATION_HALLUCINATION,
                        query=query_text,
                        context=f"Faithfulness: {generation_result.faithfulness:.2f}",
                    ))
                    errors_by_intent[intent][ErrorType.GENERATION_HALLUCINATION.value] += 1

        # Check overall metrics for issues
        if generation_result.rouge_l < self.error_thresholds["low_rouge_threshold"]:
            error_samples.append(ErrorSample(
                sample_id="overall",
                error_type=ErrorType.GENERATION_LOW_QUALITY,
                query="Overall",
                context=f"ROUGE-L: {generation_result.rouge_l:.2f}",
            ))

        # Compile error counts
        error_counts = Counter([e.error_type.value for e in error_samples])
        error_rates = {
            k: v / total_queries for k, v in error_counts.items()
        } if total_queries > 0 else {}

        # Get top errors
        top_errors = [
            {"type": k, "count": v, "rate": error_rates.get(k, 0)}
            for k, v in error_counts.most_common(5)
        ]

        # Generate recommendations
        recommendations = self._generate_generation_recommendations(
            generation_result, error_counts
        )

        return ErrorAnalysis(
            error_counts=dict(error_counts),
            error_rates=error_rates,
            top_errors=top_errors,
            worst_cases=[{
                "sample_id": e.sample_id,
                "error_type": e.error_type.value,
                "query": e.query,
                "context": e.context,
            } for e in error_samples[:10]],
            errors_by_intent=dict(errors_by_intent),
            recommendations=recommendations,
        )

    def analyze_intent(
        self,
        intent_result,
    ) -> ErrorAnalysis:
        """
        Analyze intent classification errors.

        Args:
            intent_result: IntentResult from IntentEval

        Returns:
            ErrorAnalysis with intent-specific insights
        """
        error_samples = []
        errors_by_intent = defaultdict(lambda: defaultdict(int))

        # Analyze misclassified samples
        for sample in intent_result.misclassified_samples:
            true_intent = sample["true_intent"]
            predicted_intent = sample["predicted_intent"]

            error_samples.append(ErrorSample(
                sample_id=sample["query_id"],
                error_type=ErrorType.INTENT_MISCLASSIFICATION,
                query=sample["query_text"],
                true_label=true_intent,
                predicted_label=predicted_intent,
                metadata={
                    "true_intent": true_intent,
                    "predicted_intent": predicted_intent,
                },
            ))

            errors_by_intent[true_intent][ErrorType.INTENT_MISCLASSIFICATION.value] += 1

        # Compile error counts
        error_counts = Counter([e.error_type.value for e in error_samples])

        total_samples = len(intent_result.per_sample_results)
        error_rates = {
            k: v / total_samples for k, v in error_counts.items()
        } if total_samples > 0 else {}

        # Get top errors
        top_errors = [
            {"type": k, "count": v, "rate": error_rates.get(k, 0)}
            for k, v in error_counts.most_common(5)
        ]

        # Generate recommendations
        recommendations = self._generate_intent_recommendations(
            intent_result, error_counts
        )

        return ErrorAnalysis(
            error_counts=dict(error_counts),
            error_rates=error_rates,
            top_errors=top_errors,
            worst_cases=[{
                "sample_id": e.sample_id,
                "error_type": e.error_type.value,
                "query": e.query,
                "true_intent": e.true_label,
                "predicted_intent": e.predicted_label,
            } for e in error_samples[:10]],
            errors_by_intent=dict(errors_by_intent),
            recommendations=recommendations,
        )

    def _generate_retrieval_recommendations(
        self,
        error_counts: Counter,
        total_queries: int,
    ) -> List[str]:
        """Generate retrieval-specific recommendations."""
        recommendations = []

        complete_miss_rate = (
            error_counts.get(ErrorType.RETRIEVAL_COMPLETE_MISS.value, 0) / total_queries
            if total_queries > 0 else 0
        )

        if complete_miss_rate > 0.1:
            recommendations.append(
                f"⚠️ High complete miss rate ({complete_miss_rate:.1%}): "
                "Consider improving embedding model or expanding knowledge base"
            )

        low_rank_count = error_counts.get(ErrorType.RETRIEVAL_LOW_RANK.value, 0)
        if low_rank_count > 0:
            recommendations.append(
                f"⚠️ {low_rank_count} queries have relevant docs ranked low: "
                "Consider improving reranking strategy"
            )

        if not recommendations:
            recommendations.append("✅ Retrieval quality is good")

        return recommendations

    def _generate_generation_recommendations(
        self,
        generation_result,
        error_counts: Counter,
    ) -> List[str]:
        """Generate generation-specific recommendations."""
        recommendations = []

        # Check ROUGE-L
        if generation_result.rouge_l < self.error_thresholds["low_rouge_threshold"]:
            recommendations.append(
                f"⚠️ Low ROUGE-L ({generation_result.rouge_l:.2f}): "
                "Consider fine-tuning the generator with more domain data"
            )

        # Check hallucination
        if generation_result.faithfulness is not None:
            if generation_result.faithfulness < self.error_thresholds["hallucination_threshold"]:
                recommendations.append(
                    f"⚠️ High hallucination rate (Faithfulness: {generation_result.faithfulness:.2f}): "
                    "Consider adding stricter context constraints"
                )

        # Check BERTScore
        if generation_result.bertscore < 0.8:
            recommendations.append(
                f"⚠️ Low BERTScore ({generation_result.bertscore:.2f}): "
                "Generated responses may lack semantic similarity to references"
            )

        if not recommendations:
            recommendations.append("✅ Generation quality is good")

        return recommendations

    def _generate_intent_recommendations(
        self,
        intent_result,
        error_counts: Counter,
    ) -> List[str]:
        """Generate intent-specific recommendations."""
        recommendations = []

        # Check overall accuracy
        if intent_result.accuracy < 0.85:
            recommendations.append(
                f"⚠️ Low intent accuracy ({intent_result.accuracy:.2f}): "
                "Consider adding more training data for misclassified intents"
            )

        # Check per-class F1
        low_f1_intents = [
            intent for intent, f1 in intent_result.per_class_f1.items()
            if f1 < self.error_thresholds["low_f1_threshold"]
        ]

        if low_f1_intents:
            recommendations.append(
                f"⚠️ Low F1 for intents: {', '.join(low_f1_intents)}: "
                "Consider rebalancing training data or improving feature engineering"
            )

        # Check for confusion patterns
        if intent_result.confusion_matrix:
            # Find most confused pairs (simplified)
            recommendations.append(
                "ℹ️ Review confusion matrix for systematic misclassification patterns"
            )

        if not recommendations:
            recommendations.append("✅ Intent classification quality is good")

        return recommendations
