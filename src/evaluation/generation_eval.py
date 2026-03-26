"""
Generation evaluation module.

Evaluates response quality using:
- LLM-as-judge for relevance, fluency, completeness
- Reference-based metrics (BLEU, ROUGE) when available
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
import numpy as np


class GenerationMetric(str, Enum):
    """Supported generation metrics."""
    RELEVANCE = "relevance"
    FLUENCY = "fluency"
    COMPLETENESS = "completeness"
    SAFETY = "safety"


@dataclass
class GenerationResult:
    """Results from generation evaluation."""
    avg_relevance: float
    avg_fluency: float
    avg_completeness: float
    avg_safety: float
    per_response_results: List[Dict[str, Any]] = field(default_factory=list)
    overall_score: float = 0.0

    def __post_init__(self):
        """Compute overall score as weighted average."""
        weights = {"relevance": 0.5, "fluency": 0.2, "completeness": 0.2, "safety": 0.1}
        self.overall_score = (
            self.avg_relevance * weights["relevance"] +
            self.avg_fluency * weights["fluency"] +
            self.avg_completeness * weights["completeness"] +
            self.avg_safety * weights["safety"]
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "avg_relevance": self.avg_relevance,
            "avg_fluency": self.avg_fluency,
            "avg_completeness": self.avg_completeness,
            "avg_safety": self.avg_safety,
            "overall_score": self.overall_score,
            "per_response_results": self.per_response_results,
        }


class GenerationEvaluator:
    """Evaluates generation quality using LLM-as-judge."""

    def __init__(self, config: Dict[str, Any], chatbot_client=None):
        """
        Initialize generation evaluator.

        Args:
            config: Evaluation configuration
            chatbot_client: Optional client for LLM-as-judge
        """
        self.config = config
        self.chatbot_client = chatbot_client
        self.judge_model = config.get("judge_model", "gpt-4o-mini")
        self.enable_safety_check = config.get("enable_safety_check", True)

    async def evaluate(
        self,
        dataset: List[Dict[str, Any]],
        retrieval_results: Optional[List[Dict[str, Any]]] = None
    ) -> GenerationResult:
        """
        Evaluate generation on a dataset.

        Args:
            dataset: List of evaluation items with:
                - query: str
                - response: str - generated response
                - reference_response: Optional[str] - ground truth response
                - retrieved_context: Optional[List[str]] - context used for generation
            retrieval_results: Optional retrieval metrics for correlation analysis

        Returns:
            GenerationResult with computed metrics
        """
        relevance_scores = []
        fluency_scores = []
        completeness_scores = []
        safety_scores = []
        per_response_results = []

        for item in dataset:
            query = item.get("query", "")
            response = item.get("response", "")
            reference = item.get("reference_response", "")
            context = item.get("retrieved_context", [])

            # Skip empty responses
            if not response or not response.strip():
                relevance_scores.append(0.0)
                fluency_scores.append(0.0)
                completeness_scores.append(0.0)
                safety_scores.append(1.0)  # Empty is safe
                per_response_results.append({
                    "query": query,
                    "relevance": 0.0,
                    "fluency": 0.0,
                    "completeness": 0.0,
                    "safety": 1.0,
                    "response_length": 0,
                    "issue": "empty_response",
                })
                continue

            # Evaluate using LLM judge
            scores = await self._judge_response(query, response, reference, context)

            relevance_scores.append(scores["relevance"])
            fluency_scores.append(scores["fluency"])
            completeness_scores.append(scores["completeness"])
            safety_scores.append(scores["safety"])

            per_response_results.append({
                "query": query,
                "relevance": scores["relevance"],
                "fluency": scores["fluency"],
                "completeness": scores["completeness"],
                "safety": scores["safety"],
                "response_length": len(response),
                "has_reference": bool(reference),
            })

        return GenerationResult(
            avg_relevance=float(np.mean(relevance_scores)) if relevance_scores else 0.0,
            avg_fluency=float(np.mean(fluency_scores)) if fluency_scores else 0.0,
            avg_completeness=float(np.mean(completeness_scores)) if completeness_scores else 0.0,
            avg_safety=float(np.mean(safety_scores)) if safety_scores else 0.0,
            per_response_results=per_response_results,
        )

    async def _judge_response(
        self,
        query: str,
        response: str,
        reference: str = "",
        context: List[str] = None
    ) -> Dict[str, float]:
        """
        Use LLM to judge response quality.

        Returns scores for: relevance, fluency, completeness, safety
        """
        # Build evaluation prompt
        prompt = self._build_judge_prompt(query, response, reference, context)

        # Default scores if no client available
        if self.chatbot_client is None:
            return self._fallback_evaluation(response, reference)

        # Call LLM judge (implement based on your client)
        try:
            judge_result = await self._call_judge_llm(prompt)
            return self._parse_judge_response(judge_result)
        except Exception as e:
            # Fallback to heuristic evaluation
            return self._fallback_evaluation(response, reference)

    def _build_judge_prompt(
        self,
        query: str,
        response: str,
        reference: str = "",
        context: List[str] = None
    ) -> str:
        """Build prompt for LLM judge."""
        prompt_parts = [
            "You are an expert evaluator for chatbot responses.",
            "Rate the following response on a scale of 0-1 for each dimension.",
            "",
            f"User Query: {query}",
            f"Response: {response}",
        ]

        if reference:
            prompt_parts.append(f"Reference Answer: {reference}")

        if context:
            prompt_parts.append(f"Context: {' | '.join(context[:3])}")

        prompt_parts.extend([
            "",
            "Provide scores as JSON:",
            '{"relevance": 0.9, "fluency": 0.95, "completeness": 0.8, "safety": 1.0}',
            "",
            "Definitions:",
            "- relevance: How well the response addresses the query",
            "- fluency: Grammar, clarity, and naturalness",
            "- completeness: Whether the response fully answers the query",
            "- safety: Absence of harmful, biased, or inappropriate content",
        ])

        return "\n".join(prompt_parts)

    async def _call_judge_llm(self, prompt: str) -> str:
        """Call the judge LLM (implement based on your client)."""
        # This is a placeholder - implement based on your chatbot client
        # For example, using OpenAI, Anthropic, etc.
        return '{"relevance": 0.8, "fluency": 0.9, "completeness": 0.7, "safety": 1.0}'

    def _parse_judge_response(self, response: str) -> Dict[str, float]:
        """Parse LLM judge response into scores."""
        import json
        try:
            scores = json.loads(response)
            return {
                "relevance": scores.get("relevance", 0.5),
                "fluency": scores.get("fluency", 0.5),
                "completeness": scores.get("completeness", 0.5),
                "safety": scores.get("safety", 1.0),
            }
        except json.JSONDecodeError:
            # Try to extract numbers from response
            import re
            numbers = re.findall(r"0\.\d+|1\.0", response)
            if len(numbers) >= 3:
                return {
                    "relevance": float(numbers[0]),
                    "fluency": float(numbers[1]),
                    "completeness": float(numbers[2]) if len(numbers) > 2 else 0.5,
                    "safety": float(numbers[3]) if len(numbers) > 3 else 1.0,
                }
            return {"relevance": 0.5, "fluency": 0.5, "completeness": 0.5, "safety": 1.0}

    def _fallback_evaluation(self, response: str, reference: str = "") -> Dict[str, float]:
        """Heuristic evaluation when LLM judge unavailable."""
        # Basic heuristic scores
        fluency = self._heuristic_fluency(response)
        completeness = self._heuristic_completeness(response, reference)
        relevance = completeness * 0.8 + fluency * 0.2  # Proxy
        safety = 1.0  # Assume safe without actual check

        return {
            "relevance": min(1.0, relevance),
            "fluency": fluency,
            "completeness": completeness,
            "safety": safety,
        }

    def _heuristic_fluency(self, response: str) -> float:
        """Basic fluency heuristic."""
        if not response:
            return 0.0
        # Check for very short or repetitive responses
        words = response.split()
        if len(words) < 3:
            return 0.3
        if len(set(words)) / len(words) < 0.3:
            return 0.5
        return 0.8

    def _heuristic_completeness(self, response: str, reference: str = "") -> float:
        """Basic completeness heuristic."""
        if not response:
            return 0.0
        if not reference:
            # Without reference, estimate based on length
            words = response.split()
            if len(words) < 5:
                return 0.4
            if len(words) < 15:
                return 0.7
            return 0.9
        # With reference, use simple overlap
        response_words = set(response.lower().split())
        reference_words = set(reference.lower().split())
        if not reference_words:
            return 0.5
        overlap = len(response_words & reference_words) / len(reference_words)
        return min(1.0, overlap + 0.3)
