"""
Intent classification evaluation module.

Computes intent classification metrics:
- Accuracy
- Precision, Recall, F1 (macro, micro, weighted)
- Confusion Matrix
- Per-class metrics
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple
from collections import Counter
import numpy as np
from enum import Enum


@dataclass
class IntentResult:
    """Results from intent classification evaluation."""
    accuracy: float
    f1_macro: float
    f1_micro: float
    f1_weighted: float
    confusion_matrix: Dict[str, Dict[str, int]]
    per_class_metrics: Dict[str, Dict[str, float]]
    error_cases: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "accuracy": self.accuracy,
            "f1_macro": self.f1_macro,
            "f1_micro": self.f1_micro,
            "f1_weighted": self.f1_weighted,
            "confusion_matrix": self.confusion_matrix,
            "per_class_metrics": self.per_class_metrics,
            "error_cases": self.error_cases,
        }


class IntentEvaluator:
    """Evaluates intent classification performance."""

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize intent evaluator.

        Args:
            config: Evaluation configuration
        """
        self.config = config or {}

    async def evaluate(
        self,
        predictions: List[str],
        ground_truths: List[str]
    ) -> IntentResult:
        """
        Evaluate intent classification.

        Args:
            predictions: Predicted intent labels
            ground_truths: Ground truth intent labels

        Returns:
            IntentResult with computed metrics
        """
        if len(predictions) != len(ground_truths):
            raise ValueError(
                f"Predictions ({len(predictions)}) and ground truths "
                f"({len(ground_truths)}) must have same length"
            )

        # Compute basic metrics
        accuracy = self._compute_accuracy(predictions, ground_truths)

        # Get unique classes
        all_classes = sorted(set(predictions) | set(ground_truths))

        # Compute per-class metrics
        per_class_metrics = {}
        for cls in all_classes:
            metrics = self._compute_class_metrics(predictions, ground_truths, cls)
            per_class_metrics[cls] = metrics

        # Compute F1 scores
        f1_macro = np.mean([m["f1"] for m in per_class_metrics.values()])
        f1_weighted = self._compute_f1_weighted(predictions, ground_truths, per_class_metrics)
        f1_micro = accuracy  # For single-label, micro F1 = accuracy

        # Build confusion matrix
        confusion_matrix = self._build_confusion_matrix(predictions, ground_truths)

        # Identify error cases
        error_cases = self._identify_errors(predictions, ground_truths)

        return IntentResult(
            accuracy=accuracy,
            f1_macro=float(f1_macro),
            f1_micro=f1_micro,
            f1_weighted=f1_weighted,
            confusion_matrix=confusion_matrix,
            per_class_metrics=per_class_metrics,
            error_cases=error_cases,
        )

    def _compute_accuracy(self, predictions: List[str], ground_truths: List[str]) -> float:
        """Compute overall accuracy."""
        correct = sum(p == g for p, g in zip(predictions, ground_truths))
        return correct / len(ground_truths) if ground_truths else 0.0

    def _compute_class_metrics(
        self,
        predictions: List[str],
        ground_truths: List[str],
        cls: str
    ) -> Dict[str, float]:
        """Compute precision, recall, F1 for a single class."""
        tp = sum(p == cls and g == cls for p, g in zip(predictions, ground_truths))
        fp = sum(p == cls and g != cls for p, g in zip(predictions, ground_truths))
        fn = sum(p != cls and g == cls for p, g in zip(predictions, ground_truths))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        support = sum(g == cls for g in ground_truths)

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

    def _compute_f1_weighted(
        self,
        predictions: List[str],
        ground_truths: List[str],
        per_class_metrics: Dict[str, Dict[str, float]]
    ) -> float:
        """Compute weighted average F1 score."""
        total_support = sum(m["support"] for m in per_class_metrics.values())
        if total_support == 0:
            return 0.0

        weighted_f1 = sum(
            m["f1"] * m["support"] for m in per_class_metrics.values()
        ) / total_support
        return weighted_f1

    def _build_confusion_matrix(
        self,
        predictions: List[str],
        ground_truths: List[str]
    ) -> Dict[str, Dict[str, int]]:
        """Build confusion matrix as nested dict."""
        all_classes = sorted(set(predictions) | set(ground_truths))
        matrix = {actual: {predicted: 0 for predicted in all_classes} for actual in all_classes}

        for actual, predicted in zip(ground_truths, predictions):
            matrix[actual][predicted] += 1

        return matrix

    def _identify_errors(
        self,
        predictions: List[str],
        ground_truths: List[str]
    ) -> List[Dict[str, Any]]:
        """Identify misclassified examples."""
        errors = []

        # Get class distribution for context
        class_counts = Counter(ground_truths)

        for i, (pred, truth) in enumerate(zip(predictions, ground_truths)):
            if pred != truth:
                errors.append({
                    "index": i,
                    "predicted": pred,
                    "actual": truth,
                    "class_support": class_counts[truth],
                })

        # Sort by class support (prioritize errors on rare classes)
        errors.sort(key=lambda x: x["class_support"])

        return errors[:50]  # Limit to top 50 error cases

    def get_confusion_summary(self, confusion_matrix: Dict[str, Dict[str, int]]) -> Dict[str, Any]:
        """
        Generate a summary of confusion patterns.

        Returns:
            Dict with most confused pairs and per-class accuracy
        """
        confused_pairs = []
        per_class_accuracy = {}

        for actual, predictions in confusion_matrix.items():
            total = sum(predictions.values())
            correct = predictions.get(actual, 0)
            per_class_accuracy[actual] = correct / total if total > 0 else 0.0

            # Find most common misclassifications
            for predicted, count in predictions.items():
                if predicted != actual and count > 0:
                    confused_pairs.append({
                        "actual": actual,
                        "predicted": predicted,
                        "count": count,
                    })

        # Sort by frequency
        confused_pairs.sort(key=lambda x: x["count"], reverse=True)

        return {
            "top_confusions": confused_pairs[:10],
            "per_class_accuracy": per_class_accuracy,
        }
