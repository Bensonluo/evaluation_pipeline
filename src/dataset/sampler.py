"""
Stratified Sampler Module

Implements stratified sampling to create representative test subsets.
Ensures balanced distribution across intents and other categories.

Design Reference: DESIGN.md Chapter 13
"""

import logging
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class SamplingResult:
    """Result of sampling operation."""

    samples: List[Dict]
    original_size: int
    sampled_size: int
    sampling_ratio: float
    stratum_counts: Dict[str, int]
    metadata: Dict[str, Any]


class StratifiedSampler:
    """
    Stratified sampler for creating representative test subsets.

    Supports stratification by:
    - Intent type (default for QA/intent datasets)
    - Custom key functions
    - Multiple strata layers

    Example:
        >>> sampler = StratifiedSampler(stratify_key="intent")
        >>> result = sampler.sample(samples, n=100)
    """

    def __init__(
        self,
        stratify_key: str = "intent",
        seed: Optional[int] = None,
        min_samples_per_stratum: int = 1,
    ):
        """
        Initialize sampler.

        Args:
            stratify_key: Key to stratify by (e.g., 'intent')
            seed: Random seed for reproducibility
            min_samples_per_stratum: Minimum samples to include per stratum
        """
        self.stratify_key = stratify_key
        self.rng = random.Random(seed)
        self.min_samples_per_stratum = min_samples_per_stratum

    def sample(
        self,
        samples: List[Dict],
        n: Optional[int] = None,
        ratio: Optional[float] = None,
        strata_counts: Optional[Dict[str, int]] = None,
    ) -> SamplingResult:
        """
        Perform stratified sampling.

        Args:
            samples: Input sample list
            n: Target sample size (exclusive with ratio and strata_counts)
            ratio: Sampling ratio (e.g., 0.2 for 20%)
            strata_counts: Exact counts per stratum

        Returns:
            SamplingResult with sampled data and statistics

        Raises:
            ValueError: If sampling parameters are invalid
        """
        if not samples:
            return SamplingResult(
                samples=[],
                original_size=0,
                sampled_size=0,
                sampling_ratio=0.0,
                stratum_counts={},
                metadata={},
            )

        # Group by stratum
        strata = self._group_by_stratum(samples)

        # Determine target counts
        if strata_counts is not None:
            target_counts = strata_counts
        elif n is not None:
            target_counts = self._compute_counts_proportional(strata, n)
        elif ratio is not None:
            n_target = max(1, int(len(samples) * ratio))
            target_counts = self._compute_counts_proportional(strata, n_target)
        else:
            raise ValueError("Must specify one of: n, ratio, or strata_counts")

        # Apply minimum samples constraint
        target_counts = self._apply_minimum_constraint(strata, target_counts)

        # Sample from each stratum
        sampled = []
        actual_counts: Dict[str, int] = defaultdict(int)

        for stratum, stratum_samples in strata.items():
            count = min(target_counts.get(stratum, 0), len(stratum_samples))
            if count > 0:
                sampled_stratum = self.rng.sample(stratum_samples, count)
                sampled.extend(sampled_stratum)
                actual_counts[stratum] = count

        # Shuffle final result
        self.rng.shuffle(sampled)

        return SamplingResult(
            samples=sampled,
            original_size=len(samples),
            sampled_size=len(sampled),
            sampling_ratio=len(sampled) / len(samples),
            stratum_counts=dict(actual_counts),
            metadata={
                "strata_available": list(strata.keys()),
                "original_counts": {k: len(v) for k, v in strata.items()},
                "target_counts": target_counts,
            },
        )

    def _group_by_stratum(self, samples: List[Dict]) -> Dict[str, List[Dict]]:
        """Group samples by stratification key."""
        strata: Dict[str, List[Dict]] = defaultdict(list)

        for sample in samples:
            key = sample.get(self.stratify_key, "UNKNOWN")
            strata[key].append(sample)

        return dict(strata)

    def _compute_counts_proportional(
        self,
        strata: Dict[str, List[Dict]],
        n_target: int,
    ) -> Dict[str, int]:
        """Compute proportional counts per stratum."""
        total = sum(len(v) for v in strata.values())
        if total == 0:
            return {}

        counts: Dict[str, int] = {}
        allocated = 0

        # Compute proportional counts
        for stratum, stratum_samples in strata.items():
            proportion = len(stratum_samples) / total
            count = max(1, int(proportion * n_target))
            counts[stratum] = count
            allocated += count

        # Distribute remainder to largest strata
        remainder = n_target - allocated
        if remainder > 0:
            sorted_strata = sorted(
                strata.items(),
                key=lambda x: len(x[1]),
                reverse=True,
            )
            for stratum, _ in sorted_strata[:remainder]:
                counts[stratum] += 1

        return counts

    def _apply_minimum_constraint(
        self,
        strata: Dict[str, List[Dict]],
        target_counts: Dict[str, int],
    ) -> Dict[str, int]:
        """Apply minimum samples per stratum constraint."""
        adjusted = target_counts.copy()

        for stratum, stratum_samples in strata.items():
            available = len(stratum_samples)
            current = adjusted.get(stratum, 0)

            if current < self.min_samples_per_stratum and available >= self.min_samples_per_stratum:
                adjusted[stratum] = self.min_samples_per_stratum
            elif current > available:
                adjusted[stratum] = available

        return adjusted

    def resample_with_replacement(
        self,
        samples: List[Dict],
        n: int,
        weights: Optional[Dict[str, float]] = None,
    ) -> SamplingResult:
        """
        Sample with replacement (bootstrap).

        Args:
            samples: Input sample list
            n: Target sample size
            weights: Optional weights per stratum for biased sampling

        Returns:
            SamplingResult
        """
        strata = self._group_by_stratum(samples)
        stratum_names = list(strata.keys())

        # Compute sampling weights
        if weights is None:
            # Proportional to stratum size
            weights = {
                name: len(stratum_samples) / len(samples)
                for name, stratum_samples in strata.items()
            }

        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}

        # Sample with replacement
        sampled = []
        actual_counts: Dict[str, int] = defaultdict(int)

        for _ in range(n):
            # Choose stratum by weighted random
            stratum = self.rng.choices(
                stratum_names,
                weights=[weights.get(s, 0) for s in stratum_names],
            )[0]

            # Sample from stratum with replacement
            stratum_samples = strata[stratum]
            sample = self.rng.choice(stratum_samples)
            sampled.append(sample)
            actual_counts[stratum] += 1

        return SamplingResult(
            samples=sampled,
            original_size=len(samples),
            sampled_size=n,
            sampling_ratio=n / len(samples),
            stratum_counts=dict(actual_counts),
            metadata={
                "sampling_method": "with_replacement",
                "weights": weights,
            },
        )

    def create_train_val_split(
        self,
        samples: List[Dict],
        val_ratio: float = 0.2,
        seed: Optional[int] = None,
    ) -> tuple[List[Dict], List[Dict]]:
        """
        Create stratified train/validation split.

        Args:
            samples: Input sample list
            val_ratio: Validation set ratio
            seed: Random seed for split

        Returns:
            Tuple of (train_samples, val_samples)
        """
        if seed is not None:
            self.rng = random.Random(seed)

        val_result = self.sample(samples, ratio=val_ratio)

        # Get indices to determine train set
        val_ids = {s.get("query_id", i) for i, s in enumerate(val_result.samples)}
        train_samples = [
            s for i, s in enumerate(samples)
            if s.get("query_id", i) not in val_ids
        ]

        return train_samples, val_result.samples

    def balance_dataset(
        self,
        samples: List[Dict],
        target_size: Optional[int] = None,
        method: str = "upsample",
    ) -> List[Dict]:
        """
        Balance dataset across strata.

        Args:
            samples: Input sample list
            target_size: Target size per stratum (default: max current size)
            method: 'upsample' or 'downsample'

        Returns:
            Balanced sample list
        """
        strata = self._group_by_stratum(samples)

        if target_size is None:
            target_size = max(len(v) for v in strata.values())

        balanced = []

        for stratum, stratum_samples in strata.items():
            if method == "upsample":
                # Sample with replacement to reach target
                while len(stratum_samples) < target_size:
                    stratum_samples.extend(stratum_samples[:target_size - len(stratum_samples)])
                balanced.extend(stratum_samples[:target_size])
            else:  # downsample
                # Sample without replacement
                count = min(len(stratum_samples), target_size)
                balanced.extend(self.rng.sample(stratum_samples, count))

        self.rng.shuffle(balanced)
        return balanced


# Convenience function
def sample_dataset(
    samples: List[Dict],
    n: Optional[int] = None,
    ratio: Optional[float] = None,
    stratify_key: str = "intent",
    seed: Optional[int] = None,
) -> SamplingResult:
    """
    Sample dataset with stratification.

    Args:
        samples: Input sample list
        n: Target sample size
        ratio: Sampling ratio
        stratify_key: Key to stratify by
        seed: Random seed

    Returns:
        SamplingResult
    """
    sampler = StratifiedSampler(stratify_key=stratify_key, seed=seed)
    return sampler.sample(samples, n=n, ratio=ratio)
