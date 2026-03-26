"""
Dataset Loader Module

Loads test datasets from JSONL files or database.
Supports QA pairs, retrieval labels, and intent classification data.

Design Reference: DESIGN.md Chapter 13.2
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class DatasetInfo:
    """Dataset metadata."""

    dataset_name: str
    dataset_type: str  # 'qa', 'retrieval', 'intent'
    version: str = "1.0"
    description: str = ""
    total_samples: int = 0
    stats: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QASample:
    """QA pair sample."""

    query: str
    intent: str
    ground_truth: str
    query_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalLabel:
    """Retrieval ground truth label."""

    query: str
    relevant_docs: List[str]
    query_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IntentSample:
    """Intent classification sample."""

    query: str
    intent: str
    confidence: Optional[float] = None
    query_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class DatasetLoader:
    """
    Dataset loader supporting multiple formats and sources.

    Supports:
    - JSONL files (qa, retrieval, intent)
    - PostgreSQL database (test_datasets table)
    """

    def __init__(self, db_connection: Optional[Any] = None):
        """
        Initialize dataset loader.

        Args:
            db_connection: Database connection object (psycopg2/sqlalchemy)
        """
        self.db_connection = db_connection

    def load_jsonl(
        self,
        file_path: Union[str, Path],
        dataset_type: str = "qa",
    ) -> tuple[List[Dict], DatasetInfo]:
        """
        Load dataset from JSONL file.

        Args:
            file_path: Path to JSONL file
            dataset_type: Type of dataset ('qa', 'retrieval', 'intent')

        Returns:
            Tuple of (samples, dataset_info)

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If dataset_type is invalid
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        samples = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    sample = json.loads(line)
                    samples.append(sample)
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse line {line_num}: {e}")

        if not samples:
            raise ValueError(f"No valid samples found in {file_path}")

        # Validate and normalize based on dataset type
        validated_samples = self._validate_samples(samples, dataset_type)

        # Create dataset info
        info = DatasetInfo(
            dataset_name=file_path.stem,
            dataset_type=dataset_type,
            version="1.0",
            total_samples=len(validated_samples),
            stats=self._compute_stats(validated_samples, dataset_type),
        )

        logger.info(f"Loaded {len(validated_samples)} samples from {file_path}")
        return validated_samples, info

    def _validate_samples(
        self,
        samples: List[Dict],
        dataset_type: str,
    ) -> List[Dict]:
        """Validate and normalize samples based on dataset type."""
        validated = []

        if dataset_type == "qa":
            required_fields = ["query", "intent", "ground_truth"]
            for sample in samples:
                if all(k in sample for k in required_fields):
                    validated.append({
                        "query": sample["query"],
                        "intent": sample["intent"],
                        "ground_truth": sample["ground_truth"],
                        "query_id": sample.get("query_id"),
                        "metadata": sample.get("metadata", {}),
                    })
                else:
                    missing = [k for k in required_fields if k not in sample]
                    logger.warning(f"Sample missing fields {missing}, skipping")

        elif dataset_type == "retrieval":
            required_fields = ["query", "relevant_docs"]
            for sample in samples:
                if all(k in sample for k in required_fields):
                    if isinstance(sample["relevant_docs"], list):
                        validated.append({
                            "query": sample["query"],
                            "relevant_docs": sample["relevant_docs"],
                            "query_id": sample.get("query_id"),
                            "metadata": sample.get("metadata", {}),
                        })
                    else:
                        logger.warning("relevant_docs must be a list, skipping")
                else:
                    missing = [k for k in required_fields if k not in sample]
                    logger.warning(f"Sample missing fields {missing}, skipping")

        elif dataset_type == "intent":
            required_fields = ["query", "intent"]
            for sample in samples:
                if all(k in sample for k in required_fields):
                    validated.append({
                        "query": sample["query"],
                        "intent": sample["intent"],
                        "confidence": sample.get("confidence"),
                        "query_id": sample.get("query_id"),
                        "metadata": sample.get("metadata", {}),
                    })
                else:
                    missing = [k for k in required_fields if k not in sample]
                    logger.warning(f"Sample missing fields {missing}, skipping")

        else:
            raise ValueError(
                f"Invalid dataset_type: {dataset_type}. "
                "Must be 'qa', 'retrieval', or 'intent'"
            )

        return validated

    def _compute_stats(self, samples: List[Dict], dataset_type: str) -> Dict:
        """Compute dataset statistics."""
        stats = {"total": len(samples)}

        if dataset_type in ("qa", "intent"):
            # Intent distribution
            intent_counts: Dict[str, int] = {}
            for sample in samples:
                intent = sample.get("intent", "UNKNOWN")
                intent_counts[intent] = intent_counts.get(intent, 0) + 1
            stats["intent_distribution"] = intent_counts

        elif dataset_type == "retrieval":
            # Relevant docs count distribution
            doc_counts = [len(s.get("relevant_docs", [])) for s in samples]
            stats["avg_relevant_docs"] = sum(doc_counts) / len(doc_counts) if doc_counts else 0
            stats["max_relevant_docs"] = max(doc_counts) if doc_counts else 0
            stats["min_relevant_docs"] = min(doc_counts) if doc_counts else 0

        return stats

    def load_from_db(
        self,
        dataset_name: str,
        version: Optional[str] = None,
    ) -> tuple[List[Dict], DatasetInfo]:
        """
        Load dataset from database.

        Args:
            dataset_name: Name of the dataset
            version: Optional version filter

        Returns:
            Tuple of (samples, dataset_info)

        Raises:
            ValueError: If database connection not available
        """
        if self.db_connection is None:
            raise ValueError("Database connection not configured")

        # Query test_datasets table
        query = "SELECT data, stats, metadata, dataset_type, version FROM test_datasets WHERE dataset_name = %s"
        params = [dataset_name]

        if version:
            query += " AND version = %s"
            params.append(version)

        query += " ORDER BY created_at DESC LIMIT 1"

        cursor = self.db_connection.cursor()
        cursor.execute(query, params)
        row = cursor.fetchone()

        if row is None:
            raise ValueError(f"Dataset '{dataset_name}' not found")

        data, stats, metadata, dataset_type, db_version = row

        info = DatasetInfo(
            dataset_name=dataset_name,
            dataset_type=dataset_type,
            version=db_version,
            total_samples=len(data) if data else 0,
            stats=stats or {},
            metadata=metadata or {},
        )

        logger.info(f"Loaded {info.total_samples} samples from database: {dataset_name}")
        return data or [], info

    def save_to_db(
        self,
        samples: List[Dict],
        info: DatasetInfo,
    ) -> int:
        """
        Save dataset to database.

        Args:
            samples: List of sample dictionaries
            info: Dataset metadata

        Returns:
            Dataset ID

        Raises:
            ValueError: If database connection not available
        """
        if self.db_connection is None:
            raise ValueError("Database connection not configured")

        query = """
            INSERT INTO test_datasets (dataset_name, dataset_type, version, description, data, stats, metadata)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (dataset_name, version) DO UPDATE
            SET data = EXCLUDED.data, stats = EXCLUDED.stats, metadata = EXCLUDED.metadata
            RETURNING id
        """

        cursor = self.db_connection.cursor()
        cursor.execute(
            query,
            (
                info.dataset_name,
                info.dataset_type,
                info.version,
                info.description,
                json.dumps(samples),
                json.dumps(info.stats),
                json.dumps(info.metadata),
            ),
        )

        dataset_id = cursor.fetchone()[0]
        self.db_connection.commit()

        logger.info(f"Saved dataset '{info.dataset_name}' (id={dataset_id}) to database")
        return dataset_id


# Convenience functions
def load_jsonl(
    file_path: Union[str, Path],
    dataset_type: str = "qa",
) -> tuple[List[Dict], DatasetInfo]:
    """Load dataset from JSONL file."""
    loader = DatasetLoader()
    return loader.load_jsonl(file_path, dataset_type)


def load_dataset_from_db(
    dataset_name: str,
    db_connection: Any,
    version: Optional[str] = None,
) -> tuple[List[Dict], DatasetInfo]:
    """Load dataset from database."""
    loader = DatasetLoader(db_connection)
    return loader.load_from_db(dataset_name, version)
