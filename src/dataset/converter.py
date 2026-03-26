"""
Format Converter Module

Converts between different dataset formats for compatibility
with various evaluation tools and databases.

Design Reference: DESIGN.md Chapter 13
"""

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class FormatConverter:
    """
    Convert between dataset formats.

    Supported conversions:
    - JSONL (file) <-> Internal (list of dicts)
    - JSON (file) <-> Internal
    - Database format <-> Internal
    - RAGAS format <-> Internal
    """

    # Intent types from DESIGN.md Section 12.2
    INTENT_TYPES = [
        "HOW_TO",
        "POLICY",
        "PRODUCT",
        "BILLING",
        "COMPLAINT",
        "REFUND",
        "FEEDBACK",
        "GREETING",
        "OTHER",
    ]

    @classmethod
    def to_internal_format(
        cls,
        data: Union[List[Dict], str, Path],
        source_format: str = "jsonl",
    ) -> List[Dict]:
        """
        Convert data to internal format (list of dicts).

        Args:
            data: Input data (list, file path, or raw string)
            source_format: Source format ('jsonl', 'json', 'ragas')

        Returns:
            List of sample dictionaries in internal format

        Raises:
            ValueError: If format is unsupported
        """
        if source_format == "jsonl":
            if isinstance(data, (str, Path)):
                return cls._jsonl_file_to_internal(data)
            return cls._jsonl_string_to_internal(data)

        elif source_format == "json":
            if isinstance(data, (str, Path)):
                return cls._json_file_to_internal(data)
            return data  # Already internal

        elif source_format == "ragas":
            return cls._ragas_to_internal(data)

        else:
            raise ValueError(f"Unsupported source format: {source_format}")

    @classmethod
    def from_internal_format(
        cls,
        samples: List[Dict],
        target_format: str,
        output_path: Optional[Union[str, Path]] = None,
    ) -> Optional[Union[str, List[Dict]]]:
        """
        Convert from internal format to target format.

        Args:
            samples: Internal format samples
            target_format: Target format ('jsonl', 'json', 'ragas', 'db')
            output_path: Optional file path to write output

        Returns:
            Converted data (or None if written to file)

        Raises:
            ValueError: If format is unsupported
        """
        if target_format == "jsonl":
            result = cls._internal_to_jsonl(samples)
            if output_path:
                cls._write_file(output_path, result)
                return None
            return result

        elif target_format == "json":
            result = json.dumps(samples, ensure_ascii=False, indent=2)
            if output_path:
                cls._write_file(output_path, result)
                return None
            return result

        elif target_format == "ragas":
            result = cls._internal_to_ragas(samples)
            if output_path:
                cls._write_file(output_path, json.dumps(result, ensure_ascii=False, indent=2))
                return None
            return result

        elif target_format == "db":
            return cls._internal_to_db_format(samples)

        else:
            raise ValueError(f"Unsupported target format: {target_format}")

    @staticmethod
    def _jsonl_file_to_internal(file_path: Union[str, Path]) -> List[Dict]:
        """Convert JSONL file to internal format."""
        samples = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        samples.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse line: {e}")
        return samples

    @staticmethod
    def _jsonl_string_to_internal(data: Union[str, List]) -> List[Dict]:
        """Convert JSONL string to internal format."""
        if isinstance(data, list):
            return data

        samples = []
        for line in data.split("\n"):
            line = line.strip()
            if line:
                try:
                    samples.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return samples

    @staticmethod
    def _json_file_to_internal(file_path: Union[str, Path]) -> List[Dict]:
        """Convert JSON file to internal format."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and "samples" in data:
            return data["samples"]
        else:
            raise ValueError(f"Invalid JSON format in {file_path}")

    @staticmethod
    def _internal_to_jsonl(samples: List[Dict]) -> str:
        """Convert internal format to JSONL string."""
        lines = []
        for sample in samples:
            lines.append(json.dumps(sample, ensure_ascii=False))
        return "\n".join(lines)

    @classmethod
    def _ragas_to_internal(cls, data: Union[List[Dict], str, Path]) -> List[Dict]:
        """
        Convert RAGAS format to internal format.

        RAGAS format:
        {
            "question": "How to reset password?",
            "contexts": ["context1", "context2"],
            "ground_truth": "Step 1: Click forgot password...",
            "answer": "generated answer"
        }
        """
        if isinstance(data, (str, Path)):
            with open(data, "r", encoding="utf-8") as f:
                data = json.load(f)

        if not isinstance(data, list):
            data = [data]

        internal = []
        for item in data:
            internal.append({
                "query": item.get("question", ""),
                "ground_truth": item.get("ground_truth", ""),
                "contexts": item.get("contexts", []),
                "generated_answer": item.get("answer", ""),
                "intent": item.get("intent", "OTHER"),
                "metadata": item.get("metadata", {}),
            })

        return internal

    @classmethod
    def _internal_to_ragas(cls, samples: List[Dict]) -> List[Dict]:
        """Convert internal format to RAGAS format."""
        ragas_data = []

        for sample in samples:
            ragas_data.append({
                "question": sample.get("query", ""),
                "contexts": sample.get("contexts", []),
                "ground_truth": sample.get("ground_truth", ""),
                "answer": sample.get("generated_answer", ""),
                "metadata": sample.get("metadata", {}),
            })

        return ragas_data

    @staticmethod
    def _internal_to_db_format(samples: List[Dict]) -> Dict[str, Any]:
        """
        Convert internal format to database-compatible format.

        Returns dict matching test_datasets table schema.
        """
        # Compute intent distribution
        intent_counts: Dict[str, int] = {}
        for sample in samples:
            intent = sample.get("intent", "UNKNOWN")
            intent_counts[intent] = intent_counts.get(intent, 0) + 1

        return {
            "data": samples,
            "stats": {
                "total_samples": len(samples),
                "intent_distribution": intent_counts,
            },
            "metadata": {
                "format_version": "1.0",
                "schema": "internal_v1",
            },
        }

    @staticmethod
    def _write_file(file_path: Union[str, Path], content: str) -> None:
        """Write content to file."""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

        logger.info(f"Written to {file_path}")

    @classmethod
    def validate_intent(cls, intent: str) -> bool:
        """Check if intent is valid."""
        return intent in cls.INTENT_TYPES

    @classmethod
    def normalize_intent(cls, intent: str) -> str:
        """Normalize intent to uppercase valid value."""
        intent_upper = intent.upper()
        if cls.validate_intent(intent_upper):
            return intent_upper
        return "OTHER"

    @classmethod
    def convert_retrieval_labels_to_qa(
        cls,
        retrieval_labels: List[Dict],
        knowledge_base: Dict[str, Dict],
    ) -> List[Dict]:
        """
        Convert retrieval labels to QA format using knowledge base.

        Args:
            retrieval_labels: List of {"query": str, "relevant_docs": [str]}
            knowledge_base: Dict mapping doc_id to content

        Returns:
            List of QA samples
        """
        qa_samples = []

        for label in retrieval_labels:
            query = label.get("query", "")
            relevant_docs = label.get("relevant_docs", [])

            # Construct ground truth from relevant docs
            contexts = []
            for doc_id in relevant_docs:
                if doc_id in knowledge_base:
                    contexts.append(knowledge_base[doc_id].get("content", ""))

            qa_samples.append({
                "query": query,
                "ground_truth": " ".join(contexts),
                "relevant_docs": relevant_docs,
                "intent": label.get("intent", "OTHER"),
                "metadata": label.get("metadata", {}),
            })

        return qa_samples


# Convenience functions
def convert_to_internal_format(
    data: Union[List[Dict], str, Path],
    source_format: str = "jsonl",
) -> List[Dict]:
    """Convert data to internal format."""
    return FormatConverter.to_internal_format(data, source_format)


def convert_from_internal(
    samples: List[Dict],
    target_format: str,
    output_path: Optional[Union[str, Path]] = None,
) -> Optional[Union[str, List[Dict]]]:
    """Convert from internal format to target format."""
    return FormatConverter.from_internal_format(samples, target_format, output_path)
