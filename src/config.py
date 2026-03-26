"""Configuration management for evaluation pipeline.

Loads YAML configuration files with environment variable substitution.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, validator


@dataclass(frozen=True)
class DatabaseConfig:
    """Database connection configuration."""

    host: str
    port: int
    database: str
    user: str
    password: str
    pool_min: int = 1
    pool_max: int = 10
    command_timeout: int = 30

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DatabaseConfig:
        return cls(
            host=data.get("host", "localhost"),
            port=data.get("port", 5432),
            database=data["database"],
            user=data["user"],
            password=data["password"],
            pool_min=data.get("pool_min", 1),
            pool_max=data.get("pool_max", 10),
            command_timeout=data.get("command_timeout", 30),
        )

    @property
    def dsn(self) -> str:
        """Return PostgreSQL DSN."""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


@dataclass(frozen=True)
class DatasetConfig:
    """Dataset configuration."""

    name: str = "customer_service_qa_v2"
    version: str = "2.0"
    type: str = "qa"
    file_path: str = "data/test_dataset.jsonl"

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> DatasetConfig:
        if data is None:
            return cls()
        return cls(
            name=data.get("name", "customer_service_qa_v2"),
            version=data.get("version", "2.0"),
            type=data.get("type", "qa"),
            file_path=data.get("file_path", "data/test_dataset.jsonl"),
        )


@dataclass(frozen=True)
class QdrantConfig:
    """Qdrant vector database configuration."""

    host: str
    port: int
    collection_name: str
    api_key: str | None = None
    https: bool = False

    @property
    def base_url(self) -> str:
        """Return base URL for Qdrant client."""
        scheme = "https" if self.https else "http"
        return f"{scheme}://{self.host}:{self.port}"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QdrantConfig:
        return cls(
            host=data.get("host", "localhost"),
            port=data.get("port", 6333),
            collection_name=data["collection_name"],
            api_key=data.get("api_key"),
            https=data.get("https", False),
        )


@dataclass(frozen=True)
class LatencyThresholds:
    """Latency threshold configuration."""

    retrieval_p95_max_ms: int = 200
    retrieval_p99_max_ms: int = 500
    generation_p95_max_ms: int = 3000
    generation_p99_max_ms: int = 10000
    total_p95_max_ms: int = 5000
    total_p99_max_ms: int = 15000

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> LatencyThresholds:
        if data is None:
            return cls()
        return cls(
            retrieval_p95_max_ms=data.get("retrieval", {}).get("p95_max_ms", 200),
            retrieval_p99_max_ms=data.get("retrieval", {}).get("p99_max_ms", 500),
            generation_p95_max_ms=data.get("generation", {}).get("p95_max_ms", 3000),
            generation_p99_max_ms=data.get("generation", {}).get("p99_max_ms", 10000),
            total_p95_max_ms=data.get("total", {}).get("p95_max_ms", 5000),
            total_p99_max_ms=data.get("total", {}).get("p99_max_ms", 15000),
        )


@dataclass(frozen=True)
class MetricThresholds:
    """Metric threshold configuration for evaluation."""

    # Degradation thresholds (percentage)
    precision_drop_threshold: float = 0.05
    hallucination_rate_max: float = 0.1
    p95_latency_increase_max_ms: int = 500

    # Minimum acceptable values
    min_mrr5: float = 0.6
    min_ndcg5: float = 0.6
    min_hitrate3: float = 0.7
    min_faithfulness: float = 0.8
    min_answer_relevance: float = 3.5

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> MetricThresholds:
        if data is None:
            return cls()
        return cls(
            precision_drop_threshold=data.get("precision_drop_threshold", 0.05),
            hallucination_rate_max=data.get("hallucination_rate_max", 0.1),
            p95_latency_increase_max_ms=data.get("p95_latency_increase_max_ms", 500),
            min_mrr5=data.get("min_mrr5", 0.6),
            min_ndcg5=data.get("min_ndcg5", 0.6),
            min_hitrate3=data.get("min_hitrate3", 0.7),
            min_faithfulness=data.get("min_faithfulness", 0.8),
            min_answer_relevance=data.get("min_answer_relevance", 3.5),
        )


@dataclass(frozen=True)
class LLMJudgeConfig:
    """LLM-as-a-Judge configuration."""

    model: str = "gpt-4"
    temperature: float = 0.0
    max_tokens: int = 500
    api_key: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> LLMJudgeConfig:
        if data is None:
            return cls()
        return cls(
            model=data.get("model", "gpt-4"),
            temperature=data.get("temperature", 0.0),
            max_tokens=data.get("max_tokens", 500),
            api_key=data.get("api_key") or os.getenv("OPENAI_API_KEY"),
        )


@dataclass(frozen=True)
class ChatbotAPIConfig:
    """Chatbot API configuration for generation evaluation."""

    base_url: str
    timeout_ms: int = 10000
    api_key: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChatbotAPIConfig:
        return cls(
            base_url=data["base_url"],
            timeout_ms=data.get("timeout_ms", 10000),
            api_key=data.get("api_key"),
        )


class EvaluationConfig(BaseModel):
    """Main evaluation configuration."""

    # Core components
    database: DatabaseConfig
    qdrant: QdrantConfig
    chatbot_api: ChatbotAPIConfig
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)

    # Thresholds
    latency_thresholds: LatencyThresholds = Field(default_factory=LatencyThresholds)
    metric_thresholds: MetricThresholds = Field(default_factory=MetricThresholds)

    # LLM Judge
    llm_judge: LLMJudgeConfig = Field(default_factory=LLMJudgeConfig)

    # Evaluation settings
    bootstrap_samples: int = 1000
    confidence_level: float = 0.95
    max_concurrent_requests: int = 10

    @validator("confidence_level")
    def validate_confidence_level(cls, v: float) -> float:
        if not 0 < v < 1:
            raise ValueError("confidence_level must be between 0 and 1")
        return v


class ConfigLoader:
    """Load and manage configuration from YAML files."""

    _ENV_VAR_PATTERN = re.compile(r"\$\{([^}]+)\}")

    @classmethod
    def load(cls, config_path: str | Path) -> EvaluationConfig:
        """Load configuration from YAML file with environment variable substitution.

        Args:
            config_path: Path to YAML configuration file.

        Returns:
            EvaluationConfig: Validated configuration object.

        Raises:
            FileNotFoundError: If config file doesn't exist.
            ValueError: If configuration is invalid.
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        raw_data = yaml.safe_load(config_path.read_text())
        substituted = cls._substitute_env_vars(raw_data)

        # Parse nested configs
        database = DatabaseConfig.from_dict(substituted.get("database", {}))
        qdrant = QdrantConfig.from_dict(substituted.get("qdrant", {}))
        chatbot_api = ChatbotAPIConfig.from_dict(substituted.get("chatbot_api", {}))
        dataset = DatasetConfig.from_dict(substituted.get("dataset"))
        latency_thresholds = LatencyThresholds.from_dict(
            substituted.get("latency_thresholds")
        )
        metric_thresholds = MetricThresholds.from_dict(
            substituted.get("metric_thresholds")
        )
        llm_judge = LLMJudgeConfig.from_dict(substituted.get("llm_judge"))

        return EvaluationConfig(
            database=database,
            qdrant=qdrant,
            chatbot_api=chatbot_api,
            dataset=dataset,
            latency_thresholds=latency_thresholds,
            metric_thresholds=metric_thresholds,
            llm_judge=llm_judge,
            bootstrap_samples=substituted.get("bootstrap_samples", 1000),
            confidence_level=substituted.get("confidence_level", 0.95),
            max_concurrent_requests=substituted.get("max_concurrent_requests", 10),
        )

    @classmethod
    def _substitute_env_vars(cls, data: Any) -> Any:
        """Recursively substitute ${VAR_NAME} patterns with environment variables.

        Args:
            data: Raw configuration data (dict, list, or scalar).

        Returns:
            Data with environment variables substituted.
        """
        if isinstance(data, dict):
            return {k: cls._substitute_env_vars(v) for k, v in data.items()}
        if isinstance(data, list):
            return [cls._substitute_env_vars(item) for item in data]
        if isinstance(data, str):
            return cls._ENV_VAR_PATTERN.sub(
                lambda m: os.getenv(m.group(1), m.group(0)), data
            )
        return data


def load_config(config_path: str | Path | None = None) -> EvaluationConfig:
    """Load configuration from default or specified path.

    Args:
        config_path: Optional path to config file. Defaults to
                    config/evaluation_config.yaml.

    Returns:
        EvaluationConfig: Validated configuration.
    """
    if config_path is None:
        # Default path relative to project root
        project_root = Path(__file__).parent.parent
        config_path = project_root / "config" / "evaluation_config.yaml"

    return ConfigLoader.load(config_path)
