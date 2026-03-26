"""PostgreSQL database connection and CRUD operations.

Manages connections to the evaluation pipeline database with connection pooling.
Provides CRUD operations for evaluation_baselines, evaluation_results, and test_datasets.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import psycopg2
from psycopg2 import sql
from psycopg2.extras import RealDictCursor
from psycopg2.pool import ThreadedConnectionPool


class DatabaseManager:
    """Manage PostgreSQL connections and CRUD operations for evaluation pipeline."""

    def __init__(self, dsn: str, min_size: int = 1, max_size: int = 10):
        """Initialize database connection pool.

        Args:
            dsn: PostgreSQL connection string.
            min_size: Minimum number of connections in pool.
            max_size: Maximum number of connections in pool.
        """
        self._pool = ThreadedConnectionPool(
            min_size,
            max_size,
            dsn,
        )

    @property
    def pool(self) -> ThreadedConnectionPool:
        """Get the connection pool."""
        return self._pool

    def close(self) -> None:
        """Close the connection pool."""
        self._pool.closeall()

    # ==========================================================================
    # Evaluation Baselines CRUD
    # ==========================================================================

    def create_baseline(
        self,
        baseline_name: str,
        model_version: str | None = None,
        finetune_target: str | None = None,
        description: str | None = None,
        retrieval_metrics: dict[str, float] | None = None,
        generation_metrics: dict[str, float] | None = None,
        ragas_metrics: dict[str, float] | None = None,
        intent_metrics: dict[str, float] | None = None,
        business_metrics: dict[str, float] | None = None,
        latency_metrics: dict[str, int] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Create a new evaluation baseline.

        Args:
            baseline_name: Unique name for the baseline (e.g., "v1.0", "pre-finetune").
            model_version: Model version identifier.
            finetune_target: What was fine-tuned ('embedding', 'llm', 'intent', 'all').
            description: Human-readable description.
            retrieval_metrics: Dict with mrr5, ndcg5, hitrate3, precision5.
            generation_metrics: Dict with rouge_l, bleu4, bertscore.
            ragas_metrics: Dict with faithfulness, answer_relevance, context_relevance.
            intent_metrics: Dict with intent_accuracy, intent_f1_macro.
            business_metrics: Dict with accuracy, response_rate, no_hallucination_rate.
            latency_metrics: Dict with p50_ms, p95_ms, p99_ms.
            metadata: Additional metadata as JSON.

        Returns:
            int: The ID of the created baseline.

        Raises:
            psycopg2.errors.UniqueViolation: If baseline_name already exists.
        """
        retrieval_metrics = retrieval_metrics or {}
        generation_metrics = generation_metrics or {}
        ragas_metrics = ragas_metrics or {}
        intent_metrics = intent_metrics or {}
        business_metrics = business_metrics or {}
        latency_metrics = latency_metrics or {}
        metadata = metadata or {}

        query = sql.SQL("""
            INSERT INTO evaluation_baselines (
                baseline_name, model_version, finetune_target, description,
                retrieval_mrr5, retrieval_ndcg5, retrieval_hitrate3, retrieval_precision5,
                generation_rouge_l, generation_bleu4, generation_bertscore,
                ragas_faithfulness, ragas_answer_relevance, ragas_context_relevance,
                intent_accuracy, intent_f1_macro,
                business_accuracy, business_response_rate, business_no_hallucination_rate,
                latency_p50_ms, latency_p95_ms, latency_p99_ms,
                metadata
            ) VALUES (
                %s, %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s,
                %s, %s, %s,
                %s, %s,
                %s, %s, %s,
                %s, %s, %s,
                %s
            )
            RETURNING id
        """)

        conn = self._pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    query,
                    (
                        baseline_name,
                        model_version,
                        finetune_target,
                        description,
                        retrieval_metrics.get("mrr5"),
                        retrieval_metrics.get("ndcg5"),
                        retrieval_metrics.get("hitrate3"),
                        retrieval_metrics.get("precision5"),
                        generation_metrics.get("rouge_l"),
                        generation_metrics.get("bleu4"),
                        generation_metrics.get("bertscore"),
                        ragas_metrics.get("faithfulness"),
                        ragas_metrics.get("answer_relevance"),
                        ragas_metrics.get("context_relevance"),
                        intent_metrics.get("accuracy"),
                        intent_metrics.get("f1_macro"),
                        business_metrics.get("accuracy"),
                        business_metrics.get("response_rate"),
                        business_metrics.get("no_hallucination_rate"),
                        latency_metrics.get("p50_ms"),
                        latency_metrics.get("p95_ms"),
                        latency_metrics.get("p99_ms"),
                        json.dumps(metadata),
                    ),
                )
                result = cur.fetchone()
                conn.commit()
                return result[0] if result else None
        finally:
            self._pool.putconn(conn)

    def get_baseline(self, baseline_id: int | None = None, baseline_name: str | None = None) -> dict[str, Any] | None:
        """Retrieve a baseline by ID or name.

        Args:
            baseline_id: Baseline ID to retrieve.
            baseline_name: Baseline name to retrieve (alternative to ID).

        Returns:
            dict: Baseline data including all metrics, or None if not found.
        """
        if baseline_id:
            query = "SELECT * FROM evaluation_baselines WHERE id = %s"
            params = (baseline_id,)
        elif baseline_name:
            query = "SELECT * FROM evaluation_baselines WHERE baseline_name = %s"
            params = (baseline_name,)
        else:
            raise ValueError("Must provide either baseline_id or baseline_name")

        conn = self._pool.getconn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, params)
                return cur.fetchone()
        finally:
            self._pool.putconn(conn)

    def list_baselines(
        self, finetune_target: str | None = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        """List all baselines, optionally filtered by finetune_target.

        Args:
            finetune_target: Filter by finetune target.
            limit: Maximum number of results.

        Returns:
            list[dict]: List of baseline records.
        """
        if finetune_target:
            query = """
                SELECT * FROM evaluation_baselines
                WHERE finetune_target = %s
                ORDER BY created_at DESC
                LIMIT %s
            """
            params = (finetune_target, limit)
        else:
            query = """
                SELECT * FROM evaluation_baselines
                ORDER BY created_at DESC
                LIMIT %s
            """
            params = (limit,)

        conn = self._pool.getconn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, params)
                return cur.fetchall()
        finally:
            self._pool.putconn(conn)

    def update_baseline(
        self,
        baseline_id: int,
        **updates: Any,
    ) -> bool:
        """Update baseline fields.

        Args:
            baseline_id: ID of baseline to update.
            **updates: Fields to update (metrics, description, etc.).

        Returns:
            bool: True if update was successful.
        """
        if not updates:
            return True

        # Map Python field names to database columns
        field_map = {
            "description": "description",
            "model_version": "model_version",
            "finetune_target": "finetune_target",
        }

        set_clauses = []
        values = []

        for key, value in updates.items():
            if key in field_map:
                col = field_map[key]
                set_clauses.append(f"{col} = %s")
                values.append(value)

        if not set_clauses:
            return True

        values.append(baseline_id)
        query = f"UPDATE evaluation_baselines SET {', '.join(set_clauses)} WHERE id = %s"

        conn = self._pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(query, values)
                conn.commit()
                return cur.rowcount > 0
        finally:
            self._pool.putconn(conn)

    def delete_baseline(self, baseline_id: int) -> bool:
        """Delete a baseline.

        Args:
            baseline_id: ID of baseline to delete.

        Returns:
            bool: True if deletion was successful.
        """
        query = "DELETE FROM evaluation_baselines WHERE id = %s"

        conn = self._pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(query, (baseline_id,))
                conn.commit()
                return cur.rowcount > 0
        finally:
            self._pool.putconn(conn)

    # ==========================================================================
    # Evaluation Results CRUD
    # ==========================================================================

    def create_evaluation_result(
        self,
        eval_name: str,
        dataset_id: int | None = None,
        baseline_id: int | None = None,
        mode: str | None = None,
        finetune_target: str | None = None,
        retrieval_metrics: dict[str, Any] | None = None,
        generation_metrics: dict[str, Any] | None = None,
        ragas_metrics: dict[str, Any] | None = None,
        intent_metrics: dict[str, Any] | None = None,
        business_metrics: dict[str, Any] | None = None,
        latency_p50_ms: int | None = None,
        latency_p95_ms: int | None = None,
        latency_p99_ms: int | None = None,
        error_analysis: dict[str, Any] | None = None,
        comparison_mode: bool = False,
        delta_metrics: dict[str, Any] | None = None,
        significance_tests: dict[str, Any] | None = None,
        status: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Create a new evaluation result record.

        Args:
            eval_name: Name of the evaluation run.
            dataset_id: ID of the test dataset used.
            baseline_id: ID of the baseline for comparison.
            mode: Evaluation mode ('full', 'retrieval_only', etc.).
            finetune_target: What was fine-tuned.
            retrieval_metrics: JSONB of retrieval metrics.
            generation_metrics: JSONB of generation metrics.
            ragas_metrics: JSONB of RAGAS metrics.
            intent_metrics: JSONB of intent metrics.
            business_metrics: JSONB of business metrics.
            latency_p50_ms: P50 latency in milliseconds.
            latency_p95_ms: P95 latency in milliseconds.
            latency_p99_ms: P99 latency in milliseconds.
            error_analysis: JSONB of error analysis.
            comparison_mode: Whether this is a comparison evaluation.
            delta_metrics: JSONB of delta metrics vs baseline.
            significance_tests: JSONB of statistical significance test results.
            status: Overall status ('PASS', 'FAIL', 'IMPROVED', 'DEGRADED', etc.).
            metadata: Additional metadata.

        Returns:
            int: The ID of the created evaluation result.
        """
        query = sql.SQL("""
            INSERT INTO evaluation_results (
                eval_name, dataset_id, baseline_id, mode, finetune_target,
                retrieval_metrics, generation_metrics, ragas_metrics, intent_metrics, business_metrics,
                latency_p50_ms, latency_p95_ms, latency_p99_ms,
                error_analysis, comparison_mode, delta_metrics, significance_tests, status, metadata
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """)

        conn = self._pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    query,
                    (
                        eval_name,
                        dataset_id,
                        baseline_id,
                        mode,
                        finetune_target,
                        json.dumps(retrieval_metrics) if retrieval_metrics else None,
                        json.dumps(generation_metrics) if generation_metrics else None,
                        json.dumps(ragas_metrics) if ragas_metrics else None,
                        json.dumps(intent_metrics) if intent_metrics else None,
                        json.dumps(business_metrics) if business_metrics else None,
                        latency_p50_ms,
                        latency_p95_ms,
                        latency_p99_ms,
                        json.dumps(error_analysis) if error_analysis else None,
                        comparison_mode,
                        json.dumps(delta_metrics) if delta_metrics else None,
                        json.dumps(significance_tests) if significance_tests else None,
                        status,
                        json.dumps(metadata) if metadata else None,
                    ),
                )
                result = cur.fetchone()
                conn.commit()
                return result[0] if result else None
        finally:
            self._pool.putconn(conn)

    def get_evaluation_result(self, result_id: int) -> dict[str, Any] | None:
        """Retrieve an evaluation result by ID.

        Args:
            result_id: ID of the evaluation result.

        Returns:
            dict: Evaluation result data, or None if not found.
        """
        query = "SELECT * FROM evaluation_results WHERE id = %s"

        conn = self._pool.getconn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, (result_id,))
                return cur.fetchone()
        finally:
            self._pool.putconn(conn)

    def list_evaluation_results(
        self,
        eval_name: str | None = None,
        dataset_id: int | None = None,
        baseline_id: int | None = None,
        status: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """List evaluation results with optional filters.

        Args:
            eval_name: Filter by evaluation name.
            dataset_id: Filter by dataset ID.
            baseline_id: Filter by baseline ID.
            status: Filter by status.
            limit: Maximum number of results.

        Returns:
            list[dict]: List of evaluation result records.
        """
        conditions = []
        params = []
        param_count = 0

        if eval_name:
            param_count += 1
            conditions.append(f"eval_name = %s")
            params.append(eval_name)

        if dataset_id:
            param_count += 1
            conditions.append(f"dataset_id = %s")
            params.append(dataset_id)

        if baseline_id:
            param_count += 1
            conditions.append(f"baseline_id = %s")
            params.append(baseline_id)

        if status:
            param_count += 1
            conditions.append(f"status = %s")
            params.append(status)

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        query = f"""
            SELECT * FROM evaluation_results
            {where_clause}
            ORDER BY eval_at DESC
            LIMIT %s
        """
        params.append(limit)

        conn = self._pool.getconn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, params)
                return cur.fetchall()
        finally:
            self._pool.putconn(conn)

    def get_latest_evaluation_result(
        self, eval_name: str | None = None
    ) -> dict[str, Any] | None:
        """Get the most recent evaluation result.

        Args:
            eval_name: Optional filter by evaluation name.

        Returns:
            dict: Latest evaluation result, or None if none exist.
        """
        if eval_name:
            query = """
                SELECT * FROM evaluation_results
                WHERE eval_name = %s
                ORDER BY eval_at DESC
                LIMIT 1
            """
            params = (eval_name,)
        else:
            query = """
                SELECT * FROM evaluation_results
                ORDER BY eval_at DESC
                LIMIT 1
            """
            params = ()

        conn = self._pool.getconn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, params)
                return cur.fetchone()
        finally:
            self._pool.putconn(conn)

    # ==========================================================================
    # Test Datasets CRUD
    # ==========================================================================

    def create_test_dataset(
        self,
        dataset_name: str,
        dataset_type: str,
        data: list[dict[str, Any]] | dict[str, Any],
        version: str | None = None,
        description: str | None = None,
        stats: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Create a new test dataset.

        Args:
            dataset_name: Unique name for the dataset.
            dataset_type: Type of dataset ('qa', 'retrieval', 'intent').
            data: The actual test data (list of samples or nested dict).
            version: Optional version string.
            description: Human-readable description.
            stats: Statistics about the dataset (sample count, distribution, etc.).
            metadata: Additional metadata.

        Returns:
            int: The ID of the created dataset.

        Raises:
            psycopg2.errors.UniqueViolation: If dataset_name+version already exists.
        """
        query = sql.SQL("""
            INSERT INTO test_datasets (
                dataset_name, dataset_type, version, description, data, stats, metadata
            ) VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """)

        conn = self._pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    query,
                    (
                        dataset_name,
                        dataset_type,
                        version,
                        description,
                        json.dumps(data),
                        json.dumps(stats) if stats else None,
                        json.dumps(metadata) if metadata else None,
                    ),
                )
                result = cur.fetchone()
                conn.commit()
                return result[0] if result else None
        finally:
            self._pool.putconn(conn)

    def get_test_dataset(
        self, dataset_id: int | None = None, dataset_name: str | None = None, version: str | None = None
    ) -> dict[str, Any] | None:
        """Retrieve a test dataset by ID or name.

        Args:
            dataset_id: Dataset ID to retrieve.
            dataset_name: Dataset name to retrieve.
            version: Dataset version (required with dataset_name).

        Returns:
            dict: Dataset data, or None if not found.
        """
        if dataset_id:
            query = "SELECT * FROM test_datasets WHERE id = %s"
            params = (dataset_id,)
        elif dataset_name:
            if version:
                query = "SELECT * FROM test_datasets WHERE dataset_name = %s AND version = %s"
                params = (dataset_name, version)
            else:
                query = "SELECT * FROM test_datasets WHERE dataset_name = %s ORDER BY created_at DESC LIMIT 1"
                params = (dataset_name,)
        else:
            raise ValueError("Must provide either dataset_id or dataset_name")

        conn = self._pool.getconn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, params)
                return cur.fetchone()
        finally:
            self._pool.putconn(conn)

    def list_test_datasets(
        self, dataset_type: str | None = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        """List test datasets, optionally filtered by type.

        Args:
            dataset_type: Filter by dataset type ('qa', 'retrieval', 'intent').
            limit: Maximum number of results.

        Returns:
            list[dict]: List of dataset records (metadata only, not full data).
        """
        if dataset_type:
            query = """
                SELECT id, dataset_name, dataset_type, version, description,
                       created_at, stats, metadata
                FROM test_datasets
                WHERE dataset_type = %s
                ORDER BY created_at DESC
                LIMIT %s
            """
            params = (dataset_type, limit)
        else:
            query = """
                SELECT id, dataset_name, dataset_type, version, description,
                       created_at, stats, metadata
                FROM test_datasets
                ORDER BY created_at DESC
                LIMIT %s
            """
            params = (limit,)

        conn = self._pool.getconn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, params)
                return cur.fetchall()
        finally:
            self._pool.putconn(conn)

    def delete_test_dataset(self, dataset_id: int) -> bool:
        """Delete a test dataset.

        Args:
            dataset_id: ID of dataset to delete.

        Returns:
            bool: True if deletion was successful.
        """
        query = "DELETE FROM test_datasets WHERE id = %s"

        conn = self._pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(query, (dataset_id,))
                conn.commit()
                return cur.rowcount > 0
        finally:
            self._pool.putconn(conn)

    # ==========================================================================
    # Health Check
    # ==========================================================================

    def health_check(self) -> bool:
        """Check if database connection is healthy.

        Returns:
            bool: True if connection is healthy.
        """
        try:
            conn = self._pool.getconn()
            try:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    return cur.fetchone()[0] == 1
            finally:
                self._pool.putconn(conn)
        except Exception:
            return False
