"""
Airflow DAG for RAG Chatbot Evaluation Pipeline.

This DAG defines the complete evaluation workflow:
1. data_preparation - Load and validate test datasets
2. retrieval_evaluation - Evaluate retrieval quality (MRR, NDCG, HitRate)
3. generation_evaluation - Evaluate generation quality (ROUGE, BLEU, BERTScore)
4. baseline_comparison - Compare results against baseline (conditional)
5. report_generation - Generate evaluation reports (JSON/HTML/Markdown)

Supported DAG modes:
- rag_evaluation_daily: Daily monitoring (@daily)
- rag_evaluation_weekly: Weekly monitoring (@weekly)
- rag_evaluation_ft_compare: Manual trigger for finetune comparison

Trigger examples:
    # Daily evaluation
    airflow dags trigger rag_evaluation_daily

    # Weekly evaluation
    airflow dags trigger rag_evaluation_weekly

    # Finetune comparison with baseline
    airflow dags trigger rag_evaluation_ft_compare \
        --conf '{"baseline_name": "v1.0", "comparison_mode": true, "mode": "full"}'

    # Retrieval-only evaluation (for embedding finetune)
    airflow dags trigger rag_evaluation_ft_compare \
        --conf '{"baseline_name": "v1.0", "mode": "retrieval_only", "finetune_target": "embedding"}'
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timedelta, UTC
from pathlib import Path
from typing import Any

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.trigger_rule import TriggerRule

import sys

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.database import DatabaseManager
from src.evaluation.retrieval_eval import RetrievalEvaluator, RetrievalResult
from src.evaluation.generation_eval import GenerationEvaluator, GenerationResult
from src.evaluation.intent_eval import IntentEvaluator, IntentResult
from src.reporting.json_reporter import JSONReporter
from src.reporting.html_generator import HTMLGenerator
from src.reporting.delta_reporter import DeltaReporter
from src.dataset.loader import DatasetLoader


# =============================================================================
# Default Arguments
# =============================================================================

DEFAULT_ARGS = {
    "owner": "rag-team",
    "depends_on_past": False,
    "start_date": datetime(2026, 1, 1),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "catchup": False,
    "max_active_runs": 1,
}


# =============================================================================
# Task Functions
# =============================================================================

def prepare_data(**context) -> dict[str, Any]:
    """Load and validate test dataset.

    Pushes dataset_id to XCom for downstream tasks.

    Args:
        **context: Airflow context with dag_run.config containing:
            - dataset_name: Name of dataset to load
            - dataset_version: Optional version string
            - dataset_type: Type ('qa', 'retrieval', 'intent')

    Returns:
        dict: Dataset metadata including id and stats.
    """
    dag_run = context.get("dag_run")
    conf = dag_run.conf if dag_run else {}

    # Load configuration
    config_path = os.getenv("EVAL_CONFIG_PATH", "configs/evaluation_config.yaml")
    config = load_config(config_path)

    # Get dataset parameters from config or DAG run config
    dataset_name = conf.get("dataset_name") or config.dataset.name
    dataset_version = conf.get("dataset_version") or config.dataset.version
    dataset_type = conf.get("dataset_type") or config.dataset.type

    # Initialize database
    db = DatabaseManager(
        config.database.dsn,
        min_size=config.database.pool_min,
        max_size=config.database.pool_max,
    )

    # Try to load existing dataset from database
    dataset = db.get_test_dataset(dataset_name=dataset_name, version=dataset_version)

    if dataset:
        print(f"Loaded existing dataset: {dataset_name} v{dataset_version} (id={dataset['id']})")
        dataset_id = dataset["id"]
        data = dataset.get("data", [])
    else:
        # Load from file if not in database
        file_path = conf.get("file_path") or config.dataset.file_path
        loader = DatasetLoader()
        data, info = loader.load_jsonl(file_path)
        stats = info.stats

        # Validate and save to database
        dataset_id = db.create_test_dataset(
            dataset_name=dataset_name,
            dataset_type=dataset_type,
            data=data,
            version=dataset_version,
            stats=stats,
        )
        print(f"Created new dataset: {dataset_name} v{dataset_version} (id={dataset_id})")

    db.close()

    # Push dataset_id to XCom
    context["task_instance"].xcom_push(key="dataset_id", value=dataset_id)

    return {
        "dataset_id": dataset_id,
        "dataset_name": dataset_name,
        "dataset_version": dataset_version,
        "dataset_type": dataset_type,
        "sample_count": len(data) if isinstance(data, list) else len(data.get("samples", [])),
    }


def evaluate_retrieval(**context) -> dict[str, Any]:
    """Evaluate retrieval quality using MRR, NDCG, HitRate, Precision.

    Pushes retrieval_metrics to XCom.

    Args:
        **context: Airflow context.

    Returns:
        dict: Retrieval evaluation results.
    """
    dag_run = context.get("dag_run")
    conf = dag_run.conf if dag_run else {}
    ti = context["task_instance"]

    # Get dataset_id from XCom
    dataset_id = ti.xcom_pull(task_ids="data_preparation", key="dataset_id")

    # Load configuration
    config_path = os.getenv("EVAL_CONFIG_PATH", "configs/evaluation_config.yaml")
    config = load_config(config_path)

    # Initialize database and get dataset
    db = DatabaseManager(config.database.dsn)
    dataset = db.get_test_dataset(dataset_id=dataset_id)
    test_data = dataset.get("data", [])

    # Initialize retrieval evaluator
    from src.api.qdrant import QdrantClient
    qdrant_client = QdrantClient(
        base_url=config.qdrant.base_url,
        collection_name=config.qdrant.collection_name,
    )

    evaluator = RetrievalEvaluator(
        config={
            "mrr_k": 5,
            "ndcg_k": 5,
            "hitrate_k": 3,
            "precision_k": 5,
        },
        qdrant_client=qdrant_client,
    )

    # Run evaluation (async function)
    result: RetrievalResult = asyncio.run(evaluator.evaluate(test_data))

    # Convert to dict format for storage
    metrics = {
        "mrr_at_5": result.mrr_at_5,
        "ndcg_at_5": result.ndcg_at_5,
        "hitrate_at_3": result.hitrate_at_3,
        "precision_at_5": result.precision_at_5,
        "per_query_results": result.per_query_results,
    }

    # Push to XCom
    ti.xcom_push(key="retrieval_metrics", value=metrics)

    db.close()

    return metrics


def evaluate_generation(**context) -> dict[str, Any]:
    """Evaluate generation quality using ROUGE, BLEU, BERTScore.

    Pushes generation_metrics to XCom.

    Args:
        **context: Airflow context.

    Returns:
        dict: Generation evaluation results.
    """
    dag_run = context.get("dag_run")
    conf = dag_run.conf if dag_run else {}
    ti = context["task_instance"]

    # Get dataset_id from XCom
    dataset_id = ti.xcom_pull(task_ids="data_preparation", key="dataset_id")

    # Load configuration
    config_path = os.getenv("EVAL_CONFIG_PATH", "configs/evaluation_config.yaml")
    config = load_config(config_path)

    # Initialize database and get dataset
    db = DatabaseManager(config.database.dsn)
    dataset = db.get_test_dataset(dataset_id=dataset_id)
    test_data = dataset.get("data", [])

    # Initialize generation evaluator
    evaluator = GenerationEvaluator(
        config={
            "api_base_url": config.chatbot_api.base_url,
            "timeout_ms": config.chatbot_api.timeout_ms,
            "llm_judge": {
                "model": config.llm_judge.model,
                "temperature": config.llm_judge.temperature,
                "api_key": config.llm_judge.api_key,
            },
        }
    )

    # Run evaluation (async function)
    result: GenerationResult = asyncio.run(evaluator.evaluate(test_data))

    # Convert to dict format for storage
    metrics = {
        "avg_relevance": result.avg_relevance,
        "avg_fluency": result.avg_fluency,
        "avg_completeness": result.avg_completeness,
        "avg_safety": result.avg_safety,
        "overall_score": result.overall_score,
        "per_response_results": result.per_response_results,
    }

    # Push to XCom
    ti.xcom_push(key="generation_metrics", value=metrics)

    db.close()

    return metrics


def compare_baseline(**context) -> dict[str, Any] | None:
    """Compare current results with baseline (conditional task).

    Only executes when comparison_mode=True in DAG config.

    Pushes delta_metrics and status to XCom.

    Args:
        **context: Airflow context with dag_run.config containing:
            - comparison_mode: bool, whether to run comparison
            - baseline_name: str, name of baseline to compare against

    Returns:
        dict | None: Delta metrics if comparison run, None otherwise.
    """
    dag_run = context.get("dag_run")
    conf = dag_run.conf if dag_run else {}
    ti = context["task_instance"]

    # Check if comparison mode is enabled
    comparison_mode = conf.get("comparison_mode", False)
    if not comparison_mode:
        print("Comparison mode disabled, skipping baseline comparison")
        ti.xcom_push(key="comparison_skipped", value=True)
        return None

    # Get baseline name from config
    baseline_name = conf.get("baseline_name")
    if not baseline_name:
        print("No baseline_name specified, skipping comparison")
        ti.xcom_push(key="comparison_skipped", value=True)
        return None

    # Load configuration
    config_path = os.getenv("EVAL_CONFIG_PATH", "configs/evaluation_config.yaml")
    config = load_config(config_path)

    # Initialize database
    db = DatabaseManager(config.database.dsn)

    # Load baseline
    baseline = db.get_baseline(baseline_name=baseline_name)
    if not baseline:
        raise ValueError(f"Baseline '{baseline_name}' not found in database")

    # Get current metrics from XCom
    retrieval_metrics = ti.xcom_pull(task_ids="retrieval_evaluation", key="retrieval_metrics")
    generation_metrics = ti.xcom_pull(task_ids="generation_evaluation", key="generation_metrics")

    # Initialize delta reporter
    reporter = DeltaReporter(config=config)

    # Compute delta metrics
    delta_metrics = reporter.compute_delta(
        baseline_metrics={
            "retrieval": {
                "mrr_at_5": baseline.get("retrieval_mrr5"),
                "ndcg_at_5": baseline.get("retrieval_ndcg5"),
                "hitrate_at_3": baseline.get("retrieval_hitrate3"),
                "precision_at_5": baseline.get("retrieval_precision5"),
            },
            "generation": {
                "relevance": baseline.get("generation_rouge_l"),  # Mapping field
                "fluency": baseline.get("generation_bleu4"),
                "completeness": baseline.get("generation_bertscore"),
            },
        },
        current_metrics={
            "retrieval": retrieval_metrics or {},
            "generation": generation_metrics or {},
        },
    )

    # Determine overall status
    status = reporter.determine_status(delta_metrics, config.metric_thresholds)

    result = {
        "baseline_id": baseline["id"],
        "baseline_name": baseline_name,
        "delta_metrics": delta_metrics,
        "status": status,
    }

    # Push to XCom
    ti.xcom_push(key="delta_metrics", value=delta_metrics)
    ti.xcom_push(key="comparison_status", value=status)
    ti.xcom_push(key="comparison_skipped", value=False)

    db.close()

    return result


def generate_report(**context) -> dict[str, Any]:
    """Generate evaluation report (JSON, HTML, Markdown).

    Args:
        **context: Airflow context.

    Returns:
        dict: Report metadata including file paths.
    """
    dag_run = context.get("dag_run")
    conf = dag_run.conf if dag_run else {}
    ti = context["task_instance"]

    # Get evaluation mode
    eval_mode = conf.get("mode", "full")
    eval_name = conf.get("eval_name", f"rag_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    # Load configuration
    config_path = os.getenv("EVAL_CONFIG_PATH", "configs/evaluation_config.yaml")
    config = load_config(config_path)

    # Initialize database
    db = DatabaseManager(config.database.dsn)

    # Collect all metrics from XCom
    dataset_info = ti.xcom_pull(task_ids="data_preparation")
    retrieval_metrics = ti.xcom_pull(task_ids="retrieval_evaluation", key="retrieval_metrics")
    generation_metrics = ti.xcom_pull(task_ids="generation_evaluation", key="generation_metrics")
    delta_result = ti.xcom_pull(task_ids="baseline_comparison")

    # Determine baseline info
    baseline_id = None
    comparison_mode = False
    delta_metrics = None
    status = "COMPLETED"

    if delta_result and not delta_result.get("comparison_skipped"):
        baseline_id = delta_result.get("baseline_id")
        comparison_mode = True
        delta_metrics = delta_result.get("delta_metrics")
        status = delta_result.get("status", "COMPLETED")

    # Build evaluation result
    eval_result = {
        "eval_name": eval_name,
        "eval_at": datetime.now(UTC).isoformat(),
        "dataset_id": dataset_info.get("dataset_id"),
        "mode": eval_mode,
        "finetune_target": conf.get("finetune_target"),
        "retrieval_metrics": retrieval_metrics,
        "generation_metrics": generation_metrics,
        "comparison_mode": comparison_mode,
        "delta_metrics": delta_metrics,
        "status": status,
    }

    # Save to database
    result_id = db.create_evaluation_result(
        eval_name=eval_name,
        dataset_id=dataset_info.get("dataset_id"),
        baseline_id=baseline_id,
        mode=eval_mode,
        finetune_target=conf.get("finetune_target"),
        retrieval_metrics=retrieval_metrics,
        generation_metrics=generation_metrics,
        comparison_mode=comparison_mode,
        delta_metrics=delta_metrics,
        status=status,
    )

    # Generate JSON report (strip per-query/response data for summary)
    output_dir = Path("outputs/reports")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract top-level metrics only (no per-query lists)
    retrieval_summary = {k: v for k, v in (retrieval_metrics or {}).items() if k != "per_query_results"}
    generation_summary = {k: v for k, v in (generation_metrics or {}).items() if k != "per_response_results"}

    json_path = output_dir / f"{eval_name}.json"
    reporter = JSONReporter(eval_name=eval_name, dataset_info={"dataset_id": dataset_info.get("dataset_id")})
    report_data = reporter.generate_report(
        retrieval_metrics=retrieval_summary,
        generation_metrics=generation_summary,
        delta_metrics=delta_metrics,
    )
    reporter.save_report(report_data, json_path)

    # Generate HTML report
    html_generator = HTMLGenerator(eval_name=eval_name, dataset_info={"dataset_id": dataset_info.get("dataset_id")})
    html_path = output_dir / f"{eval_name}.html"
    html_content = html_generator.generate_report(
        retrieval_metrics=retrieval_summary,
        generation_metrics=generation_summary,
        delta_metrics=delta_metrics,
    )
    html_generator.save_report(html_content, html_path)

    db.close()

    return {
        "result_id": result_id,
        "eval_name": eval_name,
        "json_report": str(json_path),
        "html_report": str(html_path),
        "status": status,
    }


# =============================================================================
# DAG Creation Functions
# =============================================================================

def create_dag(
    dag_id: str,
    schedule_interval: str | None,
    description: str,
    tags: list[str] | None = None,
) -> DAG:
    """Create RAG evaluation DAG with specified schedule.

    Args:
        dag_id: Unique DAG identifier.
        schedule_interval: Cron expression or preset (e.g., '@daily', None).
        description: DAG description.
        tags: List of tags for DAG organization.

    Returns:
        DAG: Configured Airflow DAG.
    """
    tags = tags or ["rag", "evaluation", "chatbot"]

    dag = DAG(
        dag_id=dag_id,
        description=description,
        schedule_interval=schedule_interval,
        default_args=DEFAULT_ARGS,
        tags=tags,
    )

    # Define tasks
    data_prep_task = PythonOperator(
        task_id="data_preparation",
        python_callable=prepare_data,
        dag=dag,
    )

    retrieval_eval_task = PythonOperator(
        task_id="retrieval_evaluation",
        python_callable=evaluate_retrieval,
        dag=dag,
    )

    generation_eval_task = PythonOperator(
        task_id="generation_evaluation",
        python_callable=evaluate_generation,
        dag=dag,
    )

    baseline_compare_task = PythonOperator(
        task_id="baseline_comparison",
        python_callable=compare_baseline,
        dag=dag,
    )

    report_gen_task = PythonOperator(
        task_id="report_generation",
        python_callable=generate_report,
        dag=dag,
    )

    # Define task dependencies
    # data_preparation → retrieval_evaluation
    # data_preparation → generation_evaluation
    # retrieval_evaluation → baseline_comparison
    # generation_evaluation → baseline_comparison
    # baseline_comparison → report_generation
    # (baseline_comparison can be skipped, so report_generation uses ONE_SUCCESS)
    data_prep_task >> [retrieval_eval_task, generation_eval_task]
    [retrieval_eval_task, generation_eval_task] >> baseline_compare_task
    baseline_compare_task >> report_gen_task

    return dag


# =============================================================================
# DAG Instances
# =============================================================================

# Daily evaluation DAG for monitoring
rag_evaluation_daily = create_dag(
    dag_id="rag_evaluation_daily",
    schedule_interval="@daily",
    description="Daily RAG evaluation monitoring",
    tags=["rag", "evaluation", "chatbot", "daily"],
)

# Weekly evaluation DAG for trending
rag_evaluation_weekly = create_dag(
    dag_id="rag_evaluation_weekly",
    schedule_interval="@weekly",
    description="Weekly RAG evaluation trending",
    tags=["rag", "evaluation", "chatbot", "weekly"],
)

# Manual trigger DAG for finetune comparison
rag_evaluation_ft_compare = create_dag(
    dag_id="rag_evaluation_ft_compare",
    schedule_interval=None,  # Manual trigger only
    description="Manual trigger DAG for finetune comparison",
    tags=["rag", "evaluation", "chatbot", "finetune", "comparison"],
)
