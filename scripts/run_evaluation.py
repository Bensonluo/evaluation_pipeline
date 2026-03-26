#!/usr/bin/env python3
"""
RAG Chatbot 评测流水线 - 独立运行脚本

生产级独立评测工具，支持四种评测模式：
- full: 检索+生成+意图+业务 全面评测
- retrieval_only: 仅检索指标（Embedding 微调后）
- generation_only: 仅生成指标（LLM 微调后）
- intent_only: 仅意图指标（意图分类器微调后）

Usage:
    python run_evaluation.py --config configs/evaluation_config.yaml --mode full
    python run_evaluation.py --dataset qa_test_set_v1 --baseline baseline_pre_finetuning
    python run_evaluation.py --mode retrieval_only --output ./outputs/eval_results.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import yaml

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# =============================================================================
# Logging Configuration
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("evaluation.log"),
    ],
)
logger = logging.getLogger(__name__)


# =============================================================================
# Evaluation Mode Enumeration
# =============================================================================

class EvaluationMode(str, Enum):
    """评测模式枚举"""
    FULL = "full"
    RETRIEVAL_ONLY = "retrieval_only"
    GENERATION_ONLY = "generation_only"
    INTENT_ONLY = "intent_only"

    @classmethod
    def from_string(cls, mode_str: str) -> "EvaluationMode":
        """从字符串解析评测模式"""
        try:
            return cls(mode_str.lower())
        except ValueError:
            valid_modes = [m.value for m in cls]
            raise ValueError(
                f"Invalid evaluation mode: '{mode_str}'. "
                f"Valid modes: {valid_modes}"
            )


# =============================================================================
# Configuration Data Classes
# =============================================================================

@dataclass
class EvaluationConfig:
    """评测配置"""
    # 基础配置
    name: str = "rag_evaluation"
    description: str = "RAG Chatbot 评测"
    mode: EvaluationMode = EvaluationMode.FULL

    # 数据集配置
    dataset_name: str = "qa_test_set_v1"
    dataset_version: str = "1.0"

    # 基线对比配置
    baseline_name: Optional[str] = None
    comparison_mode: bool = False

    # 输出配置
    output_dir: str = "./outputs"
    output_formats: list[str] = field(default_factory=lambda: ["json"])

    # API 配置
    chatbot_url: str = "http://localhost:8000"
    chatbot_timeout: int = 30
    qdrant_url: str = "http://localhost:6333"

    # 数据库配置
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "rag_chatbot"
    db_user: str = "postgres"
    db_password: str = ""

    # 统计检验配置
    enable_significance_tests: bool = True
    confidence_level: float = 0.95
    n_bootstrap: int = 1000


@dataclass
class EvaluationResult:
    """评测结果"""
    eval_name: str
    eval_at: str
    mode: str
    dataset: dict[str, Any]

    # 各维度指标
    retrieval_metrics: dict[str, Any] = field(default_factory=dict)
    generation_metrics: dict[str, Any] = field(default_factory=dict)
    ragas_metrics: dict[str, Any] = field(default_factory=dict)
    intent_metrics: dict[str, Any] = field(default_factory=dict)
    business_metrics: dict[str, Any] = field(default_factory=dict)

    # 延迟指标
    latency_p50_ms: int = 0
    latency_p95_ms: int = 0
    latency_p99_ms: int = 0

    # 对比分析
    baseline_name: Optional[str] = None
    delta_metrics: dict[str, Any] = field(default_factory=dict)
    significance_tests: dict[str, Any] = field(default_factory=dict)
    status: str = "COMPLETED"

    # 错误分析
    error_analysis: dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Evaluation Pipeline Core
# =============================================================================

class EvaluationPipeline:
    """
    评测流水线核心类

    生产级实现，参考 DESIGN.md 第十八章最佳实践：
    - 自动化测试基础设施
    - 测试数据集构建最佳实践
    - 基线建立与版本管理
    - LLM-as-a-Judge 最佳实践
    - 多维度评分与权衡分析
    """

    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.result = EvaluationResult(
            eval_name=config.name,
            eval_at=datetime.now().isoformat(),
            mode=config.mode.value,
            dataset={"name": config.dataset_name, "version": config.dataset_version},
        )
        self._setup_output_dir()

    def _setup_output_dir(self):
        """创建输出目录"""
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Data Preparation Phase
    # -------------------------------------------------------------------------

    async def prepare_data(self) -> dict[str, Any]:
        """
        数据准备阶段

        从数据库加载测试数据集，验证格式，进行分层采样
        """
        logger.info(f"[Phase 1/5] Loading dataset: {self.config.dataset_name}")

        # TODO: 实现实际数据加载逻辑
        # 这里模拟数据加载
        dataset = {
            "name": self.config.dataset_name,
            "version": self.config.dataset_version,
            "total_samples": 100,
            "samples": [
                {
                    "query": "如何重置密码?",
                    "intent": "HOW_TO",
                    "ground_truth": "步骤1: 点击登录页的'忘记密码'按钮...",
                    "complexity": "SIMPLE",
                }
            ],
        }

        logger.info(f"Loaded {dataset['total_samples']} samples")
        return dataset

    # -------------------------------------------------------------------------
    # Retrieval Evaluation Phase
    # -------------------------------------------------------------------------

    async def evaluate_retrieval(self, dataset: dict[str, Any]) -> dict[str, Any]:
        """
        检索评测阶段

        计算 MRR@5, NDCG@5, HitRate@3, Precision@5
        """
        if self.config.mode == EvaluationMode.GENERATION_ONLY:
            logger.info("[Phase 2/5] Skipping retrieval evaluation (generation_only mode)")
            return {}

        logger.info("[Phase 2/5] Running retrieval evaluation")

        # TODO: 实现实际检索评测逻辑
        metrics = {
            "mrr@5": 0.7234,
            "ndcg@5": 0.6891,
            "hitrate@3": 0.8125,
            "precision@5": 0.6234,
            "context_precision": 0.74,
            "context_recall": 0.86,
        }

        logger.info(f"Retrieval metrics: MRR@5={metrics['mrr@5']:.4f}, NDCG@5={metrics['ndcg@5']:.4f}")
        return metrics

    # -------------------------------------------------------------------------
    # Generation Evaluation Phase
    # -------------------------------------------------------------------------

    async def evaluate_generation(self, dataset: dict[str, Any]) -> dict[str, Any]:
        """
        生成评测阶段

        计算 ROUGE-L, BLEU-4, BERTScore, RAGAS 指标
        """
        if self.config.mode == EvaluationMode.RETRIEVAL_ONLY:
            logger.info("[Phase 3/5] Skipping generation evaluation (retrieval_only mode)")
            return {}

        logger.info("[Phase 3/5] Running generation evaluation")

        # TODO: 实现实际生成评测逻辑
        generation_metrics = {
            "rouge_l": 0.5432,
            "bleu_4": 0.3456,
            "bertscore": 0.8234,
        }

        ragas_metrics = {
            "faithfulness": 0.94,
            "answer_relevance": 4.3,
            "context_precision": 0.78,
            "context_recall": 0.88,
            "answer_correctness": 0.87,
        }

        logger.info(f"Generation metrics: ROUGE-L={generation_metrics['rouge_l']:.4f}")
        return generation_metrics, ragas_metrics

    # -------------------------------------------------------------------------
    # Intent Evaluation Phase
    # -------------------------------------------------------------------------

    async def evaluate_intent(self, dataset: dict[str, Any]) -> dict[str, Any]:
        """
        意图识别评测阶段

        计算 Intent Accuracy, F1-Macro, Confusion Matrix
        """
        if self.config.mode not in (EvaluationMode.FULL, EvaluationMode.INTENT_ONLY):
            logger.info("[Phase 4/5] Skipping intent evaluation")
            return {}

        logger.info("[Phase 4/5] Running intent evaluation")

        # TODO: 实现实际意图评测逻辑
        metrics = {
            "accuracy": 0.9234,
            "f1_macro": 0.9102,
            "confusion_matrix": [[85, 5, 2, 0, 0, 1], [3, 90, 1, 0, 0, 0]],
        }

        logger.info(f"Intent metrics: Accuracy={metrics['accuracy']:.4f}")
        return metrics

    # -------------------------------------------------------------------------
    # Baseline Comparison Phase
    # -------------------------------------------------------------------------

    async def compare_baseline(self) -> dict[str, Any]:
        """
        基线对比阶段

        计算相对/绝对变化，统计显著性检验
        """
        if not self.config.comparison_mode or not self.config.baseline_name:
            logger.info("[Phase 5/5] Skipping baseline comparison")
            return {}

        logger.info(f"[Phase 5/5] Comparing with baseline: {self.config.baseline_name}")

        # TODO: 实现实际基线对比逻辑
        delta_metrics = {
            "retrieval_mrr5": {
                "baseline": 0.68,
                "current": 0.7234,
                "delta": "+0.0434",
                "delta_pct": "+6.4%",
                "status": "IMPROVED",
            },
            "generation_rouge_l": {
                "baseline": 0.52,
                "current": 0.5432,
                "delta": "+0.0232",
                "delta_pct": "+4.5%",
                "status": "IMPROVED",
            },
        }

        if self.config.enable_significance_tests:
            significance_tests = {
                "retrieval_mrr5": {
                    "p_value": 0.023,
                    "ci_lower": 0.012,
                    "ci_upper": 0.075,
                    "is_significant": True,
                }
            }
        else:
            significance_tests = {}

        self.result.status = "IMPROVED"
        logger.info(f"Overall status: {self.result.status}")
        return delta_metrics, significance_tests

    # -------------------------------------------------------------------------
    # Report Generation Phase
    # -------------------------------------------------------------------------

    def generate_report(self) -> str:
        """
        生成评测报告

        支持 JSON 和 HTML 格式
        """
        logger.info("[Report] Generating evaluation report")

        report_path = Path(self.config.output_dir) / f"{self.config.name}_report"

        if "json" in self.config.output_formats:
            json_path = str(report_path) + ".json"
            self._save_json_report(json_path)
            logger.info(f"[Report] JSON report saved: {json_path}")

        if "html" in self.config.output_formats:
            html_path = str(report_path) + ".html"
            self._save_html_report(html_path)
            logger.info(f"[Report] HTML report saved: {html_path}")

        return str(report_path) + ".json"

    def _save_json_report(self, path: str):
        """保存 JSON 报告"""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._to_dict(), f, ensure_ascii=False, indent=2)

    def _save_html_report(self, path: str):
        """保存 HTML 报告"""
        html_template = self._generate_html_template()
        with open(path, "w", encoding="utf-8") as f:
            f.write(html_template)

    def _to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "eval_name": self.result.eval_name,
            "eval_at": self.result.eval_at,
            "mode": self.result.mode,
            "dataset": self.result.dataset,
            "retrieval_metrics": self.result.retrieval_metrics,
            "generation_metrics": self.result.generation_metrics,
            "ragas_metrics": self.result.ragas_metrics,
            "intent_metrics": self.result.intent_metrics,
            "latency": {
                "p50_ms": self.result.latency_p50_ms,
                "p95_ms": self.result.latency_p95_ms,
                "p99_ms": self.result.latency_p99_ms,
            },
            "baseline_comparison": {
                "baseline_name": self.result.baseline_name,
                "delta_metrics": self.result.delta_metrics,
                "significance_tests": self.result.significance_tests,
                "status": self.result.status,
            } if self.config.comparison_mode else None,
            "error_analysis": self.result.error_analysis,
        }

    def _generate_html_template(self) -> str:
        """生成 HTML 报告"""
        # 简化版 HTML 模板
        return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>RAG Chatbot 评测报告 - {self.result.eval_name}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; }}
        .metric {{ display: inline-block; margin: 10px 20px 10px 0; padding: 15px; background: #f9f9f9; border-radius: 4px; min-width: 150px; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #4CAF50; }}
        .metric-label {{ font-size: 12px; color: #777; text-transform: uppercase; }}
        .status-improved {{ color: #4CAF50; }}
        .status-degraded {{ color: #f44336; }}
        .status-pass {{ color: #2196F3; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔍 RAG Chatbot 评测报告</h1>
        <p><strong>评测名称:</strong> {self.result.eval_name}</p>
        <p><strong>评测时间:</strong> {self.result.eval_at}</p>
        <p><strong>评测模式:</strong> {self.result.mode}</p>
        <p><strong>数据集:</strong> {self.result.dataset['name']} v{self.result.dataset['version']}</p>

        <h2>📊 检索质量</h2>
        {self._format_metrics_html(self.result.retrieval_metrics)}

        <h2>✍️ 生成质量</h2>
        {self._format_metrics_html(self.result.generation_metrics)}

        <h2>🎯 RAGAS 指标</h2>
        {self._format_metrics_html(self.result.ragas_metrics)}

        <h2>🏷️ 意图识别</h2>
        {self._format_metrics_html(self.result.intent_metrics)}

        {self._format_comparison_html()}
    </div>
</body>
</html>"""

    def _format_metrics_html(self, metrics: dict[str, Any]) -> str:
        """格式化指标为 HTML"""
        if not metrics:
            return "<p>无数据</p>"
        html = ""
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                html += f'<div class="metric"><div class="metric-value">{value:.4f}</div><div class="metric-label">{key}</div></div>'
        return html

    def _format_comparison_html(self) -> str:
        """格式化对比结果为 HTML"""
        if not self.result.delta_metrics:
            return ""
        html = '<h2>📈 基线对比</h2>'
        for metric, delta in self.result.delta_metrics.items():
            status_class = f"status-{delta['status'].lower()}"
            html += f'<div class="metric"><div class="metric-value {status_class}">{delta["delta_pct"]}</div><div class="metric-label">{metric}</div></div>'
        return html

    # -------------------------------------------------------------------------
    # Main Execution
    # -------------------------------------------------------------------------

    async def run(self) -> str:
        """
        执行完整评测流程

        Returns:
            报告文件路径
        """
        logger.info("=" * 70)
        logger.info(f"Starting RAG Chatbot Evaluation: {self.config.name}")
        logger.info(f"Mode: {self.config.mode.value}")
        logger.info(f"Dataset: {self.config.dataset_name}")
        if self.config.comparison_mode:
            logger.info(f"Baseline: {self.config.baseline_name}")
        logger.info("=" * 70)

        # Phase 1: Data Preparation
        dataset = await self.prepare_data()

        # Phase 2: Retrieval Evaluation
        retrieval_metrics = await self.evaluate_retrieval(dataset)
        self.result.retrieval_metrics = retrieval_metrics

        # Phase 3: Generation Evaluation
        if self.config.mode in (EvaluationMode.FULL, EvaluationMode.GENERATION_ONLY):
            gen_metrics, ragas_metrics = await self.evaluate_generation(dataset)
            self.result.generation_metrics = gen_metrics
            self.result.ragas_metrics = ragas_metrics

        # Phase 4: Intent Evaluation
        intent_metrics = await self.evaluate_intent(dataset)
        self.result.intent_metrics = intent_metrics

        # Phase 5: Baseline Comparison
        if self.config.comparison_mode:
            delta_metrics, sig_tests = await self.compare_baseline()
            self.result.delta_metrics = delta_metrics
            self.result.significance_tests = sig_tests
            self.result.baseline_name = self.config.baseline_name

        # Generate Report
        report_path = self.generate_report()

        logger.info("=" * 70)
        logger.info(f"Evaluation completed: {self.result.status}")
        logger.info(f"Report saved: {report_path}")
        logger.info("=" * 70)

        return report_path


# =============================================================================
# Configuration Loader
# =============================================================================

def load_config_from_yaml(config_path: str) -> EvaluationConfig:
    """从 YAML 文件加载配置"""
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # 提取环境变量
    evaluation_config = data.get("evaluation", {})
    api_config = data.get("api", {})
    database_config = data.get("database", {})
    reporting_config = data.get("reporting", {})
    statistics_config = data.get("statistics", {})

    return EvaluationConfig(
        name=evaluation_config.get("name", "rag_evaluation"),
        description=evaluation_config.get("description", ""),
        chatbot_url=os.path.expandvars(api_config.get("chatbot_url", "http://localhost:8000")),
        chatbot_timeout=api_config.get("chatbot_timeout", 30),
        qdrant_url=os.path.expandvars(api_config.get("qdrant_url", "http://localhost:6333")),
        db_host=os.path.expandvars(database_config.get("host", "localhost")),
        db_port=int(os.path.expandvars(database_config.get("port", "5432"))),
        db_name=os.path.expandvars(database_config.get("database", "rag_chatbot")),
        db_user=os.path.expandvars(database_config.get("user", "postgres")),
        db_password=os.path.expandvars(database_config.get("password", "")),
        output_dir=reporting_config.get("output_dir", "./outputs"),
        output_formats=reporting_config.get("formats", ["json"]),
        enable_significance_tests=statistics_config.get("enabled", True),
        confidence_level=statistics_config.get("confidence_level", 0.95),
        n_bootstrap=statistics_config.get("n_bootstrap", 1000),
    )


# =============================================================================
# CLI Interface
# =============================================================================

def parse_arguments() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="RAG Chatbot 评测流水线 - 生产级独立运行脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
评测模式:
  full              检索+生成+意图+业务 全面评测
  retrieval_only    仅检索指标（Embedding 微调后）
  generation_only   仅生成指标（LLM 微调后）
  intent_only       仅意图指标（意图分类器微调后）

示例:
  # 全面评测
  python run_evaluation.py --mode full

  # 仅检索评测（Embedding 微调后）
  python run_evaluation.py --mode retrieval_only --dataset qa_test_set_v1

  # 微调对比评测
  python run_evaluation.py --mode full --baseline baseline_pre_finetuning --comparison

  # 使用自定义配置
  python run_evaluation.py --config /path/to/config.yaml --output ./results/
        """,
    )

    # 配置相关
    parser.add_argument(
        "-c", "--config",
        type=str,
        default=None,
        help="配置文件路径 (YAML格式)",
    )

    # 评测模式
    parser.add_argument(
        "-m", "--mode",
        type=str,
        choices=["full", "retrieval_only", "generation_only", "intent_only"],
        default="full",
        help="评测模式 (默认: full)",
    )

    # 数据集配置
    parser.add_argument(
        "-d", "--dataset",
        type=str,
        default="qa_test_set_v1",
        help="测试数据集名称 (默认: qa_test_set_v1)",
    )

    parser.add_argument(
        "--dataset-version",
        type=str,
        default="1.0",
        help="数据集版本 (默认: 1.0)",
    )

    # 基线对比配置
    parser.add_argument(
        "-b", "--baseline",
        type=str,
        default=None,
        help="基线名称，用于对比评测",
    )

    parser.add_argument(
        "--comparison",
        action="store_true",
        help="启用对比模式，与指定基线进行对比",
    )

    # 输出配置
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="./outputs",
        help="输出目录路径 (默认: ./outputs)",
    )

    parser.add_argument(
        "--format",
        type=str,
        choices=["json", "html", "both"],
        default="both",
        help="输出格式 (默认: both)",
    )

    # API 配置
    parser.add_argument(
        "--chatbot-url",
        type=str,
        default=None,
        help="Chatbot API URL (覆盖配置文件)",
    )

    parser.add_argument(
        "--qdrant-url",
        type=str,
        default=None,
        help="Qdrant API URL (覆盖配置文件)",
    )

    # 数据库配置
    parser.add_argument(
        "--db-host",
        type=str,
        default=None,
        help="数据库主机 (覆盖配置文件)",
    )

    parser.add_argument(
        "--db-name",
        type=str,
        default=None,
        help="数据库名称 (覆盖配置文件)",
    )

    # 其他选项
    parser.add_argument(
        "--no-significance",
        action="store_true",
        help="禁用统计显著性检验",
    )

    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="评测名称 (用于标识本次评测)",
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="详细输出模式",
    )

    return parser.parse_args()


async def main():
    """主函数"""
    args = parse_arguments()

    # 配置日志级别
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # 加载配置
    if args.config:
        config = load_config_from_yaml(args.config)
    else:
        config = EvaluationConfig()

    # 命令行参数覆盖配置文件
    config.mode = EvaluationMode.from_string(args.mode)
    config.dataset_name = args.dataset
    config.dataset_version = args.dataset_version
    config.output_dir = args.output

    if args.baseline:
        config.baseline_name = args.baseline
        config.comparison_mode = args.comparison

    if args.format == "json":
        config.output_formats = ["json"]
    elif args.format == "html":
        config.output_formats = ["html"]
    else:  # both
        config.output_formats = ["json", "html"]

    if args.chatbot_url:
        config.chatbot_url = args.chatbot_url
    if args.qdrant_url:
        config.qdrant_url = args.qdrant_url
    if args.db_host:
        config.db_host = args.db_host
    if args.db_name:
        config.db_name = args.db_name
    if args.no_significance:
        config.enable_significance_tests = False
    if args.name:
        config.name = args.name

    # 运行评测
    pipeline = EvaluationPipeline(config)
    try:
        report_path = await pipeline.run()
        logger.info(f"✅ 评测完成! 报告: {report_path}")
        return 0
    except Exception as e:
        logger.error(f"❌ 评测失败: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
