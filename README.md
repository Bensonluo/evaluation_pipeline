# RAG Chatbot 端到端评测流水线

## 设计概述

本评测流水线用于评估 RAG Chatbot (智能客服) 的质量,重点关注**模型微调前后的指标变化**。

### 评测维度

| 维度 | 指标 | 说明 |
|------|------|------|
| **检索质量** | MRR@5, NDCG@5, HitRate@K, Context Precision/Recall | 向量检索准确性 |
| **生成质量** | ROUGE-L, BLEU-4, BERTScore, RAGAS (Faithfulness, Answer Relevance) | 回答生成质量 |
| **意图识别** | Intent Accuracy, Intent F1 (Macro) | 分类准确性 |
| **延迟指标** | P50/P95/P99 Latency, Cost per Query | 性能与成本 |
| **端到端质量** | 准确率, 回复率, 无幻觉率 | 业务核心指标 |
| **微调效果** | Delta Metrics + 统计显著性检验 | 微调前后指标变化 |

### Pipeline 架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Airflow DAG Schedule                               │
│                         (daily / weekly / on-demand)                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  1. data_preparation                                                        │
│     - 加载测试数据集 (Q&A pairs, 检索标注)                                   │
│     - 数据验证和格式转换                                                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  2. retrieval_evaluation                                                    │
│     - 对每个 Query 执行检索                                                  │
│     - 计算 MRR, NDCG, HitRate                                               │
│     - 保存检索结果用于生成评测                                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  3. generation_evaluation                                                   │
│     - 使用 Chatbot API 生成回答                                              │
│     - 计算 ROUGE, BLEU, BERTScore                                           │
│     - 人工标注结果(如提供)                                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  4. baseline_comparison  (Branch: 微调模式才执行)                            │
│     - 加载历史基线数据                                                       │
│     - 计算 Delta Metrics (微调后 vs 微调前)                                  │
│     - 判断指标提升是否显著                                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  5. report_generation                                                        │
│     - 生成 HTML/JSON 报告                                                    │
│     - 上传指标到 MLflow/Weights & Biases                                    │
│     - 告警(如指标下降)                                                       │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 文件结构

```
evaluation_pipeline/
├── src/                              # 核心源码
│   ├── config.py                     # 配置管理
│   ├── database.py                   # PostgreSQL 连接
│   ├── metrics/                      # 指标计算
│   │   ├── retrieval.py              # 检索指标 (MRR, NDCG, HitRate)
│   │   ├── generation.py             # 生成指标 (ROUGE, BLEU, BERTScore)
│   │   ├── ragas.py                  # RAGAS 评估 (Faithfulness, Relevance)
│   │   └── statistics.py             # 统计分析 (Bootstrap, p-value)
│   ├── dataset/                      # 数据集管理
│   │   ├── loader.py                 # 数据加载
│   │   ├── sampler.py                # 分层采样
│   │   └── converter.py              # 格式转换
│   ├── api/                          # API 客户端
│   │   ├── chatbot.py                # Chatbot API
│   │   └── qdrant.py                 # Qdrant 检索
│   ├── evaluation/                   # 评测执行
│   │   ├── retrieval_eval.py         # 检索评测
│   │   ├── generation_eval.py        # 生成评测
│   │   ├── intent_eval.py            # 意图评测
│   │   └── error_analyzer.py         # 错误分析
│   └── reporting/                    # 报告生成
│       ├── delta_reporter.py         # Delta 变化报告
│       └── html_generator.py          # HTML 报告
├── dags/                             # Airflow DAG
│   └── rag_evaluation_dag.py
├── configs/                          # 配置文件
│   ├── default_metrics.yaml
│   ├── ragas_prompts.yaml            # RAGAS Prompt 模板
│   └── evaluation_config.yaml
├── sql/                              # 数据库
│   └── init_evaluation_db.sql
├── scripts/
│   └── run_evaluation.py             # 独立运行脚本
└── tests/
    ├── unit/
    └── integration/
```

## 快速开始

### 1. 安装依赖

```bash
cd evaluation_pipeline
pip install -r requirements.txt
```

### 2. 配置环境

```bash
export AIRFLOW_HOME=/path/to/airflow
export CHATBOT_API_URL=http://localhost:8000
export CHATBOT_API_KEY=your-api-key
```

### 3. 启动 Airflow

```bash
airflow webserver -p 8080 &
airflow scheduler &
```

### 4. 手动触发评测

```bash
airflow dags trigger rag_evaluation
```

## 微调效果评测

### 评测模式

| 模式 | 评测范围 | 使用场景 |
|------|----------|----------|
| `full` | 检索+生成+意图+业务 | 全面评测 |
| `retrieval_only` | 仅检索指标 | Embedding 微调后 |
| `generation_only` | 仅生成指标 | LLM 微调后 |
| `intent_only` | 仅意图指标 | 意图分类器微调后 |

### 对比模式

当设置 `comparison_mode=True` 时,pipeline 会:

1. **加载基线数据**: 从 `evaluation_baselines` 表读取历史评测结果
2. **执行当前评测**: 在新模型/配置下运行完整评测
3. **计算 Delta**: 对比每个指标的相对/绝对变化
4. **统计显著性检验**: Bootstrap 置信区间 + p-value
5. **生成报告**: 包含变化百分比和统计显著性分析

### Delta Metrics 示例

```
| Metric        | Baseline | Current | Delta  | p-value | Status  |
|---------------|----------|---------|--------|---------|---------|
| MRR@5         | 0.723    | 0.789   | +9.1%  | 0.023   | ✅ SIGNIFICANT_IMPROVED |
| NDCG@5        | 0.681    | 0.712   | +4.5%  | 0.045   | ✅ MARGINAL_IMPROVED |
| ROUGE-L       | 0.452    | 0.438   | -3.1%  | 0.120   | ➖ NOT_SIGNIFICANT |
| Accuracy      | 0.823    | 0.851   | +3.4%  | 0.031   | ✅ IMPROVED |
```

## 测试数据格式

### Q&A 测试集 (test_dataset.jsonl)

```json
{"query": "如何重置密码?", "intent": "HOW_TO", "ground_truth": "步骤1: 点击登录页的'忘记密码'...", "complexity": "SIMPLE"}
{"query": "你们的退款政策是什么?", "intent": "POLICY", "ground_truth": "我们支持7天内无理由退款...", "complexity": "SIMPLE"}
{"query": "产品A和竞品B比较有什么优势?", "intent": "PRODUCT", "ground_truth": "产品A相比竞品B的优势是...", "complexity": "COMPLEX"}
```

### 意图类型

| 意图 | 说明 |
|------|------|
| `HOW_TO` | 操作指南类 |
| `POLICY` | 政策规则类 |
| `PRODUCT` | 产品咨询类 |
| `BILLING` | 账单支付类 |
| `COMPLAINT` | 投诉建议类 |
| `REFUND` | 退款售后类 |

### 检索标注 (retrieval_labels.json)

```json
{"query": "产品A有哪些特性?", "relevant_docs": ["doc_id_1", "doc_id_3"]}
```

## 调度策略

| DAG ID | Schedule | 用途 |
|--------|----------|------|
| `rag_evaluation_daily` | `@daily` | 日常监控 |
| `rag_evaluation_weekly` | `@weekly` | 周度报告 |
| `rag_evaluation_ft_compare` | None (手动) | 微调对比实验 |

### 触发示例

```bash
# 全面评测
airflow dags trigger rag_evaluation --conf '{"mode": "full"}'

# 仅评测检索（Embedding 微调后）
airflow dags trigger rag_evaluation --conf '{"mode": "retrieval_only", "finetune_target": "embedding"}'

# 微调对比
airflow dags trigger rag_evaluation --conf '{"mode": "full", "baseline_name": "v1.0", "comparison_mode": true}'
```

## 延迟指标

| 指标 | 说明 | 阈值 |
|------|------|------|
| P50 Latency | 中位数响应时间 | < 300ms |
| P95 Latency | 95% 分位数 | < 1000ms |
| P99 Latency | 99% 分位数 | < 3000ms |
| Cost per Query | 每query成本 | < $0.003 |
