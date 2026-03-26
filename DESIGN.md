# RAG Chatbot 评测流水线架构设计

## 一、业务背景

- **目标**: 评测 RAG Chatbot (智能客服) 的质量
- **核心需求**: 评估模型微调前后的指标变化
- **触发方式**: 手动触发

---

## 二、评测指标体系

### 2.1 检索阶段 (Retrieval)

| 指标 | 计算方式 | 说明 |
|------|----------|------|
| **MRR@5** | MRR = Σ(1/rank_i)/N | 平均倒数排名，检索结果相关性 |
| **NDCG@5** | NDCG = DCG/IDCG | 归一化折损累计增益，排序质量 |
| **HitRate@3** | HitRate = Hits/N | Top3 结果包含正确答案的比例 |
| **Precision@5** | P@K = Relevant∩Retrieved/K | 精确率 |

### 2.2 生成阶段 (Generation)

| 指标 | 说明 | 业界使用 |
|------|------|---------|
| **ROUGE-L** | 最长公共子序列匹配率 | NLP 标准，摘要/问答 |
| **BLEU-4** | 4-gram 精确率的加权组合 | 机器翻译标配 |
| **BERTScore** | 基于 BERT 的语义相似度 | 上下文感知评价 |

### 2.3 端到端 RAG 评测 (RAGAS Framework)

| 指标 | 含义 | 业务价值 |
|------|------|----------|
| **Faithfulness** | 回答是否忠于检索内容 (1-5分) | 幻觉检测，客服准确性 |
| **Answer Relevance** | 回答与问题的相关性 (1-5分) | 回答有效性 |
| **Context Relevance** | 检索上下文与问题的相关性 (1-5分) | 检索质量 |

### 2.4 业务指标

| 指标 | 说明 |
|------|------|
| **准确率** | 人工标注：回答是否正确解决用户问题 |
| **回复率** | 能给出有效回答的比例（非 "不知道"） |
| **无幻觉率** | Faithfulness >= 3 的比例 |

---

## 三、基线数据管理

### 3.1 数据库设计

**PostgreSQL 表结构**

```sql
-- 评测基线表：存储每次微调后的基准指标
CREATE TABLE evaluation_baselines (
    id              SERIAL PRIMARY KEY,
    baseline_name   VARCHAR(100) NOT NULL,      -- 如 "v1.0", "pre-finetune"
    model_version   VARCHAR(100),                -- 模型版本标识
    description     TEXT,
    created_at      TIMESTAMP DEFAULT NOW(),

    -- 检索指标
    retrieval_mrr5          DECIMAL(5,4),
    retrieval_ndcg5         DECIMAL(5,4),
    retrieval_hitrate3      DECIMAL(5,4),
    retrieval_precision5    DECIMAL(5,4),

    -- 生成指标
    generation_rouge_l      DECIMAL(5,4),
    generation_bleu4        DECIMAL(5,4),
    generation_bertscore    DECIMAL(5,4),

    -- RAGAS 指标
    ragas_faithfulness      DECIMAL(3,2),
    ragas_answer_relevance   DECIMAL(3,2),
    ragas_context_relevance  DECIMAL(3,2),

    -- 业务指标
    business_accuracy        DECIMAL(5,4),
    business_response_rate  DECIMAL(5,4),
    business_no_hallucination_rate DECIMAL(5,4),

    metadata                JSONB,                -- 额外元数据
    UNIQUE(baseline_name)
);

-- 评测结果表：每次评测的详细结果
CREATE TABLE evaluation_results (
    id              SERIAL PRIMARY KEY,
    baseline_id     INTEGER REFERENCES evaluation_baselines(id),
    eval_name       VARCHAR(100) NOT NULL,
    eval_at         TIMESTAMP DEFAULT NOW(),

    -- 各维度指标汇总 (JSONB 存储详细结果)
    retrieval_metrics   JSONB,
    generation_metrics  JSONB,
    ragas_metrics       JSONB,
    business_metrics    JSONB,

    -- 微调对比模式
    comparison_mode      BOOLEAN DEFAULT FALSE,
    delta_metrics        JSONB,                   -- 相比基线的变化
    status              VARCHAR(20),              -- PASS/FAIL/IMPROVED/DEGRADED
    metadata            JSONB
);

-- 测试数据集表
CREATE TABLE test_datasets (
    id              SERIAL PRIMARY KEY,
    dataset_name     VARCHAR(100) NOT NULL,
    dataset_type     VARCHAR(20),                  -- 'qa', 'retrieval', 'human_annotated'
    description      TEXT,
    created_at       TIMESTAMP DEFAULT NOW(),
    data             JSONB NOT NULL,               -- 测试数据内容
    metadata         JSONB
);
```

---

## 四、Pipeline 架构

### 4.1 Airflow DAG 设计

**触发方式**: 手动触发 (`airflow dags trigger rag_evaluation`)

**DAG 类型**:
1. `rag_evaluation_daily` - 日常监控
2. `rag_evaluation_ft_compare` - 微调对比实验 (主要)

### 4.2 Pipeline 流程

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Step 1: data_preparation                                                   │
│  ─────────────────────────────────────────────────────────────────────────  │
│  输入:  test_dataset.jsonl (Q&A pairs)                                      │
│        retrieval_labels.json (检索标注)                                      │
│  输出:  validated_test_data (存入 test_datasets 表)                         │
│  任务:                                                                       │
│    - 加载并验证测试数据格式                                                  │
│    - 数据清洗和预处理                                                        │
│    - 写入 PostgreSQL test_datasets 表                                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Step 2: retrieval_evaluation                                               │
│  ─────────────────────────────────────────────────────────────────────────  │
│  输入:  test_dataset (Query)                                                 │
│  输出:  retrieval_results (MRR, NDCG, HitRate)                              │
│  任务:                                                                       │
│    - 对每个 Query 执行向量检索 (Qdrant)                                      │
│    - 计算 MRR@5, NDCG@5, HitRate@3, Precision@5                            │
│    - 保存检索结果                                                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Step 3: generation_evaluation                                               │
│  ─────────────────────────────────────────────────────────────────────────  │
│  输入:  retrieval_results + chatbot_api                                     │
│  输出:  generation_results (ROUGE, BLEU, BERTScore)                         │
│  任务:                                                                       │
│    - 调用 Chatbot API 生成回答                                               │
│    - 计算 ROUGE-L, BLEU-4, BERTScore                                        │
│    - (可选) 调用 LLM 评估 RAGAS 指标                                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Step 4: baseline_comparison  [仅 comparison_mode=True 时执行]              │
│  ─────────────────────────────────────────────────────────────────────────  │
│  输入:  current_results + historical_baseline                                │
│  输出:  delta_metrics                                                        │
│  任务:                                                                       │
│    - 从 evaluation_baselines 表加载指定基线                                  │
│    - 计算每个指标的 Delta (相对变化 % 和绝对变化)                             │
│    - 判断 status: IMPROVED / DEGRADED / PASS / FAIL                        │
│    - 阈值判断：指标下降 > 5% 标记为 DEGRADED                                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Step 5: report_generation                                                   │
│  ─────────────────────────────────────────────────────────────────────────  │
│  输入:  all_metrics + delta_metrics                                          │
│  输出:  evaluation_report.json/html                                         │
│  任务:                                                                       │
│    - 生成结构化报告 (JSON + HTML)                                            │
│    - 写入 evaluation_results 表                                             │
│    - 告警通知 (如果 DEGRADED)                                                │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.3 Delta Metrics 计算规则

```
Delta(%) = (current_value - baseline_value) / baseline_value * 100

Status 判断:
- IMPROVED:  delta > 0 且 绝对值变化 > 1%
- DEGRADED:  delta < -5%
- PASS:      -5% <= delta <= 0
- FAIL:      指标低于业务最低阈值
```

---

## 五、数据流设计

### 5.1 测试数据格式

**test_dataset.jsonl**
```json
{"query": "如何重置密码?", "intent": "HOW_TO", "ground_truth": "步骤1: 点击忘记密码..."}
{"query": "退款政策是什么?", "intent": "POLICY", "ground_truth": "支持7天无理由退款..."}
```

**retrieval_labels.json**
```json
{"query": "产品A有哪些特性?", "relevant_docs": ["doc_id_1", "doc_id_3"]}
```

### 5.2 评测结果格式

```json
{
  "eval_name": "finetune_v2_compare",
  "eval_at": "2026-03-25T10:00:00Z",
  "retrieval_metrics": {
    "mrr@5": 0.789,
    "ndcg@5": 0.712,
    "hitrate@3": 0.856,
    "precision@5": 0.623
  },
  "generation_metrics": {
    "rouge_l": 0.452,
    "bleu_4": 0.312,
    "bertscore": 0.891
  },
  "ragas_metrics": {
    "faithfulness": 4.2,
    "answer_relevance": 4.5,
    "context_relevance": 3.8
  },
  "delta_metrics": {
    "retrieval_mrr5": {"baseline": 0.723, "current": 0.789, "delta": "+9.1%"},
    "generation_rouge_l": {"baseline": 0.452, "current": 0.438, "delta": "-3.1%"}
  },
  "status": "IMPROVED"
}
```

---

## 六、技术选型

| 组件 | 技术选型 | 说明 |
|------|---------|------|
| 调度框架 | Apache Airflow 2.x | 成熟稳定，支持 DAG |
| 数据库 | PostgreSQL | 基线存储，已有的 rag_chatbot 生态 |
| 检索服务 | Qdrant | 向量数据库，已有的 rag_chatbot 生态 |
| 指标计算 | Python (ragas, rouge-score, bert-score) | 成熟库 |
| 报告生成 | JSON + HTML | 结构化输出 |

---

## 七、文件结构

```
evaluation_pipeline/
├── dags/
│   ├── rag_evaluation_dag.py              # 主 DAG 定义
│   └── operators/
│       ├── data_preparation.py           # 数据准备
│       ├── retrieval_eval.py              # 检索评测
│       ├── generation_eval.py             # 生成评测
│       ├── baseline_compare.py            # 基线对比
│       └── report_generator.py            # 报告生成
├── scripts/
│   ├── metrics/
│   │   ├── retrieval_metrics.py           # 检索指标
│   │   ├── generation_metrics.py         # 生成指标
│   │   └── ragas_metrics.py               # RAGAS 指标
│   ├── dataset/
│   │   ├── loader.py                     # 数据加载
│   │   └── converter.py                  # 格式转换
│   └── api_client.py                     # Chatbot API
├── config/
│   ├── default_metrics.yaml               # 指标配置
│   └── evaluation_config.yaml             # 评测任务配置
├── sql/
│   └── init_evaluation_db.sql            # 数据库初始化
├── tests/
│   ├── unit/
│   └── integration/
├── requirements.txt
└── README.md
```

---

## 八、使用流程

### 8.1 初始化数据库

```bash
psql -h localhost -U postgres -d rag_chatbot -f sql/init_evaluation_db.sql
```

### 8.2 手动触发评测

```bash
# 日常评测 (使用最新模型)
airflow dags trigger rag_evaluation_daily

# 微调对比 (对比 v1.0 基线)
airflow dags trigger rag_evaluation_ft_compare \
    --conf '{"baseline_name": "v1.0", "comparison_mode": true}'
```

### 8.3 查看结果

```sql
-- 查看最新评测结果
SELECT * FROM evaluation_results ORDER BY eval_at DESC LIMIT 5;

-- 查看指标变化
SELECT
    b.baseline_name,
    r.eval_at,
    r.delta_metrics
FROM evaluation_results r
JOIN evaluation_baselines b ON r.baseline_id = b.id
ORDER BY r.eval_at DESC;
```

---

## 九、报告示例

```
============================================================
       RAG Chatbot 评测报告 - 微调对比
============================================================
评测名称: finetune_v2_vs_v1
评测时间: 2026-03-25 10:00:00
基线版本: v1.0 (2026-03-20)
------------------------------------------------------------

【检索质量】
指标           基线      当前      变化      状态
MRR@5          0.723     0.789     +9.1%     ✅ IMPROVED
NDCG@5         0.681     0.712     +4.5%     ✅ IMPROVED
HitRate@3      0.812     0.856     +5.4%     ✅ IMPROVED

【生成质量】
指标           基线      当前      变化      状态
ROUGE-L        0.452     0.438     -3.1%     ⚠️ DEGRADED
BLEU-4         0.312     0.298     -4.5%     ⚠️ DEGRADED
BERTScore      0.891     0.887     -0.4%     ✅ PASS

【RAGAS 指标】
指标           基线      当前      变化      状态
Faithfulness   4.1       4.2       +2.4%     ✅ IMPROVED
Answer Relev.  4.3       4.5       +4.7%     ✅ IMPROVED

【业务指标】
指标           基线      当前      变化      状态
准确率         0.823     0.851     +3.4%     ✅ IMPROVED

------------------------------------------------------------
总体状态: ⚠️ PARTIAL_IMPROVED
        检索和业务指标提升，但生成指标有轻微下降
        建议：检查生成器是否需要继续微调
============================================================
```

---

## 十、RAGAS 完整指标体系（生产级）

### 10.1 RAG 四象限评估模型

根据 2025 年生产级 RAG 评测最佳实践，评估分为四个核心维度：

```
                        ┌───────────────────────────────────────┐
                        │           RAG 四象限评估               │
                        ├───────────────────────────────────────┤
                        │                                       │
                        │   ┌─────────────────┬─────────────┐   │
                        │   │   检索质量       │   延迟成本   │   │
                        │   │  (Retrieval)     │  (Latency)  │   │
                        │   ├─────────────────┼─────────────┤   │
                        │   │   生成质量       │   业务指标   │   │
                        │   │  (Generation)    │  (Business) │   │
                        │   └─────────────────┴─────────────┘   │
                        │                                       │
                        └───────────────────────────────────────┘
```

| 象限 | 核心指标 | 业务价值 |
|------|----------|----------|
| **检索质量** | Context Precision, Context Recall, MRR, NDCG | 检索是否找到正确答案 |
| **生成质量** | Faithfulness, Answer Relevance, Answer Correctness | 回答是否正确、忠实 |
| **延迟成本** | P50/P95/P99 Latency, Cost per Query | 性能与成本平衡 |
| **业务指标** | Accuracy, Response Rate, Task Completion | 用户实际满意度 |

### 10.2 RAGAS 核心指标详解

| 指标 | 定义 | 计算方式 | 阈值建议 |
|------|------|----------|----------|
| **Context Precision** | 检索上下文中有多少相关块 | 正确块数 / 总块数 (Top-K) | ≥ 0.7 |
| **Context Recall** | 正确答案是否被检索到 | 覆盖的 GT 实体 / GT 总实体 | ≥ 0.8 |
| **Faithfulness** | 回答是否忠实于上下文 | 忠实陈述数 / 总陈述数 | ≥ 0.9 |
| **Answer Relevance** | 回答与问题的相关程度 | LLM 评估 (1-5) | ≥ 4.0 |
| **Answer Correctness** | 回答的完整正确性 | 准确率 + 相关性综合 | ≥ 0.85 |
| **Context Entity Recall** | 上下文中召回的实体比例 | 召回实体 / GT 实体 | ≥ 0.7 |

### 10.3 RAGAS Prompt 模板（增强版）

```python
# RAGAS Context Precision Prompt
CONTEXT_PRECISION_PROMPT = """Given a question and a context, evaluate whether the context contains relevant information to answer the question.

Question: {question}
Context: {context}

Determine the precision of the context (0.0 to 1.0), where 1.0 means all information in the context is relevant to the question.

Output JSON:
{{"precision": float, "reasoning": str}}"""

# RAGAS Context Recall Prompt
CONTEXT_RECALL_PROMPT = """Given a question, the ground truth answer, and the retrieved context, determine what portion of the ground truth answer can be attributed to the context.

Ground Truth Answer: {ground_truth}
Retrieved Context: {context}

Calculate the recall (0.0 to 1.0) representing how much of the ground truth is supported by the context.

Output JSON:
{{"recall": float, "attributed_statements": [str], "missing_statements": [str]}}"""

# RAGAS Answer Correctness Prompt
ANSWER_CORRECTNESS_PROMPT = """Given a question, ground truth answer, and generated answer, evaluate the correctness of the generated answer.

Question: {question}
Ground Truth: {ground_truth}
Generated Answer: {answer}

Evaluate answer correctness on two dimensions:
1. Semantic similarity (0.0 to 1.0)
2. Factual accuracy (0.0 to 1.0)

Combined score: (similarity + accuracy) / 2

Output JSON:
{{"correctness": float, "similarity": float, "accuracy": float, "feedback": str}}"""
```

---

## 十一、微调目标配置（Finetune Target）

### 11.1 三类微调目标

| 微调对象 | 影响范围 | 评测重点 | 关键指标 |
|----------|----------|----------|----------|
| **Embedding 模型** | 检索质量 | 向量检索准确性 | MRR@5, NDCG@5, HitRate@K |
| **LLM 生成器** | 回答质量 | 生成内容质量 | ROUGE-L, RAGAS, 业务指标 |
| **意图分类器** | 意图识别 | 分类准确性 | Intent Accuracy, Intent F1 |

### 11.2 评测模式配置

```python
class EvaluationMode(Enum):
    FULL = "full"              # 检索+生成+意图+业务，全方位评测
    RETRIEVAL_ONLY = "retrieval_only"  # 仅检索指标（Embedding 微调后）
    GENERATION_ONLY = "generation_only"  # 仅生成指标（LLM 微调后）
    INTENT_ONLY = "intent_only"  # 仅意图指标（意图分类器微调后）
```

**触发方式**：
```bash
# 微调目标为 Embedding 时，仅评测检索
airflow dags trigger rag_evaluation \
    --conf '{"mode": "retrieval_only", "finetune_target": "embedding"}'

# 微调目标为 LLM 时，评测生成+RAGAS
airflow dags trigger rag_evaluation \
    --conf '{"mode": "generation_only", "finetune_target": "llm"}'
```

### 11.3 评测流程条件分支

```
                    ┌─────────────────────┐
                    │  finetune_target    │
                    └─────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│   embedding   │     │      llm      │     │     intent     │
└───────────────┘     └───────────────┘     └───────────────┘
        │                     │                     │
        ▼                     ▼                     ▼
┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│ retrieval_eval│     │ generation_eval│     │ intent_eval   │
│ + baseline    │     │ + baseline     │     │ + baseline    │
└───────────────┘     └───────────────┘     └───────────────┘
```

---

## 十二、意图识别评测（Intent Detection）

### 11.1 评测指标

| 指标 | 说明 | 计算方式 |
|------|------|----------|
| **Intent Accuracy** | 意图分类准确率 | Correct / Total |
| **Intent F1 (Macro)** | 各意图类别 F1 的平均值 | unweighted mean of per-class F1 |
| **Intent Confusion Matrix** | 混淆矩阵 | 统计各意图的分类情况 |

### 11.2 意图类型定义

```python
INTENT_TYPES = [
    "HOW_TO",      # 操作指南类
    "POLICY",      # 政策规则类
    "PRODUCT",     # 产品咨询类
    "BILLING",     # 账单支付类
    "COMPLAINT",   # 投诉建议类
    "REFUND",      # 退款售后类
    "FEEDBACK",    # 反馈评价类
    "GREETING",    # 问候寒暄类
    "OTHER",       # 其他/未知
]
```

### 11.3 意图分类评测实现

```python
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from typing import List, Dict

def evaluate_intent(
    predicted_intents: List[str],
    ground_truth_intents: List[str],
    intent_types: List[str]
) -> Dict:
    """意图分类评测"""
    return {
        "intent_accuracy": accuracy_score(ground_truth_intents, predicted_intents),
        "intent_f1_macro": f1_score(
            ground_truth_intents,
            predicted_intents,
            labels=intent_types,
            average="macro"
        ),
        "confusion_matrix": confusion_matrix(
            ground_truth_intents,
            predicted_intents,
            labels=intent_types
        ).tolist(),
    }
```

---

## 十三、数据管理统一规范

### 12.1 设计原则

**统一使用数据库管理，文件作为导入/导出格式**

```
┌─────────────────────────────────────────────────────────┐
│                    数据管理架构                          │
├─────────────────────────────────────────────────────────┤
│                                                         │
│   ┌──────────────┐         ┌──────────────────────┐   │
│   │  .jsonl 文件  │ ──────▶ │   test_datasets 表    │   │
│   │  (导入来源)   │ import  │   (主数据源)          │   │
│   └──────────────┘         └──────────────────────┘   │
│           │                           │                 │
│           │ export                    │ query           │
│           ▼                           ▼                 │
│   ┌──────────────┐         ┌──────────────────────┐   │
│   │  .jsonl 文件  │ ◀────── │   evaluation_results │   │
│   │  (备份/归档)  │         │   表 (结果存储)       │   │
│   └──────────────┘         └──────────────────────┘   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 12.2 数据流向

| 操作 | 数据源 | 目标 |
|------|--------|------|
| 导入测试集 | `test_dataset.jsonl` | `test_datasets` 表 |
| 执行评测 | `test_datasets` 表 | `evaluation_results` 表 |
| 导出报告 | `evaluation_results` 表 | `report.json` |
| 基线备份 | `evaluation_results` 表 | `baseline_results/*.json` |

### 12.3 数据库表说明

```sql
-- 测试数据集表（主数据源）
CREATE TABLE test_datasets (
    id              SERIAL PRIMARY KEY,
    dataset_name    VARCHAR(100) NOT NULL,
    dataset_type    VARCHAR(20) NOT NULL,  -- 'qa', 'retrieval', 'intent'
    version         VARCHAR(20),
    created_at      TIMESTAMP DEFAULT NOW(),
    data            JSONB NOT NULL,       -- 完整数据集
    stats           JSONB,                -- 数据集统计：样本数、各意图分布等
    metadata        JSONB,
    UNIQUE(dataset_name, version)
);

-- 评测结果表
CREATE TABLE evaluation_results (
    id              SERIAL PRIMARY KEY,
    eval_name       VARCHAR(100) NOT NULL,
    eval_at         TIMESTAMP DEFAULT NOW(),
    dataset_id      INTEGER REFERENCES test_datasets(id),

    -- 评测配置
    mode            VARCHAR(20),           -- 'full', 'retrieval_only', etc.
    finetune_target VARCHAR(50),           -- 'embedding', 'llm', 'intent', 'all'

    -- 各维度指标
    retrieval_metrics   JSONB,
    generation_metrics  JSONB,
    intent_metrics       JSONB,
    business_metrics     JSONB,

    -- 错误分析
    error_analysis       JSONB,

    -- 延迟指标
    latency_p50_ms       INTEGER,
    latency_p95_ms       INTEGER,
    latency_p99_ms       INTEGER,

    -- 基线对比
    comparison_mode      BOOLEAN DEFAULT FALSE,
    baseline_id          INTEGER REFERENCES evaluation_baselines(id),
    delta_metrics        JSONB,
    status               VARCHAR(20),
    significance_tests   JSONB,            -- p-value 等统计检验结果

    metadata            JSONB
);
```

---

## 十四、统计显著性检验

### 13.1 目的

判断微调前后的指标差异是否具有统计显著性，避免因样本波动误判。

### 13.2 方法：Bootstrap 置信区间

```python
import numpy as np
from typing import List, Tuple

def bootstrap_confidence_interval(
    baseline_scores: List[float],
    current_scores: List[float],
    confidence_level: float = 0.95,
    n_bootstrap: int = 1000
) -> dict:
    """
    使用 Bootstrap 方法计算指标变化的置信区间和 p 值

    Returns:
        {
            "delta_mean": float,        # 平均变化
            "ci_lower": float,          # 置信区间下限
            "ci_upper": float,          # 置信区间上限
            "p_value": float,            # 显著性 p 值
            "is_significant": bool       # 是否统计显著
        }
    """
    original_delta = np.mean(current_scores) - np.mean(baseline_scores)

    # 合并池化 Bootstrap
    pooled = np.concatenate([baseline_scores, current_scores])
    deltas = []

    for _ in range(n_bootstrap):
        # 有放回采样，保持原样本量
        baseline_sample = np.random.choice(pooled, size=len(baseline_scores), replace=True)
        current_sample = np.random.choice(pooled, size=len(current_scores), replace=True)
        deltas.append(np.mean(current_sample) - np.mean(baseline_sample))

    deltas = np.array(deltas)

    # 计算置信区间
    alpha = 1 - confidence_level
    ci_lower = np.percentile(deltas, alpha / 2 * 100)
    ci_upper = np.percentile(deltas, (1 - alpha / 2) * 100)

    # 计算 p 值（双尾）
    p_value = np.mean(np.abs(deltas) >= np.abs(original_delta))

    # 显著判断：p < 0.05 且置信区间不包含 0
    is_significant = p_value < 0.05 and (ci_lower > 0 or ci_upper < 0)

    return {
        "delta_mean": original_delta,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value": p_value,
        "is_significant": is_significant
    }
```

### 13.3 Delta Status 判断

| delta 方向 | 统计显著 | |delta| > 1% | Status |
|------------|---------|-------------|------------------------|
| 上升 | 是 | 是 | `SIGNIFICANT_IMPROVED` |
| 上升 | 是 | 否 | `MARGINAL_IMPROVED` |
| 下降 | 是 | 是 | `SIGNIFICANT_DEGRADED` |
| 下降 | 是 | 否 | `MARGINAL_DEGRADED` |
| - | 否 | - | `NOT_SIGNIFICANT` |

### 13.4 报告输出示例

```json
{
  "delta_metrics": {
    "mrr@5": {
      "baseline": 0.723,
      "current": 0.789,
      "delta": "+0.066",
      "delta_pct": "+9.1%",
      "significance": {
        "p_value": 0.023,
        "ci_lower": 0.032,
        "ci_upper": 0.101,
        "is_significant": true
      },
      "status": "SIGNIFICANT_IMPROVED"
    }
  }
}
```

---

## 十五、延迟指标（Latency Metrics）

### 14.1 指标定义

| 指标 | 说明 | 业务意义 |
|------|------|----------|
| **P50 Latency** | 中位数响应时间 | 典型用户体验 |
| **P95 Latency** | 95% 分位数响应时间 | 尾部用户体验 |
| **P99 Latency** | 99% 分位数响应时间 | 极端情况监控 |
| **Retrieval Latency** | 检索阶段耗时 | 向量检索性能 |
| **Generation Latency** | 生成阶段耗时 | LLM 推理性能 |

### 14.2 延迟记录实现

```python
import time
import statistics
from typing import List, Dict

class LatencyTracker:
    def __init__(self):
        self.retrieval_times: List[float] = []
        self.generation_times: List[float] = []
        self.total_times: List[float] = []

    def record_retrieval(self, duration_ms: float):
        self.retrieval_times.append(duration_ms)

    def record_generation(self, duration_ms: float):
        self.generation_times.append(duration_ms)

    def record_total(self, duration_ms: float):
        self.total_times.append(duration_ms)

    def compute_percentiles(self, times: List[float], percentiles=[50, 95, 99]) -> Dict:
        """计算分位数"""
        sorted_times = sorted(times)
        return {
            f"p{p}": sorted_times[int(len(sorted_times) * p / 100)]
            for p in percentiles
        }

    def get_summary(self) -> Dict:
        return {
            "retrieval": self.compute_percentiles(self.retrieval_times),
            "generation": self.compute_percentiles(self.generation_times),
            "total": self.compute_percentiles(self.total_times)
        }
```

### 14.3 延迟阈值告警

```yaml
# configs/evaluation_config.yaml
latency_thresholds:
  retrieval:
    p95_max_ms: 200      # 检索 P95 不超过 200ms
    p99_max_ms: 500
  generation:
    p95_max_ms: 3000     # 生成 P95 不超过 3s
    p99_max_ms: 10000
  total:
    p95_max_ms: 5000     # 端到端 P95 不超过 5s
    p99_max_ms: 15000
```

---

## 十六、Ground Truth 来源与质量保证

### 15.1 检索标注获取方式

| 方式 | 成本 | 质量 | 适用场景 |
|------|------|------|----------|
| **人工标注** | 高 | 高 | 重要数据集首次构建 |
| **LLM 辅助标注** | 中 | 中高 | 迭代扩展标注 |
| **规则+弱监督** | 低 | 中 | 大规模数据预处理 |
| **用户反馈回流** | 低 | 动态 | 在线持续优化 |

### 15.2 LLM 辅助标注流程

```python
# 使用 GPT-4 进行初步标注，人工抽检校正

LLM_ANNOTATION_PROMPT = """给定用户问题，识别最相关的知识库文档。

问题：{query}

知识库文档：
{doc_list}

要求：
1. 选择所有与问题相关的文档 ID
2. 考虑语义相关性，不仅仅是关键词匹配
3. 返回 JSON 格式：{{"relevant_docs": ["doc_id_1", "doc_id_2"]}}

请标注："""
```

### 15.3 标注质量控制

```python
# 1. 抽样人工抽检
def quality_check(annotations: List[Dict], sample_size: int = 50) -> float:
    """抽样检查标注一致性，返回人工确认的准确率"""
    sampled = random.sample(annotations, min(sample_size, len(annotations)))
    # ... 人工标注对比 ...
    return confirmed_accuracy

# 2. 多标注员一致性
def inter_annotator_agreement(annotations_list: List[List[str]]) -> float:
    """计算多标注员之间的 Fleiss' Kappa 系数"""
    # ... 计算 Kappa ...
    return kappa_score

# 3. 阈值触发重新标注
if quality_check(annotations) < 0.85:
    # 自动触发人工重新标注
    trigger_human_review()
```

### 15.4 知识库文档格式

```json
{
  "doc_id": "kb_policy_refund_001",
  "category": "POLICY",
  "title": "退款政策细则",
  "content": "我们支持7天内无理由退款...",
  "intent": ["REFUND", "POLICY"],
  "keywords": ["退款", "退货", "售后"],
  "metadata": {
    "created_at": "2026-01-15",
    "updated_at": "2026-03-01",
    "version": "2.1"
  }
}
```

---

## 十七、完整数据库 Schema

```sql
-- 评测基线表：存储每次微调后的基准指标
CREATE TABLE evaluation_baselines (
    id                  SERIAL PRIMARY KEY,
    baseline_name       VARCHAR(100) NOT NULL,
    model_version       VARCHAR(100),
    finetune_target     VARCHAR(50),         -- 'embedding', 'llm', 'intent', 'all'
    description         TEXT,
    created_at          TIMESTAMP DEFAULT NOW(),

    -- 检索指标
    retrieval_mrr5          DECIMAL(5,4),
    retrieval_ndcg5         DECIMAL(5,4),
    retrieval_hitrate3      DECIMAL(5,4),
    retrieval_precision5    DECIMAL(5,4),

    -- 生成指标
    generation_rouge_l      DECIMAL(5,4),
    generation_bleu4       DECIMAL(5,4),
    generation_bertscore    DECIMAL(5,4),

    -- RAGAS 指标
    ragas_faithfulness         DECIMAL(3,2),
    ragas_answer_relevance     DECIMAL(3,2),
    ragas_context_relevance    DECIMAL(3,2),

    -- 意图识别指标
    intent_accuracy         DECIMAL(5,4),
    intent_f1_macro         DECIMAL(5,4),

    -- 业务指标
    business_accuracy           DECIMAL(5,4),
    business_response_rate      DECIMAL(5,4),
    business_no_hallucination_rate DECIMAL(5,4),

    -- 延迟指标
    latency_p50_ms          INTEGER,
    latency_p95_ms          INTEGER,
    latency_p99_ms          INTEGER,

    metadata                JSONB,
    UNIQUE(baseline_name)
);

-- 评测结果表：每次评测的详细结果
CREATE TABLE evaluation_results (
    id                  SERIAL PRIMARY KEY,
    baseline_id         INTEGER REFERENCES evaluation_baselines(id),
    eval_name           VARCHAR(100) NOT NULL,
    eval_at             TIMESTAMP DEFAULT NOW(),
    dataset_id          INTEGER REFERENCES test_datasets(id),

    -- 评测配置
    mode                VARCHAR(20),           -- 'full', 'retrieval_only', etc.
    finetune_target     VARCHAR(50),

    -- 各维度指标汇总
    retrieval_metrics   JSONB,
    generation_metrics  JSONB,
    ragas_metrics       JSONB,
    intent_metrics      JSONB,
    business_metrics    JSONB,

    -- 延迟指标
    latency_p50_ms      INTEGER,
    latency_p95_ms      INTEGER,
    latency_p99_ms      INTEGER,

    -- 错误分析
    error_analysis      JSONB,

    -- 微调对比模式
    comparison_mode     BOOLEAN DEFAULT FALSE,
    delta_metrics       JSONB,
    significance_tests  JSONB,
    status              VARCHAR(30),

    metadata            JSONB
);

-- 测试数据集表
CREATE TABLE test_datasets (
    id              SERIAL PRIMARY KEY,
    dataset_name    VARCHAR(100) NOT NULL,
    dataset_type     VARCHAR(20) NOT NULL,      -- 'qa', 'retrieval', 'intent'
    version          VARCHAR(20),
    description      TEXT,
    created_at       TIMESTAMP DEFAULT NOW(),
    data             JSONB NOT NULL,
    stats            JSONB,                      -- 样本数、意图分布等
    metadata         JSONB,
    UNIQUE(dataset_name, version)
);
```

---

## 十八、生产级最佳实践

### 17.1 自动化测试基础设施

根据 2025 年生产级 RAG 评测指南，自动化测试基础设施是关键：

```python
# 自动化评测触发器
AUTOMATION_CONFIG = {
    # CI/CD 集成
    "ci_trigger": {
        "on_pr": True,           # PR 时运行快速检查
        "on_merge": True,         # 合并时运行完整评测
        "on_schedule": "0 2 * * *"  # 每天凌晨 2 点
    },

    # 自动失败门禁
    "fail_gates": {
        "precision@5_drop_threshold": 0.05,    # Precision@5 下降超过 5% 阻止部署
        "hallucination_rate_max": 0.1,          # 幻觉率超过 10% 阻止部署
        "p95_latency_increase_max_ms": 500     # P95 延迟增加超过 500ms 警告
    },

    # 持续监控
    "drift_detection": {
        "enabled": True,
        "metric_window": "7d",                  # 7 天滚动窗口
        "alert_on_drift": True
    }
}
```

### 17.2 测试数据集构建最佳实践

```python
# 多样性查询集设计
QUERY_DIVERSITY = {
    "query_shapes": [
        {"type": "factual", "desc": "事实型问题", "example": "退款政策是什么?"},
        {"type": "procedural", "desc": "操作步骤型", "example": "如何重置密码?"},
        {"type": "comparative", "desc": "对比型", "example": "套餐A和套餐B有什么区别?"},
        {"type": "troubleshooting", "desc": "故障排除型", "example": "无法登录怎么办?"},
        {"type": "vague", "desc": "模糊型", "example": "那个问题"}
    ],

    # 边缘案例覆盖
    "edge_cases": [
        {"type": "typo", "desc": "拼写错误", "example": "怎麼重置密码?"},
        {"type": "cross_lingual", "desc": "中英混合", "example": "怎么 order 这个 product?"},
        {"type": "incomplete", "desc": "信息不完整", "example": "退款?"},
        {"type": "ambiguous", "desc": "歧义性", "example": "它支持吗?"}
    ]
}
```

### 17.3 基线建立与版本管理

```python
# 基线版本管理策略
BASELINE_STRATEGY = {
    # 仅在重大变更时重新建立基线
    "rebaseline_triggers": [
        "index_schema_major_version",   # 索引 Schema 大版本变更
        "embedding_model_family",       # Embedding 模型族变更
        "llm_model_major_version"       # LLM 大版本升级
    ],

    # 保持功能基线和运营基线分离
    "baseline_categories": {
        "functional": ["accuracy", "groundedness", "faithfulness"],  # 功能质量指标
        "operational": ["latency", "cost_per_query", "throughput"]   # 运营效率指标
    },

    # 至少运行两个完整周期确认指标稳定
    "stability_check_cycles": 2
}
```

### 17.4 LLM-as-a-Judge 最佳实践

```python
# LLM-as-a-Judge 配置
LLM_JUDGE_CONFIG = {
    "model": "gpt-4",
    "temperature": 0,
    "prompt_template": """Evaluate the following RAG response on a scale of 1-5:

Question: {question}
Context: {context}
Response: {response}

Criteria:
- Faithfulness: Does the response stick to the context?
- Relevance: Does the response answer the question?
- Completeness: Is the response comprehensive?

Provide a JSON output:
{{"score": int, "faithfulness": int, "relevance": int, "completeness": int, "reasoning": str}}""",

    # 偏见缓解
    "bias_mitigation": [
        "使用平衡的测试集避免偏向特定风格",
        "多次评估取平均减少随机性",
        "定期校准 Judge 提示词"
    ]
}
```

### 17.5 多维度评分与权衡分析

```python
# 综合评分卡
SCORECARD_TEMPLATE = {
    "retrieval": {
        "precision@5": {"value": 0.78, "target": 0.7, "status": "pass"},
        "recall@10": {"value": 0.92, "target": 0.85, "status": "pass"},
        "ndcg@10": {"value": 0.81, "target": 0.8, "status": "pass"}
    },
    "generation": {
        "faithfulness": {"value": 0.94, "target": 0.9, "status": "pass"},
        "answer_relevance": {"value": 4.2, "target": 4.0, "status": "pass"},
        "answer_correctness": {"value": 0.88, "target": 0.85, "status": "pass"}
    },
    "latency": {
        "p50_ms": {"value": 245, "target": 300, "status": "pass"},
        "p95_ms": {"value": 890, "target": 1000, "status": "pass"},
        "p99_ms": {"value": 2100, "target": 3000, "status": "pass"}
    },
    "cost": {
        "per_query_usd": {"value": 0.002, "target": 0.003, "status": "pass"}
    }
}

# 权衡分析示例
TRADEOFF_ANALYSIS = """
指标权衡示例：
1. 召回率 vs 延迟：Recall@10 从 0.85 → 0.92，延迟增加 200ms（可接受）
2. 精确度 vs 幻觉率：Precision 降低 3% 但 Hallucination 降低 50%（有利）
3. 成本 vs 质量：Cost 增加 15% 但 Accuracy 提升 8%（合理）
"""
```

---

## 十九、完整评测报告模板

```json
{
  "report_metadata": {
    "eval_name": "rag_evaluation_20260325",
    "eval_version": "2.0",
    "generated_at": "2026-03-25T10:00:00Z",
    "dataset": {
      "name": "customer_service_qa_v2",
      "total_samples": 500,
      "intents_covered": ["HOW_TO", "POLICY", "PRODUCT", "BILLING", "COMPLAINT"],
      "edge_cases_ratio": 0.15
    }
  },

  "scorecard": {
    "overall_status": "PASS",
    "summary": {
      "retrieval_quality": "EXCELLENT",
      "generation_quality": "GOOD",
      "latency": "ACCEPTABLE",
      "business_impact": "POSITIVE"
    }
  },

  "metrics": {
    "retrieval": {
      "mrr@5": {"value": 0.82, "target": 0.75, "delta": "+9.3%", "status": "IMPROVED"},
      "ndcg@5": {"value": 0.78, "target": 0.70, "delta": "+11.4%", "status": "IMPROVED"},
      "hitrate@3": {"value": 0.91, "target": 0.85, "delta": "+7.1%", "status": "IMPROVED"},
      "context_precision": {"value": 0.74, "target": 0.70, "delta": "+5.7%", "status": "IMPROVED"},
      "context_recall": {"value": 0.86, "target": 0.80, "delta": "+7.5%", "status": "IMPROVED"}
    },
    "generation": {
      "faithfulness": {"value": 0.94, "target": 0.90, "delta": "+4.4%", "status": "IMPROVED"},
      "answer_relevance": {"value": 4.3, "target": 4.0, "delta": "+7.5%", "status": "IMPROVED"},
      "answer_correctness": {"value": 0.87, "target": 0.85, "delta": "+2.4%", "status": "IMPROVED"},
      "rouge_l": {"value": 0.48, "target": 0.40, "delta": "+20%", "status": "IMPROVED"}
    },
    "intent": {
      "intent_accuracy": {"value": 0.94, "target": 0.90, "delta": "+4.4%", "status": "IMPROVED"},
      "intent_f1_macro": {"value": 0.91, "target": 0.85, "delta": "+7.1%", "status": "IMPROVED"}
    },
    "latency": {
      "p50_ms": {"value": 245, "target": 300, "status": "PASS"},
      "p95_ms": {"value": 890, "target": 1000, "status": "PASS"},
      "p99_ms": {"value": 2100, "target": 3000, "status": "PASS"}
    },
    "business": {
      "accuracy": {"value": 0.89, "target": 0.85, "delta": "+4.7%", "status": "IMPROVED"},
      "response_rate": {"value": 0.96, "target": 0.95, "delta": "+1.0%", "status": "IMPROVED"},
      "no_hallucination_rate": {"value": 0.94, "target": 0.90, "delta": "+4.4%", "status": "IMPROVED"}
    }
  },

  "stratified_results": {
    "by_intent": {
      "HOW_TO": {"accuracy": 0.92, "count": 120},
      "POLICY": {"accuracy": 0.95, "count": 85},
      "PRODUCT": {"accuracy": 0.88, "count": 95},
      "BILLING": {"accuracy": 0.91, "count": 110},
      "COMPLAINT": {"accuracy": 0.82, "count": 90}
    },
    "by_complexity": {
      "SIMPLE": {"accuracy": 0.95, "count": 150},
      "MODERATE": {"accuracy": 0.89, "count": 250},
      "COMPLEX": {"accuracy": 0.78, "count": 100}
    }
  },

  "error_analysis": {
    "top_errors": [
      {"type": "retrieval_complete_miss", "count": 12, "rate": 0.024},
      {"type": "generation_hallucination", "count": 8, "rate": 0.016},
      {"type": "intent_misclassification", "count": 15, "rate": 0.030}
    ],
    "worst_cases": [
      {"query": "...", "expected_intent": "REFUND", "predicted_intent": "COMPLAINT", "issue": "语义相似导致误分类"}
    ]
  },

  "significance_tests": {
    "mrr@5": {"p_value": 0.023, "ci": [0.032, 0.101], "significant": true},
    "faithfulness": {"p_value": 0.045, "ci": [0.012, 0.068], "significant": true}
  },

  "recommendations": [
    "✅ 检索质量显著提升，建议持续监控 Embedding 模型更新",
    "⚠️ COMPLAINT 意图准确率偏低(0.82)，建议补充该类训练数据",
    "ℹ️ P95 延迟接近阈值，建议优化检索排序策略"
  ]
}
```

---

## 二十、后续扩展点

1. **自动化调度**: 接入 GitHub Webhook，代码推送时自动触发
2. **A/B 在线测试**: 扩展支持线上流量分流
3. **告警集成**: 接入 Slack/钉钉，DEGRADED 时自动通知
4. **模型注册**: 与模型版本管理平台集成
5. **可视化面板**: 接入 Grafana/Kibana 可视化指标趋势
6. **多语言支持**: 扩展支持中英混合评测
7. **流式评测**: 支持实时流式输出评测
8. **增量评测**: 仅评测新增/修改的数据
