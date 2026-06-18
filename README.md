# RAG Chatbot 端到端评测流水线

## 设计概述

本评测流水线用于评估 **RAG Chatbot(智能客服)** 的质量。它把 chatbot 当作**黑盒**——唯一访问入口是公开的 `POST /api/v1/chat`,不直连 Qdrant、不查 DB、不调内部服务。这样测到的是用户真实看到的效果(混合检索 + rerank + 可选 GraphRAG + LangGraph 对话引擎的端到端表现),而不是某个孤立组件。

> 约束:**本流水线只读 chatbot,不修改 chatbot 的任何代码。** rag_chatbot 作为被测系统独立运行。

### 评测维度

| 维度 | 指标 | 说明 |
|------|------|------|
| **检索质量** | MRR@5, NDCG@5, HitRate@3, Precision@5 | 对 chatbot 返回的 `sources` 评分(黑盒,不直连向量库) |
| **生成质量** | LLM-as-a-Judge(relevance/fluency/completeness/safety) + ROUGE/BLEU | judge 默认走 GLM OpenAI 兼容网关 |
| **意图识别** | Intent Accuracy, Intent F1 (Macro) | 意图取自 chatbot 响应的 `intent` 字段 |
| **多轮对话** | 槽位 Precision/Recall/F1、平均澄清轮数、意图切换准确率、任务完成率 | 经 `converse()` 跑多轮(同 session_id),覆盖 LangGraph 的槽位填充/状态栈 |
| **延迟指标** | P50/P95/P99 Latency | 阈值按客服 RAG 真实场景校准(端到端,非裸向量检索) |
| **微调效果** | Delta Metrics + 统计显著性检验 | 微调前后指标变化 |

### 黑盒契约(被测系统接口)

流水线通过 `POST /api/v1/chat` 与 chatbot 交互:

```
请求: {"message": str, "session_id": int (>0), "user_id"?: int, "max_tokens"?: int}
响应: {
  "content": str,                         # 回答文本(检索/生成评测的主输入)
  "session_id": int,
  "intent": str,                          # 意图(意图评测的输入)
  "sources": [str]|null,                  # 检索到的文档 id 列表(检索评测的输入)
  "metadata": {
    "filled_slots": {...},                # 已填槽位(多轮槽位评测的输入)
    "pending_slots": [...],               # 待填槽位(澄清轮数评测的输入)
    "confidence": float
  },
  "dialogue_state": {"phase", "pending_slots", "filled_slots"}
}
```

关键黑盒行为(已内置于客户端):
- **匿名访问**:不带 `Authorization` 头即可(服务端回退到 user_id=0);不要带坏 token(会 401),也不要调 `POST /sessions`(匿名会 500)。`session_id` 是任意正整数,无需预先创建。
- **session_id = 对话线程**:服务端按 `thread_id=session_id` 做进程内记忆。复用同一 id 即多轮连续;每条独立样本用不同 id 避免记忆串扰。
- **限流**:服务端约 10 req/min/IP。客户端默认节流到 9 rpm 留余量,见 `ChatbotClient.DEFAULT_RATE_LIMIT_RPM`。

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
│     - 加载测试数据集 (Q&A、检索标注、多轮对话)                              │
│     - 数据验证和格式转换                                                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
┌──────────────────────┐ ┌─────────────────────┐ ┌─────────────────────────┐
│ 2. retrieval_eval    │ │ 3. generation_eval  │ │ 4. dialogue_eval        │
│  打 /api/v1/chat     │ │  打 /api/v1/chat    │ │  打 /api/v1/chat 多轮   │
│  对 sources 评分     │ │  LLM-as-a-Judge     │ │  槽位/澄清/意图切换     │
│  MRR/NDCG/HitRate    │ │  ROUGE/BLEU         │ │  任务完成率             │
└──────────────────────┘ └─────────────────────┘ └─────────────────────────┘
                    └───────────────┼───────────────┘
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  5. baseline_comparison  (微调对比模式才执行)                               │
│     - Delta Metrics (微调后 vs 微调前) + 统计显著性检验                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  6. report_generation                                                        │
│     - 生成 HTML/JSON 报告,包含检索/生成/对话三类指标                        │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 文件结构

```
evaluation_pipeline/
├── src/                              # 核心源码
│   ├── config.py                     # 配置管理(黑盒 endpoint / GLM judge / 节流)
│   ├── database.py                   # PostgreSQL 连接
│   ├── metrics/                      # 指标计算
│   │   ├── retrieval.py              # 检索指标 (MRR, NDCG, HitRate)
│   │   ├── generation.py             # 生成指标 (ROUGE, BLEU)
│   │   ├── ragas.py                  # RAGAS 评估 + LLMJudgeClient(GLM)
│   │   └── statistics.py             # 统计分析 (Bootstrap, p-value)
│   ├── dataset/                      # 数据集管理
│   │   ├── loader.py                 # 数据加载
│   │   ├── sampler.py                # 分层采样
│   │   └── converter.py              # 格式转换 + 意图归一化
│   ├── api/                          # 黑盒 API 客户端
│   │   └── chatbot.py                # /api/v1/chat 客户端(session/节流/多轮)
│   ├── evaluation/                   # 评测执行
│   │   ├── retrieval_eval.py         # 检索评测(黑盒 sources)
│   │   ├── generation_eval.py        # 生成评测(LLM-as-a-Judge)
│   │   ├── dialogue_eval.py          # 多轮对话评测(槽位/意图切换/完成率)
│   │   ├── intent_eval.py            # 意图评测
│   │   └── error_analyzer.py         # 错误分析
│   └── reporting/                    # 报告生成
│       ├── delta_reporter.py         # Delta 变化报告
│       ├── html_generator.py         # HTML 报告
│       └── json_reporter.py          # JSON 报告
├── dags/                             # Airflow DAG
│   └── rag_evaluation_dag.py         # 含 dialogue_evaluation 任务
├── configs/                          # 配置文件
│   ├── default_metrics.yaml
│   ├── ragas_prompts.yaml            # RAGAS Prompt 模板
│   └── evaluation_config.yaml        # 黑盒 endpoint / 意图枚举 / judge / 节流 / 阈值
├── data/                             # 测试数据集(.gitignore,本地 fixture)
│   ├── test_dataset.jsonl            # 单轮 Q&A
│   ├── retrieval_labels.json         # 检索标注
│   └── dialogue_dataset.jsonl        # 多轮对话 case
├── sql/                              # 数据库
│   └── init_evaluation_db.sql
├── scripts/
│   └── run_evaluation.py             # 独立运行脚本
└── tests/
    ├── unit/
    │   ├── test_blackbox_eval.py     # 黑盒客户端/检索/judge/意图 单测
    │   └── test_dialogue_eval.py     # 多轮对话评测单测
    └── integration/
```

## 快速开始

### 1. 安装依赖

```bash
cd evaluation_pipeline
pip install -r requirements.txt
```

### 2. 配置环境

复制 `.env.example` 为 `.env` 并填写。匿名访问可不填 chatbot key:

```bash
export AIRFLOW_HOME=/path/to/airflow

# 被测 chatbot(黑盒,匿名访问可不填 key)
export CHATBOT_API_URL=http://localhost:8000
export CHATBOT_API_KEY=                      # 可空;启用鉴权再填
export CHATBOT_RATE_LIMIT_RPM=9              # 留在服务端 ~10 req/min/IP 之下

# LLM-as-a-Judge(默认 GLM OpenAI 兼容网关;也可 JUDGE_PROVIDER=openai)
export JUDGE_PROVIDER=glm
export JUDGE_MODEL=glm-5.2
export GLM_API_KEY=your-glm-key
export GLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4
```

### 3. 启动被测 chatbot 与 Airflow

```bash
# 先启动 rag_chatbot(被测系统,默认监听 8000)
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
| `full` | 检索+生成+意图+对话+业务 | 全面评测 |
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
| Metric              | Baseline | Current | Delta  | p-value | Status  |
|---------------------|----------|---------|--------|---------|---------|
| MRR@5               | 0.723    | 0.789   | +9.1%  | 0.023   | ✅ SIGNIFICANT_IMPROVED |
| NDCG@5              | 0.681    | 0.712   | +4.5%  | 0.045   | ✅ MARGINAL_IMPROVED |
| Slot F1 (对话)      | 0.820    | 0.871   | +6.2%  | 0.018   | ✅ IMPROVED |
| Judge Relevance     | 0.452    | 0.438   | -3.1%  | 0.120   | ➖ NOT_SIGNIFICANT |
```

## 测试数据格式

> 数据文件在 `.gitignore` 中,作为本地 fixture 管理。仓库内置了示例数据可直接跑通。

### 单轮 Q&A (`data/test_dataset.jsonl`)

意图标签必须与 chatbot 的 `Intent` 枚举一致(全小写):

```json
{"query": "退款政策是什么?", "intent": "policy", "ground_truth": "7天无理由退款:购买后7天内可申请退款..."}
{"query": "我要退款,订单号 SO20260617001", "intent": "refund", "ground_truth": "请提供以下信息以处理退款..."}
{"query": "你好", "intent": "greeting", "ground_truth": "您好!很高兴为您服务..."}
```

### 意图类型(对齐 rag_chatbot 的 `Intent` 枚举)

| 组 | 意图 | 说明 |
|----|------|------|
| Task | `refund` `return` `query_order` `track_shipping` `complaint` | 任务型(需槽位填充) |
| RAG | `faq` `policy` | 知识问答 |
| Direct | `chitchat` `greeting` | 直接回复 |
| Meta | `confirm` `deny` `cancel` `unknown` | 元对话 |
| Graph | `relationship_query` `global_summary` `entity_lookup` | 图谱查询(需 GRAPH_RAG_ENABLED) |

### 检索标注 (`data/retrieval_labels.json`,JSONL)

`relevant_docs` 的 id 需与 chatbot 实际返回的 `sources` 命名一致才能命中:

```json
{"query": "如何重置密码?", "intent": "faq", "relevant_docs": ["doc_password_reset_001", "doc_account_settings_002"]}
```

### 多轮对话 (`data/dialogue_dataset.jsonl`)

每行一个对话 case,支持紧凑的 per-turn 写法。`required_slots` 决定任务完成率,`expect_switch` 标注是否含意图切换:

```json
{"case_id": "refund_clarify_3turn", "turns": [
  {"message": "我想退货退款", "intent": "refund", "slots": {}},
  {"message": "订单号是 SO20260617002", "intent": "refund", "slots": {"order_id": "SO20260617002"}},
  {"message": "原因是质量问题", "intent": "refund", "slots": {"order_id": "SO20260617002", "reason": "质量问题"}}
], "required_slots": ["order_id", "reason"], "expect_switch": false}
```

## 多轮对话评测方法论

采用 PARADISE / MultiWOZ 范式的任务导向对话评测,针对 LangGraph 对话引擎设计:

| 指标 | 定义 | 对应 LangGraph 节点 |
|------|------|-------------------|
| 槽位 Precision/Recall/F1 | 最终轮 `filled_slots` vs `required_slots` | `collect_slots` |
| 平均澄清轮数 | 必需槽全部填满前的轮数(越少越好) | `collect_slots` → `generate_response` |
| 每轮意图准确率 | 检测意图 vs 标注意图 | `detect_intent` |
| 意图切换准确率 | 切换轮是否跟随新任务 | `handle_switch` + `state_stack` |
| 任务完成率 | 末轮必需槽是否填齐 | 端到端 |

## 调度策略

| DAG ID | Schedule | 用途 |
|--------|----------|------|
| `rag_evaluation_daily` | `@daily` | 日常监控 |
| `rag_evaluation_weekly` | `@weekly` | 周度报告 |
| `rag_evaluation_ft_compare` | None (手动) | 微调对比实验 |

### 触发示例

```bash
# 全面评测(含多轮对话)
airflow dags trigger rag_evaluation --conf '{"mode": "full"}'

# 仅评测检索(Embedding 微调后)
airflow dags trigger rag_evaluation --conf '{"mode": "retrieval_only", "finetune_target": "embedding"}'

# 微调对比
airflow dags trigger rag_evaluation --conf '{"mode": "full", "baseline_name": "v1.0", "comparison_mode": true}'

# 指定多轮对话数据集
airflow dags trigger rag_evaluation --conf '{"dialogue_file": "data/dialogue_dataset_v2.jsonl"}'
```

## 延迟阈值(端到端,按客服 RAG 真实场景校准)

| 指标 | 阈值 | 说明 |
|------|------|------|
| 检索 P95 | < 2000ms | 含 hybrid + rerank |
| 生成 P95 | < 10000ms | 含 LLM 生成 + 可选 GraphRAG |
| 端到端 P95 | < 12000ms | /api/v1/chat 全程 |

阈值可在 `configs/evaluation_config.yaml` 调整。

## 测试

```bash
.venv/bin/python -m pytest tests/unit/test_blackbox_eval.py tests/unit/test_dialogue_eval.py -q
```

覆盖:黑盒契约解析、session/节流、黑盒检索评分、judge 回落、意图归一化、多轮槽位/澄清/意图切换/完成率。26 个单测,无需网络。

## 备注

- **检索评测 vs 直连向量库**:主流路径用 chatbot 返回的 `sources`(测真实检索)。直连 Qdrant 作为可选的"纯向量召回离线基线",默认关闭(`retrieval.vector_db.enabled: false`),且 embedding 须与线上 BGE-M3 一致。
- **doc_id 对齐**:`retrieval_labels.json` 的 `relevant_docs` 需与 chatbot 实际返回的 `sources` id 命名一致。建议先跑一次拿几条真实 `sources` 校准标注。
- **限流与批量**:大批量评测受 9 rpm 约束,DAG 任务 timeout 已按此估(检索 1800s / 生成 3600s / 对话 1800s)。若在 chatbot 侧调高限流,同步上调 `CHATBOT_RATE_LIMIT_RPM`。
