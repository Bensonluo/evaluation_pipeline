<div align="center">

# RAG Evaluation Pipeline

**An automated, black-box evaluation pipeline for RAG systems — covering retrieval, generation, intent, and multi-turn dialogue quality. Built on Apache Airflow with statistical significance testing.**

[![GitHub stars](https://img.shields.io/github/stars/Bensonluo/evaluation_pipeline?style=for-the-badge)](https://github.com/Bensonluo/evaluation_pipeline/stargazers)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)
[![Companion Project](https://img.shields.io/badge/Companion-rag__chatbot-blueviolet?style=for-the-badge)](https://github.com/Bensonluo/rag_chatbot)

[![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Apache Airflow](https://img.shields.io/badge/Airflow-2.8+-017CEE?logo=apache-airflow&logoColor=white)](https://airflow.apache.org/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-4169E1?logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)

<!-- 🎬 录制说明:录 Airflow DAG 触发 → 各 task 执行 → 最终 HTML 报告生成 -->
<img src="docs/assets/demo.gif" alt="Evaluation Pipeline Demo" width="80%">

*🎬 Replace this with a 30s GIF of the pipeline run — see [Recording Guide](#-demo-recording-guide) below*

</div>

---

## 📌 Table of Contents

- [Why This Project](#-why-this-project)
- [Key Highlights](#-key-highlights)
- [Black-Box Philosophy](#-black-box-philosophy)
- [Evaluation Dimensions](#-evaluation-dimensions)
- [Quick Start](#-quick-start)
- [Fine-tune Comparison](#-fine-tune-comparison)
- [Pipeline Architecture](#-pipeline-architecture)
- [中文说明](#-中文说明)

---

## 💡 Why This Project

RAG evaluation is the silent killer of LLM projects. Most teams ship a chatbot, get positive vibes in demos, then discover in production that:

- ❌ Retrieval misses the right docs 30% of the time — but no one measured it
- ❌ Switching embedding models "felt better" — but was it actually better?
- ❌ Multi-turn dialogue breaks after intent switches — undetected
- ❌ There's no baseline, so improvements can't be proven

This project solves all of them:

> 🎯 **A black-box evaluation pipeline that treats your RAG system as an opaque endpoint** — measuring retrieval quality, generation quality, intent accuracy, and multi-turn dialogue robustness. With before/after comparison and statistical significance testing.

It's the **missing half** of any serious RAG project. Pairs perfectly with [rag_chatbot](https://github.com/Bensonluo/rag_chatbot).

---

## ✨ Key Highlights

<div align="center">

| 📊 Retrieval | ✍️ Generation | 🎯 Intent |
|:---:|:---:|:---:|
| MRR@5, NDCG@5 | ROUGE, BLEU | Intent Accuracy |
| HitRate@3, Precision@5 | LLM-as-a-Judge | Intent F1 (Macro) |
| Black-box `sources` scoring | Relevance/fluency/safety | Per-turn accuracy |

| 💬 Multi-turn Dialogue | 📈 Comparison | 📋 Output |
|:---:|:---:|:---:|
| Slot Precision/Recall/F1 | Delta Metrics | HTML reports |
| Avg clarification turns | Bootstrap confidence | JSON exports |
| Intent switch accuracy | p-value significance | Executive summary |

| 📈 Stats | | |
|:---:|:---:|:---:|
| **6** evaluation dimensions | **8+** metrics | **26** unit tests |
| **Black-box** by design | **Airflow** orchestrated | **Stat-sig** tested |

</div>

### 🧠 What makes it different

1. **Black-box by design** — only talks to the public `/api/v1/chat` endpoint, never touches vector DBs or internal services. Measures what users actually experience.
2. **Multi-turn dialogue evaluation** — PARADISE / MultiWOZ-style metrics specifically designed for LangGraph dialogue engines (slot filling, intent switch, task completion)
3. **Fine-tune comparison mode** — load baseline, run new evaluation, compute deltas with Bootstrap confidence intervals and p-values
4. **LLM-as-a-Judge** — uses GLM (or OpenAI) to grade response quality on relevance, fluency, completeness, safety
5. **Airflow-orchestrated** — scheduled daily/weekly or triggered on-demand for A/B experiments

---

## 📦 Black-Box Philosophy

The pipeline treats the RAG system as an **opaque endpoint**:

```
┌──────────────────────┐         ┌──────────────────────┐
│                      │  POST   │                      │
│  Evaluation Pipeline │ ──────> │   RAG Chatbot        │
│                      │  /chat  │   (black box)        │
│  - measures          │ <────── │                      │
│  - compares          │  JSON   │  - retrieval         │
│  - reports           │         │  - generation        │
└──────────────────────┘         │  - dialogue engine   │
                                 └──────────────────────┘
```

**Why black-box?**
- ✅ Measures **end-to-end** user experience (hybrid retrieval + rerank + GraphRAG + dialogue)
- ✅ **Zero coupling** — works with any RAG system exposing a chat endpoint
- ✅ Won't break when the chatbot's internals change
- ✅ Easy to point at staging vs production for environment comparison

### Black-box contract

```http
POST /api/v1/chat
Content-Type: application/json

{
  "message": "我要退款",
  "session_id": 12345
}
```

```json
{
  "content": "请提供您的订单号...",
  "intent": "refund",
  "sources": ["doc_refund_policy_001"],
  "metadata": {
    "filled_slots": {"order_id": "..."},
    "pending_slots": ["reason"]
  }
}
```

The pipeline reads `content`, `sources`, `intent`, and slot metadata — nothing else.

---

## 📐 Evaluation Dimensions

| Dimension | Metrics | What it tells you |
|-----------|---------|-------------------|
| **Retrieval** | MRR@5, NDCG@5, HitRate@3, Precision@5 | Are we surfacing the right docs? |
| **Generation** | LLM-as-a-Judge + ROUGE/BLEU | Is the answer good? |
| **Intent** | Accuracy, F1 (Macro) | Does intent classification work? |
| **Multi-turn** | Slot F1, avg clarification turns, intent switch accuracy, task completion | Does the dialogue engine hold up? |
| **Latency** | P50 / P95 / P99 | Is it fast enough for production? |
| **Fine-tune effect** | Delta Metrics + p-value | Did the fine-tune actually help? |

---

## 🚀 Quick Start

### Option 1: Full setup

```bash
git clone https://github.com/Bensonluo/evaluation_pipeline.git
cd evaluation_pipeline

pip install -r requirements.txt

# Configure
export AIRFLOW_HOME=/path/to/airflow

# Point at your RAG chatbot (anonymous access supported)
export CHATBOT_API_URL=http://localhost:8000
export CHATBOT_RATE_LIMIT_RPM=9          # Stay under server's ~10 rpm limit

# LLM-as-a-Judge (GLM recommended)
export JUDGE_PROVIDER=glm
export JUDGE_MODEL=glm-5.2
export GLM_API_KEY=your-glm-key
export GLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4
```

### Option 2: Run without Airflow

```bash
python scripts/run_evaluation.py --mode full
```

### Option 3: Trigger via Airflow

```bash
# Start Airflow
airflow webserver -p 8080 &
airflow scheduler &

# Trigger full evaluation
airflow dags trigger rag_evaluation --conf '{"mode": "full"}'
```

---

## 🔬 Fine-tune Comparison

The killer feature for ML teams: prove that fine-tune actually helped.

### How it works

1. **Load baseline** — read historical results from `evaluation_baselines` table
2. **Run current evaluation** — execute full pipeline on new model/config
3. **Compute deltas** — relative + absolute change per metric
4. **Statistical significance** — Bootstrap confidence intervals + p-values
5. **Generate report** — percentage change + significance verdict

### Sample Delta Report

```
| Metric              | Baseline | Current | Delta  | p-value | Status                |
|---------------------|----------|---------|--------|---------|-----------------------|
| MRR@5               | 0.723    | 0.789   | +9.1%  | 0.023   | ✅ SIGNIFICANT_IMPROVED |
| NDCG@5              | 0.681    | 0.712   | +4.5%  | 0.045   | ✅ MARGINAL_IMPROVED   |
| Slot F1 (dialogue)  | 0.820    | 0.871   | +6.2%  | 0.018   | ✅ IMPROVED            |
| Judge Relevance     | 0.452    | 0.438   | -3.1%  | 0.120   | ➖ NOT_SIGNIFICANT     |
```

### Evaluation Modes

| Mode | Scope | Use case |
|------|-------|----------|
| `full` | All dimensions | Comprehensive evaluation |
| `retrieval_only` | Retrieval metrics | After embedding fine-tune |
| `generation_only` | Generation metrics | After LLM fine-tune |
| `intent_only` | Intent metrics | After intent classifier fine-tune |

```bash
# Compare against baseline v1.0
airflow dags trigger rag_evaluation --conf '{
  "mode": "full",
  "baseline_name": "v1.0",
  "comparison_mode": true
}'
```

---

## 🏗️ Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Airflow DAG Schedule                               │
│                         (daily / weekly / on-demand)                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  1. data_preparation                                                        │
│     Load test datasets (Q&A, retrieval labels, multi-turn dialogue cases)  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
┌──────────────────────┐ ┌─────────────────────┐ ┌─────────────────────────┐
│ 2. retrieval_eval    │ │ 3. generation_eval  │ │ 4. dialogue_eval        │
│  calls /api/v1/chat  │ │  calls /api/v1/chat │ │  calls /api/v1/chat xN  │
│  scores `sources`    │ │  LLM-as-a-Judge     │ │  slots / switch / done  │
│  MRR/NDCG/HitRate    │ │  ROUGE/BLEU         │ │  task completion rate   │
└──────────────────────┘ └─────────────────────┘ └─────────────────────────┘
                    └───────────────┼───────────────┘
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  5. baseline_comparison  (only in comparison mode)                          │
│     Delta Metrics (current vs baseline) + statistical significance test     │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  6. report_generation                                                        │
│     HTML/JSON reports with retrieval / generation / dialogue metrics        │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Multi-turn Dialogue Evaluation Methodology

Based on PARADISE / MultiWOZ paradigms, designed for LangGraph dialogue engines:

| Metric | Definition | LangGraph Node |
|--------|------------|----------------|
| Slot Precision/Recall/F1 | final `filled_slots` vs `required_slots` | `collect_slots` |
| Avg clarification turns | turns to fill all required slots (lower is better) | `collect_slots` → `generate_response` |
| Per-turn intent accuracy | detected intent vs labeled intent | `detect_intent` |
| Intent switch accuracy | switch turn follows new task? | `handle_switch` + state stack |
| Task completion rate | all required slots filled by final turn | end-to-end |

---

## 📁 Project Structure

```
evaluation_pipeline/
├── src/
│   ├── config.py             # Black-box endpoint / judge / throttling config
│   ├── database.py           # PostgreSQL connection
│   ├── metrics/
│   │   ├── retrieval.py      # MRR, NDCG, HitRate
│   │   ├── generation.py     # ROUGE, BLEU
│   │   ├── ragas.py          # RAGAS + LLMJudgeClient (GLM)
│   │   └── statistics.py     # Bootstrap, p-value
│   ├── dataset/              # Loader, sampler, converter
│   ├── api/
│   │   └── chatbot.py        # Black-box /api/v1/chat client (session/throttle)
│   ├── evaluation/           # Retrieval / generation / dialogue / intent / error
│   └── reporting/            # Delta, HTML, JSON reporters
├── dags/
│   └── rag_evaluation_dag.py # Airflow DAG
├── configs/
│   ├── default_metrics.yaml
│   ├── ragas_prompts.yaml
│   └── evaluation_config.yaml  # endpoint / intents / judge / thresholds
├── data/                     # Test datasets (gitignored local fixtures)
├── sql/
│   └── init_evaluation_db.sql
├── scripts/
│   └── run_evaluation.py     # Standalone runner (no Airflow needed)
└── tests/                    # 26 unit tests, no network required
```

---

## 🧪 Testing

```bash
pytest tests/unit/test_blackbox_eval.py tests/unit/test_dialogue_eval.py -q
```

Covers: black-box contract parsing, session/throttling, retrieval scoring, judge fallback, intent normalization, multi-turn slot/clarification/intent-switch/completion. **26 unit tests, no network required.**

---

## 🗺️ Roadmap

- [x] Black-box evaluation (retrieval / generation / intent / dialogue)
- [x] LLM-as-a-Judge (GLM / OpenAI)
- [x] Multi-turn dialogue evaluation (slots / intent switch / completion)
- [x] Fine-tune comparison with statistical significance
- [x] Airflow orchestration + standalone runner
- [x] 26 unit tests
- [ ] RAGAS integration for faithfulness / answer relevancy
- [ ] Automated regression alerts (Slack / webhook)
- [ ] Visual dashboard for metric trends over time

---

## 🤝 Companion Project

This pipeline is designed to evaluate [**rag_chatbot**](https://github.com/Bensonluo/rag_chatbot) — a LangGraph-based enterprise customer service chatbot. Together they form a complete develop → evaluate → iterate loop:

```
rag_chatbot  ──(improvements)──>  evaluation_pipeline
     ▲                                    │
     └──────────(feedback)────────────────┘
```

---

## 🤝 Contributing

PRs welcome — especially:
- 📊 New evaluation metrics (faithfulness, answer relevancy)
- 🌍 New dataset formats or domains
- 📈 Visualization improvements for reports
- 🐛 Bug fixes with a failing test

---

## 📜 License

[MIT](LICENSE) — free for personal and commercial use.

If this project helped you ship a better RAG system, please ⭐ star the repo.

---

## 📬 Contact

- 💼 **Portfolio**: [benluo.art](https://benluo.art)
- 🐙 **GitHub**: [@Bensonluo](https://github.com/Bensonluo)
- 💬 **Issues**: [GitHub Issues](https://github.com/Bensonluo/evaluation_pipeline/issues)

---

## 🇨🇳 中文说明

**RAG 端到端评测流水线** — 基于 Airflow 的黑盒评测系统。

### 核心亮点

- **黑盒设计**:只通过 `POST /api/v1/chat` 访问被测系统,测真实用户体验
- **六大评测维度**:检索质量(MRR/NDCG/HitRate)、生成质量(ROUGE/BLEU/LLM-Judge)、意图识别、多轮对话、延迟、微调效果
- **多轮对话评测**:基于 PARADISE/MultiWOZ 范式,专门为 LangGraph 对话引擎设计
  - 槽位 Precision/Recall/F1
  - 平均澄清轮数
  - 意图切换准确率
  - 任务完成率
- **微调对比模式**:Delta Metrics + Bootstrap 置信区间 + p-value 统计显著性检验
- **LLM-as-a-Judge**:默认走 GLM OpenAI 兼容网关
- **Airflow 编排**:支持 daily/weekly 调度 + 手动触发
- **26 个单测**:无需网络,覆盖黑盒契约/检索评分/多轮评测

### 快速开始

```bash
git clone https://github.com/Bensonluo/evaluation_pipeline.git
cd evaluation_pipeline
pip install -r requirements.txt

# 配置被测系统(支持匿名访问)
export CHATBOT_API_URL=http://localhost:8000
export CHATBOT_RATE_LIMIT_RPM=9

# 配置 Judge
export JUDGE_PROVIDER=glm
export GLM_API_KEY=your-glm-key

# 运行
python scripts/run_evaluation.py --mode full
```

### 配套项目

本流水线专为评测 [rag_chatbot](https://github.com/Bensonluo/rag_chatbot) 设计,两者共同构成「开发 → 评测 → 迭代」闭环。

---

<details>
<summary>🎬 Demo Recording Guide (for maintainers)</summary>

### How to record the hero GIF

1. **Tool**: [Kap](https://getkap.co/) (Mac) or [licecap](https://www.cockos.com/licecap/)
2. **Content** (~30s):
   - 0-5s: Open Airflow UI, show the `rag_evaluation` DAG
   - 5-15s: Trigger DAG with `{"mode": "full"}`, show tasks lighting up green
   - 15-25s: Open the generated HTML report, scroll through metric tables
   - 25-30s: Highlight a Delta comparison table (baseline vs current with p-values)
3. **Save to**: `docs/assets/demo.gif` (keep under 5MB)

</details>

<!--
RECORDING_TODO:
1. Record demo.gif → docs/assets/demo.gif
2. Replace placeholder img tag in hero section
3. Consider adding a Live Demo link (deploy Airflow + expose read-only?)
-->
