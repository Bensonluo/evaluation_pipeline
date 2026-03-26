-- ============================================================================
-- 评测系统数据库初始化脚本
-- ============================================================================
-- 创建日期: 2026-03-25
-- 描述: 微调评测系统的数据库 Schema，包含基线管理、评测结果、测试数据集
-- ============================================================================

-- ----------------------------------------------------------------------------
-- 测试数据集表
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS test_datasets (
    id              SERIAL PRIMARY KEY,
    dataset_name    VARCHAR(100) NOT NULL,
    dataset_type    VARCHAR(20) NOT NULL,      -- 'qa', 'retrieval', 'intent'
    version         VARCHAR(20),
    description     TEXT,
    created_at      TIMESTAMP DEFAULT NOW(),
    data            JSONB NOT NULL,
    stats           JSONB,                      -- 样本数、意图分布等
    metadata        JSONB,
    UNIQUE(dataset_name, version)
);

COMMENT ON TABLE test_datasets IS '测试数据集表，存储QA、检索、意图识别等测试数据';
COMMENT ON COLUMN test_datasets.dataset_type IS '数据集类型: qa(问答), retrieval(检索), intent(意图识别)';
COMMENT ON COLUMN test_datasets.data IS '测试数据JSON，包含questions, contexts, labels等';
COMMENT ON COLUMN test_datasets.stats IS '数据集统计信息: 样本数、类别分布、平均长度等';

-- ----------------------------------------------------------------------------
-- 评测基线表：存储每次微调后的基准指标
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS evaluation_baselines (
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
    generation_bleu4        DECIMAL(5,4),
    generation_bertscore    DECIMAL(5,4),

    -- RAGAS 指标
    ragas_faithfulness      DECIMAL(3,2),
    ragas_answer_relevance  DECIMAL(3,2),
    ragas_context_relevance DECIMAL(3,2),

    -- 意图识别指标
    intent_accuracy         DECIMAL(5,4),
    intent_f1_macro         DECIMAL(5,4),

    -- 业务指标
    business_accuracy       DECIMAL(5,4),
    business_response_rate  DECIMAL(5,4),
    business_no_hallucination_rate DECIMAL(5,4),

    -- 延迟指标
    latency_p50_ms      INTEGER,
    latency_p95_ms      INTEGER,
    latency_p99_ms      INTEGER,

    metadata            JSONB,
    UNIQUE(baseline_name)
);

COMMENT ON TABLE evaluation_baselines IS '评测基线表，存储每次微调后的基准性能指标';
COMMENT ON COLUMN evaluation_baselines.finetune_target IS '微调目标: embedding(向量), llm(生成模型), intent(意图识别), all(全栈)';
COMMENT ON COLUMN evaluation_baselines.retrieval_mrr5 IS '检索平均倒数排名@5';
COMMENT ON COLUMN evaluation_baselines.retrieval_ndcg5 IS '检索归一化折损累计增益@5';
COMMENT ON COLUMN evaluation_baselines.retrieval_hitrate3 IS '检索命中率@3';
COMMENT ON COLUMN evaluation_baselines.retrieval_precision5 IS '检索精确率@5';
COMMENT ON COLUMN evaluation_baselines.generation_rouge_l IS '生成ROUGE-L分数';
COMMENT ON COLUMN evaluation_baselines.generation_bleu4 IS '生成BLEU-4分数';
COMMENT ON COLUMN evaluation_baselines.generation_bertscore IS '生成BERTScore分数';
COMMENT ON COLUMN evaluation_baselines.ragas_faithfulness IS 'RAGAS忠实度分数(0-1)';
COMMENT ON COLUMN evaluation_baselines.ragas_answer_relevance IS 'RAGAS答案相关性(0-1)';
COMMENT ON COLUMN evaluation_baselines.ragas_context_relevance IS 'RAGAS上下文相关性(0-1)';
COMMENT ON COLUMN evaluation_baselines.intent_accuracy IS '意图识别准确率';
COMMENT ON COLUMN evaluation_baselines.intent_f1_macro IS '意图识别F1-macro分数';
COMMENT ON COLUMN evaluation_baselines.business_accuracy IS '业务准确率';
COMMENT ON COLUMN evaluation_baselines.business_response_rate IS '业务响应率';
COMMENT ON COLUMN evaluation_baselines.business_no_hallucination_rate IS '无幻觉率';

-- ----------------------------------------------------------------------------
-- 评测结果表：每次评测的详细结果
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS evaluation_results (
    id                  SERIAL PRIMARY KEY,
    baseline_id         INTEGER REFERENCES evaluation_baselines(id) ON DELETE SET NULL,
    eval_name           VARCHAR(100) NOT NULL,
    eval_at             TIMESTAMP DEFAULT NOW(),
    dataset_id          INTEGER REFERENCES test_datasets(id) ON DELETE SET NULL,

    -- 评测配置
    mode                VARCHAR(20),           -- 'full', 'retrieval_only', etc.
    finetune_target     VARCHAR(50),

    -- 各维度指标汇总 (JSON格式存储详细分解指标)
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

COMMENT ON TABLE evaluation_results IS '评测结果表，记录每次评测的详细数据和对比分析';
COMMENT ON COLUMN evaluation_results.mode IS '评测模式: full(全流程), retrieval_only(仅检索), generation_only(仅生成), intent_only(仅意图)';
COMMENT ON COLUMN evaluation_results.comparison_mode IS '是否为对比模式(对比两个基线)';
COMMENT ON COLUMN evaluation_results.delta_metrics IS '指标差异: {"metric": delta, ...}';
COMMENT ON COLUMN evaluation_results.significance_tests IS '显著性检验结果';
COMMENT ON COLUMN evaluation_results.error_analysis IS '错误分析: 失败案例、错误类型分布等';

-- ----------------------------------------------------------------------------
-- 创建索引
-- ----------------------------------------------------------------------------

-- 评测结果索引
CREATE INDEX IF NOT EXISTS idx_evaluation_results_eval_at ON evaluation_results(eval_at DESC);
CREATE INDEX IF NOT EXISTS idx_evaluation_results_baseline_id ON evaluation_results(baseline_id);
CREATE INDEX IF NOT EXISTS idx_evaluation_results_dataset_id ON evaluation_results(dataset_id);
CREATE INDEX IF NOT EXISTS idx_evaluation_results_status ON evaluation_results(status);
CREATE INDEX IF NOT EXISTS idx_evaluation_results_mode ON evaluation_results(mode);

-- 评测基线索引
CREATE INDEX IF NOT EXISTS idx_evaluation_baselines_created_at ON evaluation_baselines(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_evaluation_baselines_finetune_target ON evaluation_baselines(finetune_target);

-- 测试数据集索引
CREATE INDEX IF NOT EXISTS idx_test_datasets_name ON test_datasets(dataset_name);
CREATE INDEX IF NOT EXISTS idx_test_datasets_type ON test_datasets(dataset_type);

-- JSONB 字段索引用于高效查询
CREATE INDEX IF NOT EXISTS idx_evaluation_results_metadata ON evaluation_results USING GIN(metadata);
CREATE INDEX IF NOT EXISTS idx_evaluation_baselines_metadata ON evaluation_baselines USING GIN(metadata);
CREATE INDEX IF NOT EXISTS idx_test_datasets_metadata ON test_datasets USING GIN(metadata);

-- ----------------------------------------------------------------------------
-- 创建视图
-- ----------------------------------------------------------------------------

-- 最新基线概览视图
CREATE OR REPLACE VIEW v_latest_baselines AS
SELECT 
    id,
    baseline_name,
    model_version,
    finetune_target,
    description,
    created_at,
    -- 检索指标摘要
    ROUND(retrieval_ndcg5::NUMERIC, 4) as ndcg5,
    -- 生成指标摘要
    ROUND(generation_rouge_l::NUMERIC, 4) as rouge_l,
    -- RAGAS指标摘要
    ROUND(ragas_faithfulness::NUMERIC, 2) as faithfulness,
    -- 业务指标摘要
    ROUND(business_accuracy::NUMERIC, 4) as accuracy,
    ROUND(latency_p95_ms) as latency_p95_ms
FROM evaluation_baselines
ORDER BY created_at DESC;

COMMENT ON VIEW v_latest_baselines IS '最新基线概览视图，用于快速查看关键指标';

-- 评测结果趋势视图
CREATE OR REPLACE VIEW v_evaluation_trends AS
SELECT 
    er.id,
    er.eval_name,
    er.eval_at,
    er.mode,
    eb.baseline_name,
    eb.finetune_target,
    eb.model_version,
    -- 提取JSONB中的关键指标
    (er.retrieval_metrics->>'ndcg5')::DECIMAL(5,4) as ndcg5,
    (er.generation_metrics->>'rouge_l')::DECIMAL(5,4) as rouge_l,
    (er.ragas_metrics->>'faithfulness')::DECIMAL(3,2) as faithfulness,
    er.latency_p95_ms,
    er.status
FROM evaluation_results er
LEFT JOIN evaluation_baselines eb ON er.baseline_id = eb.id
ORDER BY er.eval_at DESC;

COMMENT ON VIEW v_evaluation_trends IS '评测趋势视图，用于追踪性能变化';

-- ----------------------------------------------------------------------------
-- 初始化示例数据 (可选)
-- ----------------------------------------------------------------------------

-- 插入默认基线记录 (微调前基准)
INSERT INTO evaluation_baselines (
    baseline_name,
    model_version,
    finetune_target,
    description,
    retrieval_mrr5,
    retrieval_ndcg5,
    retrieval_hitrate3,
    retrieval_precision5,
    generation_rouge_l,
    generation_bleu4,
    ragas_faithfulness,
    ragas_answer_relevance,
    intent_accuracy,
    business_accuracy,
    latency_p95_ms
) VALUES (
    'baseline_pre_finetuning',
    'qwen2.5-7b-instruct',
    'all',
    '微调前的基准性能指标',
    0.6234,
    0.5891,
    0.7125,
    0.4567,
    0.5432,
    0.3456,
    0.75,
    0.68,
    0.8234,
    0.7890,
    250
) ON CONFLICT (baseline_name) DO NOTHING;

-- 插入示例测试数据集
INSERT INTO test_datasets (
    dataset_name,
    dataset_type,
    version,
    description,
    data,
    stats
) VALUES (
    'qa_test_set_v1',
    'qa',
    '1.0',
    '问答系统测试数据集',
    '{"samples": [{"question": "如何注册账号？", "context": "...", "answer": "..."}]}'::JSONB,
    '{"total_samples": 100, "avg_question_length": 15, "avg_answer_length": 50}'::JSONB
) ON CONFLICT (dataset_name, version) DO NOTHING;

-- ============================================================================
-- 初始化完成
-- ============================================================================
