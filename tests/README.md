# Test Suite for RAG Chatbot Evaluation Pipeline

This directory contains the complete test suite for the evaluation pipeline.

## Test Structure

```
tests/
├── __init__.py
├── unit/
│   ├── __init__.py
│   ├── test_retrieval_metrics.py    # Tests for MRR, NDCG, HitRate, Precision, Recall
│   ├── test_generation_metrics.py   # Tests for ROUGE, BLEU, BERTScore
│   └── test_statistics.py           # Tests for Bootstrap CI, p-value, significance testing
└── integration/
    ├── __init__.py
    └── test_pipeline.py             # End-to-end pipeline tests
```

## Running Tests

### Install Dependencies

First, ensure you have pytest and required dependencies:

```bash
# From evaluation_pipeline directory
pip install -r requirements.txt
pip install pytest numpy scipy
```

### Run All Tests

```bash
# From evaluation_pipeline directory
python -m pytest tests/ -v
```

### Run Specific Test Files

```bash
# Run only retrieval metrics tests
python -m pytest tests/unit/test_retrieval_metrics.py -v

# Run only generation metrics tests
python -m pytest tests/unit/test_generation_metrics.py -v

# Run only statistics tests
python -m pytest tests/unit/test_statistics.py -v

# Run integration tests
python -m pytest tests/integration/test_pipeline.py -v
```

### Run Specific Test Classes or Functions

```bash
# Run only MRR tests
python -m pytest tests/unit/test_retrieval_metrics.py::TestMRR -v

# Run a specific test
python -m pytest tests/unit/test_retrieval_metrics.py::TestMRR::test_mrr_perfect_ranking -v
```

### Run with Coverage

```bash
# Install coverage tool
pip install pytest-cov

# Run tests with coverage report
python -m pytest tests/ --cov=src --cov-report=html --cov-report=term
```

## Test Coverage

### Unit Tests

- **test_retrieval_metrics.py**: Tests for all retrieval metrics
  - MRR (Mean Reciprocal Rank)
  - NDCG (Normalized Discounted Cumulative Gain)
  - HitRate
  - Precision@K
  - Recall@K
  - Combined metrics computation

- **test_generation_metrics.py**: Tests for generation metrics
  - ROUGE-L, ROUGE-1, ROUGE-2
  - BLEU-4 with different n-grams
  - BERTScore (placeholder without package)
  - Combined metrics computation
  - Edge cases (empty strings, unicode, etc.)

- **test_statistics.py**: Tests for statistical analysis
  - Bootstrap confidence intervals
  - P-value calculation (t-test, Wilcoxon, Mann-Whitney)
  - Paired bootstrap tests
  - Delta status computation
  - Multiple comparison correction (Bonferroni, Holm, FDR)
  - Multi-metric significance testing

### Integration Tests

- **test_pipeline.py**: End-to-end pipeline tests
  - Retrieval evaluation pipeline
  - Generation evaluation pipeline
  - Statistical analysis workflow
  - Complete pipeline execution
  - Multi-model comparison
  - Edge cases and error handling

## Test Fixtures

Tests use pytest fixtures for sample data:

- `sample_retrieval_results`: Sample retrieval data for testing
- `sample_generation_data`: Sample prediction/reference pairs
- `sample_metric_scores`: Baseline and current metric scores
- `pipeline_data`: Complete pipeline test data
- `multi_model_comparison_data`: Data for comparing multiple models

## Writing New Tests

When adding new tests, follow these patterns:

```python
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from module import function_to_test

class TestFeature:
    """Test feature X."""

    def test_basic_case(self):
        """Test basic functionality."""
        result = function_to_test(input_data)
        assert result == expected_output

    def test_edge_case(self):
        """Test edge case."""
        result = function_to_test(edge_case_input)
        assert result == expected_edge_output

    @pytest.fixture
    def sample_data(self):
        """Sample data for tests."""
        return {...}
```

## Continuous Integration

To run tests in CI:

```yaml
# Example GitHub Actions workflow
- name: Run tests
  run: |
    pip install pytest numpy scipy
    python -m pytest tests/ -v --tb=short
```

## Troubleshooting

### Import Errors

If you get import errors, ensure you're running from the `evaluation_pipeline` directory:

```bash
cd /path/to/chatbot_test_pipeline/evaluation_pipeline
python -m pytest tests/ -v
```

### Missing Dependencies

If tests fail due to missing packages:

```bash
pip install numpy scipy pytest
```

For ROUGE/BLEU tests, the fallback implementations are used by default.
To use official packages:

```bash
pip install rouge-score sacrebleu
```

For BERTScore (optional):

```bash
pip install bert-score
```
