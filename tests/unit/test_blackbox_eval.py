"""Unit tests for the black-box chatbot client and evaluation adapters.

These tests do not make network calls. They cover:
- GenerateResponse.from_api parsing of the new /api/v1/chat contract
- ChatbotClient session id allocation + throttle (no real HTTP)
- RetrievalEvaluator.evaluate_via_api against a stub chatbot client
- GenerationEvaluator heuristic fallback when no judge is configured
- FormatConverter intent normalization against the customer-service enum
"""

from __future__ import annotations

import asyncio
from typing import List

import pytest

from src.api.chatbot import ChatbotClient, GenerateResponse
from src.dataset.converter import FormatConverter
from src.evaluation.generation_eval import GenerationEvaluator
from src.evaluation.retrieval_eval import RetrievalEvaluator


# ── GenerateResponse.from_api ────────────────────────────────────────────────


def test_from_api_parses_new_contract():
    data = {
        "content": "您好，请提供订单号。",
        "session_id": 7,
        "intent": "refund",
        "sources": ["doc_a", "doc_b"],
        "metadata": {
            "confidence": 0.91,
            "filled_slots": {"order_id": "SO123"},
            "pending_slots": ["reason"],
        },
        "dialogue_state": {
            "phase": "refund",
            "pending_slots": ["reason"],
            "filled_slots": {"order_id": "SO123"},
        },
    }
    resp = GenerateResponse.from_api(data, latency_ms=42)
    assert resp.response == "您好，请提供订单号。"
    assert resp.intent == "refund"
    assert resp.sources == ["doc_a", "doc_b"]
    assert resp.filled_slots == {"order_id": "SO123"}
    assert resp.pending_slots == ["reason"]
    assert resp.confidence == 0.91
    assert resp.status == "success"


def test_from_api_handles_missing_fields():
    # Non-RAG intents may return no sources/metadata.
    resp = GenerateResponse.from_api({"content": "你好！", "session_id": 1, "intent": "greeting"}, 5)
    assert resp.response == "你好！"
    assert resp.sources == []
    assert resp.filled_slots == {}
    assert resp.pending_slots == []


def test_error_response_factory():
    resp = GenerateResponse.error_response("boom")
    assert resp.status == "error"
    assert resp.error == "boom"
    assert resp.response == ""


# ── ChatbotClient session allocation + throttle ──────────────────────────────


def test_allocate_session_id_is_unique_and_positive():
    client = ChatbotClient("http://x", rate_limit_rpm=1000)
    ids = {client.allocate_session_id() for _ in range(50)}
    assert len(ids) == 50
    assert all(i > 0 for i in ids)


def test_throttle_enforces_min_interval(monkeypatch):
    # The throttle sleeps when (last_request_at + min_interval) > now.
    # We drive a virtual clock and record sleep durations without touching the
    # real event-loop clock. Return a monotonically increasing time so any
    # internal asyncio scheduling stays happy.
    clock = {"t": 0.0}

    def fake_monotonic():
        return clock["t"]

    slept = []

    async def fake_sleep(seconds):
        slept.append(seconds)
        clock["t"] += seconds  # advance virtual clock by the waited amount

    monkeypatch.setattr("time.monotonic", fake_monotonic)
    monkeypatch.setattr("asyncio.sleep", fake_sleep)

    client = ChatbotClient("http://x", rate_limit_rpm=120)  # min_interval = 0.5s
    # First call: last_request_at=0, now=0 → wait 0.5.
    asyncio.run(client._throttle())
    # Second call: last_request_at=0.5 (set after first), now=0.5 → wait 0.5.
    asyncio.run(client._throttle())
    assert len(slept) == 2
    assert all(abs(s - 0.5) < 1e-9 for s in slept)


# ── RetrievalEvaluator.evaluate_via_api (stub client) ────────────────────────


class _StubChatbot:
    """Minimal stub matching the subset of ChatbotClient the evaluator uses."""

    def __init__(self, responses: List[GenerateResponse]):
        self._responses = responses
        self._n = 0

    def allocate_session_id(self) -> int:
        self._n += 1
        return self._n

    async def batch_generate(self, messages, session_ids=None, max_concurrency=3, **kwargs):
        from src.api.chatbot import BatchGenerateResponse

        return BatchGenerateResponse(
            responses=self._responses,
            total_latency_ms=10,
            success_count=sum(1 for r in self._responses if r.status == "success"),
            failure_count=sum(1 for r in self._responses if r.status != "success"),
        )


def test_evaluate_via_api_scores_sources():
    dataset = [
        {"query": "q1", "relevant_docs": ["doc_a", "doc_c"]},
        {"query": "q2", "relevant_docs": ["doc_z"]},
    ]
    # q1: sources rank doc_a first → hit; q2: sources miss → no hit.
    responses = [
        GenerateResponse(response="r1", latency_ms=1, sources=["doc_a", "doc_b"]),
        GenerateResponse(response="r2", latency_ms=1, sources=["doc_x"]),
    ]
    evaluator = RetrievalEvaluator({"mrr_k": 5, "ndcg_k": 5, "hitrate_k": 3, "precision_k": 5})
    result = asyncio.run(evaluator.evaluate_via_api(dataset, _StubChatbot(responses)))

    assert result.hitrate_at_3 == 0.5  # 1 of 2 queries hit
    assert result.mrr_at_5 == pytest.approx(0.5)  # 1.0 + 0.0 / 2
    assert result.empty_source_count == 0
    assert len(result.per_query_results) == 2


def test_evaluate_via_api_counts_empty_sources():
    dataset = [{"query": "q", "relevant_docs": ["doc_a"]}]
    # A failed response yields no sources → counted as empty.
    responses = [GenerateResponse.error_response("HTTP 500")]
    evaluator = RetrievalEvaluator({"mrr_k": 5, "ndcg_k": 5, "hitrate_k": 3, "precision_k": 5})
    result = asyncio.run(evaluator.evaluate_via_api(dataset, _StubChatbot(responses)))
    assert result.empty_source_count == 1
    assert result.mrr_at_5 == 0.0


# ── GenerationEvaluator heuristic fallback ───────────────────────────────────


def test_generation_eval_heuristic_fallback_without_judge():
    evaluator = GenerationEvaluator({"judge_model": "glm-5.2"}, llm_judge_client=None)
    dataset = [
        {"query": "退款政策?", "response": "7天无理由退款，请提供订单号。",
         "reference_response": "7天无理由退款"},
    ]
    result = asyncio.run(evaluator.evaluate(dataset))
    assert 0.0 <= result.avg_relevance <= 1.0
    assert 0.0 <= result.avg_fluency <= 1.0
    assert result.avg_safety == 1.0  # heuristic assumes safe
    assert len(result.per_response_results) == 1


# ── Intent normalization (customer-service enum) ─────────────────────────────


@pytest.mark.parametrize("raw,expected", [
    ("Refund", "refund"),
    ("REFUND", "refund"),
    ("HOW_TO", "unknown"),       # legacy label → unknown (not in new enum)
    ("query_order", "query_order"),
    ("bogus", "unknown"),
    (None, "unknown"),
])
def test_normalize_intent(raw, expected):
    assert FormatConverter.normalize_intent(raw) == expected


def test_validate_intent_case_insensitive():
    assert FormatConverter.validate_intent("FAQ") is True
    assert FormatConverter.validate_intent("Track_Shipping") is True
    assert FormatConverter.validate_intent("OTHER") is False
