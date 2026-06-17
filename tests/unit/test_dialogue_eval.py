"""Unit tests for multi-turn dialogue evaluation (black-box).

No network calls. Covers:
- DialogueCase parsing (compact per-turn-object + explicit-list forms)
- Slot precision/recall/F1 against final filled_slots
- Clarification-turn counting
- Per-turn intent accuracy
- Intent-switch detection
- Task completion rate
- evaluate_via_api with a stub chatbot client
- dialogue_dataset.jsonl loads and parses cleanly
"""

from __future__ import annotations

import asyncio
from typing import List

import pytest

from src.api.chatbot import GenerateResponse
from src.dataset.loader import DatasetLoader
from src.evaluation.dialogue_eval import (
    DialogueCase,
    DialogueEvaluator,
)


# ── DialogueCase parsing ─────────────────────────────────────────────────────


def test_case_from_compact_turn_objects():
    data = {
        "case_id": "c1",
        "turns": [
            {"message": "退款，订单 SO1", "intent": "refund", "slots": {"order_id": "SO1"}},
            {"message": "原因是质量问题", "intent": "refund", "slots": {"order_id": "SO1", "reason": "质量问题"}},
        ],
        "required_slots": ["order_id", "reason"],
    }
    case = DialogueCase.from_dict(data)
    assert case.case_id == "c1"
    assert case.turns == ["退款，订单 SO1", "原因是质量问题"]
    assert case.expected_intent == ["refund", "refund"]
    assert case.expected_slots[1] == {"order_id": "SO1", "reason": "质量问题"}
    assert case.required_slot_set == {"order_id", "reason"}


def test_case_from_explicit_lists():
    data = {
        "case_id": "c2",
        "turns": ["你好", "查订单 SO2"],
        "expected_intent": ["greeting", "query_order"],
        "expected_slots": [{}, {"order_id": "SO2"}],
    }
    case = DialogueCase.from_dict(data)
    # required_slots defaults to the union of asserted slot keys.
    assert case.required_slot_set == {"order_id"}
    assert case.expected_intent == ["greeting", "query_order"]


def test_case_length_mismatch_raises():
    with pytest.raises(ValueError, match="expected_intent length"):
        DialogueCase(
            case_id="x",
            turns=["a", "b"],
            expected_intent=["refund"],          # wrong length
            expected_slots=[{}, {}],
        )


# ── Slot scoring + completion ────────────────────────────────────────────────


def _resp(intent: str, filled: dict) -> GenerateResponse:
    return GenerateResponse(
        response="ok", latency_ms=1, intent=intent, filled_slots=dict(filled)
    )


def test_slot_scoring_perfect_completion():
    case = DialogueCase(
        case_id="c",
        turns=["t1", "t2"],
        expected_intent=["refund", "refund"],
        expected_slots=[{"order_id": "SO1"}, {"order_id": "SO1", "reason": "破损"}],
        required_slots=["order_id", "reason"],
    )
    responses = [
        _resp("refund", {"order_id": "SO1"}),
        _resp("refund", {"order_id": "SO1", "reason": "破损"}),
    ]
    scored = DialogueEvaluator()._score_case(case, responses)
    assert scored["slot_precision"] == 1.0
    assert scored["slot_recall"] == 1.0
    assert scored["slot_f1"] == 1.0
    assert scored["task_complete"] is True
    # Turn 0 still had reason pending → 1 clarification turn before completion.
    assert scored["clarification_turns"] == 1


def test_slot_scoring_partial_recall():
    case = DialogueCase(
        case_id="c",
        turns=["t1"],
        expected_intent=["refund"],
        expected_slots=[{"order_id": "SO1", "reason": "破损"}],
        required_slots=["order_id", "reason"],
    )
    # System only filled order_id, not reason.
    responses = [_resp("refund", {"order_id": "SO1"})]
    scored = DialogueEvaluator()._score_case(case, responses)
    assert scored["slot_recall"] == 0.5
    assert scored["task_complete"] is False
    # Never completed → all turns count as clarification.
    assert scored["clarification_turns"] == 1


def test_slot_precision_penalizes_extra_slots():
    case = DialogueCase(
        case_id="c",
        turns=["t1"],
        expected_intent=["refund"],
        expected_slots=[{"order_id": "SO1"}],
        required_slots=["order_id"],
    )
    # System filled an unexpected extra slot.
    responses = [_resp("refund", {"order_id": "SO1", "surprise": "x"})]
    scored = DialogueEvaluator()._score_case(case, responses)
    # 1 of 2 filled slots was required → precision 0.5, recall 1.0.
    assert scored["slot_precision"] == 0.5
    assert scored["slot_recall"] == 1.0


# ── Intent + switch ──────────────────────────────────────────────────────────


def test_intent_accuracy_ignores_none_turns():
    case = DialogueCase(
        case_id="c",
        turns=["a", "b", "c"],
        expected_intent=[None, "refund", "track_shipping"],
        expected_slots=[{}, {}, {}],
        required_slots=["order_id"],
    )
    responses = [
        _resp("chitchat", {}),
        _resp("Refund", {"order_id": "SO1"}),   # case-insensitive match
        _resp("faq", {"order_id": "SO1"}),      # mismatch
    ]
    scored = DialogueEvaluator()._score_case(case, responses)
    # 1 of 2 asserted turns correct.
    assert scored["intent_accuracy"] == 0.5


def test_intent_switch_followed():
    case = DialogueCase(
        case_id="c",
        turns=["退款 SO1", "算了，查下物流"],
        expected_intent=["refund", "track_shipping"],
        expected_slots=[{"order_id": "SO1"}, {"order_id": "SO1"}],
        required_slots=["order_id"],
        expect_switch=True,
    )
    responses = [
        _resp("refund", {"order_id": "SO1"}),
        _resp("track_shipping", {"order_id": "SO1"}),
    ]
    scored = DialogueEvaluator()._score_case(case, responses)
    assert scored["intent_switch_ok"] is True


def test_intent_switch_not_followed():
    case = DialogueCase(
        case_id="c",
        turns=["退款 SO1", "查下物流"],
        expected_intent=["refund", "track_shipping"],
        expected_slots=[{}, {"order_id": "SO1"}],
        required_slots=["order_id"],
        expect_switch=True,
    )
    # System stayed on refund at the switching turn.
    responses = [
        _resp("refund", {}),
        _resp("refund", {"order_id": "SO1"}),
    ]
    scored = DialogueEvaluator()._score_case(case, responses)
    assert scored["intent_switch_ok"] is False


# ── evaluate_via_api with stub client ────────────────────────────────────────


class _StubChatbot:
    def __init__(self, scripted: List[List[GenerateResponse]]):
        self._scripted = scripted
        self._i = 0
        self._n = 0

    def allocate_session_id(self) -> int:
        self._n += 1
        return self._n

    async def converse(self, turns, session_id=None, **kwargs):
        resp = self._scripted[self._i]
        self._i += 1
        return resp


def test_evaluate_via_api_aggregates():
    cases = [
        DialogueCase(
            case_id="ok",
            turns=["退款 SO1", "原因破损"],
            expected_intent=["refund", "refund"],
            expected_slots=[{"order_id": "SO1"}, {"order_id": "SO1", "reason": "破损"}],
            required_slots=["order_id", "reason"],
        ),
        DialogueCase(
            case_id="partial",
            turns=["退款"],
            expected_intent=["refund"],
            expected_slots=[{"order_id": "SO1", "reason": "破损"}],
            required_slots=["order_id", "reason"],
        ),
    ]
    scripted = [
        # Case 1: completes by turn 1.
        [
            _resp("refund", {"order_id": "SO1"}),
            _resp("refund", {"order_id": "SO1", "reason": "破损"}),
        ],
        # Case 2: never fills slots.
        [_resp("refund", {})],
    ]
    result = asyncio.run(
        DialogueEvaluator().evaluate_via_api(cases, _StubChatbot(scripted))
    )
    assert result.case_count == 2
    assert result.task_completion_rate == 0.5  # 1 of 2 completed
    assert 0.0 < result.slot_recall < 1.0       # partial across cases
    assert len(result.per_case_results) == 2


# ── Dataset loads ────────────────────────────────────────────────────────────


def test_dialogue_dataset_loads_and_parses():
    # Dialogue cases use turns/message, not the QA query/intent/ground_truth
    # schema, so load them as raw JSONL rows (no QA validation) and parse.
    from src.dataset.converter import FormatConverter

    rows = FormatConverter._jsonl_file_to_internal("data/dialogue_dataset.jsonl")
    assert len(rows) >= 8
    cases = [DialogueCase.from_dict(r) for r in rows]
    switch_count = sum(1 for c in cases if c.expect_switch)
    assert switch_count >= 2, "dataset should include intent-switch cases"
    # At least one multi-turn (>=2) case.
    assert any(len(c.turns) >= 2 for c in cases)
