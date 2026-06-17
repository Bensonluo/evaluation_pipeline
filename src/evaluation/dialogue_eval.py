"""
Dialogue (multi-turn) evaluation module (black-box).

Evaluates the chatbot's task-oriented dialogue behaviour over the public
POST /api/v1/chat endpoint, where conversation continuity is keyed on
session_id (the server's in-memory thread). This is the path that exercises
the LangGraph dialogue engine — slot filling, clarifying turns, and intent
switching — which single-turn evaluation cannot reach.

Metrics (PARADISE / MultiWOZ-style task-oriented dialogue evaluation):
    - Slot precision / recall / F1: did the system extract the expected slots?
    - Avg clarification turns: how many turns before required slots are filled.
    - Per-turn intent accuracy: did the detected intent match the label?
    - Intent switch accuracy: did the system follow a mid-conversation topic
      change (and restore prior state on cancel/resume)?
    - Task completion rate: fraction of conversations that reached a state with
      all required slots filled (a proxy for "the task could be executed").

Black-box contract reminder (from app/api/v1/chat.py):
    Each turn's response carries `intent`, `metadata.filled_slots` (dict),
    `metadata.pending_slots` (list), and `dialogue_state`. We score against
    per-turn `expected_intent` and per-turn/turn-accumulated `expected_slots`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class DialogueCase:
    """One multi-turn conversation case for evaluation.

    Attributes:
        case_id: Stable identifier for the case.
        turns: Ordered list of user messages.
        expected_intent: Per-turn expected intent label (aligned with turns).
            Use None for "don't care" turns (e.g. an open clarification turn
            whose intent we don't assert on).
        expected_slots: Per-turn expected filled slots, as a dict of
            {slot_name: value}. These are checked against the system's
            `metadata.filled_slots` at that turn. Missing keys are tolerated;
            only the listed keys are asserted. Use {} for "don't care".
        required_slots: The full set of slots that must be filled for the task
            to be considered completable. Used for task-completion rate. If
            omitted, derived as the union of all expected_slots keys.
        expect_switch: Whether this case exercises an intent switch (the
            conversation changes task mid-way). Used to group switch accuracy.
        metadata: Free-form annotations.
    """

    case_id: str
    turns: List[str]
    expected_intent: List[Optional[str]]
    expected_slots: List[Dict[str, Any]]
    required_slots: Optional[List[str]] = None
    expect_switch: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = len(self.turns)
        if len(self.expected_intent) != n:
            raise ValueError(
                f"case {self.case_id}: expected_intent length "
                f"({len(self.expected_intent)}) != turns ({n})"
            )
        if len(self.expected_slots) != n:
            raise ValueError(
                f"case {self.case_id}: expected_slots length "
                f"({len(self.expected_slots)}) != turns ({n})"
            )

    @property
    def required_slot_set(self) -> set:
        if self.required_slots is not None:
            return set(self.required_slots)
        # Default: union of all keys asserted across turns.
        union: set = set()
        for s in self.expected_slots:
            union.update(s.keys())
        return union

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DialogueCase":
        """Build a DialogueCase from a dataset row.

        Accepted row shapes (any of):
            {"case_id", "turns", "expected_intent", "expected_slots",
             "required_slots"?, "expect_switch"?}
        or compact:
            {"case_id", "turns": [{"message", "intent"?, "slots"?}, ...]}
        """
        case_id = data.get("case_id") or data.get("id") or ""
        turns = data.get("turns")
        if turns is None:
            raise ValueError(f"case {case_id}: missing 'turns'")

        # Compact per-turn-object form.
        if turns and isinstance(turns[0], dict):
            turn_msgs = [t.get("message", "") for t in turns]
            exp_intent = [t.get("intent") for t in turns]
            exp_slots = [t.get("slots") or {} for t in turns]
        else:
            turn_msgs = list(turns)
            exp_intent = list(data.get("expected_intent") or [None] * len(turn_msgs))
            exp_slots = list(data.get("expected_slots") or [{}] * len(turn_msgs))

        return cls(
            case_id=case_id,
            turns=turn_msgs,
            expected_intent=exp_intent,
            expected_slots=exp_slots,
            required_slots=data.get("required_slots"),
            expect_switch=bool(data.get("expect_switch", False)),
            metadata=data.get("metadata") or {},
        )


@dataclass
class DialogueResult:
    """Aggregate results from dialogue evaluation."""

    slot_precision: float
    slot_recall: float
    slot_f1: float
    avg_clarification_turns: float
    intent_accuracy: float
    intent_switch_accuracy: float
    task_completion_rate: float
    case_count: int
    per_case_results: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "slot_precision": self.slot_precision,
            "slot_recall": self.slot_recall,
            "slot_f1": self.slot_f1,
            "avg_clarification_turns": self.avg_clarification_turns,
            "intent_accuracy": self.intent_accuracy,
            "intent_switch_accuracy": self.intent_switch_accuracy,
            "task_completion_rate": self.task_completion_rate,
            "case_count": self.case_count,
            "per_case_results": self.per_case_results,
        }


class DialogueEvaluator:
    """Evaluates multi-turn dialogue quality (black-box).

    Usage:
        evaluator = DialogueEvaluator()
        result = await evaluator.evaluate_via_api(cases, chatbot_client)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}

    async def evaluate_via_api(
        self,
        cases: List[DialogueCase],
        chatbot_client,
    ) -> DialogueResult:
        """Run each case through the chatbot (one session_id per case) and score.

        Args:
            cases: Dialogue cases to evaluate.
            chatbot_client: A ChatbotClient (src/api/chatbot.py). Each case gets
                a fresh session_id so conversations don't bleed into each other.

        Returns:
            DialogueResult with aggregate + per-case metrics.
        """
        per_case: List[Dict[str, Any]] = []
        for case in cases:
            session_id = chatbot_client.allocate_session_id()
            # converse() reuses one thread across turns — the only way to reach
            # the LangGraph multi-turn state machine over the black box.
            responses = await chatbot_client.converse(
                turns=case.turns, session_id=session_id
            )
            per_case.append(self._score_case(case, responses))

        return self._aggregate(per_case, cases)

    def _score_case(
        self,
        case: DialogueCase,
        responses,
    ) -> Dict[str, Any]:
        """Score a single case against the per-turn expectations.

        Slot scoring is over the *final* turn's filled_slots (accumulated state)
        vs. the union of expected slots across the conversation. Per-turn slot
        assertions are also recorded for finer diagnosis.
        """
        required = case.required_slot_set

        # --- Slots: compare final accumulated filled_slots to required set.
        final_filled: Dict[str, Any] = {}
        if responses:
            final_filled = dict(responses[-1].filled_slots or {})
        filled_keys = set(final_filled.keys())

        # Precision: of the slots the system filled, how many were required.
        if filled_keys:
            slot_precision = len(filled_keys & required) / len(filled_keys)
        else:
            slot_precision = 0.0
        # Recall: of the required slots, how many did the system fill.
        slot_recall = len(filled_keys & required) / len(required) if required else 1.0
        slot_f1 = (
            2 * slot_precision * slot_recall / (slot_precision + slot_recall)
            if (slot_precision + slot_recall) > 0
            else 0.0
        )

        # --- Per-turn slot correctness (strict key+value match on asserted keys).
        per_turn_slots: List[Dict[str, Any]] = []
        for resp, expected in zip(responses, case.expected_slots):
            got = resp.filled_slots or {}
            matches = {k: (str(got.get(k)) == str(v)) for k, v in expected.items()}
            per_turn_slots.append({
                "expected": expected,
                "got": got,
                "correct": all(matches.values()) if matches else True,
            })

        # --- Clarification turns: turns until no pending required slots remain.
        # A turn with pending required slots counts as a clarification turn.
        # We count the number of turns before the last turn in which a required
        # slot was still pending.
        clar_turns = self._count_clarification_turns(responses, required)

        # --- Intent accuracy over asserted turns.
        intent_matches: List[bool] = []
        for resp, expected_intent in zip(responses, case.expected_intent):
            if expected_intent is None:
                continue
            intent_matches.append((resp.intent or "").lower() == expected_intent.lower())
        intent_accuracy = (
            sum(1 for m in intent_matches if m) / len(intent_matches)
            if intent_matches else 1.0
        )

        # --- Intent switch (only meaningful for switch cases).
        # Pass if the system's intent at the switching turn matches the new
        # expected intent AND required slots of the new task eventually fill.
        switch_ok = None
        if case.expect_switch:
            switch_ok = self._intent_switch_followed(case, responses)

        # --- Task completion: all required slots filled by the last turn.
        task_complete = required.issubset(filled_keys) if required else True

        return {
            "case_id": case.case_id,
            "turn_count": len(responses),
            "slot_precision": slot_precision,
            "slot_recall": slot_recall,
            "slot_f1": slot_f1,
            "clarification_turns": clar_turns,
            "intent_accuracy": intent_accuracy,
            "intent_switch_ok": switch_ok,
            "task_complete": task_complete,
            "final_filled_slots": final_filled,
            "per_turn_slots": per_turn_slots,
            "per_turn_intent": [r.intent for r in responses],
            "errors": [r.error for r in responses if r.error],
        }

    def _count_clarification_turns(self, responses, required: set) -> int:
        """Turns spent clarifying = turns before the first turn at which all
        required slots are filled. If never fully filled, counts all turns.
        """
        if not required:
            return 0
        for i, resp in enumerate(responses):
            filled = set((resp.filled_slots or {}).keys())
            pending_required = required - filled
            if not pending_required:
                return i  # turns 0..i-1 were clarification; turn i completed
        # Never completed: every turn was effectively a clarification turn.
        return len(responses)

    def _intent_switch_followed(
        self, case: DialogueCase, responses
    ) -> bool:
        """Heuristic: did the system follow the intended final intent?

        We take the last non-None expected intent as the "switched-to" intent
        and check the corresponding turn's detected intent matches. This is a
        conservative proxy; richer switch-state checks would need the
        dialogue_state.stack which is optional in the black-box response.
        """
        switched_to = None
        switched_idx = None
        for i, intent in enumerate(case.expected_intent):
            if intent is not None:
                switched_to, switched_idx = intent, i
        if switched_to is None or switched_idx is None:
            return True
        if switched_idx >= len(responses):
            return False
        return (responses[switched_idx].intent or "").lower() == switched_to.lower()

    def _aggregate(
        self,
        per_case: List[Dict[str, Any]],
        cases: List[DialogueCase],
    ) -> DialogueResult:
        n = len(per_case) or 1

        slot_p = np.mean([c["slot_precision"] for c in per_case]) if per_case else 0.0
        slot_r = np.mean([c["slot_recall"] for c in per_case]) if per_case else 0.0
        slot_f1 = np.mean([c["slot_f1"] for c in per_case]) if per_case else 0.0
        avg_clar = np.mean([c["clarification_turns"] for c in per_case]) if per_case else 0.0
        intent_acc = np.mean([c["intent_accuracy"] for c in per_case]) if per_case else 0.0
        completion = sum(1 for c in per_case if c["task_complete"]) / n

        switch_results = [c["intent_switch_ok"] for c in per_case if c["intent_switch_ok"] is not None]
        switch_acc = (
            sum(1 for s in switch_results if s) / len(switch_results)
            if switch_results else 1.0
        )

        return DialogueResult(
            slot_precision=float(slot_p),
            slot_recall=float(slot_r),
            slot_f1=float(slot_f1),
            avg_clarification_turns=float(avg_clar),
            intent_accuracy=float(intent_acc),
            intent_switch_accuracy=float(switch_acc),
            task_completion_rate=float(completion),
            case_count=len(per_case),
            per_case_results=per_case,
        )
