"""
RAGAS Metrics for RAG Evaluation

This module implements RAGAS (Retrieval Augmented Generation Assessment) metrics
for evaluating end-to-end RAG system quality.

Metrics implemented:
    - Faithfulness: Whether the answer is faithful to the retrieved context
    - Answer Relevance: How relevant the answer is to the question
    - Context Precision: Proportion of retrieved context that is relevant
    - Context Recall: Whether the ground truth can be attributed to context

Reference: DESIGN.md Chapter 10 - RAGAS Complete Metric System
"""

import json
import re
from typing import List, Dict, Union, Optional, Any
import warnings


# RAGAS Prompt Templates (DESIGN.md Chapter 10.3)

CONTEXT_PRECISION_PROMPT = """Given a question and a context, evaluate whether the context contains relevant information to answer the question.

Question: {question}
Context: {context}

Determine the precision of the context (0.0 to 1.0), where 1.0 means all information in the context is relevant to the question.

Output JSON:
{{"precision": float, "reasoning": str}}"""

CONTEXT_RECALL_PROMPT = """Given a question, the ground truth answer, and the retrieved context, determine what portion of the ground truth answer can be attributed to the context.

Ground Truth Answer: {ground_truth}
Retrieved Context: {context}

Calculate the recall (0.0 to 1.0) representing how much of the ground truth is supported by the context.

Output JSON:
{{"recall": float, "attributed_statements": [str], "missing_statements": [str]}}"""

FAITHFULNESS_PROMPT = """Given the context and the answer, evaluate whether the answer is faithful to the context (i.e., does not contain hallucinations).

Context: {context}
Answer: {answer}

Evaluate faithfulness on two dimensions:
1. Truthfulness: Are all claims in the answer supported by the context?
2. Hallucination: Does the answer contain information not in the context?

Output JSON:
{{"faithfulness": float, "hallucination_count": int, "total_claims": int, "reasoning": str}}"""

ANSWER_RELEVANCE_PROMPT = """Given a question and an answer, evaluate how relevant the answer is to the question.

Question: {question}
Answer: {answer}

Evaluate relevance (0.0 to 1.0) based on:
- Does the answer directly address the question?
- Is the answer complete and informative?
- Does the answer avoid unnecessary tangents?

Output JSON:
{{"relevance": float, "reasoning": str}}"""

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


def _extract_json_from_response(response: str) -> Dict[str, Any]:
    """
    Extract JSON from LLM response, handling various formats.

    Args:
        response: Raw LLM response string

    Returns:
        Parsed JSON dictionary

    Raises:
        ValueError: If JSON cannot be extracted
    """
    # Try direct parsing first
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass

    # Try to find JSON in markdown code blocks
    json_match = re.search(r'```(?:json)?\s*\n?({.*?})\s*```', response, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try to find any JSON-like structure
    json_match = re.search(r'\{.*?\}', response, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not extract valid JSON from response: {response[:200]}...")


def evaluate_context_precision(
    questions: Union[str, List[str]],
    contexts: Union[str, List[str], List[List[str]]],
    llm_client: Any = None
) -> Dict[str, float]:
    """
    Evaluate Context Precision using LLM-as-a-judge.

    Context Precision measures what proportion of retrieved context is
    relevant to answering the question.

    Args:
        questions: Single question or list of questions
        contexts: Retrieved context(s) - string, list of strings, or list of context lists
        llm_client: LLM client with invoke() method for evaluation

    Returns:
        Dictionary with precision scores

    Example:
        >>> evaluate_context_precision(
        ...     "What is the refund policy?",
        ...     ["We offer 7-day refunds.", "Our store opens at 9am."]
        ... )
        {"precision": 0.5, "avg_precision": 0.5}

    Note:
        Requires an LLM client. If not provided, returns placeholder score.
    """
    if llm_client is None:
        warnings.warn("No LLM client provided. Returning placeholder score.")
        return {"precision": 0.5, "avg_precision": 0.5, "error": "No LLM client"}

    # Normalize inputs
    questions_list = [questions] if isinstance(questions, str) else questions

    if isinstance(contexts, str):
        contexts_list = [[contexts]] * len(questions_list)
    elif isinstance(contexts[0], str):
        contexts_list = [contexts] * len(questions_list)
    else:
        contexts_list = contexts

    if len(questions_list) != len(contexts_list):
        raise ValueError("Number of questions must match number of context lists")

    precisions = []

    for question, context_list in zip(questions_list, contexts_list):
        combined_context = " ".join(context_list) if isinstance(context_list, list) else context_list

        prompt = CONTEXT_PRECISION_PROMPT.format(
            question=question,
            context=combined_context
        )

        try:
            response = llm_client.invoke(prompt)
            result = _extract_json_from_response(response)
            precisions.append(result.get("precision", 0.5))
        except Exception as e:
            warnings.warn(f"Context precision evaluation failed: {e}")
            precisions.append(0.5)

    return {
        "precision": sum(precisions) / len(precisions),
        "avg_precision": sum(precisions) / len(precisions),
        "individual_precisions": precisions
    }


def evaluate_context_recall(
    questions: Union[str, List[str]],
    contexts: Union[str, List[str], List[List[str]]],
    ground_truths: Union[str, List[str]],
    llm_client: Any = None
) -> Dict[str, float]:
    """
    Evaluate Context Recall using LLM-as-a-judge.

    Context Recall measures whether the ground truth answer can be
    attributed to the retrieved context.

    Args:
        questions: Single question or list of questions
        contexts: Retrieved context(s)
        ground_truths: Ground truth answer(s)
        llm_client: LLM client with invoke() method

    Returns:
        Dictionary with recall scores

    Example:
        >>> evaluate_context_recall(
        ...     "What is the refund policy?",
        ...     ["7-day no questions asked refund policy"],
        ...     "You can get a full refund within 7 days"
        ... )
        {"recall": 0.9, "avg_recall": 0.9}
    """
    if llm_client is None:
        warnings.warn("No LLM client provided. Returning placeholder score.")
        return {"recall": 0.5, "avg_recall": 0.5, "error": "No LLM client"}

    # Normalize inputs
    questions_list = [questions] if isinstance(questions, str) else questions
    gt_list = [ground_truths] if isinstance(ground_truths, str) else ground_truths

    if isinstance(contexts, str):
        contexts_list = [[contexts]] * len(questions_list)
    elif isinstance(contexts[0], str):
        contexts_list = [contexts] * len(questions_list)
    else:
        contexts_list = contexts

    recalls = []

    for question, context_list, gt in zip(questions_list, contexts_list, gt_list):
        combined_context = " ".join(context_list) if isinstance(context_list, list) else context_list

        prompt = CONTEXT_RECALL_PROMPT.format(
            ground_truth=gt,
            context=combined_context
        )

        try:
            response = llm_client.invoke(prompt)
            result = _extract_json_from_response(response)
            recalls.append(result.get("recall", 0.5))
        except Exception as e:
            warnings.warn(f"Context recall evaluation failed: {e}")
            recalls.append(0.5)

    return {
        "recall": sum(recalls) / len(recalls),
        "avg_recall": sum(recalls) / len(recalls),
        "individual_recalls": recalls
    }


def evaluate_faithfulness(
    answers: Union[str, List[str]],
    contexts: Union[str, List[str], List[List[str]]],
    llm_client: Any = None
) -> Dict[str, float]:
    """
    Evaluate Faithfulness using LLM-as-a-judge.

    Faithfulness measures whether the answer is faithful to the retrieved
    context (i.e., contains no hallucinations).

    Args:
        answers: Generated answer(s)
        contexts: Retrieved context(s) used to generate answers
        llm_client: LLM client with invoke() method

    Returns:
        Dictionary with faithfulness scores

    Example:
        >>> evaluate_faithfulness(
        ...     "Our refund policy lasts 14 days",
        ...     ["We offer a 7-day refund policy"]
        ... )
        {"faithfulness": 0.5, "avg_faithfulness": 0.5, "hallucination_count": 1}

    Note:
        Faithfulness >= 3 (on 1-5 scale) or >= 0.9 (on 0-1 scale) is
        considered acceptable per DESIGN.md Chapter 2.3.
    """
    if llm_client is None:
        warnings.warn("No LLM client provided. Returning placeholder score.")
        return {"faithfulness": 0.5, "avg_faithfulness": 0.5, "error": "No LLM client"}

    # Normalize inputs
    answers_list = [answers] if isinstance(answers, str) else answers

    if isinstance(contexts, str):
        contexts_list = [[contexts]] * len(answers_list)
    elif isinstance(contexts[0], str):
        contexts_list = [contexts] * len(answers_list)
    else:
        contexts_list = contexts

    faithfulness_scores = []
    total_hallucinations = 0
    total_claims = 0

    for answer, context_list in zip(answers_list, contexts_list):
        combined_context = " ".join(context_list) if isinstance(context_list, list) else context_list

        prompt = FAITHFULNESS_PROMPT.format(
            context=combined_context,
            answer=answer
        )

        try:
            response = llm_client.invoke(prompt)
            result = _extract_json_from_response(response)
            faithfulness_scores.append(result.get("faithfulness", 0.5))
            total_hallucinations += result.get("hallucination_count", 0)
            total_claims += result.get("total_claims", 1)
        except Exception as e:
            warnings.warn(f"Faithfulness evaluation failed: {e}")
            faithfulness_scores.append(0.5)

    return {
        "faithfulness": sum(faithfulness_scores) / len(faithfulness_scores),
        "avg_faithfulness": sum(faithfulness_scores) / len(faithfulness_scores),
        "individual_faithfulness": faithfulness_scores,
        "hallucination_rate": total_hallucinations / max(total_claims, 1),
        "no_hallucination_rate": 1 - (total_hallucinations / max(total_claims, 1))
    }


def evaluate_answer_relevance(
    questions: Union[str, List[str]],
    answers: Union[str, List[str]],
    llm_client: Any = None
) -> Dict[str, float]:
    """
    Evaluate Answer Relevance using LLM-as-a-judge.

    Answer Relevance measures how relevant and complete the answer is
    to the asked question.

    Args:
        questions: Question(s) asked
        answers: Generated answer(s)
        llm_client: LLM client with invoke() method

    Returns:
        Dictionary with relevance scores

    Example:
        >>> evaluate_answer_relevance(
        ...     "What is the refund policy?",
        ...     "We offer a 7-day refund policy on all items."
        ... )
        {"relevance": 0.95, "avg_relevance": 0.95}
    """
    if llm_client is None:
        warnings.warn("No LLM client provided. Returning placeholder score.")
        return {"relevance": 0.5, "avg_relevance": 0.5, "error": "No LLM client"}

    # Normalize inputs
    questions_list = [questions] if isinstance(questions, str) else questions
    answers_list = [answers] if isinstance(answers, str) else answers

    if len(questions_list) != len(answers_list):
        raise ValueError("Number of questions must match number of answers")

    relevance_scores = []

    for question, answer in zip(questions_list, answers_list):
        prompt = ANSWER_RELEVANCE_PROMPT.format(
            question=question,
            answer=answer
        )

        try:
            response = llm_client.invoke(prompt)
            result = _extract_json_from_response(response)
            relevance_scores.append(result.get("relevance", 0.5))
        except Exception as e:
            warnings.warn(f"Answer relevance evaluation failed: {e}")
            relevance_scores.append(0.5)

    return {
        "relevance": sum(relevance_scores) / len(relevance_scores),
        "avg_relevance": sum(relevance_scores) / len(relevance_scores),
        "individual_relevance": relevance_scores
    }


def evaluate_answer_correctness(
    questions: Union[str, List[str]],
    answers: Union[str, List[str]],
    ground_truths: Union[str, List[str]],
    llm_client: Any = None
) -> Dict[str, float]:
    """
    Evaluate Answer Correctness using LLM-as-a-judge.

    Answer Correctness combines semantic similarity and factual accuracy
    between the generated answer and ground truth.

    Args:
        questions: Question(s) asked
        answers: Generated answer(s)
        ground_truths: Ground truth answer(s)
        llm_client: LLM client with invoke() method

    Returns:
        Dictionary with correctness scores

    Example:
        >>> evaluate_answer_correctness(
        ...     "What is the refund policy?",
        ...     "7-day refund policy",
        ...     "You can get a full refund within 7 days"
        ... )
        {"correctness": 0.92, "similarity": 0.88, "accuracy": 0.96}
    """
    if llm_client is None:
        warnings.warn("No LLM client provided. Returning placeholder score.")
        return {"correctness": 0.5, "avg_correctness": 0.5, "error": "No LLM client"}

    # Normalize inputs
    questions_list = [questions] if isinstance(questions, str) else questions
    answers_list = [answers] if isinstance(answers, str) else answers
    gt_list = [ground_truths] if isinstance(ground_truths, str) else ground_truths

    correctness_scores = []
    similarity_scores = []
    accuracy_scores = []

    for question, answer, gt in zip(questions_list, answers_list, gt_list):
        prompt = ANSWER_CORRECTNESS_PROMPT.format(
            question=question,
            ground_truth=gt,
            answer=answer
        )

        try:
            response = llm_client.invoke(prompt)
            result = _extract_json_from_response(response)
            correctness_scores.append(result.get("correctness", 0.5))
            similarity_scores.append(result.get("similarity", 0.5))
            accuracy_scores.append(result.get("accuracy", 0.5))
        except Exception as e:
            warnings.warn(f"Answer correctness evaluation failed: {e}")
            correctness_scores.append(0.5)
            similarity_scores.append(0.5)
            accuracy_scores.append(0.5)

    return {
        "correctness": sum(correctness_scores) / len(correctness_scores),
        "avg_correctness": sum(correctness_scores) / len(correctness_scores),
        "similarity": sum(similarity_scores) / len(similarity_scores),
        "accuracy": sum(accuracy_scores) / len(accuracy_scores),
        "individual_correctness": correctness_scores
    }


def compute_ragas_metrics(
    questions: List[str],
    contexts: List[List[str]],
    answers: List[str],
    ground_truths: List[str] = None,
    llm_client: Any = None
) -> Dict[str, float]:
    """
    Compute all RAGAS metrics at once.

    Args:
        questions: List of questions
        contexts: List of retrieved context lists
        answers: List of generated answers
        ground_truths: Optional ground truth answers for recall/correctness
        llm_client: LLM client for evaluation

    Returns:
        Dictionary with all RAGAS metrics

    Example:
        >>> compute_ragas_metrics(
        ...     questions=["What is the refund policy?"],
        ...     contexts=[["7-day refund policy on all items"]],
        ...     answers=["We offer 7-day refunds"],
        ...     ground_truths=["Full refund within 7 days"]
        ... )
        {
            "context_precision": 0.9,
            "context_recall": 0.95,
            "faithfulness": 0.88,
            "answer_relevance": 0.92,
            "answer_correctness": 0.87
        }
    """
    if llm_client is None:
        warnings.warn("No LLM client provided. Using placeholder scores.")
        return {
            "context_precision": 0.5,
            "context_recall": 0.5,
            "faithfulness": 0.5,
            "answer_relevance": 0.5,
            "answer_correctness": 0.5 if ground_truths else None,
            "error": "No LLM client"
        }

    metrics = {}

    # Context Precision
    try:
        cp_result = evaluate_context_precision(questions, contexts, llm_client)
        metrics["context_precision"] = cp_result["precision"]
    except Exception as e:
        warnings.warn(f"Context precision evaluation failed: {e}")
        metrics["context_precision"] = 0.0

    # Faithfulness
    try:
        f_result = evaluate_faithfulness(answers, contexts, llm_client)
        metrics["faithfulness"] = f_result["faithfulness"]
        metrics["no_hallucination_rate"] = f_result["no_hallucination_rate"]
    except Exception as e:
        warnings.warn(f"Faithfulness evaluation failed: {e}")
        metrics["faithfulness"] = 0.0
        metrics["no_hallucination_rate"] = 0.0

    # Answer Relevance
    try:
        ar_result = evaluate_answer_relevance(questions, answers, llm_client)
        metrics["answer_relevance"] = ar_result["relevance"]
    except Exception as e:
        warnings.warn(f"Answer relevance evaluation failed: {e}")
        metrics["answer_relevance"] = 0.0

    # Context Recall (requires ground truth)
    if ground_truths:
        try:
            cr_result = evaluate_context_recall(questions, contexts, ground_truths, llm_client)
            metrics["context_recall"] = cr_result["recall"]
        except Exception as e:
            warnings.warn(f"Context recall evaluation failed: {e}")
            metrics["context_recall"] = 0.0

        # Answer Correctness (requires ground truth)
        try:
            ac_result = evaluate_answer_correctness(questions, answers, ground_truths, llm_client)
            metrics["answer_correctness"] = ac_result["correctness"]
        except Exception as e:
            warnings.warn(f"Answer correctness evaluation failed: {e}")
            metrics["answer_correctness"] = 0.0

    return metrics
