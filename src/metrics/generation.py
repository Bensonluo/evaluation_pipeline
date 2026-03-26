"""
Generation Metrics for RAG Evaluation

This module implements metrics for evaluating the quality of generated responses
in a RAG system using both lexical similarity (ROUGE, BLEU) and semantic
similarity (BERTScore).

Metrics implemented:
    - ROUGE-L: Longest Common Subsequence based F-score
    - BLEU-4: 4-gram precision with brevity penalty
    - BERTScore: Contextual embedding-based similarity

Reference: DESIGN.md Chapter 2.2 - Generation Stage Metrics
"""

from typing import List, Dict, Union, Optional
import warnings


def rouge_scores(
    prediction: Union[str, List[str]],
    reference: Union[str, List[str]],
    rouge_type: str = "L"
) -> Dict[str, float]:
    """
    Calculate ROUGE scores between prediction and reference.

    ROUGE (Recall-Oriented Understudy for Gisting Evaluation) measures
    the overlap of n-grams between candidate and reference texts.

    Args:
        prediction: Generated text(s) - single string or list of strings
        reference: Reference text(s) - single string or list of strings
        rouge_type: Type of ROUGE metric ("L", "1", "2", or "all")
            - "L": Longest Common Subsequence (default)
            - "1": Unigram overlap
            - "2": Bigram overlap
            - "all": Compute all types

    Returns:
        Dictionary with precision, recall, and f-measure

    Example:
        >>> rouge_scores("The cat sat on the mat", "A cat sat on the mat")
        {"precision": 0.857, "recall": 0.857, "fmeasure": 0.857}

    Note:
        For production use, install 'rouge-score' package:
        pip install rouge-score

        Then use:
        from rouge_score import rouge_scorer
    """
    # Try to use rouge-score package if available
    try:
        from rouge_score import rouge_scorer
        from rouge_score import scoring

        scorer = rouge_scorer.RougeScorer(
            [f"rouge{rouge_type}"] if rouge_type != "all" else ["rouge1", "rouge2", "rougeL"],
            use_stemmer=True
        )

        # Normalize input to lists
        predictions = [prediction] if isinstance(prediction, str) else prediction
        references = [reference] if isinstance(reference, str) else reference

        # Calculate scores for each pair
        aggregator = scoring.BootstrapAggregator()
        for pred, ref in zip(predictions, references):
            aggregator.add_scores(scorer.score(ref, pred))

        # Get mean scores
        result = aggregator.aggregate()
        scores = {}

        if rouge_type == "all":
            for metric in ["rouge1", "rouge2", "rougeL"]:
                mid = result[metric].mid
                scores[metric] = {
                    "precision": mid.precision,
                    "recall": mid.recall,
                    "fmeasure": mid.fmeasure
                }
        else:
            metric_name = f"rouge{rouge_type}"
            mid = result[metric_name].mid
            scores = {
                "precision": mid.precision,
                "recall": mid.recall,
                "fmeasure": mid.fmeasure
            }

        return scores

    except ImportError:
        # Fallback to simplified implementation
        warnings.warn(
            "rouge-score package not installed. Using simplified implementation. "
            "For accurate ROUGE scores: pip install rouge-score"
        )
        return _simple_rouge_l(prediction, reference)


def _simple_rouge_l(
    prediction: Union[str, List[str]],
    reference: Union[str, List[str]]
) -> Dict[str, float]:
    """
    Simplified ROUGE-L implementation using Longest Common Subsequence.

    This is a fallback when rouge-score package is not available.
    """
    def _lcs_length(seq1: List[str], seq2: List[str]) -> int:
        """Calculate LCS length between two sequences."""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i - 1] == seq2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        return dp[m][n]

    # Normalize to lists
    predictions = [prediction] if isinstance(prediction, str) else prediction
    references = [reference] if isinstance(reference, str) else reference

    total_precision = 0.0
    total_recall = 0.0
    total_fmeasure = 0.0

    for pred, ref in zip(predictions, references):
        pred_tokens = pred.split()
        ref_tokens = ref.split()

        lcs_len = _lcs_length(pred_tokens, ref_tokens)

        precision = lcs_len / len(pred_tokens) if pred_tokens else 0.0
        recall = lcs_len / len(ref_tokens) if ref_tokens else 0.0

        if precision + recall > 0:
            fmeasure = (2 * precision * recall) / (precision + recall)
        else:
            fmeasure = 0.0

        total_precision += precision
        total_recall += recall
        total_fmeasure += fmeasure

    n = len(predictions)
    return {
        "precision": total_precision / n,
        "recall": total_recall / n,
        "fmeasure": total_fmeasure / n
    }


def bleu_score(
    prediction: Union[str, List[str]],
    reference: Union[str, List[str]],
    max_order: int = 4,
    smooth: bool = False
) -> Dict[str, float]:
    """
    Calculate BLEU score between prediction and reference.

    BLEU (Bilingual Evaluation Understudy) measures n-gram overlap
    with a brevity penalty for short predictions.

    Args:
        prediction: Generated text(s) - single string or list of strings
        reference: Reference text(s) - single string or list of strings
        max_order: Maximum n-gram order (default: 4 for BLEU-4)
        smooth: Whether to apply smoothing (recommended for short texts)

    Returns:
        Dictionary with BLEU score and individual n-gram precisions

    Example:
        >>> bleu_score("the cat is on the mat", "the cat is on the mat")
        {"bleu": 1.0, "precisions": [1.0, 1.0, 1.0, 1.0], "brevity_penalty": 1.0}

    Note:
        For production use, install sacreBLEU:
        pip install sacrebleu

        Then use:
        import sacrebleu
    """
    # Try to use sacrebleu if available
    try:
        import sacrebleu

        # Normalize input
        if isinstance(prediction, str):
            prediction = [prediction]
        if isinstance(reference, str):
            reference = [[reference]]
        elif isinstance(reference[0], str):
            reference = [reference]

        # Calculate BLEU
        bleu = sacrebleu.corpus_bleu(
            prediction,
            reference,
            max_order=max_order,
            smooth=smooth
        )

        return {
            "bleu": bleu.score / 100.0,  # Normalize to 0-1
            "precisions": [p / 100.0 for p in bleu.precisions],
            "brevity_penalty": bleu.bp,
            "sys_len": bleu.sys_len,
            "ref_len": bleu.ref_len
        }

    except ImportError:
        # Fallback to simplified implementation
        warnings.warn(
            "sacrebleu package not installed. Using simplified implementation. "
            "For accurate BLEU scores: pip install sacrebleu"
        )
        return _simple_bleu(prediction, reference, max_order)


def _simple_bleu(
    prediction: Union[str, List[str]],
    reference: Union[str, List[str]],
    max_order: int = 4
) -> Dict[str, float]:
    """
    Simplified BLEU implementation.
    """
    from collections import Counter
    import math

    def _get_ngrams(tokens: List[str], n: int) -> Counter:
        """Extract n-grams from token list."""
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngrams.append(tuple(tokens[i:i + n]))
        return Counter(ngrams)

    def _modified_precision(pred: List[str], ref: List[str], n: int) -> float:
        """Calculate modified n-gram precision."""
        pred_ngrams = _get_ngrams(pred, n)
        ref_ngrams = _get_ngrams(ref, n)

        if not pred_ngrams:
            return 0.0

        # Clip counts by reference max
        clipped_counts = {
            ngram: min(count, ref_ngrams.get(ngram, 0))
            for ngram, count in pred_ngrams.items()
        }

        return sum(clipped_counts.values()) / sum(pred_ngrams.values())

    # Normalize input
    predictions = [prediction] if isinstance(prediction, str) else prediction
    references = [reference] if isinstance(reference, str) else reference

    precisions = []
    for n in range(1, max_order + 1):
        total_precision = 0.0
        for pred, ref in zip(predictions, references):
            pred_tokens = pred.split()
            ref_tokens = ref.split()
            total_precision += _modified_precision(pred_tokens, ref_tokens, n)
        precisions.append(total_precision / len(predictions))

    # Calculate brevity penalty
    pred_len = sum(len(p.split()) for p in predictions)
    ref_len = sum(len(r.split()) for r in references)

    if pred_len > ref_len:
        bp = 1.0
    elif pred_len == 0:
        bp = 0.0
    else:
        bp = math.exp(1 - ref_len / pred_len) if ref_len > 0 else 0.0

    # Calculate geometric mean of precisions
    if min(precisions) > 0:
        log_precisions = sum(math.log(p) for p in precisions) / len(precisions)
        bleu = bp * math.exp(log_precisions)
    else:
        bleu = 0.0

    return {
        "bleu": bleu,
        "precisions": precisions,
        "brevity_penalty": bp
    }


def bertscore(
    predictions: Union[str, List[str]],
    references: Union[str, List[str]],
    model_type: str = "bert-base-multilingual-cased",
    num_layers: int = None,
    batch_size: int = 32
) -> Dict[str, float]:
    """
    Calculate BERTScore between predictions and references.

    BERTScore uses contextual embeddings from BERT to compute semantic
    similarity, which is more robust to paraphrasing than n-gram metrics.

    Args:
        predictions: Generated text(s) - single string or list of strings
        references: Reference text(s) - single string or list of strings
        model_type: BERT model to use (default: multilingual BERT)
        num_layers: Number of transformer layers to use (default: all)
        batch_size: Batch size for encoding

    Returns:
        Dictionary with precision, recall, and F1 scores

    Example:
        >>> bertscore(
        ...     "The feline is resting on the rug",
        ...     "The cat is sitting on the mat"
        ... )
        {"precision": 0.92, "recall": 0.89, "f1": 0.905}

    Note:
        Requires bert-score package:
        pip install bert-score

        For multilingual support:
        pip install bert-score-multilingual
    """
    try:
        from bert_score import score

        # Normalize input
        if isinstance(predictions, str):
            predictions = [predictions]
        if isinstance(references, str):
            references = [references]

        # Calculate BERTScore
        P, R, F1 = score(
            predictions,
            references,
            model_type=model_type,
            num_layers=num_layers,
            batch_size=batch_size,
            verbose=False
        )

        return {
            "precision": P.mean().item(),
            "recall": R.mean().item(),
            "f1": F1.mean().item(),
            "individual_precisions": P.tolist(),
            "individual_recalls": R.tolist(),
            "individual_f1s": F1.tolist()
        }

    except ImportError:
        warnings.warn(
            "bert-score package not installed. BERTScore requires: pip install bert-score"
        )
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "error": "bert-score package not installed"
        }
    except Exception as e:
        warnings.warn(f"BERTScore calculation failed: {e}")
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "error": str(e)
        }


def compute_generation_metrics(
    predictions: Union[str, List[str]],
    references: Union[str, List[str]],
    use_bertscore: bool = False
) -> Dict[str, float]:
    """
    Compute all generation metrics at once.

    Args:
        predictions: Generated text(s)
        references: Reference text(s)
        use_bertscore: Whether to compute BERTScore (can be slow)

    Returns:
        Dictionary with all computed metrics

    Example:
        >>> compute_generation_metrics(
        ...     ["The cat sat on the mat", "Hello world"],
        ...     ["A cat sat on the mat", "Hi world"]
        ... )
        {
            "rouge_l_precision": 0.857,
            "rouge_l_recall": 0.857,
            "rouge_l_fmeasure": 0.857,
            "bleu": 0.723,
            "bleu_1": 0.85,
            "bleu_2": 0.72,
            "bleu_3": 0.68,
            "bleu_4": 0.62
        }
    """
    # Calculate ROUGE-L
    rouge_result = rouge_scores(predictions, references, rouge_type="L")
    if "precision" in rouge_result:
        rouge_l = rouge_result
    else:
        rouge_l = rouge_result.get("rougeL", rouge_result.get("rouge1", {}))

    # Calculate BLEU-4
    bleu_result = bleu_score(predictions, references, max_order=4)

    metrics = {
        "rouge_l_precision": rouge_l.get("precision", 0.0),
        "rouge_l_recall": rouge_l.get("recall", 0.0),
        "rouge_l_fmeasure": rouge_l.get("fmeasure", 0.0),
        "bleu": bleu_result.get("bleu", 0.0),
        "bleu_1": bleu_result.get("precisions", [0, 0, 0, 0])[0],
        "bleu_2": bleu_result.get("precisions", [0, 0, 0, 0])[1],
        "bleu_3": bleu_result.get("precisions", [0, 0, 0, 0])[2],
        "bleu_4": bleu_result.get("precisions", [0, 0, 0, 0])[3],
    }

    # Optionally add BERTScore
    if use_bertscore:
        bert_result = bertscore(predictions, references)
        metrics.update({
            "bertscore_precision": bert_result.get("precision", 0.0),
            "bertscore_recall": bert_result.get("recall", 0.0),
            "bertscore_f1": bert_result.get("f1", 0.0),
        })

    return metrics
