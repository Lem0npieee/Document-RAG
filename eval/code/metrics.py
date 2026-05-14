from __future__ import annotations

import re
import unicodedata
from typing import Any


def normalize_text(text: str) -> str:
    value = unicodedata.normalize("NFKC", str(text or ""))
    value = value.strip().lower()
    value = re.sub(r"\s+", " ", value)
    return value


def levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    if len(a) < len(b):
        a, b = b, a

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            ins = cur[j - 1] + 1
            delete = prev[j] + 1
            replace = prev[j - 1] + (0 if ca == cb else 1)
            cur.append(min(ins, delete, replace))
        prev = cur
    return prev[-1]


def anls_score_single(pred: str, ref: str, threshold: float = 0.5) -> float:
    p = normalize_text(pred)
    r = normalize_text(ref)
    if not p and not r:
        return 1.0
    if not p or not r:
        return 0.0
    dist = levenshtein_distance(p, r)
    denom = max(len(p), len(r), 1)
    sim = 1.0 - (dist / denom)
    return sim if sim >= threshold else 0.0


def anls_score(pred: str, refs: list[str], threshold: float = 0.5) -> float:
    if not refs:
        return 0.0
    return max(anls_score_single(pred, ref, threshold=threshold) for ref in refs)


def exact_match(pred: str, refs: list[str]) -> float:
    p = normalize_text(pred)
    for ref in refs:
        if p == normalize_text(ref):
            return 1.0
    return 0.0


def summarize_metrics(records: list[dict[str, Any]], threshold: float = 0.5) -> dict[str, float | int]:
    if not records:
        return {"count": 0, "anls": 0.0, "em": 0.0}

    anls_values: list[float] = []
    em_values: list[float] = []
    for item in records:
        pred = str(item.get("prediction", ""))
        refs = item.get("answers", [])
        if not isinstance(refs, list):
            refs = []
        anls_values.append(anls_score(pred, refs, threshold=threshold))
        em_values.append(exact_match(pred, refs))

    n = len(records)
    return {
        "count": n,
        "anls": round(sum(anls_values) / n, 6),
        "em": round(sum(em_values) / n, 6),
    }


def evidence_page_recall(records: list[dict[str, Any]]) -> dict[str, float | int]:
    labeled = 0
    hit = 0
    for item in records:
        gt_pages = item.get("evidence_pages", [])
        pred_pages = item.get("pages", [])
        if not isinstance(gt_pages, list) or not gt_pages:
            continue
        labeled += 1
        gt = set()
        for p in gt_pages:
            try:
                ip = int(p)
            except Exception:
                continue
            if ip > 0:
                gt.add(ip)
        pred = set()
        if isinstance(pred_pages, list):
            for p in pred_pages:
                try:
                    ip = int(p)
                except Exception:
                    continue
                if ip > 0:
                    pred.add(ip)
        if gt and gt.intersection(pred):
            hit += 1

    recall = (hit / labeled) if labeled else 0.0
    return {
        "labeled_count": labeled,
        "hit_count": hit,
        "recall": round(recall, 6),
    }


def _char_ngrams(text: str, n: int = 2) -> list[str]:
    """Character n-gram tokenization, works for both Chinese and English."""
    normalized = normalize_text(text)
    if not normalized:
        return []
    if len(normalized) < n:
        return [normalized]
    return [normalized[i : i + n] for i in range(len(normalized) - n + 1)]


def containment_match(pred: str, refs: list[str]) -> float:
    """Return 1.0 if normalized ref is contained in normalized pred (or vice versa), else 0.0.

    This metric tolerates verbose predictions that embed the correct answer inside
    extra explanation — e.g. "根据文档内容，答案是Table 2" correctly matches ref "Table 2".

    For short refs (≤3 chars), uses bigram overlap to avoid false positives from
    character-level substring matching (e.g. ref="2" matching any text containing "2").
    """
    np = normalize_text(pred)
    if not np:
        return 0.0
    for ref in refs:
        nr = normalize_text(ref)
        if not nr:
            continue
        # ref inside pred — exact substring
        if len(nr) > 3 and np.find(nr) >= 0:
            return 1.0
        # pred inside ref — exact substring
        if len(np) > 3 and nr.find(np) >= 0:
            return 1.0
        # For short strings, verify via bigram overlap to reduce false positives
        ref_grams = _char_ngrams(nr, n=2) if len(nr) <= 3 else []
        if ref_grams:
            pred_grams = _char_ngrams(np, n=2)
            overlap = sum(1 for g in ref_grams if g in pred_grams)
            if overlap == len(ref_grams):
                return 1.0
        # Long-string containment check for the remaining case
        if len(nr) <= 3 and np.find(nr) >= 0:
            # Short ref found in pred, but bigram check already handled above
            # This catches cases where bigram generation fails (single char)
            if len(nr) == 1 and np.find(nr) >= 0:
                return 1.0
    return 0.0


def token_f1(pred: str, refs: list[str]) -> float:
    """Character bigram F1 between normalized prediction and best-matching reference.

    Uses character n-grams instead of word tokens so it works for Chinese text.
    """
    pred_grams = _char_ngrams(pred, n=2)
    if not pred_grams:
        return 0.0

    best_f1 = 0.0
    for ref in refs:
        ref_grams = _char_ngrams(ref, n=2)
        if not ref_grams:
            continue

        pred_counts: dict[str, int] = {}
        for g in pred_grams:
            pred_counts[g] = pred_counts.get(g, 0) + 1

        ref_counts: dict[str, int] = {}
        for g in ref_grams:
            ref_counts[g] = ref_counts.get(g, 0) + 1

        common = 0
        for g, cnt in ref_counts.items():
            common += min(cnt, pred_counts.get(g, 0))

        precision = common / len(pred_grams) if pred_grams else 0.0
        recall = common / len(ref_grams) if ref_grams else 0.0

        if precision > 0 and recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        if f1 > best_f1:
            best_f1 = f1

    return round(best_f1, 6)


def heuristic_judge_score(pred: str, refs: list[str], threshold: float = 0.5) -> float:
    """Compute an approximate LLM-judge score using heuristics when LLM call is unavailable.

    Rules:
    - Exact match after normalization: 1.0
    - containment match: 0.9
    - token F1 >= 0.8: 0.8
    - token F1 >= threshold: 0.5  (uses the same threshold as ANLS for consistency)
    - token F1 >= 0.3: 0.25
    - else: 0.0

    Note: threshold values (0.8/0.5/0.3) are heuristic and not yet calibrated
    against real evaluation data. Adjust after pilot evaluation if needed.
    """
    if exact_match(pred, refs) == 1.0:
        return 1.0
    if containment_match(pred, refs) == 1.0:
        return 0.9
    f1 = token_f1(pred, refs)
    if f1 >= 0.8:
        return 0.8
    if f1 >= threshold:
        return 0.5
    if f1 >= 0.3:
        return 0.25
    return 0.0


def score_all_metrics(pred: str, refs: list[str], threshold: float = 0.5) -> dict[str, float]:
    """Compute all available metrics for a single (prediction, reference-list) pair.

    The `threshold` parameter is passed to both anls_score (where it gates
    low-similarity results to 0.0) and llm_judge_score (where it gates the
    0.5 tier).  The em, containment, and token_f1 metrics are not affected
    by threshold — they are inherently binary or continuous.

    Returns a dict with: anls, em, containment, token_f1, llm_judge.
    """
    return {
        "anls": anls_score(pred, refs, threshold=threshold),
        "em": exact_match(pred, refs),
        "containment": containment_match(pred, refs),
        "token_f1": token_f1(pred, refs),
        "heuristic_judge": heuristic_judge_score(pred, refs, threshold=threshold),
    }

