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

    # Keep memory O(min(len(a), len(b))).
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
