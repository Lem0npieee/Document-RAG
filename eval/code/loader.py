from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class DocVQASample:
    question_id: str
    question: str
    answers: list[str]
    image: str
    image_path: Path
    source_hint: str


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    return []


def _extract_records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]

    if not isinstance(payload, dict):
        return []

    for key in ("data", "questions", "samples", "annotations", "items"):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]

    return []


def _extract_answers(record: dict[str, Any]) -> list[str]:
    candidates: list[str] = []

    raw_answers = record.get("answers")
    if isinstance(raw_answers, list):
        for item in raw_answers:
            if isinstance(item, str) and item.strip():
                candidates.append(item.strip())
            elif isinstance(item, dict):
                text = str(
                    item.get("text", "")
                    or item.get("answer", "")
                    or item.get("value", "")
                ).strip()
                if text:
                    candidates.append(text)

    single_answer = record.get("answer")
    if isinstance(single_answer, str) and single_answer.strip():
        candidates.append(single_answer.strip())

    gt_answers = record.get("gt_answers")
    if isinstance(gt_answers, list):
        for item in gt_answers:
            if isinstance(item, str) and item.strip():
                candidates.append(item.strip())

    return list(dict.fromkeys(candidates))


def _extract_image_name(record: dict[str, Any]) -> str:
    for key in ("image", "image_path", "file_name", "filename", "file", "document"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _extract_question_id(record: dict[str, Any], index: int) -> str:
    for key in ("question_id", "questionId", "qid", "id"):
        value = record.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return f"sample_{index:06d}"


def _extract_question_text(record: dict[str, Any]) -> str:
    for key in ("question", "query", "text"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def load_docvqa_samples(
    annotations_path: str | Path,
    images_root: str | Path,
    require_image_exists: bool = True,
    keep_unanswered: bool = False,
) -> tuple[list[DocVQASample], list[str]]:
    annotations = Path(annotations_path)
    images_dir = Path(images_root)

    if not annotations.exists():
        raise FileNotFoundError(f"Annotations file not found: {annotations}")

    payload = json.loads(annotations.read_text(encoding="utf-8"))
    records = _extract_records(payload)
    if not records:
        raise ValueError(
            "No sample records found in annotations JSON. "
            "Expected list or dict with key one of: data/questions/samples/annotations/items."
        )

    warnings: list[str] = []
    samples: list[DocVQASample] = []

    for idx, record in enumerate(records, start=1):
        qid = _extract_question_id(record, idx)
        question = _extract_question_text(record)
        answers = _extract_answers(record)
        image_name = _extract_image_name(record)

        if not question:
            warnings.append(f"[{qid}] empty question, skipped")
            continue
        if not image_name:
            warnings.append(f"[{qid}] missing image field, skipped")
            continue
        if not answers and not keep_unanswered:
            warnings.append(f"[{qid}] no reference answers, skipped")
            continue

        image_path = Path(image_name)
        if not image_path.is_absolute():
            image_path = images_dir / image_path
        image_path = image_path.resolve()

        if require_image_exists and not image_path.exists():
            warnings.append(f"[{qid}] image not found: {image_path}, skipped")
            continue

        source_hint = image_path.name
        samples.append(
            DocVQASample(
                question_id=qid,
                question=question,
                answers=answers,
                image=image_name,
                image_path=image_path,
                source_hint=source_hint,
            )
        )

    return samples, warnings
