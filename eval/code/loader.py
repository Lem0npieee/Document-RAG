from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class PDFQASample:
    question_id: str
    question: str
    answers: list[str]
    doc_name: str
    doc_path: Path
    source_hint: str
    category: str
    dataset: str
    question_type: str
    evidence_pages: list[int]
    annotation_file: str


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _as_records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if not isinstance(payload, dict):
        return []

    for key in ("qa_pairs", "qas", "questions", "items", "records", "examples", "data"):
        value = payload.get(key)
        if isinstance(value, list):
            return [x for x in value if isinstance(x, dict)]

    for value in payload.values():
        if isinstance(value, list) and value and isinstance(value[0], dict):
            return [x for x in value if isinstance(x, dict)]
    return []


def _to_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _to_answers(item: dict[str, Any]) -> list[str]:
    candidates: list[str] = []
    for key in ("answers", "answer", "gold_answers", "reference_answers", "references", "label"):
        if key not in item:
            continue
        value = item.get(key)
        if isinstance(value, list):
            candidates.extend(_to_str(x) for x in value)
        else:
            candidates.append(_to_str(value))
    cleaned = []
    for answer in candidates:
        if answer:
            cleaned.append(answer)
    return cleaned


def _extract_question(item: dict[str, Any]) -> str:
    for key in ("question", "query", "prompt", "instruction"):
        value = _to_str(item.get(key))
        if value:
            return value
    return ""


def _extract_question_type(item: dict[str, Any]) -> str:
    for key in ("question_type", "type", "task_type"):
        value = _to_str(item.get(key))
        if value:
            return value
    return "unknown"


def _extract_evidence_pages(item: dict[str, Any]) -> list[int]:
    pages: list[int] = []
    for key in ("evidence_pages", "pages", "page_ids", "relevant_pages"):
        value = item.get(key)
        if isinstance(value, list):
            for p in value:
                try:
                    ip = int(p)
                except Exception:
                    continue
                if ip > 0:
                    pages.append(ip)
    if not pages:
        single_page = item.get("page")
        if single_page is not None:
            try:
                ip = int(single_page)
                if ip > 0:
                    pages.append(ip)
            except Exception:
                pass
    uniq = []
    seen = set()
    for p in pages:
        if p in seen:
            continue
        seen.add(p)
        uniq.append(p)
    return uniq


def _extract_doc_name(item: dict[str, Any], fallback_stem: str) -> str:
    for key in (
        "file_name",
        "pdf",
        "pdf_file",
        "document",
        "doc",
        "source",
        "image",
        "file",
        "doc_name",
    ):
        value = _to_str(item.get(key))
        if value:
            return value
    return fallback_stem


def _build_doc_index(pdfs_root: Path) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    if not pdfs_root.exists():
        return index
    for p in pdfs_root.rglob("*.pdf"):
        base = p.name.lower()
        stem = p.stem.lower()
        index.setdefault(base, []).append(p.resolve())
        index.setdefault(stem, []).append(p.resolve())
    return index


def _resolve_doc_path(doc_name: str, pdfs_root: Path, doc_index: dict[str, list[Path]]) -> Path | None:
    candidate = Path(doc_name)
    direct = (pdfs_root / candidate).resolve()
    if direct.exists() and direct.is_file():
        return direct

    basename = candidate.name.lower()
    stem = candidate.stem.lower()
    for key in (basename, stem):
        if key not in doc_index:
            continue
        choices = doc_index[key]
        if choices:
            return choices[0]

    if not candidate.suffix:
        alt = f"{candidate.name}.pdf".lower()
        if alt in doc_index and doc_index[alt]:
            return doc_index[alt][0]
    return None


def _category_from_path(path: Path, annotations_root: Path) -> tuple[str, str]:
    rel_parts = path.resolve().relative_to(annotations_root.resolve()).parts
    lower_parts = [x.lower() for x in rel_parts]

    category = "unknown"
    for token in lower_parts:
        if token in {"real-pdfqa", "real"}:
            category = "real"
            break
        if token in {"syn-pdfqa", "syn", "synthetic"}:
            category = "syn"
            break

    dataset = "unknown"
    if len(rel_parts) >= 2:
        dataset = rel_parts[1]
    return category, dataset


def load_pdfqa_samples(
    annotations_root: Path,
    pdfs_root: Path,
    category: str = "real",
    qa_split: str = "all",
    require_doc_exists: bool = True,
    keep_unanswered: bool = False,
    max_samples: int = 0,
) -> tuple[list[PDFQASample], list[str]]:
    warnings: list[str] = []
    category = _to_str(category).lower() or "real"
    qa_split = _to_str(qa_split).lower() or "all"
    allow_all = category == "all"
    allow_all_split = qa_split == "all"

    if annotations_root.is_file():
        annotation_files = [annotations_root.resolve()]
    elif annotations_root.exists():
        annotation_files = sorted(p.resolve() for p in annotations_root.rglob("*.json"))
    else:
        raise FileNotFoundError(f"annotations root not found: {annotations_root}")

    doc_index = _build_doc_index(pdfs_root)
    samples: list[PDFQASample] = []

    for ann_file in annotation_files:
        ann_category, dataset = _category_from_path(ann_file, annotations_root)
        if not allow_all and ann_category != category:
            continue
        if not allow_all_split:
            rel_text = str(ann_file).lower()
            if qa_split == "raw" and "_rawqa" not in rel_text:
                continue
            if qa_split == "vf" and "_vfqa" not in rel_text:
                continue
            if qa_split == "cf" and "_cfqa" not in rel_text:
                continue

        try:
            payload = _read_json(ann_file)
        except Exception as exc:
            warnings.append(f"failed to read {ann_file}: {exc}")
            continue

        records = _as_records(payload)
        if not records:
            warnings.append(f"no records parsed from {ann_file}")
            continue

        doc_fallback = ann_file.stem
        for idx, item in enumerate(records, start=1):
            question = _extract_question(item)
            if not question:
                warnings.append(f"skip empty question: {ann_file}#{idx}")
                continue

            answers = _to_answers(item)
            if not answers and not keep_unanswered:
                continue

            doc_name_raw = _extract_doc_name(item, doc_fallback)
            doc_path = _resolve_doc_path(doc_name_raw, pdfs_root, doc_index)
            if doc_path is None:
                if require_doc_exists:
                    warnings.append(
                        f"missing pdf for record: {ann_file}#{idx}, doc={doc_name_raw}"
                    )
                    continue
                source_hint = Path(doc_name_raw).name if doc_name_raw else doc_fallback
                doc_path = (pdfs_root / source_hint).resolve()
            else:
                source_hint = doc_path.name

            qid = _to_str(item.get("question_id") or item.get("qid") or item.get("id"))
            if not qid:
                rel = ann_file.resolve().relative_to(annotations_root.resolve())
                qid = f"{rel.as_posix()}#{idx}"

            sample = PDFQASample(
                question_id=qid,
                question=question,
                answers=answers,
                doc_name=source_hint,
                doc_path=doc_path,
                source_hint=source_hint,
                category=ann_category,
                dataset=dataset,
                question_type=_extract_question_type(item),
                evidence_pages=_extract_evidence_pages(item),
                annotation_file=str(ann_file),
            )
            samples.append(sample)
            if max_samples and len(samples) >= max_samples:
                return samples, warnings

    return samples, warnings
