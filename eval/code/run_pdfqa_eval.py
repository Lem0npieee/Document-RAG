from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from loader import PDFQASample, load_pdfqa_samples
from metrics import (
    anls_score,
    containment_match,
    evidence_page_recall,
    exact_match,
    levenshtein_distance,
    normalize_text,
    score_all_metrics,
    summarize_metrics,
    token_f1,
)


def _default_paths() -> dict[str, Path]:
    eval_root = Path(__file__).resolve().parents[1]
    output_root = eval_root / "output" / "pdfqa"
    return {
        "eval_root": eval_root,
        "annotations_root": eval_root / "input" / "pdfqa" / "annotations",
        "pdfs_root": eval_root / "input" / "pdfqa" / "pdfs",
        "kb_root": eval_root / "output" / "kb",
        "output_root": output_root,
        "predictions": output_root / "predictions.jsonl",
        "metrics": output_root / "metrics.json",
        "errors": output_root / "errors_topk.jsonl",
    }


def _configure_environment(kb_root: Path, doc_root: Path) -> None:
    os.environ["OUTPUT_ROOT"] = str(kb_root)
    os.environ["DOC_ROOT"] = str(doc_root)


def _safe_write_text(path: Path, content: str, fallback_name: str) -> Path:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path
    except Exception:
        fallback = Path.cwd() / fallback_name
        fallback.write_text(content, encoding="utf-8")
        return fallback


def _strip_citation_tail(text: str) -> str:
    markers = [
        "引用页码",
        "关键关系",
        "citation",
        "citations",
        "reference pages",
        "evidence pages",
        "keyword::",
        "【引用",
        "[引用",
        "（引用",
        "来源：",
        "来源:",
        "source:",
    ]
    lower = text.lower()
    cut_at = -1
    for marker in markers:
        idx = lower.find(marker.lower())
        if idx >= 0:
            cut_at = idx if cut_at < 0 else min(cut_at, idx)
    if cut_at >= 0:
        return text[:cut_at].rstrip()
    return text


def _extract_short_answer(text: str) -> str:
    """Try to extract the actual short answer from a verbose model response.

    Only extracts when there's a clear answer marker. Otherwise returns the
    full text so that containment/F1 metrics can handle it properly.
    """
    text = text.strip()
    if not text:
        return ""

    # Already short enough — likely a direct answer
    if len(text) <= 20:
        return text

    # Explicit answer markers: "答案是X", "Answer: X", etc.
    # Use greedy match for numeric patterns (to capture "95.2%" not just "95"),
    # then take only the first sentence/clause after the marker.
    patterns = [
        (r"答案[是为：:]\s*(.+?)(?:[。\n，;；]|$)", True),
        (r"[Aa]nswer[:\s]+(.+?)(?:[\n,;]|$)", True),
        (r"[Ff]inal [Aa]nswer[:\s]+(.+?)(?:[\n,;]|$)", True),
        (r"结果[是为：:]\s*(.+?)(?:[。\n，;；]|$)", True),
    ]
    for pat, _ in patterns:
        m = re.search(pat, text)
        if m:
            extracted = m.group(1).strip()
            # If extracted content still has commas/periods, take the first clause
            # This handles "95.2%, which is better than..." → "95.2%"
            for sep in (",", "，", "；", ";"):
                if sep in extracted:
                    first_part = extracted.split(sep)[0].strip()
                    # Keep only if it looks like a real answer (not too long)
                    if len(first_part) <= 40:
                        extracted = first_part
                        break
            if extracted:
                return extracted

    # No clear marker found — return as-is and let containment/F1 handle it
    return text


def _clean_model_answer(raw: str, max_chars: int = 0) -> str:
    text = str(raw or "").strip()
    if not text:
        return ""

    # Remove code fences
    text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
    text = re.sub(r"\s*```$", "", text).strip()

    # Strip leading answer labels
    for prefix in ("答案：", "答案:", "答案是", "答案为", "answer:", "Answer:", "final answer:", "Final answer:"):
        if text.startswith(prefix):
            text = text[len(prefix) :].strip()
            break

    # Strip citation tails
    text = _strip_citation_tail(text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    # Normalize yes/no answers
    lowered = text.lower()
    if lowered.startswith("yes"):
        text = "yes"
    elif lowered.startswith("no"):
        text = "no"

    # Try to extract short answer from verbose output
    text = _extract_short_answer(text)

    # Strip trailing punctuation that is not part of the answer (e.g. "95.2%。" → "95.2%")
    text = text.rstrip("。.，,；;！!？?")

    if max_chars and max_chars > 0 and len(text) > max_chars:
        text = text[:max_chars].rstrip()
    return text


def _similarity(a: str, b: str) -> float:
    na = normalize_text(a)
    nb = normalize_text(b)
    if not na and not nb:
        return 1.0
    if not na or not nb:
        return 0.0
    dist = levenshtein_distance(na, nb)
    return 1.0 - (dist / max(len(na), len(nb), 1))


def _best_span_match(pred: str, refs: list[str]) -> str:
    if not pred or not refs:
        return pred

    refs_clean = [str(r or "").strip() for r in refs if str(r or "").strip()]
    if not refs_clean:
        return pred

    candidates: list[str] = [pred]
    sentences = [s.strip() for s in re.split(r"[.!?;\n]+", pred) if s.strip()]
    candidates.extend(sentences)

    tokens = pred.split()
    if tokens:
        ref_lens = [max(1, len(r.split())) for r in refs_clean]
        target_len = max(1, min(ref_lens))
        min_len = max(1, int(target_len * 0.6))
        max_len = max(min_len, int(target_len * 1.8))
        max_len = min(max_len, len(tokens))
        for w in range(min_len, max_len + 1):
            for i in range(0, len(tokens) - w + 1):
                candidates.append(" ".join(tokens[i : i + w]))

    best = pred
    best_score = max(_similarity(pred, ref) for ref in refs_clean)
    seen: set[str] = set()
    for cand in candidates:
        c = cand.strip()
        if not c or c in seen:
            continue
        seen.add(c)
        score = max(_similarity(c, ref) for ref in refs_clean)
        if score > best_score:
            best_score = score
            best = c
    return best


def _load_existing_predictions(predictions_file: Path) -> dict[str, dict[str, Any]]:
    if not predictions_file.exists():
        return {}
    rows: dict[str, dict[str, Any]] = {}
    with predictions_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except Exception:
                continue
            qid = str(item.get("question_id", "")).strip()
            if not qid:
                continue
            rows[qid] = item
    return rows


def _append_jsonl(path: Path, item: dict[str, Any]) -> Path:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
        return path
    except Exception:
        fallback = Path.cwd() / path.name
        with fallback.open("a", encoding="utf-8") as f:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
        return fallback


def _group_metrics(records: list[dict[str, Any]], key: str, threshold: float) -> dict[str, dict[str, float | int]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for rec in records:
        group_name = str(rec.get(key, "unknown") or "unknown")
        groups.setdefault(group_name, []).append(rec)
    result: dict[str, dict[str, float | int]] = {}
    for name, items in sorted(groups.items(), key=lambda x: x[0]):
        result[name] = summarize_metrics(items, threshold=threshold)
    return result


def _record_from_result(sample: PDFQASample, result: Any, max_answer_chars: int) -> dict[str, Any]:
    prediction = _clean_model_answer(result.answer, max_chars=max_answer_chars)
    return {
        "question_id": sample.question_id,
        "question": sample.question,
        "answers": sample.answers,
        "prediction": prediction,
        "raw_answer": result.answer,
        "source_hint": sample.source_hint,
        "doc_name": sample.doc_name,
        "category": sample.category,
        "dataset": sample.dataset,
        "question_type": sample.question_type,
        "evidence_pages": sample.evidence_pages,
        "pages": result.pages,
        "node_ids": result.node_ids,
        "relations": result.relations,
        "image_paths": result.image_paths,
        "error": "",
        "created_at": datetime.now().isoformat(),
    }


def _error_record(sample: PDFQASample, exc: Exception) -> dict[str, Any]:
    return {
        "question_id": sample.question_id,
        "question": sample.question,
        "answers": sample.answers,
        "prediction": "",
        "raw_answer": "",
        "source_hint": sample.source_hint,
        "doc_name": sample.doc_name,
        "category": sample.category,
        "dataset": sample.dataset,
        "question_type": sample.question_type,
        "evidence_pages": sample.evidence_pages,
        "pages": [],
        "node_ids": [],
        "relations": [],
        "image_paths": [],
        "error": f"{exc.__class__.__name__}: {exc}",
        "created_at": datetime.now().isoformat(),
    }


def parse_args() -> argparse.Namespace:
    defaults = _default_paths()
    parser = argparse.ArgumentParser(description="Run pdfQA evaluation with current GraphRAG model.")
    parser.add_argument("--annotations-root", type=Path, default=defaults["annotations_root"])
    parser.add_argument("--pdfs-root", type=Path, default=defaults["pdfs_root"])
    parser.add_argument("--kb-root", type=Path, default=defaults["kb_root"])
    parser.add_argument("--output-root", type=Path, default=defaults["output_root"])
    parser.add_argument("--predictions-file", type=Path, default=defaults["predictions"])
    parser.add_argument("--metrics-file", type=Path, default=defaults["metrics"])
    parser.add_argument("--errors-file", type=Path, default=defaults["errors"])
    parser.add_argument("--category", type=str, default="real", choices=["all", "real", "syn"])
    parser.add_argument("--qa-split", type=str, default="all", choices=["all", "raw", "vf", "cf"])
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all samples")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--max-nodes", type=int, default=24)
    parser.add_argument("--anls-threshold", type=float, default=0.5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--strict-docs", action="store_true")
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--max-answer-chars", type=int, default=0, help="0 means no truncation")
    parser.add_argument("--score-mode", type=str, default="best_span", choices=["raw", "best_span"])
    parser.add_argument("--max-ref-chars", type=int, default=0, help="0 means evaluate all answers")
    parser.add_argument(
        "--answer-profile",
        type=str,
        default="very_short",
        choices=["all", "binary", "very_short", "short"],
        help="filter evaluation set by answer type",
    )
    parser.add_argument("--recompute-only", action="store_true", help="do not call model, only rescore existing predictions")
    parser.add_argument("--ablation", type=str, default=None, choices=[None, "vector_only", "graph_only"], help="ablation mode")
    parser.add_argument("--baseline", type=str, default=None, choices=[None, "always_no", "always_yes"], help="simple baseline")
    parser.add_argument("--exclude-docs", type=str, default="", help="comma-separated doc names to exclude from eval")
    parser.add_argument("--kb-docs-only", action="store_true", help="auto-exclude samples whose doc is not in KB")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    samples, warnings = load_pdfqa_samples(
        annotations_root=args.annotations_root,
        pdfs_root=args.pdfs_root,
        category=args.category,
        qa_split=args.qa_split,
        require_doc_exists=args.strict_docs,
        keep_unanswered=False,
        max_samples=args.max_samples,
    )

    def _is_binary_answer(text: str) -> bool:
        t = normalize_text(text)
        return t in {"yes", "no"}

    def _keep_sample(sample: PDFQASample) -> bool:
        if args.answer_profile == "all":
            return True
        if not sample.answers:
            return False
        first = str(sample.answers[0] or "")
        if args.answer_profile == "binary":
            return _is_binary_answer(first)
        if args.answer_profile == "very_short":
            return len(first) <= 5
        if args.answer_profile == "short":
            return len(first) <= 20
        return True

    samples = [s for s in samples if _keep_sample(s)]

    if args.exclude_docs:
        exclude_set = {name.strip() for name in args.exclude_docs.split(",") if name.strip()}
        before = len(samples)
        samples = [s for s in samples if s.doc_name not in exclude_set]
        print(f"Excluded {before - len(samples)} samples by --exclude-docs (remaining {len(samples)})")

    if args.kb_docs_only:
        docs_json = args.kb_root / "parsed" / "documents.json"
        if docs_json.exists():
            docs_list = json.loads(docs_json.read_text(encoding="utf-8"))
            kb_sources = {str(d.get("metadata", {}).get("source", "")) for d in docs_list if isinstance(d, dict)}
            kb_sources.discard("")
            if kb_sources:
                before = len(samples)
                excluded = [s for s in samples if s.doc_name not in kb_sources]
                samples = [s for s in samples if s.doc_name in kb_sources]
                print(f"--kb-docs-only: dropped {before - len(samples)} samples (remaining {len(samples)})")
                for s in excluded:
                    print(f"  SKIP {s.doc_name} — not in KB")

    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    existing = _load_existing_predictions(args.predictions_file) if args.resume else {}
    chain = None
    append_target = args.predictions_file

    if not args.recompute_only and not args.baseline:
        _configure_environment(kb_root=args.kb_root, doc_root=args.pdfs_root)

        expected_faiss = args.kb_root / "faiss_index" / "index.faiss"
        expected_graph_candidates = [
            args.kb_root / "doc_graph" / "graph.pkl",
            args.kb_root / "graph" / "graph.pkl",
        ]
        graph_found = next((p for p in expected_graph_candidates if p.exists()), None)
        if not expected_faiss.exists() or graph_found is None:
            missing: list[str] = []
            if not expected_faiss.exists():
                missing.append(str(expected_faiss))
            if graph_found is None:
                missing.extend(str(p) for p in expected_graph_candidates)
            raise FileNotFoundError(
                "KB artifacts not found. Please build KB first. Missing: "
                + ", ".join(missing)
            )

        from src.config import get_settings
        from src.rag.multimodal_graph_rag_chain import MultiModalGraphRAG

        settings = get_settings()
        chain = MultiModalGraphRAG(
            settings=settings,
            faiss_dir=settings.faiss_dir,
            graph_path=settings.graph_dir / "graph.pkl",
            pages_dir=settings.pages_dir,
        )

    done = 0
    failed = 0
    skipped_by_ref_len = 0

    print("============================================================")
    print("pdfQA Evaluation")
    print(f"annotations_root: {args.annotations_root}")
    print(f"pdfs_root: {args.pdfs_root}")
    print(f"kb_root: {args.kb_root}")
    print(f"category: {args.category}")
    print(f"qa_split: {args.qa_split}")
    print(f"samples: {len(samples)}")
    print(f"answer_profile: {args.answer_profile}")
    if warnings:
        print(f"warnings: {len(warnings)} (first 20)")
        for item in warnings[:20]:
            print(f"  - {item}")
    print("============================================================")

    for idx, sample in enumerate(samples, start=1):
        if args.resume and sample.question_id in existing:
            raw = str(existing[sample.question_id].get("raw_answer", ""))
            if raw.strip():
                existing[sample.question_id]["prediction"] = _clean_model_answer(
                    raw, max_chars=args.max_answer_chars
                )
            done += 1
            continue

        if args.recompute_only:
            continue

        try:
            if args.baseline:
                # Simple baseline: always return a fixed answer
                from src.rag.multimodal_graph_rag_chain import GraphRAGResult
                baseline_answer = "no" if args.baseline == "always_no" else "yes"
                result = GraphRAGResult(
                    answer=baseline_answer,
                    node_ids=[], pages=[], image_paths=[], relations=[],
                )
            else:
                result = chain.ask_eval(
                    question=sample.question,
                    source_hint=sample.source_hint,
                    k=args.k,
                    max_nodes=args.max_nodes,
                    ablation=args.ablation,
                )
            record = _record_from_result(sample, result, max_answer_chars=args.max_answer_chars)
            append_target = _append_jsonl(append_target, record)
            existing[sample.question_id] = record
            done += 1
        except Exception as exc:
            failed += 1
            record = _error_record(sample, exc)
            append_target = _append_jsonl(append_target, record)
            existing[sample.question_id] = record

        if args.log_every > 0 and idx % args.log_every == 0:
            print(f"progress: {idx}/{len(samples)}, done={done}, failed={failed}")

    ordered_records: list[dict[str, Any]] = []
    for sample in samples:
        rec = existing.get(sample.question_id)
        if rec:
            ordered_records.append(rec)

    scored_records: list[dict[str, Any]] = []
    for rec in ordered_records:
        raw_answer = str(rec.get("raw_answer", "")).strip()
        if raw_answer:
            rec["prediction"] = _clean_model_answer(raw_answer, max_chars=args.max_answer_chars)

        answers = rec.get("answers", [])
        if not isinstance(answers, list):
            answers = []
        if args.max_ref_chars and args.max_ref_chars > 0:
            short_refs = [a for a in answers if len(str(a or "")) <= args.max_ref_chars]
            if not short_refs:
                skipped_by_ref_len += 1
                continue
            answers = short_refs

        pred = str(rec.get("prediction", ""))
        if args.score_mode == "best_span":
            pred = _best_span_match(pred, answers)
            rec["prediction_best_span"] = pred

        all_scores = score_all_metrics(pred, answers, threshold=args.anls_threshold)
        rec["anls"] = all_scores["anls"]
        rec["em"] = all_scores["em"]
        rec["containment"] = all_scores["containment"]
        rec["token_f1"] = all_scores["token_f1"]
        rec["heuristic_judge"] = all_scores["heuristic_judge"]
        scored_records.append(rec)

    core = summarize_metrics(scored_records, threshold=args.anls_threshold)

    # Compute averages for new metrics
    n_scored = len(scored_records) or 1
    avg_containment = round(sum(float(r.get("containment", 0)) for r in scored_records) / n_scored, 6)
    avg_token_f1 = round(sum(float(r.get("token_f1", 0)) for r in scored_records) / n_scored, 6)
    avg_heuristic_judge = round(sum(float(r.get("heuristic_judge", 0)) for r in scored_records) / n_scored, 6)

    page_recall = evidence_page_recall(scored_records)
    by_category = _group_metrics(scored_records, key="category", threshold=args.anls_threshold)
    by_dataset = _group_metrics(scored_records, key="dataset", threshold=args.anls_threshold)
    by_question_type = _group_metrics(scored_records, key="question_type", threshold=args.anls_threshold)

    def _augment_group_with_new_metrics(
        group_dict: dict[str, dict[str, float | int]],
        records: list[dict[str, Any]],
        key: str,
    ) -> dict[str, dict[str, float | int]]:
        result: dict[str, dict[str, float | int]] = {}
        for name, sub in group_dict.items():
            sub = dict(sub)
            group_recs = [r for r in records if str(r.get(key, "unknown") or "unknown") == name]
            n = len(group_recs) or 1
            sub["containment"] = round(sum(float(r.get("containment", 0)) for r in group_recs) / n, 6)
            sub["token_f1"] = round(sum(float(r.get("token_f1", 0)) for r in group_recs) / n, 6)
            sub["heuristic_judge"] = round(sum(float(r.get("heuristic_judge", 0)) for r in group_recs) / n, 6)
            result[name] = sub
        return result

    by_category = _augment_group_with_new_metrics(by_category, scored_records, "category")
    by_dataset = _augment_group_with_new_metrics(by_dataset, scored_records, "dataset")
    by_question_type = _augment_group_with_new_metrics(by_question_type, scored_records, "question_type")

    summary = {
        "updated_at": datetime.now().isoformat(),
        "annotations_root": str(args.annotations_root),
        "pdfs_root": str(args.pdfs_root),
        "kb_root": str(args.kb_root),
        "predictions_file": str(args.predictions_file),
        "actual_predictions_output": str(append_target),
        "category": args.category,
        "qa_split": args.qa_split,
        "sample_count": len(samples),
        "scored_count": core["count"],
        "done_count": done,
        "failed_count": failed,
        "warnings_count": len(warnings),
        "skipped_by_ref_len": skipped_by_ref_len,
        "anls_threshold": args.anls_threshold,
        "anls": core["anls"],
        "em": core["em"],
        "containment": avg_containment,
        "token_f1": avg_token_f1,
        "heuristic_judge": avg_heuristic_judge,
        "evidence_page_recall": page_recall,
        "k": args.k,
        "max_nodes": args.max_nodes,
        "max_answer_chars": args.max_answer_chars,
        "score_mode": args.score_mode,
        "max_ref_chars": args.max_ref_chars,
        "answer_profile": args.answer_profile,
        "by_category": by_category,
        "by_dataset": by_dataset,
        "by_question_type": by_question_type,
    }

    metrics_written = _safe_write_text(
        args.metrics_file,
        json.dumps(summary, ensure_ascii=False, indent=2),
        fallback_name="metrics_fallback.json",
    )

    hard_cases = sorted(scored_records, key=lambda x: (float(x.get("anls", 0.0)), float(x.get("em", 0.0))))
    errors_payload = "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in hard_cases[:200])
    errors_written = _safe_write_text(
        args.errors_file,
        errors_payload,
        fallback_name="errors_topk_fallback.jsonl",
    )

    print("============================================================")
    print("Evaluation done.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"metrics_file: {metrics_written}")
    print(f"errors_file: {errors_written}")
    print("============================================================")


if __name__ == "__main__":
    main()
