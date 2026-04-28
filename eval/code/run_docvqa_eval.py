from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from loader import load_docvqa_samples
from metrics import anls_score, exact_match, summarize_metrics


def _default_paths() -> dict[str, Path]:
    eval_root = Path(__file__).resolve().parents[1]
    output_root = eval_root / "output" / "docvqa"
    return {
        "eval_root": eval_root,
        "annotations": eval_root / "input" / "docvqa" / "val.json",
        "images_root": eval_root / "input" / "docvqa" / "images",
        "kb_root": eval_root / "output" / "kb",
        "output_root": output_root,
        "predictions": output_root / "predictions.jsonl",
        "metrics": output_root / "metrics.json",
        "errors": output_root / "errors_topk.jsonl",
    }


def _configure_environment(kb_root: Path, doc_root: Path) -> None:
    os.environ["OUTPUT_ROOT"] = str(kb_root)
    os.environ["DOC_ROOT"] = str(doc_root)


def _clean_model_answer(raw: str) -> str:
    text = str(raw or "").strip()
    if not text:
        return ""

    text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    # Try to strip common wrappers.
    for prefix in ("答案：", "答案:", "answer:", "Answer:", "final answer:", "Final answer:"):
        if text.startswith(prefix):
            text = text[len(prefix) :].strip()
            break

    # For evaluation, keep a compact answer span.
    if "\n" in text:
        first = text.splitlines()[0].strip()
        if first:
            text = first
    if len(text) > 300:
        text = text[:300].strip()
    return text


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


def _append_jsonl(path: Path, item: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    defaults = _default_paths()
    parser = argparse.ArgumentParser(description="Run DocVQA evaluation with current GraphRAG model.")
    parser.add_argument("--annotations", type=Path, default=defaults["annotations"])
    parser.add_argument("--images-root", type=Path, default=defaults["images_root"])
    parser.add_argument("--kb-root", type=Path, default=defaults["kb_root"])
    parser.add_argument("--output-root", type=Path, default=defaults["output_root"])
    parser.add_argument("--predictions-file", type=Path, default=defaults["predictions"])
    parser.add_argument("--metrics-file", type=Path, default=defaults["metrics"])
    parser.add_argument("--errors-file", type=Path, default=defaults["errors"])
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all samples")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--max-nodes", type=int, default=24)
    parser.add_argument("--anls-threshold", type=float, default=0.5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--strict-images", action="store_true")
    parser.add_argument("--log-every", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    samples, warnings = load_docvqa_samples(
        annotations_path=args.annotations,
        images_root=args.images_root,
        require_image_exists=args.strict_images,
        keep_unanswered=False,
    )
    if args.max_samples and args.max_samples > 0:
        samples = samples[: args.max_samples]

    _configure_environment(kb_root=args.kb_root, doc_root=args.images_root)

    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from src.config import get_settings
    from src.rag.multimodal_graph_rag_chain import MultiModalGraphRAG

    settings = get_settings()
    chain = MultiModalGraphRAG(
        settings=settings,
        faiss_dir=settings.faiss_dir,
        graph_path=settings.graph_dir / "graph.pkl",
        pages_dir=settings.pages_dir,
    )

    existing = _load_existing_predictions(args.predictions_file) if args.resume else {}
    done = 0
    failed = 0

    print("============================================================")
    print("DocVQA Evaluation")
    print(f"annotations: {args.annotations}")
    print(f"images_root: {args.images_root}")
    print(f"kb_root: {args.kb_root}")
    print(f"predictions_file: {args.predictions_file}")
    print(f"samples: {len(samples)}")
    if warnings:
        print(f"warnings: {len(warnings)} (first 10)")
        for item in warnings[:10]:
            print(f"  - {item}")
    print("============================================================")

    for idx, sample in enumerate(samples, start=1):
        if args.resume and sample.question_id in existing:
            done += 1
            continue

        try:
            result = chain.ask_eval(
                question=sample.question,
                source_hint=sample.source_hint,
                k=args.k,
                max_nodes=args.max_nodes,
            )
            prediction = _clean_model_answer(result.answer)
            item = {
                "question_id": sample.question_id,
                "question": sample.question,
                "answers": sample.answers,
                "prediction": prediction,
                "raw_answer": result.answer,
                "image": sample.image,
                "source_hint": sample.source_hint,
                "pages": result.pages,
                "node_ids": result.node_ids,
                "relations": result.relations,
                "image_paths": result.image_paths,
                "error": "",
                "created_at": datetime.now().isoformat(),
            }
            _append_jsonl(args.predictions_file, item)
            existing[sample.question_id] = item
            done += 1
        except Exception as exc:
            failed += 1
            item = {
                "question_id": sample.question_id,
                "question": sample.question,
                "answers": sample.answers,
                "prediction": "",
                "raw_answer": "",
                "image": sample.image,
                "source_hint": sample.source_hint,
                "pages": [],
                "node_ids": [],
                "relations": [],
                "image_paths": [],
                "error": f"{exc.__class__.__name__}: {exc}",
                "created_at": datetime.now().isoformat(),
            }
            _append_jsonl(args.predictions_file, item)
            existing[sample.question_id] = item

        if args.log_every > 0 and idx % args.log_every == 0:
            print(f"progress: {idx}/{len(samples)}, done={done}, failed={failed}")

    # Keep order consistent with annotation order.
    ordered_records: list[dict[str, Any]] = []
    for sample in samples:
        rec = existing.get(sample.question_id)
        if rec:
            ordered_records.append(rec)

    scored_records: list[dict[str, Any]] = []
    for rec in ordered_records:
        answers = rec.get("answers", [])
        if not isinstance(answers, list):
            answers = []
        pred = str(rec.get("prediction", ""))
        rec["anls"] = anls_score(pred, answers, threshold=args.anls_threshold)
        rec["em"] = exact_match(pred, answers)
        scored_records.append(rec)

    metric_core = summarize_metrics(scored_records, threshold=args.anls_threshold)
    summary = {
        "updated_at": datetime.now().isoformat(),
        "annotations": str(args.annotations),
        "images_root": str(args.images_root),
        "kb_root": str(args.kb_root),
        "predictions_file": str(args.predictions_file),
        "sample_count": len(samples),
        "scored_count": metric_core["count"],
        "done_count": done,
        "failed_count": failed,
        "anls_threshold": args.anls_threshold,
        "anls": metric_core["anls"],
        "em": metric_core["em"],
        "k": args.k,
        "max_nodes": args.max_nodes,
    }
    args.metrics_file.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    hard_cases = sorted(scored_records, key=lambda x: (float(x.get("anls", 0.0)), float(x.get("em", 0.0))))
    with args.errors_file.open("w", encoding="utf-8") as f:
        for item in hard_cases[:200]:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("============================================================")
    print("Evaluation done.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"errors_file: {args.errors_file}")
    print("============================================================")


if __name__ == "__main__":
    main()
