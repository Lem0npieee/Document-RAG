from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from loader import load_pdfqa_samples


def _default_paths() -> dict[str, Path]:
    eval_root = Path(__file__).resolve().parents[1]
    return {
        "eval_root": eval_root,
        "annotations_root": eval_root / "input" / "pdfqa" / "annotations",
        "pdfs_root": eval_root / "input" / "pdfqa" / "pdfs",
        "kb_root": eval_root / "output" / "kb",
        "report_root": eval_root / "output" / "pdfqa",
    }


def _configure_environment(kb_root: Path, doc_root: Path) -> None:
    os.environ["OUTPUT_ROOT"] = str(kb_root)
    os.environ["DOC_ROOT"] = str(doc_root)


def _load_progress(progress_file: Path) -> dict[str, Any]:
    if not progress_file.exists():
        return {"done_sources": [], "failures": {}}
    try:
        return json.loads(progress_file.read_text(encoding="utf-8"))
    except Exception:
        return {"done_sources": [], "failures": {}}


def _save_progress(progress_file: Path, payload: dict[str, Any]) -> None:
    progress_file.parent.mkdir(parents=True, exist_ok=True)
    progress_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    defaults = _default_paths()
    parser = argparse.ArgumentParser(description="Build GraphRAG KB for pdfQA documents.")
    parser.add_argument("--annotations-root", type=Path, default=defaults["annotations_root"])
    parser.add_argument("--pdfs-root", type=Path, default=defaults["pdfs_root"])
    parser.add_argument("--kb-root", type=Path, default=defaults["kb_root"])
    parser.add_argument("--report-root", type=Path, default=defaults["report_root"])
    parser.add_argument("--progress-file", type=Path, default=None)
    parser.add_argument("--category", type=str, default="real", choices=["all", "real", "syn", "custom"])
    parser.add_argument("--qa-split", type=str, default="all", choices=["all", "raw", "vf", "cf"])
    parser.add_argument("--doc-name", type=str, default="", help="Only build PDFs with this file name or stem")
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all samples")
    parser.add_argument("--max-docs", type=int, default=0, help="0 means all docs")
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--strict-docs", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.report_root.mkdir(parents=True, exist_ok=True)
    progress_file = args.progress_file or (args.report_root / "pdfqa_build_progress.json")
    os.environ["MODEL_PROVIDER"] = "dashscope"
    os.environ["EMBEDDING_PROVIDER"] = "dashscope"
    os.environ["EMBEDDING_MODEL"] = "text-embedding-v3"
    os.environ["DOCRAG_FORCE_DASHSCOPE_API"] = "1"

    samples, warnings = load_pdfqa_samples(
        annotations_root=args.annotations_root,
        pdfs_root=args.pdfs_root,
        category=args.category,
        qa_split=args.qa_split,
        require_doc_exists=args.strict_docs,
        keep_unanswered=False,
        max_samples=args.max_samples,
    )

    unique_docs: dict[str, Path] = {}
    for sample in samples:
        unique_docs[str(sample.doc_path.resolve())] = sample.doc_path.resolve()
    doc_paths = sorted(unique_docs.values())
    if args.doc_name:
        requested = Path(args.doc_name).name.lower()
        requested_stem = Path(requested).stem.lower()
        doc_paths = [
            p
            for p in doc_paths
            if p.name.lower() == requested or p.stem.lower() == requested_stem
        ]
        if not doc_paths:
            raise FileNotFoundError(f"No PDF matched --doc-name {args.doc_name!r}")
    if args.max_docs and args.max_docs > 0:
        doc_paths = doc_paths[: args.max_docs]

    _configure_environment(kb_root=args.kb_root, doc_root=args.pdfs_root)

    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from build_multimodal_graphrag import build_knowledge_base

    progress = _load_progress(progress_file) if args.resume else {"done_sources": [], "failures": {}}
    done_sources = set(str(x) for x in progress.get("done_sources", []))
    failures = dict(progress.get("failures", {}))

    print("============================================================")
    print("pdfQA KB Build")
    print(f"annotations_root: {args.annotations_root}")
    print(f"pdfs_root: {args.pdfs_root}")
    print(f"kb_root: {args.kb_root}")
    print(f"category: {args.category}")
    print(f"qa_split: {args.qa_split}")
    if args.doc_name:
        print(f"doc_name: {args.doc_name}")
    print(f"samples: {len(samples)}, unique docs: {len(doc_paths)}")
    if warnings:
        print(f"warnings: {len(warnings)} (first 20)")
        for item in warnings[:20]:
            print(f"  - {item}")
    print("============================================================")

    built = 0
    skipped = 0
    failed = 0

    for idx, doc_path in enumerate(doc_paths, start=1):
        source_name = doc_path.name
        if args.resume and (source_name in done_sources) and (not args.force_rebuild):
            skipped += 1
            continue

        print(f"[{idx}/{len(doc_paths)}] ingest {source_name}")
        try:
            build_knowledge_base(str(doc_path), force_rebuild=args.force_rebuild)
            done_sources.add(source_name)
            failures.pop(source_name, None)
            built += 1
        except Exception as exc:
            failed += 1
            failures[source_name] = f"{exc.__class__.__name__}: {exc}"
            print(f"  FAILED: {source_name} -> {failures[source_name]}")
        finally:
            _save_progress(
                progress_file,
                {
                    "updated_at": datetime.now().isoformat(),
                    "done_sources": sorted(done_sources),
                    "failures": failures,
                },
            )

    summary = {
        "updated_at": datetime.now().isoformat(),
        "annotations_root": str(args.annotations_root),
        "pdfs_root": str(args.pdfs_root),
        "kb_root": str(args.kb_root),
        "category": args.category,
        "qa_split": args.qa_split,
        "doc_name": args.doc_name,
        "total_samples": len(samples),
        "total_unique_docs": len(doc_paths),
        "built_docs": built,
        "skipped_docs": skipped,
        "failed_docs": failed,
        "failed_sources": failures,
        "progress_file": str(progress_file),
        "warnings_count": len(warnings),
    }
    report_file = args.report_root / "pdfqa_build_report.json"
    report_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("============================================================")
    print("Build done.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("============================================================")


if __name__ == "__main__":
    main()
