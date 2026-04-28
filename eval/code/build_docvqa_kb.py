from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from loader import load_docvqa_samples


def _default_paths() -> dict[str, Path]:
    eval_root = Path(__file__).resolve().parents[1]
    return {
        "eval_root": eval_root,
        "annotations": eval_root / "input" / "docvqa" / "val.json",
        "images_root": eval_root / "input" / "docvqa" / "images",
        "kb_root": eval_root / "output" / "kb",
        "report_root": eval_root / "output" / "docvqa",
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


def _unique_images(samples: list[Any]) -> list[Path]:
    seen: set[str] = set()
    images: list[Path] = []
    for sample in samples:
        path = str(sample.image_path.resolve())
        if path in seen:
            continue
        seen.add(path)
        images.append(Path(path))
    return images


def _ensure_unique_source_names(images: list[Path]) -> None:
    by_name: dict[str, Path] = {}
    collisions: list[str] = []
    for p in images:
        key = p.name.lower()
        if key in by_name and by_name[key] != p:
            collisions.append(f"{by_name[key]} <-> {p}")
        else:
            by_name[key] = p
    if collisions:
        detail = "\n".join(collisions[:20])
        raise RuntimeError(
            "Duplicate image basenames detected. Current KB uses basename as source key.\n"
            f"Conflicts:\n{detail}"
        )


def parse_args() -> argparse.Namespace:
    defaults = _default_paths()
    parser = argparse.ArgumentParser(description="Build GraphRAG KB for DocVQA images.")
    parser.add_argument("--annotations", type=Path, default=defaults["annotations"])
    parser.add_argument("--images-root", type=Path, default=defaults["images_root"])
    parser.add_argument("--kb-root", type=Path, default=defaults["kb_root"])
    parser.add_argument("--report-root", type=Path, default=defaults["report_root"])
    parser.add_argument("--progress-file", type=Path, default=None)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all samples")
    parser.add_argument("--max-images", type=int, default=0, help="0 means all images")
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--strict-images", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.report_root.mkdir(parents=True, exist_ok=True)
    progress_file = args.progress_file or (args.report_root / "build_progress.json")

    samples, warnings = load_docvqa_samples(
        annotations_path=args.annotations,
        images_root=args.images_root,
        require_image_exists=args.strict_images,
        keep_unanswered=False,
    )
    if args.max_samples and args.max_samples > 0:
        samples = samples[: args.max_samples]

    images = _unique_images(samples)
    if args.max_images and args.max_images > 0:
        images = images[: args.max_images]
    _ensure_unique_source_names(images)

    _configure_environment(kb_root=args.kb_root, doc_root=args.images_root)

    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from build_multimodal_graphrag import build_knowledge_base

    progress = _load_progress(progress_file) if args.resume else {"done_sources": [], "failures": {}}
    done_sources = set(str(x) for x in progress.get("done_sources", []))
    failures = dict(progress.get("failures", {}))

    print("============================================================")
    print("DocVQA KB Build")
    print(f"annotations: {args.annotations}")
    print(f"images_root: {args.images_root}")
    print(f"kb_root: {args.kb_root}")
    print(f"samples: {len(samples)}, unique images: {len(images)}")
    if warnings:
        print(f"warnings: {len(warnings)} (first 10)")
        for item in warnings[:10]:
            print(f"  - {item}")
    print("============================================================")

    built = 0
    skipped = 0
    failed = 0

    for idx, image_path in enumerate(images, start=1):
        source_name = image_path.name
        if args.resume and (source_name in done_sources) and (not args.force_rebuild):
            skipped += 1
            continue

        print(f"[{idx}/{len(images)}] ingest {source_name}")
        try:
            build_knowledge_base(str(image_path), force_rebuild=args.force_rebuild)
            done_sources.add(source_name)
            if source_name in failures:
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
        "annotations": str(args.annotations),
        "images_root": str(args.images_root),
        "kb_root": str(args.kb_root),
        "total_samples": len(samples),
        "total_unique_images": len(images),
        "built_images": built,
        "skipped_images": skipped,
        "failed_images": failed,
        "failed_sources": failures,
        "progress_file": str(progress_file),
    }
    report_file = args.report_root / "build_report.json"
    report_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("============================================================")
    print("Build done.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("============================================================")


if __name__ == "__main__":
    main()
