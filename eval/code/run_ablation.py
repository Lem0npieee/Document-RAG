from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


GROUPS = [
    {
        "id": "none",
        "name": "None组",
        "desc": "不使用 FAISS，不使用知识图谱，不上传页面总览图。",
    },
    {
        "id": "vector_only",
        "name": "vector_only组",
        "desc": "只使用 FAISS 向量库，不使用知识图谱。",
    },
    {
        "id": "graph_only",
        "name": "graph_only组",
        "desc": "只使用知识图谱，不使用 FAISS 向量库。",
    },
    {
        "id": "no_image",
        "name": "no_image组",
        "desc": "使用 FAISS 和知识图谱，但不上传 PDF 每页总览图。",
    },
    {
        "id": "full",
        "name": "full组",
        "desc": "完整流程：FAISS + 知识图谱 + 页面总览图。",
    },
]


def _default_paths() -> dict[str, Path]:
    eval_root = Path(__file__).resolve().parents[1]
    output_root = eval_root / "output" / "pdfqa" / "ablation"
    return {
        "eval_root": eval_root,
        "annotations_root": eval_root / "input" / "pdfqa" / "annotations",
        "pdfs_root": eval_root / "input" / "pdfqa" / "pdfs",
        "kb_root": eval_root / "output" / "kb",
        "output_root": output_root,
    }


def _choose_group() -> str:
    print("请选择要运行的消融实验组别：")
    for idx, group in enumerate(GROUPS, start=1):
        print(f"  {idx}. {group['name']}：{group['desc']}")

    while True:
        raw = input("输入数字 1-5：").strip()
        if raw.isdigit():
            index = int(raw)
            if 1 <= index <= len(GROUPS):
                return GROUPS[index - 1]["id"]
        print("输入无效，请重新输入 1-5。")


def parse_args() -> argparse.Namespace:
    defaults = _default_paths()
    group_ids = [group["id"] for group in GROUPS]
    parser = argparse.ArgumentParser(description="Interactive entry for pdfQA ablation evaluation.")
    parser.add_argument("--group", choices=group_ids, default="", help="ablation group; omit to choose interactively")
    parser.add_argument("--all", action="store_true", help="run all ablation groups sequentially")
    parser.add_argument("--annotations-root", type=Path, default=defaults["annotations_root"])
    parser.add_argument("--pdfs-root", type=Path, default=defaults["pdfs_root"])
    parser.add_argument("--kb-root", type=Path, default=defaults["kb_root"])
    parser.add_argument("--output-root", type=Path, default=defaults["output_root"])
    parser.add_argument("--category", type=str, default="real", choices=["all", "real", "syn"])
    parser.add_argument("--qa-split", type=str, default="all", choices=["all", "raw", "vf", "cf"])
    parser.add_argument("--answer-profile", type=str, default="very_short", choices=["all", "binary", "very_short", "short"])
    parser.add_argument("--max-samples", type=int, default=20, help="0 means all samples")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--max-nodes", type=int, default=24)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-strict-docs", action="store_true", help="do not require matching PDFs")
    parser.add_argument("--no-kb-docs-only", action="store_true", help="do not filter samples to docs present in KB")
    return parser.parse_args()


def _build_command(args: argparse.Namespace, group: str) -> list[str]:
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    predictions_file = output_root / f"{group}_predictions.jsonl"
    metrics_file = output_root / f"{group}_metrics.json"
    errors_file = output_root / f"{group}_errors_topk.jsonl"
    if not args.resume:
        for path in (predictions_file, metrics_file, errors_file):
            if path.exists():
                path.unlink()

    script = Path(__file__).resolve().with_name("run_pdfqa_eval.py")
    command = [
        sys.executable,
        "-B",
        str(script),
        "--annotations-root",
        str(args.annotations_root),
        "--pdfs-root",
        str(args.pdfs_root),
        "--kb-root",
        str(args.kb_root),
        "--output-root",
        str(output_root),
        "--predictions-file",
        str(predictions_file),
        "--metrics-file",
        str(metrics_file),
        "--errors-file",
        str(errors_file),
        "--category",
        args.category,
        "--qa-split",
        args.qa_split,
        "--answer-profile",
        args.answer_profile,
        "--max-samples",
        str(args.max_samples),
        "--k",
        str(args.k),
        "--max-nodes",
        str(args.max_nodes),
        "--ablation",
        group,
    ]

    if args.resume:
        command.append("--resume")
    if not args.no_strict_docs:
        command.append("--strict-docs")
    if not args.no_kb_docs_only:
        command.append("--kb-docs-only")

    return command


def main() -> None:
    args = parse_args()
    selected_groups = [group["id"] for group in GROUPS] if args.all else [args.group or _choose_group()]

    for group in selected_groups:
        print("=" * 60)
        print(f"运行消融实验组：{group}")
        print("=" * 60)
        command = _build_command(args, group)
        print("命令：")
        print(" ".join(f'"{part}"' if " " in part else part for part in command))
        result = subprocess.run(command, cwd=Path(__file__).resolve().parents[2])
        if result.returncode != 0:
            raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
