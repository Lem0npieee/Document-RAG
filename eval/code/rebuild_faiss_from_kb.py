from __future__ import annotations

import os
import sys
from pathlib import Path


def parse_args():
    import argparse

    eval_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Rebuild only FAISS index from an existing parsed KB.")
    parser.add_argument("--kb-root", type=Path, default=eval_root / "output" / "kb")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    os.environ["OUTPUT_ROOT"] = str(args.kb_root)

    from build_multimodal_graphrag import load_documents_from_json
    from src.config import get_settings
    from src.indexing.faiss_store import build_faiss_index

    settings = get_settings()
    settings.output_root = args.kb_root
    settings.embedding_provider = "dashscope"
    settings.embedding_model = "text-embedding-v3"

    docs_path = args.kb_root / "parsed" / "documents.json"
    if not docs_path.exists():
        raise FileNotFoundError(f"documents.json not found: {docs_path}")

    documents = load_documents_from_json(docs_path)
    if not documents:
        raise RuntimeError(f"No documents found in {docs_path}")

    print("============================================================")
    print("Rebuild FAISS Only")
    print(f"kb_root: {args.kb_root}")
    print(f"documents: {len(documents)}")
    print(f"embedding_provider: {settings.embedding_provider}")
    print(f"embedding_model: {settings.embedding_model}")
    print("============================================================")

    build_faiss_index(
        documents=documents,
        api_key=settings.dashscope_api_key,
        embedding_model=settings.embedding_model,
        output_dir=args.kb_root / "faiss_index",
        embedding_provider=settings.embedding_provider,
    )

    print("============================================================")
    print("FAISS rebuild done.")
    print("============================================================")


if __name__ == "__main__":
    main()
