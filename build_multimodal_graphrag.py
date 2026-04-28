from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

from langchain.schema import Document

from src.config import get_settings
from src.graph.builder import build_document_graph, save_graph
from src.indexing.faiss_store import build_faiss_index
from src.parsing.pipeline import (
    parse_images_to_documents,
    prepare_input_as_images,
    save_parsing_outputs,
)
from src.vl_client import DashScopeVLClient


def _resolve_input_path(input_file: str | Path, settings) -> Path:
    raw = Path(input_file)
    candidates: list[Path] = []

    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.append(raw)
        candidates.append(settings.doc_dir / raw)

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()

    attempts = "\n".join(f"- {c}" for c in candidates)
    raise FileNotFoundError(
        f"Input file not found: {input_file}\nTried:\n{attempts}\n"
        f"Tip: put documents under {settings.doc_dir} or pass an absolute path."
    )


def _compute_file_signature(input_file: str | Path) -> dict[str, str | float | int]:
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    sha256 = hashlib.sha256()
    with input_path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            sha256.update(chunk)

    stat = input_path.stat()
    return {
        "input_file": str(input_path.resolve()),
        "source_name": input_path.name,
        "sha256": sha256.hexdigest(),
        "size": int(stat.st_size),
        "mtime": float(stat.st_mtime),
    }


def _artifact_paths(settings) -> dict[str, Path]:
    return {
        "meta": settings.output_root / "build_meta.json",
        "docs": settings.parsed_dir / "documents.json",
        "graph_data": settings.parsed_dir / "graph_data.json",
        "faiss_faiss": settings.faiss_dir / "index.faiss",
        "faiss_pkl": settings.faiss_dir / "index.pkl",
        "graph_pkl": settings.graph_dir / "graph.pkl",
    }


def _load_meta(meta_path: Path) -> dict:
    if not meta_path.exists():
        return {}
    with meta_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_graph_stats(graph_pkl_path: Path) -> dict[str, float]:
    from src.graph.builder import load_graph

    graph = load_graph(graph_pkl_path)
    node_count = float(graph.number_of_nodes())
    edge_count = float(graph.number_of_edges())
    avg_degree = (sum(dict(graph.degree()).values()) / node_count) if node_count else 0.0
    return {
        "nodes": node_count,
        "edges": edge_count,
        "avg_degree": avg_degree,
    }


def _bootstrap_meta_file(meta_path: Path, signature: dict[str, str | float | int]) -> None:
    meta_data = {
        **signature,
        "build_time": datetime.now().isoformat(),
        "meta_bootstrapped": True,
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta_data, f, ensure_ascii=False, indent=2)


def _legacy_artifact_match(
    paths: dict[str, Path], signature: dict[str, str | float | int]
) -> bool:
    """Fallback for builds created before build_meta.json existed."""
    required_paths = [
        paths["docs"],
        paths["graph_data"],
        paths["faiss_faiss"],
        paths["faiss_pkl"],
        paths["graph_pkl"],
    ]
    if not all(p.exists() for p in required_paths):
        return False

    try:
        graph_data = load_graph_data_from_json(paths["graph_data"])
    except Exception:
        return False

    source_candidates: list[str] = []
    if isinstance(graph_data, dict):
        source_candidates.append(str(graph_data.get("source", "")).strip())
        for item in graph_data.get("documents", []):
            if isinstance(item, dict):
                source_candidates.append(str(item.get("source", "")).strip())

    source_candidates = [s for s in source_candidates if s]
    if str(signature["source_name"]) not in source_candidates:
        return False

    # If input file is newer than existing artifacts, do not reuse legacy outputs.
    artifact_mtime = min(p.stat().st_mtime for p in required_paths)
    input_mtime = float(signature["mtime"])
    return input_mtime <= artifact_mtime


def check_knowledge_base_status(input_file: str | Path) -> dict[str, str | bool | float]:
    """Check whether input document is already indexed in the global KB."""
    settings = get_settings()
    paths = _artifact_paths(settings)
    signature = _compute_file_signature(input_file)

    registry = _load_registry(paths["meta"])
    docs_registry = registry.get("documents", []) if isinstance(registry, dict) else []
    document_known = any(
        isinstance(item, dict)
        and item.get("sha256") == signature["sha256"]
        and int(item.get("size", -1)) == int(signature["size"])
        for item in docs_registry
    )

    artifacts_ready = all(
        p.exists()
        for p in [
            paths["docs"],
            paths["graph_data"],
            paths["faiss_faiss"],
            paths["faiss_pkl"],
            paths["graph_pkl"],
        ]
    )
    can_reuse = document_known and artifacts_ready
    match_mode = "registry"
    bootstrap_meta = False

    if not can_reuse and not docs_registry:
        legacy_match = _legacy_artifact_match(paths, signature)
        if legacy_match:
            can_reuse = True
            match_mode = "legacy"
            bootstrap_meta = True
            document_known = True

    stats = {"nodes": 0.0, "edges": 0.0, "avg_degree": 0.0}
    if can_reuse:
        try:
            stats = _load_graph_stats(paths["graph_pkl"])
        except Exception:
            can_reuse = False

    if can_reuse:
        reason = "matched"
    elif not artifacts_ready:
        reason = "artifacts_missing"
    elif not docs_registry:
        reason = "meta_missing"
    elif not document_known:
        reason = "document_not_indexed"
    else:
        reason = "unknown"

    return {
        "can_reuse": can_reuse,
        "document_known": document_known,
        "kb_ready": artifacts_ready,
        "reason": reason,
        "match_mode": match_mode,
        "bootstrap_meta": bootstrap_meta,
        "document_count": float(len(docs_registry)),
        "docs_json": str(paths["docs"]),
        "graph_data_json": str(paths["graph_data"]),
        "faiss_index": str(settings.faiss_dir),
        "graph_pkl": str(paths["graph_pkl"]),
        "nodes": stats["nodes"],
        "edges": stats["edges"],
        "avg_degree": stats["avg_degree"],
    }


def load_documents_from_json(docs_path: Path) -> list[Document]:
    """从JSON文件加载文档列表"""
    if not docs_path.exists():
        return []

    with open(docs_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    documents = []
    for item in data:
        doc = Document(
            page_content=item["page_content"],
            metadata=item["metadata"]
        )
        documents.append(doc)

    return documents


def load_graph_data_from_json(graph_data_path: Path) -> dict:
    """从JSON文件加载图数据"""
    if not graph_data_path.exists():
        return {}

    with open(graph_data_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _load_registry(meta_path: Path) -> dict:
    if not meta_path.exists():
        return {"documents": []}

    with meta_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, dict) and isinstance(raw.get("documents"), list):
        return raw

    # Backward compatibility: old single-doc meta format.
    if isinstance(raw, dict) and raw.get("sha256"):
        return {
            "documents": [
                {
                    "input_file": raw.get("input_file", ""),
                    "source_name": raw.get("source_name", ""),
                    "sha256": raw.get("sha256", ""),
                    "size": int(raw.get("size", 0)),
                    "mtime": float(raw.get("mtime", 0.0)),
                    "added_at": raw.get("build_time", datetime.now().isoformat()),
                }
            ]
        }

    return {"documents": []}


def _save_registry(meta_path: Path, registry: dict) -> None:
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(registry, f, ensure_ascii=False, indent=2)


def _normalize_graph_data(raw: dict) -> dict:
    if not isinstance(raw, dict):
        return {"documents": []}

    if isinstance(raw.get("documents"), list):
        normalized_docs = []
        for item in raw["documents"]:
            if not isinstance(item, dict):
                continue
            source = str(item.get("source", "")).strip()
            pages = item.get("pages", []) if isinstance(item.get("pages", []), list) else []
            for page in pages:
                if isinstance(page, dict) and not page.get("source"):
                    page["source"] = source
            normalized_docs.append({**item, "pages": pages})
        return {"documents": normalized_docs}

    pages = raw.get("pages", []) if isinstance(raw.get("pages", []), list) else []
    source = str(raw.get("source", "unknown_source")).strip() or "unknown_source"
    for page in pages:
        if isinstance(page, dict) and not page.get("source"):
            page["source"] = source
    return {
        "documents": [
            {
                "source": source,
                "pages": pages,
            }
        ]
    }


def _flatten_pages(graph_container: dict) -> list[dict]:
    pages: list[dict] = []
    for item in graph_container.get("documents", []):
        if not isinstance(item, dict):
            continue
        source = str(item.get("source", ""))
        for page in item.get("pages", []):
            if isinstance(page, dict):
                if not page.get("source"):
                    page["source"] = source
                pages.append(page)
    return pages


def _flatten_cross_page_links(graph_container: dict) -> list[dict]:
    links: list[dict] = []
    for item in graph_container.get("documents", []):
        if not isinstance(item, dict):
            continue
        source = str(item.get("source", ""))
        cross_links = item.get("cross_page_links", [])
        if not isinstance(cross_links, list):
            continue
        for link in cross_links:
            if not isinstance(link, dict):
                continue
            normalized = dict(link)
            if not normalized.get("source"):
                normalized["source"] = source
            links.append(normalized)
    return links


def _flatten_document_keywords(graph_container: dict) -> list[dict]:
    keywords: list[dict] = []
    for item in graph_container.get("documents", []):
        if not isinstance(item, dict):
            continue
        source = str(item.get("source", ""))
        keyword_items = item.get("document_keywords", [])
        if not isinstance(keyword_items, list):
            continue
        for kw in keyword_items:
            if not isinstance(kw, dict):
                continue
            normalized = dict(kw)
            if not normalized.get("source"):
                normalized["source"] = source
            keywords.append(normalized)
    return keywords


def _save_merged_outputs(
    documents: list[Document], graph_container: dict, parsed_dir: Path
) -> tuple[Path, Path]:
    parsed_dir.mkdir(parents=True, exist_ok=True)
    docs_path = parsed_dir / "documents.json"
    graph_data_path = parsed_dir / "graph_data.json"

    docs_payload = [
        {
            "page_content": doc.page_content,
            "metadata": doc.metadata,
        }
        for doc in documents
    ]

    docs_path.write_text(json.dumps(docs_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    graph_data_path.write_text(
        json.dumps(graph_container, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return docs_path, graph_data_path


def check_existing_knowledge_base() -> dict[str, str | bool | float]:
    settings = get_settings()
    paths = _artifact_paths(settings)
    registry = _load_registry(paths["meta"])
    registry_docs = registry.get("documents", []) if isinstance(registry.get("documents"), list) else []
    artifacts_ready = all(
        p.exists()
        for p in [
            paths["docs"],
            paths["graph_data"],
            paths["faiss_faiss"],
            paths["faiss_pkl"],
            paths["graph_pkl"],
        ]
    )

    doc_count = float(len(registry_docs))
    if doc_count == 0 and paths["graph_data"].exists():
        graph_container = _normalize_graph_data(load_graph_data_from_json(paths["graph_data"]))
        doc_count = float(len(graph_container.get("documents", [])))

    stats = {"nodes": 0.0, "edges": 0.0, "avg_degree": 0.0}
    if artifacts_ready:
        try:
            stats = _load_graph_stats(paths["graph_pkl"])
        except Exception:
            artifacts_ready = False

    return {
        "kb_ready": artifacts_ready,
        "document_count": doc_count,
        "docs_json": str(paths["docs"]),
        "graph_data_json": str(paths["graph_data"]),
        "faiss_index": str(settings.faiss_dir),
        "graph_pkl": str(paths["graph_pkl"]),
        "nodes": stats["nodes"],
        "edges": stats["edges"],
        "avg_degree": stats["avg_degree"],
    }


def build_knowledge_base(input_file: str, force_rebuild: bool = False) -> dict[str, str | float]:
    settings = get_settings()
    resolved_input = _resolve_input_path(input_file, settings)
    source_name = resolved_input.name
    paths = _artifact_paths(settings)
    signature = _compute_file_signature(resolved_input)

    print(f"[1/7] 开始处理文档入库，输入文件: {source_name}")
    print(f"  解析后的输入路径: {resolved_input}")
    print(f"  输出目录: {settings.output_root}")
    print(f"  强制重建索引: {'是' if force_rebuild else '否'}")

    registry = _load_registry(paths["meta"])
    documents = load_documents_from_json(paths["docs"])
    graph_container = _normalize_graph_data(load_graph_data_from_json(paths["graph_data"]))

    existing = check_knowledge_base_status(resolved_input)
    if not force_rebuild and bool(existing.get("can_reuse", False)) and bool(existing.get("kb_ready", False)):
        print("  文档已在知识库中，直接复用现有索引与图谱")
        if bool(existing.get("bootstrap_meta", False)):
            docs = registry.get("documents", [])
            docs.append(
                {
                    "input_file": signature["input_file"],
                    "source_name": signature["source_name"],
                    "sha256": signature["sha256"],
                    "size": signature["size"],
                    "mtime": signature["mtime"],
                    "added_at": datetime.now().isoformat(),
                }
            )
            registry["documents"] = docs
            _save_registry(paths["meta"], registry)
            print(f"  检测到旧版产物，已迁移元数据: {paths['meta']}")
        reused_result = {
            "docs_json": str(paths["docs"]),
            "graph_data_json": str(paths["graph_data"]),
            "faiss_index": str(settings.faiss_dir),
            "graph_pkl": str(paths["graph_pkl"]),
            "nodes": float(existing["nodes"]),
            "edges": float(existing["edges"]),
            "avg_degree": float(existing["avg_degree"]),
            "reused": True,
        }
        return reused_result

    document_known = bool(existing.get("document_known", False))
    newly_added = False

    if force_rebuild:
        print("  启用强制重建：将重新解析该文档并重建索引/图谱")
        documents = [doc for doc in documents if str(doc.metadata.get("source", "")) != source_name]
        graph_docs = graph_container.get("documents", [])
        if isinstance(graph_docs, list):
            graph_container["documents"] = [
                item
                for item in graph_docs
                if not (isinstance(item, dict) and str(item.get("source", "")) == source_name)
            ]
        registry_docs = registry.get("documents", [])
        if isinstance(registry_docs, list):
            registry["documents"] = [
                item
                for item in registry_docs
                if not (
                    isinstance(item, dict)
                    and (
                        str(item.get("source_name", "")) == source_name
                        or str(item.get("sha256", "")) == str(signature["sha256"])
                    )
                )
            ]
        document_known = False

    if not document_known:
        print("  该文档未入库，开始增量解析并追加")
        doc_page_dir = settings.pages_dir / str(signature["sha256"])[:12]

        print(f"[2/7] 准备输入图片...")
        image_paths = prepare_input_as_images(resolved_input, doc_page_dir)
        print(f"  已生成 {len(image_paths)} 张图片")

        print(f"[3/7] 初始化视觉语言客户端，使用模型: {settings.vl_model}")
        vl_client = DashScopeVLClient(
            api_key=settings.api_key,
            model=settings.vl_model,
        )

        print(f"[4/7] 解析图片内容...")
        new_documents, new_graph_data = parse_images_to_documents(
            image_paths=image_paths,
            source_name=source_name,
            vl_client=vl_client,
        )
        print(f"  解析完成，共新增 {len(new_documents)} 个文档片段")

        documents.extend(new_documents)
        graph_container.setdefault("documents", []).append(
            {
                "source": source_name,
                "input_file": signature["input_file"],
                "sha256": signature["sha256"],
                "size": signature["size"],
                "mtime": signature["mtime"],
                "page_dir": str(doc_page_dir),
                "added_at": datetime.now().isoformat(),
                "pages": new_graph_data.get("pages", []),
                "cross_page_links": new_graph_data.get("cross_page_links", []),
                "document_keywords": new_graph_data.get("document_keywords", []),
            }
        )

        docs_list = registry.get("documents", []) if isinstance(registry.get("documents"), list) else []
        docs_list.append(
            {
                "input_file": signature["input_file"],
                "source_name": signature["source_name"],
                "sha256": signature["sha256"],
                "size": signature["size"],
                "mtime": signature["mtime"],
                "added_at": datetime.now().isoformat(),
            }
        )
        registry["documents"] = docs_list
        newly_added = True
    else:
        print("  该文档已入库，本次仅加载并确保索引图谱可用")

    if not documents:
        raise RuntimeError("No documents available to build knowledge base.")

    print(f"[5/7] 保存解析结果...")
    docs_path, graph_data_path = _save_merged_outputs(
        documents=documents,
        graph_container=graph_container,
        parsed_dir=settings.parsed_dir,
    )
    print(f"  文档保存至: {docs_path}")
    print(f"  图数据保存至: {graph_data_path}")
    _save_registry(paths["meta"], registry)
    print(f"  元数据已保存至: {paths['meta']}")

    print(f"[6/7] 构建FAISS向量索引...")
    faiss_path = build_faiss_index(
        documents=documents,
        api_key=settings.api_key,
        embedding_model=settings.embedding_model,
        output_dir=settings.faiss_dir,
    )
    print(f"  FAISS索引保存至: {faiss_path}")

    print(f"[7/7] 构建文档知识图谱...")
    graph = build_document_graph(
        documents=documents,
        graph_data={
            "pages": _flatten_pages(graph_container),
            "cross_page_links": _flatten_cross_page_links(graph_container),
            "document_keywords": _flatten_document_keywords(graph_container),
        },
    )
    graph_path, stats = save_graph(graph, settings.graph_dir)
    print(f"  图谱保存至: {graph_path}")
    print(f"  图谱统计: {stats['nodes']} 个节点, {stats['edges']} 条边, 平均度数: {stats['avg_degree']:.2f}")

    return {
        "docs_json": str(docs_path),
        "graph_data_json": str(graph_data_path),
        "faiss_index": str(faiss_path),
        "graph_pkl": str(graph_path),
        "nodes": stats["nodes"],
        "edges": stats["edges"],
        "avg_degree": stats["avg_degree"],
        "reused": not newly_added,
        "added": newly_added,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build multimodal GraphRAG artifacts")
    parser.add_argument(
        "--input",
        required=True,
        help="Path or filename of input PDF/image. Relative names are searched under doc/ by default.",
    )
    parser.add_argument("--force-rebuild", action="store_true",
                       help="Force rebuild all artifacts even if they exist")
    args = parser.parse_args()

    start_time = datetime.now()
    print(f"\n{'='*60}")
    print(f"多模态GraphRAG构建工具")
    print(f"输入文件: {args.input}")
    print(f"开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    result = build_knowledge_base(args.input, force_rebuild=args.force_rebuild)

    end_time = datetime.now()
    duration = end_time - start_time
    minutes, seconds = divmod(duration.total_seconds(), 60)

    print(f"\n{'='*60}")
    print(f"构建完成!")
    print(f"开始时间: {start_time.strftime('%H:%M:%S')}")
    print(f"结束时间: {end_time.strftime('%H:%M:%S')}")
    print(f"总耗时: {int(minutes)}分{seconds:.1f}秒")
    print(f"输出文件:")
    print(f"  - 文档JSON: {result['docs_json']}")
    print(f"  - 图数据JSON: {result['graph_data_json']}")
    print(f"  - FAISS索引: {result['faiss_index']}")
    print(f"  - 知识图谱: {result['graph_pkl']}")
    print(f"图谱统计:")
    print(f"  - 节点数: {result['nodes']}")
    print(f"  - 边数: {result['edges']}")
    print(f"  - 平均度数: {result['avg_degree']:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
