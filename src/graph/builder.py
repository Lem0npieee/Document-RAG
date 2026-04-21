from __future__ import annotations

import pickle
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

import networkx as nx
from langchain.schema import Document


def _entity_node_id(name: str) -> str:
    normalized = (name or "unknown_entity").strip().replace(" ", "_")
    return f"entity::{normalized}"


def _find_or_create_entity(G: nx.DiGraph, name: str, page: int | None = None) -> str:
    node_id = _entity_node_id(name)
    if not G.has_node(node_id):
        G.add_node(node_id, type="entity", name=name, page=page)
    return node_id


def build_document_graph(documents: list[Document], graph_data: dict[str, Any]) -> nx.DiGraph:
    print(f"    开始构建知识图谱，文档数: {len(documents)}")
    G = nx.DiGraph()

    by_page: dict[tuple[str, int], list[str]] = defaultdict(list)
    figure_by_page: dict[tuple[str, int], dict[str, str]] = defaultdict(dict)
    conclusion_by_page: dict[tuple[str, int], list[tuple[str, str]]] = defaultdict(list)

    print(f"    处理文档节点...")
    for doc in documents:
        metadata = doc.metadata
        node_id = metadata["node_id"]
        page = int(metadata.get("page", 0))
        source = str(metadata.get("source", ""))
        page_key = (source, page)

        node_attrs = {
            "page": page,
            "image_path": metadata.get("image_path", ""),
            "type": metadata.get("type", "text"),
            "source": source,
            "fig_id": metadata.get("fig_id", ""),
            "content": doc.page_content,
        }
        G.add_node(node_id, **node_attrs)
        by_page[page_key].append(node_id)

        if metadata.get("type") == "figure":
            fig_id = metadata.get("fig_id", "")
            if fig_id:
                figure_by_page[page_key][fig_id] = node_id

        if metadata.get("type") == "conclusion":
            conclusion_by_page[page_key].append((node_id, doc.page_content))

    for _, nodes in by_page.items():
        for left, right in combinations(nodes, 2):
            G.add_edge(left, right, relation="同页")
            G.add_edge(right, left, relation="同页")

    for page_key, conclusions in conclusion_by_page.items():
        fig_map = figure_by_page.get(page_key, {})
        for conclusion_node_id, text in conclusions:
            for fig_id, fig_node_id in fig_map.items():
                if fig_id in text:
                    G.add_edge(fig_node_id, conclusion_node_id, relation="支撑结论")

    for page_item in graph_data.get("pages", []):
        page = int(page_item.get("page", 0))
        source = str(page_item.get("source", ""))

        for entity in page_item.get("entities", []):
            if not isinstance(entity, dict):
                continue
            name = str(entity.get("name", "")).strip()
            if not name:
                continue
            node_id = _find_or_create_entity(G, name, page=page)
            G.nodes[node_id]["entity_type"] = entity.get("type", "")

        fig_id_map = page_item.get("figure_id_map", {}) if isinstance(page_item, dict) else {}

        for relation in page_item.get("relations", []):
            if not isinstance(relation, dict):
                continue

            src_name = str(relation.get("from", "")).strip()
            rel_name = str(relation.get("relation", "相关")).strip() or "相关"
            dst_name = str(relation.get("to", "")).strip()
            if not src_name or not dst_name:
                continue

            src_node = fig_id_map.get(src_name) or _find_or_create_entity(G, src_name, page=page)
            dst_node = fig_id_map.get(dst_name) or _find_or_create_entity(G, dst_name, page=page)
            G.add_edge(src_node, dst_node, relation=rel_name)

    node_count = G.number_of_nodes()
    edge_count = G.number_of_edges()
    print(f"    知识图谱构建完成: {node_count} 个节点, {edge_count} 条边")

    # 统计节点类型
    type_counts = {}
    for _, attrs in G.nodes(data=True):
        node_type = attrs.get("type", "unknown")
        type_counts[node_type] = type_counts.get(node_type, 0) + 1

    print(f"    节点类型分布:")
    for node_type, count in type_counts.items():
        print(f"      {node_type}: {count} 个")

    return G


def save_graph(G: nx.DiGraph, graph_dir: str | Path) -> tuple[Path, dict[str, float]]:
    output_dir = Path(graph_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "graph.pkl"

    print(f"    正在保存知识图谱到: {output_path}")
    nx.write_gpickle(G, output_path)

    node_count = float(G.number_of_nodes())
    edge_count = float(G.number_of_edges())
    avg_degree = (sum(dict(G.degree()).values()) / node_count) if node_count else 0.0

    stats = {
        "nodes": node_count,
        "edges": edge_count,
        "avg_degree": avg_degree,
    }

    print(f"    图谱已保存，文件大小: {output_path.stat().st_size / 1024:.1f} KB")
    return output_path, stats


def load_graph(graph_path: str | Path) -> nx.DiGraph:
    return nx.read_gpickle(str(graph_path))
