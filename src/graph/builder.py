from __future__ import annotations

import re
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

import networkx as nx
from langchain_core.documents import Document


def _normalize_key(value: str) -> str:
    raw = str(value or "").strip().casefold()
    raw = re.sub(r"\s+", "_", raw)
    raw = re.sub(r"[^\w\u4e00-\u9fff]+", "_", raw)
    raw = raw.strip("_")
    return raw or "unknown"


def _keyword_node_id(term: str) -> str:
    return f"keyword::{_normalize_key(term)}"


def _find_or_create_keyword(
    G: nx.DiGraph,
    term: str,
    keyword_type: str = "concept",
    page: int | None = None,
    source: str = "",
) -> str:
    node_id = _keyword_node_id(term)
    if not G.has_node(node_id):
        G.add_node(
            node_id,
            type="keyword",
            name=term,
            keyword_type=keyword_type,
            page=page,
            pages=[page] if isinstance(page, int) and page > 0 else [],
            source=source,
            sources=[source] if source else [],
            aliases=[term],
            content=term,
        )
        return node_id

    aliases = G.nodes[node_id].get("aliases", [])
    if term and term not in aliases:
        aliases.append(term)
        G.nodes[node_id]["aliases"] = aliases

    existing_type = str(G.nodes[node_id].get("keyword_type", "")).strip()
    if (not existing_type or existing_type == "other") and keyword_type:
        G.nodes[node_id]["keyword_type"] = keyword_type
    if isinstance(page, int) and page > 0:
        pages = G.nodes[node_id].get("pages", [])
        if page not in pages:
            pages.append(page)
            G.nodes[node_id]["pages"] = sorted(pages)
    if source:
        sources = G.nodes[node_id].get("sources", [])
        if source not in sources:
            sources.append(source)
            G.nodes[node_id]["sources"] = sorted(sources)
    return node_id


def _page_anchor_nodes(
    page_key: tuple[str, int],
    by_page: dict[tuple[str, int], list[str]],
    text_nodes_by_page: dict[tuple[str, int], list[str]],
    conclusion_nodes_by_page: dict[tuple[str, int], list[str]],
    max_nodes: int = 8,
) -> list[str]:
    anchors = (
        conclusion_nodes_by_page.get(page_key, [])
        + text_nodes_by_page.get(page_key, [])
        + by_page.get(page_key, [])
    )
    unique = list(dict.fromkeys(anchors))
    return unique[:max_nodes]


def build_document_graph(documents: list[Document], graph_data: dict[str, Any]) -> nx.DiGraph:
    print(f"    Building knowledge graph, documents: {len(documents)}")
    G = nx.DiGraph()

    by_page: dict[tuple[str, int], list[str]] = defaultdict(list)
    text_nodes_by_page: dict[tuple[str, int], list[str]] = defaultdict(list)
    conclusion_nodes_by_page: dict[tuple[str, int], list[str]] = defaultdict(list)
    figure_by_page: dict[tuple[str, int], dict[str, str]] = defaultdict(dict)
    conclusion_by_page: dict[tuple[str, int], list[tuple[str, str]]] = defaultdict(list)

    for doc in documents:
        metadata = doc.metadata
        node_id = str(metadata["node_id"])
        page = int(metadata.get("page", 0))
        source = str(metadata.get("source", ""))
        page_key = (source, page)

        node_attrs = {
            "page": page,
            "page_span": metadata.get("page_span", []),
            "image_path": metadata.get("image_path", ""),
            "image_paths": metadata.get("image_paths", []),
            "type": metadata.get("type", "text"),
            "source": source,
            "fig_id": metadata.get("fig_id", ""),
            "content": doc.page_content,
        }
        G.add_node(node_id, **node_attrs)
        by_page[page_key].append(node_id)

        if metadata.get("type") == "figure":
            fig_id = str(metadata.get("fig_id", ""))
            if fig_id:
                figure_by_page[page_key][fig_id] = node_id

        if metadata.get("type") == "text":
            text_nodes_by_page[page_key].append(node_id)

        if metadata.get("type") == "conclusion":
            conclusion_by_page[page_key].append((node_id, doc.page_content))
            conclusion_nodes_by_page[page_key].append(node_id)

    # Strong local page cohesion among parsed content nodes.
    for _, nodes in by_page.items():
        for left, right in combinations(nodes, 2):
            G.add_edge(left, right, relation="same_page")
            G.add_edge(right, left, relation="same_page")

    # Figure -> conclusion support edges based on figure references in conclusion text.
    for page_key, conclusions in conclusion_by_page.items():
        fig_map = figure_by_page.get(page_key, {})
        for conclusion_node_id, text in conclusions:
            for fig_id, fig_node_id in fig_map.items():
                if fig_id in text:
                    G.add_edge(fig_node_id, conclusion_node_id, relation="supports")

    # Default cross-page continuation for consecutive text pages within same source.
    source_pages: dict[str, list[int]] = defaultdict(list)
    for source, page in text_nodes_by_page.keys():
        source_pages[source].append(page)

    for source, pages in source_pages.items():
        sorted_pages = sorted(set(pages))
        for i in range(len(sorted_pages) - 1):
            cur_page = sorted_pages[i]
            next_page = sorted_pages[i + 1]
            if next_page != cur_page + 1:
                continue
            cur_nodes = text_nodes_by_page.get((source, cur_page), [])
            next_nodes = text_nodes_by_page.get((source, next_page), [])
            if not cur_nodes or not next_nodes:
                continue
            G.add_edge(cur_nodes[-1], next_nodes[0], relation="continuation")

    for page_item in graph_data.get("pages", []):
        if not isinstance(page_item, dict):
            continue

        page = int(page_item.get("page", 0))
        source = str(page_item.get("source", ""))
        page_key = (source, page)
        anchors = _page_anchor_nodes(page_key, by_page, text_nodes_by_page, conclusion_nodes_by_page)

        # Entity nodes are intentionally disabled.
        # We only build page/content/keyword/cross-page structure.

        page_keyword_ids: list[str] = []
        for keyword in page_item.get("keywords", []):
            if isinstance(keyword, str):
                term = keyword.strip()
                keyword_type = "concept"
            elif isinstance(keyword, dict):
                term = str(keyword.get("term", "")).strip() or str(keyword.get("name", "")).strip()
                keyword_type = str(keyword.get("type", "concept")).strip() or "concept"
            else:
                continue

            if not term:
                continue

            keyword_node = _find_or_create_keyword(
                G,
                term=term,
                keyword_type=keyword_type,
                page=page,
                source=source,
            )
            page_keyword_ids.append(keyword_node)

            # Link keyword to local content anchors (important for two-hop cross-doc traversal).
            for anchor in anchors:
                if not G.has_node(anchor):
                    continue
                G.add_edge(keyword_node, anchor, relation="keyword_hit")
                G.add_edge(anchor, keyword_node, relation="keyword_hit")

        # Keyword co-occurrence within a page.
        unique_keywords = list(dict.fromkeys(page_keyword_ids))[:16]
        for left, right in combinations(unique_keywords, 2):
            G.add_edge(left, right, relation="cooccurrence")
            G.add_edge(right, left, relation="cooccurrence")

        fig_id_map = page_item.get("figure_id_map", {}) if isinstance(page_item, dict) else {}
        if not isinstance(fig_id_map, dict):
            fig_id_map = {}

        for relation in page_item.get("relations", []):
            if not isinstance(relation, dict):
                continue

            src_name = str(relation.get("from", "")).strip()
            rel_name = str(relation.get("relation", "related")).strip() or "related"
            dst_name = str(relation.get("to", "")).strip()
            if not src_name or not dst_name:
                continue

            src_node = fig_id_map.get(src_name)
            if not src_node:
                src_node = _find_or_create_keyword(
                    G,
                    term=src_name,
                    keyword_type="concept",
                    page=page,
                    source=source,
                )
            dst_node = fig_id_map.get(dst_name)
            if not dst_node:
                dst_node = _find_or_create_keyword(
                    G,
                    term=dst_name,
                    keyword_type="concept",
                    page=page,
                    source=source,
                )
            G.add_edge(src_node, dst_node, relation=rel_name)

    # Ensure document-level keywords are present even if some pages had sparse extraction.
    for kw in graph_data.get("document_keywords", []):
        if not isinstance(kw, dict):
            continue
        term = str(kw.get("term", "")).strip()
        if not term:
            continue
        keyword_type = str(kw.get("type", "concept")).strip() or "concept"
        _find_or_create_keyword(
            G,
            term=term,
            keyword_type=keyword_type,
            page=None,
            source=str(kw.get("source", "")),
        )

    # Add explicit cross-page links from parsing stage when available.
    for link in graph_data.get("cross_page_links", []):
        if not isinstance(link, dict):
            continue
        from_node_id = str(link.get("from_node_id", "")).strip()
        to_node_id = str(link.get("to_node_id", "")).strip()
        cross_node_id = str(link.get("cross_node_id", "")).strip()
        if from_node_id and to_node_id and G.has_node(from_node_id) and G.has_node(to_node_id):
            G.add_edge(from_node_id, to_node_id, relation="continuation")
        if (
            cross_node_id
            and G.has_node(cross_node_id)
            and from_node_id
            and to_node_id
            and G.has_node(from_node_id)
            and G.has_node(to_node_id)
        ):
            G.add_edge(from_node_id, cross_node_id, relation="cross_page_merge")
            G.add_edge(cross_node_id, to_node_id, relation="cross_page_merge")

    node_count = G.number_of_nodes()
    edge_count = G.number_of_edges()
    print(f"    Graph build complete: {node_count} nodes, {edge_count} edges")

    type_counts: dict[str, int] = {}
    for _, attrs in G.nodes(data=True):
        node_type = str(attrs.get("type", "unknown"))
        type_counts[node_type] = type_counts.get(node_type, 0) + 1

    print("    Node type distribution:")
    for node_type, count in sorted(type_counts.items(), key=lambda x: x[0]):
        print(f"      {node_type}: {count}")

    return G


def save_graph(G: nx.DiGraph, graph_dir: str | Path) -> tuple[Path, dict[str, float]]:
    output_dir = Path(graph_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "graph.pkl"

    print(f"    Saving graph to: {output_path}")
    nx.write_gpickle(G, output_path)

    node_count = float(G.number_of_nodes())
    edge_count = float(G.number_of_edges())
    avg_degree = (sum(dict(G.degree()).values()) / node_count) if node_count else 0.0

    stats = {
        "nodes": node_count,
        "edges": edge_count,
        "avg_degree": avg_degree,
    }

    print(f"    Graph saved, file size: {output_path.stat().st_size / 1024:.1f} KB")
    return output_path, stats


def load_graph(graph_path: str | Path) -> nx.DiGraph:
    return nx.read_gpickle(str(graph_path))
