from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx
from langchain_core.documents import Document

from src.config import Settings
from src.graph.builder import load_graph
from src.indexing.faiss_store import load_faiss_index
from src.vl_client import create_vl_client


@dataclass
class GraphRAGResult:
    answer: str
    node_ids: list[str]
    pages: list[int]
    image_paths: list[str]
    relations: list[str]


class MultiModalGraphRAG:
    def __init__(
        self,
        settings: Settings,
        faiss_dir: str | Path,
        graph_path: str | Path,
        pages_dir: str | Path,
    ) -> None:
        self.vl_client = create_vl_client(settings)
        self.vectorstore = load_faiss_index(
            api_key=settings.dashscope_api_key,
            embedding_model=settings.embedding_model,
            index_dir=faiss_dir,
        )
        self.graph = load_graph(graph_path)
        self.pages_dir = Path(pages_dir)
        self.community_profiles = self._build_community_profiles()

    def _build_community_profiles(self) -> list[dict[str, Any]]:
        if self.graph.number_of_nodes() == 0:
            return []

        undirected = self.graph.to_undirected()
        if undirected.number_of_nodes() == 0:
            return []

        try:
            communities = list(nx.algorithms.community.greedy_modularity_communities(undirected))
        except Exception:
            communities = [set(undirected.nodes())]

        profiles: list[dict[str, Any]] = []
        for idx, comm in enumerate(sorted(communities, key=len, reverse=True), start=1):
            node_ids = [nid for nid in comm if nid in self.graph]
            if not node_ids:
                continue

            pages: set[int] = set()
            entity_names: list[str] = []
            keyword_names: list[str] = []
            content_nodes: list[tuple[int, str, str]] = []

            for nid in node_ids:
                node = self.graph.nodes[nid]

                page = node.get("page")
                if isinstance(page, int) and page > 0:
                    pages.add(page)

                page_span = node.get("page_span", [])
                if isinstance(page_span, list):
                    for p in page_span:
                        if isinstance(p, int) and p > 0:
                            pages.add(p)

                node_type = str(node.get("type", ""))
                name = str(node.get("name", "")).strip()
                if node_type == "entity" and name:
                    entity_names.append(name)
                if node_type == "keyword" and name:
                    keyword_names.append(name)

                content = str(node.get("content", "")).strip()
                if content:
                    degree = int(self.graph.degree(nid))
                    content_nodes.append((degree, nid, content))

            content_nodes.sort(key=lambda x: x[0], reverse=True)
            snippet_lines: list[str] = []
            for _, nid, content in content_nodes[:8]:
                snippet = content.replace("\n", " ").strip()
                if len(snippet) > 180:
                    snippet = f"{snippet[:180]}..."
                snippet_lines.append(f"[{nid}] {snippet}")

            relation_lines: list[str] = []
            node_set = set(node_ids)
            for src in node_ids:
                for dst in self.graph.successors(src):
                    if dst not in node_set:
                        continue
                    rel = str(self.graph.edges[src, dst].get("relation", "related"))
                    relation_lines.append(f"{src} --[{rel}]--> {dst}")
                    if len(relation_lines) >= 20:
                        break
                if len(relation_lines) >= 20:
                    break

            profiles.append(
                {
                    "community_id": idx,
                    "node_ids": node_ids,
                    "size": len(node_ids),
                    "pages": sorted(pages),
                    "entities": sorted(set(entity_names))[:12],
                    "keywords": sorted(set(keyword_names))[:16],
                    "snippets": snippet_lines,
                    "relations": relation_lines,
                }
            )

        return profiles

    def _query_tokens(self, text: str) -> list[str]:
        raw = str(text).lower()
        en_words = re.findall(r"[a-z0-9_]{2,}", raw)
        zh_terms = re.findall(r"[\u4e00-\u9fff]{2,}", raw)
        single_zh = re.findall(r"[\u4e00-\u9fff]", raw)
        tokens = en_words + zh_terms + single_zh
        return list(dict.fromkeys(tokens))

    def _select_global_profiles(
        self,
        question: str,
        preferred_node_ids: list[str],
        top_n: int = 4,
    ) -> list[dict[str, Any]]:
        if not self.community_profiles:
            return []

        q_tokens = self._query_tokens(question)
        preferred = set(preferred_node_ids)
        scored: list[tuple[float, dict[str, Any]]] = []

        for profile in self.community_profiles:
            text_blob = " ".join(
                profile.get("entities", [])
                + profile.get("keywords", [])
                + profile.get("snippets", [])
                + profile.get("relations", [])
            ).lower()
            token_hits = sum(1 for t in q_tokens if t and t in text_blob)
            overlap = len(preferred.intersection(set(profile.get("node_ids", []))))
            size_bonus = min(int(profile.get("size", 0)), 80) * 0.03
            page_bonus = min(len(profile.get("pages", [])), 12) * 0.08
            score = token_hits * 2.5 + overlap * 1.2 + size_bonus + page_bonus
            scored.append((score, profile))

        scored.sort(key=lambda x: x[0], reverse=True)
        picked = [profile for score, profile in scored[:top_n] if score > 0]
        if not picked:
            picked = [profile for _, profile in scored[: max(1, min(top_n, 2))]]
        return picked

    def _render_global_context(self, profiles: list[dict[str, Any]]) -> str:
        if not profiles:
            return "无全局社区证据"

        blocks: list[str] = []
        for profile in profiles:
            cid = profile.get("community_id")
            size = profile.get("size", 0)
            pages = profile.get("pages", [])
            entities = profile.get("entities", [])
            keywords = profile.get("keywords", [])
            snippets = profile.get("snippets", [])
            relations = profile.get("relations", [])

            blocks.append(
                (
                    f"[社区#{cid}] 节点数={size} 页码={pages}\n"
                    f"关键实体: {entities}\n"
                    f"关键关键词: {keywords}\n"
                    f"代表片段:\n" + ("\n".join(snippets[:5]) if snippets else "无") + "\n"
                    f"代表关系:\n" + ("\n".join(relations[:8]) if relations else "无")
                )
            )

        return "\n\n".join(blocks)

    def _neighbor_rank(self, nid: str) -> tuple[int, int]:
        node_type = str(self.graph.nodes[nid].get("type", ""))
        priority_map = {
            "figure": 0,
            "table": 0,
            "conclusion": 1,
            "text": 1,
            "entity": 2,
            "keyword": 2,
            "cross_page_text": 3,
        }
        return (priority_map.get(node_type, 5), -int(self.graph.degree(nid)))

    def _expand_with_graph(self, seed_node_ids: list[str], max_nodes: int = 18) -> list[str]:
        selected: list[str] = []
        seen = set()

        for node_id in seed_node_ids:
            if node_id in self.graph and node_id not in seen:
                selected.append(node_id)
                seen.add(node_id)

        if not selected:
            return selected

        def collect_neighbors(candidates: list[str]) -> list[str]:
            acc: list[str] = []
            for nid in candidates:
                if nid not in self.graph:
                    continue
                near = list(self.graph.successors(nid)) + list(self.graph.predecessors(nid))
                acc.extend(near)
            uniq = [n for n in dict.fromkeys(acc) if n in self.graph and n not in seen]
            uniq.sort(key=self._neighbor_rank)
            return uniq

        # Hop-1 expansion around local retrieval seeds.
        hop1 = collect_neighbors(selected)
        for nid in hop1:
            if len(selected) >= max_nodes:
                break
            selected.append(nid)
            seen.add(nid)

        if len(selected) >= max_nodes:
            return selected

        # Hop-2 bridge expansion via entity/keyword hubs to improve cross-document linking.
        bridge_nodes = [
            nid
            for nid in selected
            if str(self.graph.nodes[nid].get("type", "")) in {"entity", "keyword"}
        ]
        hop2 = collect_neighbors(bridge_nodes)
        for nid in hop2:
            if len(selected) >= max_nodes:
                break
            selected.append(nid)
            seen.add(nid)

        return selected

    def _collect_relations(self, node_ids: list[str]) -> list[str]:
        relation_lines: list[str] = []
        node_set = set(node_ids)

        for src in node_ids:
            for dst in self.graph.successors(src):
                if dst not in node_set:
                    continue
                relation = self.graph.edges[src, dst].get("relation", "related")
                relation_lines.append(f"{src} --[{relation}]--> {dst}")

        return relation_lines

    def _collect_text_evidence(self, node_ids: list[str], retrieved_docs: list[Document]) -> str:
        lines: list[str] = []

        for node_id in node_ids:
            if node_id not in self.graph:
                continue
            node = self.graph.nodes[node_id]
            content = str(node.get("content", "")).strip()
            if content:
                page = node.get("page", "?")
                node_type = node.get("type", "text")
                fig_id = node.get("fig_id", "")
                if fig_id:
                    lines.append(f"[node={node_id} | page={page} | type={node_type} | fig_id={fig_id}] {content}")
                else:
                    lines.append(f"[node={node_id} | page={page} | type={node_type}] {content}")

        if lines:
            return "\n".join(lines)

        fallback = []
        for doc in retrieved_docs:
            page = doc.metadata.get("page", "?")
            node_type = doc.metadata.get("type", "text")
            fallback.append(f"[page={page} | type={node_type}] {doc.page_content}")
        return "\n".join(fallback)

    def _collect_pages(self, node_ids: list[str], retrieved_docs: list[Document]) -> list[int]:
        pages = set()

        for node_id in node_ids:
            if node_id not in self.graph:
                continue
            node = self.graph.nodes[node_id]
            page = node.get("page")
            if isinstance(page, int) and page > 0:
                pages.add(page)
            page_span = node.get("page_span", [])
            if isinstance(page_span, list):
                for p in page_span:
                    if isinstance(p, int) and p > 0:
                        pages.add(p)

        if not pages:
            for doc in retrieved_docs:
                page = doc.metadata.get("page")
                if isinstance(page, int) and page > 0:
                    pages.add(page)
                page_span = doc.metadata.get("page_span", [])
                if isinstance(page_span, list):
                    for p in page_span:
                        if isinstance(p, int) and p > 0:
                            pages.add(p)

        return sorted(pages)

    def _collect_image_paths(self, node_ids: list[str], retrieved_docs: list[Document]) -> list[Path]:
        paths: list[Path] = []
        seen = set()

        for node_id in node_ids:
            if node_id not in self.graph:
                continue
            image_path = str(self.graph.nodes[node_id].get("image_path", "")).strip()
            if image_path:
                p = Path(image_path)
                if p.exists() and str(p) not in seen:
                    paths.append(p)
                    seen.add(str(p))
            image_paths = self.graph.nodes[node_id].get("image_paths", [])
            if isinstance(image_paths, list):
                for item in image_paths:
                    pi = Path(str(item))
                    if pi.exists() and str(pi) not in seen:
                        paths.append(pi)
                        seen.add(str(pi))

        for doc in retrieved_docs:
            image_path = str(doc.metadata.get("image_path", "")).strip()
            if image_path:
                p = Path(image_path)
                if p.exists() and str(p) not in seen:
                    paths.append(p)
                    seen.add(str(p))
            image_paths = doc.metadata.get("image_paths", [])
            if isinstance(image_paths, list):
                for item in image_paths:
                    pi = Path(str(item))
                    if pi.exists() and str(pi) not in seen:
                        paths.append(pi)
                        seen.add(str(pi))

        return paths

    def _canonical_source_name(self, value: str | None) -> str:
        if not value:
            return ""
        return Path(str(value)).name.strip().lower()

    def _retrieve_docs(self, question: str, k: int, source_hint: str | None = None) -> list[Document]:
        if k <= 0:
            return []

        source_key = self._canonical_source_name(source_hint)
        candidate_k = max(k * 6, 30) if source_key else k
        candidates: list[Document] = []

        try:
            candidates = self.vectorstore.similarity_search(question, k=candidate_k)
        except Exception:
            retriever = self.vectorstore.as_retriever(search_kwargs={"k": candidate_k})
            candidates = retriever.invoke(question)

        if not source_key:
            return candidates[:k]

        filtered: list[Document] = []
        for doc in candidates:
            doc_source = self._canonical_source_name(str(doc.metadata.get("source", "")))
            if doc_source == source_key:
                filtered.append(doc)

        if len(filtered) >= k:
            return filtered[:k]

        # If source filtering is too strict and returns nothing, keep behavior robust
        # by falling back to top semantic candidates.
        return filtered if filtered else candidates[:k]

    def ask(
        self,
        question: str,
        k: int = 3,
        max_nodes: int = 18,
        source_hint: str | None = None,
        answer_style: str = "detailed",
    ) -> GraphRAGResult:
        print(f"\n[GraphRAG] Question: {question}")
        print(f"  Retrieval params: k={k}, max_nodes={max_nodes}")
        if source_hint:
            print(f"  Source hint: {source_hint}")

        retrieved_docs = self._retrieve_docs(question=question, k=k, source_hint=source_hint)
        print(f"  Retrieved docs: {len(retrieved_docs)}")

        seed_node_ids = [str(doc.metadata.get("node_id")) for doc in retrieved_docs if doc.metadata.get("node_id")]
        print(f"  Seed nodes: {len(seed_node_ids)}")

        expanded_node_ids = self._expand_with_graph(seed_node_ids, max_nodes=max_nodes)
        print(f"  Expanded nodes: {len(expanded_node_ids)}")

        global_profiles = self._select_global_profiles(
            question=question,
            preferred_node_ids=expanded_node_ids,
            top_n=4,
        )
        global_context = self._render_global_context(global_profiles)

        global_node_ids: list[str] = []
        for profile in global_profiles:
            for nid in profile.get("node_ids", []):
                if len(global_node_ids) >= 24:
                    break
                global_node_ids.append(str(nid))

        fused_node_ids = list(dict.fromkeys(expanded_node_ids + global_node_ids))
        print(
            f"  Global communities: {len(global_profiles)}, global supplement nodes: {len(global_node_ids)}, fused nodes: {len(fused_node_ids)}"
        )

        relation_lines = self._collect_relations(fused_node_ids)
        print(f"  Relations found: {len(relation_lines)}")

        text_evidence = self._collect_text_evidence(fused_node_ids, retrieved_docs)
        pages = self._collect_pages(fused_node_ids, retrieved_docs)
        print(f"  Related pages: {pages}")

        image_paths = self._collect_image_paths(fused_node_ids, retrieved_docs)
        if not image_paths:
            image_paths = [self.pages_dir / f"page_{page}.png" for page in pages if (self.pages_dir / f"page_{page}.png").exists()]
        print(f"  Images used: {len(image_paths)}")

        relation_block = "\n".join(relation_lines) if relation_lines else "无显式关系"

        prompt = (
            "你是文档问答助手。请融合局部检索证据、图谱关系链与全局社区摘要回答问题。\n"
            "若局部证据与全局摘要冲突，以局部证据为准并说明冲突点。\n"
            f"用户问题：{question}\n\n"
            f"图谱关系链：\n{relation_block}\n\n"
            f"全局社区证据：\n{global_context}\n\n"
            f"文本证据：\n{text_evidence}\n\n"
            "要求：回答准确、简洁；结尾给出引用页码与关键关系。"
        )
        if answer_style == "short":
            prompt += (
                "\n\nEvaluation output format:\n"
                "1) Output only the final answer string.\n"
                "2) No explanation, no citation, no extra words.\n"
                "3) If uncertain, output your best short span from the document."
            )

        print("  Generating answer...")
        answer = self.vl_client.answer_question(prompt=prompt, image_paths=image_paths)
        print(f"  Answer generated: {len(answer)} chars")

        return GraphRAGResult(
            answer=answer,
            node_ids=fused_node_ids,
            pages=pages,
            image_paths=[str(p) for p in image_paths],
            relations=relation_lines,
        )

    def ask_eval(self, question: str, source_hint: str | None = None, k: int = 5, max_nodes: int = 24) -> GraphRAGResult:
        return self.ask(
            question=question,
            k=k,
            max_nodes=max_nodes,
            source_hint=source_hint,
            answer_style="short",
        )
