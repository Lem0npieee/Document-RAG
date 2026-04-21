from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from langchain.schema import Document

from src.graph.builder import load_graph
from src.indexing.faiss_store import load_faiss_index
from src.vl_client import DashScopeVLClient


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
        api_key: str,
        vl_model: str,
        embedding_model: str,
        faiss_dir: str | Path,
        graph_path: str | Path,
        pages_dir: str | Path,
    ) -> None:
        self.vl_client = DashScopeVLClient(api_key=api_key, model=vl_model)
        self.vectorstore = load_faiss_index(
            api_key=api_key,
            embedding_model=embedding_model,
            index_dir=faiss_dir,
        )
        self.graph = load_graph(graph_path)
        self.pages_dir = Path(pages_dir)

    def _expand_with_graph(self, seed_node_ids: list[str], max_nodes: int = 8) -> list[str]:
        selected: list[str] = []
        seen = set()

        for node_id in seed_node_ids:
            if node_id in self.graph and node_id not in seen:
                selected.append(node_id)
                seen.add(node_id)

        neighbors: list[str] = []
        for seed in selected:
            if seed not in self.graph:
                continue
            near = list(self.graph.successors(seed)) + list(self.graph.predecessors(seed))
            neighbors.extend(near)

        neighbors = [node for node in neighbors if node in self.graph and node not in seen]
        neighbors.sort(
            key=lambda nid: 0
            if self.graph.nodes[nid].get("type") in {"figure", "table"}
            else 1
        )

        for node in neighbors:
            if len(selected) >= max_nodes:
                break
            selected.append(node)
            seen.add(node)

        return selected

    def _collect_relations(self, node_ids: list[str]) -> list[str]:
        relation_lines: list[str] = []
        node_set = set(node_ids)

        for src in node_ids:
            for dst in self.graph.successors(src):
                if dst not in node_set:
                    continue
                relation = self.graph.edges[src, dst].get("relation", "相关")
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
            page = self.graph.nodes[node_id].get("page")
            if isinstance(page, int) and page > 0:
                pages.add(page)

        if not pages:
            for doc in retrieved_docs:
                page = doc.metadata.get("page")
                if isinstance(page, int) and page > 0:
                    pages.add(page)

        return sorted(pages)

    def _collect_image_paths(self, node_ids: list[str], retrieved_docs: list[Document]) -> list[Path]:
        paths: list[Path] = []
        seen = set()

        for node_id in node_ids:
            if node_id not in self.graph:
                continue
            image_path = str(self.graph.nodes[node_id].get("image_path", "")).strip()
            if not image_path:
                continue
            p = Path(image_path)
            if p.exists() and str(p) not in seen:
                paths.append(p)
                seen.add(str(p))

        for doc in retrieved_docs:
            image_path = str(doc.metadata.get("image_path", "")).strip()
            if not image_path:
                continue
            p = Path(image_path)
            if p.exists() and str(p) not in seen:
                paths.append(p)
                seen.add(str(p))

        return paths

    def ask(self, question: str, k: int = 3, max_nodes: int = 8) -> GraphRAGResult:
        print(f"\n[GraphRAG问答] 问题: {question}")
        print(f"  检索参数: k={k}, max_nodes={max_nodes}")

        retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
        retrieved_docs = retriever.invoke(question)
        print(f"  检索到 {len(retrieved_docs)} 个相关文档")

        seed_node_ids = [
            str(doc.metadata.get("node_id"))
            for doc in retrieved_docs
            if doc.metadata.get("node_id")
        ]
        print(f"  种子节点: {len(seed_node_ids)} 个")

        expanded_node_ids = self._expand_with_graph(seed_node_ids, max_nodes=max_nodes)
        print(f"  扩展后节点: {len(expanded_node_ids)} 个")

        relation_lines = self._collect_relations(expanded_node_ids)
        print(f"  发现关系: {len(relation_lines)} 条")

        text_evidence = self._collect_text_evidence(expanded_node_ids, retrieved_docs)
        pages = self._collect_pages(expanded_node_ids, retrieved_docs)
        print(f"  相关页面: {pages}")

        image_paths = self._collect_image_paths(expanded_node_ids, retrieved_docs)
        if not image_paths:
            image_paths = [
                self.pages_dir / f"page_{page}.png"
                for page in pages
                if (self.pages_dir / f"page_{page}.png").exists()
            ]
        print(f"  使用图片: {len(image_paths)} 张")

        relation_block = "\n".join(relation_lines) if relation_lines else "无显式关系"

        prompt = (
            "你是文档智能助手。请根据以下多模态证据回答用户问题。\n"
            "证据包含：①检索文本 ②实体关系链 ③文档页面原图。\n"
            f"用户问题：{question}\n\n"
            f"实体关系链（知识图谱路径）：\n{relation_block}\n\n"
            f"检索到的文本与描述：\n{text_evidence}\n\n"
            "要求：答案必须准确、简洁。回答末尾必须标注 "
            "[引用：第X页, 图Y/表格Z/文本, 经由关系: R]。"
        )

        print(f"  正在生成答案...")
        answer = self.vl_client.answer_question(prompt=prompt, image_paths=image_paths)
        print(f"  答案生成完成，长度: {len(answer)} 字符")

        print(f"  问答完成\n")
        return GraphRAGResult(
            answer=answer,
            node_ids=expanded_node_ids,
            pages=pages,
            image_paths=[str(p) for p in image_paths],
            relations=relation_lines,
        )
