from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import fitz
from langchain.schema import Document

from src.utils.json_utils import ensure_page_schema, extract_json_object
from src.vl_client import DashScopeVLClient

PARSE_PROMPT = """
你是一个文档解析专家。请分析这张文档图片，提取所有内容，并严格按以下JSON格式输出：
{
  "texts": ["第1段文字...", "第2段文字..."],
  "tables": ["表格1的Markdown格式...", "表格2的Markdown格式..."],
  "figures": [
    {"fig_id": "图1", "description": "详细描述图1展示的趋势或内容..."},
    {"fig_id": "图2", "description": "..."}
  ],
  "section_conclusions": ["第1节主要结论是...", "第2节主要结论是..."],
  "entities": [
    {"name": "ResNet", "type": "模型"},
    {"name": "图3", "type": "图表"},
    {"name": "准确率", "type": "指标"}
  ],
  "relations": [
    {"from": "图3", "relation": "展示了", "to": "准确率随训练轮次上升的趋势"},
    {"from": "ResNet", "relation": "在图3中表现最优", "to": "图3"},
    {"from": "第2节结论", "relation": "由数据支撑自", "to": "表格1"}
  ]
}
如果没有某项内容，对应值设为空数组。只输出JSON，不要输出其他内容。
""".strip()


def render_pdf_to_images(pdf_path: str | Path, pages_dir: str | Path, zoom: float = 2.0) -> list[Path]:
    pages_output = Path(pages_dir)
    pages_output.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(str(pdf_path))
    total_pages = len(doc)
    print(f"    PDF共有 {total_pages} 页，开始渲染...")

    image_paths: list[Path] = []
    matrix = fitz.Matrix(zoom, zoom)

    for page_idx, page in enumerate(doc, start=1):
        if page_idx % 10 == 0 or page_idx == total_pages:
            print(f"    正在渲染第 {page_idx}/{total_pages} 页...")
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        output_path = pages_output / f"page_{page_idx}.png"
        pix.save(output_path)
        image_paths.append(output_path)

    print(f"    完成渲染，共生成 {len(image_paths)} 张图片")
    return image_paths


def prepare_input_as_images(input_path: str | Path, pages_dir: str | Path) -> list[Path]:
    input_file = Path(input_path)
    suffix = input_file.suffix.lower()

    print(f"    准备将 {input_file.name} 转换为图片，格式: {suffix}")

    if suffix == ".pdf":
        print(f"    渲染PDF为图片...")
        return render_pdf_to_images(input_file, pages_dir)

    if suffix in {".png", ".jpg", ".jpeg", ".webp"}:
        pages_output = Path(pages_dir)
        pages_output.mkdir(parents=True, exist_ok=True)
        target_path = pages_output / "page_1.png"
        print(f"    复制图片文件到: {target_path}")
        shutil.copyfile(input_file, target_path)
        return [target_path]

    raise ValueError(f"Unsupported file type: {input_file.suffix}")


def _source_tag(source_name: str) -> str:
    stem = Path(source_name).stem
    safe = "".join(ch if ch.isalnum() else "_" for ch in stem)
    return safe.strip("_")[:40] or "doc"


def _make_node_id(source_name: str, page: int, item_type: str, index: int) -> str:
    return f"{_source_tag(source_name)}_page{page}_{item_type}_{index}"


def _ensure_text(value: Any) -> str:
    return value if isinstance(value, str) else ""


def parse_images_to_documents(
    image_paths: list[Path],
    source_name: str,
    vl_client: DashScopeVLClient,
) -> tuple[list[Document], dict[str, Any]]:
    documents: list[Document] = []
    graph_pages: list[dict[str, Any]] = []

    total_pages = len(image_paths)
    print(f"    开始解析 {total_pages} 页内容...")

    for page_num, image_path in enumerate(image_paths, start=1):
        if page_num % 5 == 0 or page_num == total_pages:
            print(f"    正在解析第 {page_num}/{total_pages} 页: {image_path.name}...")
        raw_text = vl_client.extract_structured_page(PARSE_PROMPT, image_path)
        parsed = ensure_page_schema(extract_json_object(raw_text))

        page_node_ids: list[str] = []
        figure_id_map: dict[str, str] = {}
        conclusion_node_ids: list[str] = []

        for idx, text in enumerate(parsed["texts"]):
            node_id = _make_node_id(source_name, page_num, "text", idx)
            documents.append(
                Document(
                    page_content=_ensure_text(text),
                    metadata={
                        "source": source_name,
                        "page": page_num,
                        "image_path": str(image_path),
                        "type": "text",
                        "node_id": node_id,
                    },
                )
            )
            page_node_ids.append(node_id)

        for idx, table in enumerate(parsed["tables"]):
            node_id = _make_node_id(source_name, page_num, "table", idx)
            documents.append(
                Document(
                    page_content=_ensure_text(table),
                    metadata={
                        "source": source_name,
                        "page": page_num,
                        "image_path": str(image_path),
                        "type": "table",
                        "node_id": node_id,
                    },
                )
            )
            page_node_ids.append(node_id)

        for idx, figure in enumerate(parsed["figures"]):
            fig = figure if isinstance(figure, dict) else {}
            fig_id = _ensure_text(fig.get("fig_id", f"图{idx + 1}")) or f"图{idx + 1}"
            description = _ensure_text(fig.get("description", ""))
            node_id = _make_node_id(source_name, page_num, "figure", idx)
            documents.append(
                Document(
                    page_content=description,
                    metadata={
                        "source": source_name,
                        "page": page_num,
                        "image_path": str(image_path),
                        "type": "figure",
                        "fig_id": fig_id,
                        "node_id": node_id,
                    },
                )
            )
            page_node_ids.append(node_id)
            figure_id_map[fig_id] = node_id

        for idx, conclusion in enumerate(parsed["section_conclusions"]):
            node_id = _make_node_id(source_name, page_num, "conclusion", idx)
            documents.append(
                Document(
                    page_content=_ensure_text(conclusion),
                    metadata={
                        "source": source_name,
                        "page": page_num,
                        "image_path": str(image_path),
                        "type": "conclusion",
                        "node_id": node_id,
                    },
                )
            )
            page_node_ids.append(node_id)
            conclusion_node_ids.append(node_id)

        graph_pages.append(
            {
                "source": source_name,
                "page": page_num,
                "image_path": str(image_path),
                "entities": parsed["entities"],
                "relations": parsed["relations"],
                "section_conclusions": parsed["section_conclusions"],
                "figure_id_map": figure_id_map,
                "node_ids": page_node_ids,
                "conclusion_node_ids": conclusion_node_ids,
            }
        )

    graph_data = {
        "source": source_name,
        "pages": graph_pages,
    }

    # 统计不同类型的内容数量
    text_count = len([d for d in documents if d.metadata.get("type") == "text"])
    table_count = len([d for d in documents if d.metadata.get("type") == "table"])
    figure_count = len([d for d in documents if d.metadata.get("type") == "figure"])
    conclusion_count = len([d for d in documents if d.metadata.get("type") == "conclusion"])

    print(f"    解析完成统计:")
    print(f"      文本段落: {text_count} 个")
    print(f"      表格: {table_count} 个")
    print(f"      图表: {figure_count} 个")
    print(f"      结论: {conclusion_count} 个")

    return documents, graph_data


def save_parsing_outputs(
    documents: list[Document],
    graph_data: dict[str, Any],
    parsed_dir: str | Path,
) -> tuple[Path, Path]:
    parsed_output = Path(parsed_dir)
    parsed_output.mkdir(parents=True, exist_ok=True)

    print(f"    正在保存解析结果到目录: {parsed_output}")

    docs_payload = [
        {
            "page_content": doc.page_content,
            "metadata": doc.metadata,
        }
        for doc in documents
    ]

    docs_path = parsed_output / "documents.json"
    graph_path = parsed_output / "graph_data.json"

    print(f"    保存文档数据到: {docs_path.name} ({len(docs_payload)} 条记录)")
    docs_path.write_text(json.dumps(docs_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"    保存图数据到: {graph_path.name}")
    graph_path.write_text(json.dumps(graph_data, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"    保存完成")
    return docs_path, graph_path
