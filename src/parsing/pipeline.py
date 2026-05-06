from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

import fitz
from langchain_core.documents import Document

from src.utils.json_utils import ensure_page_schema, extract_json_object
from src.vl_client import VLClient

PARSE_PROMPT = """
You are a document parsing expert.
Analyze the page image and output ONLY a JSON object using this schema:
{
  "texts": [{"content": "...", "bbox": [0.10, 0.12, 0.86, 0.18]}],
  "tables": [{"content": "...", "bbox": [0.08, 0.40, 0.92, 0.58]}],
  "figures": [{"fig_id": "Figure 1", "description": "...", "bbox": [0.10, 0.26, 0.48, 0.40]}],
  "section_conclusions": [{"content": "...", "bbox": [0.10, 0.80, 0.90, 0.86]}],
  "entities": [{"name": "...", "type": "person|location|organization|method|model|dataset|metric|task|concept|other"}],
  "keywords": [{"term": "...", "type": "person|location|organization|method|model|dataset|metric|task|concept|other"}],
  "relations": [{"from": "...", "relation": "...", "to": "..."}]
}

Rules:
1) bbox is normalized [x1, y1, x2, y2] in [0,1].
2) If bbox is uncertain, omit bbox for that item.
3) If a field has no content, return [].
4) Keywords must be high-value noun phrases (people, locations, organizations, methods, models, datasets, metrics, tasks, concepts), not generic function words.
5) Keep keyword count concise (prefer 5-12 quality keywords per page).
6) Return JSON only.
""".strip()

PARSE_PROMPT_REPAIR = """
The previous extraction was too sparse.
Re-read the page carefully and extract visible content with high recall.
Output ONLY this JSON:
{
  "texts": [{"content": "...", "bbox": [0.10, 0.12, 0.86, 0.18]}],
  "tables": [{"content": "...", "bbox": [0.08, 0.40, 0.92, 0.58]}],
  "figures": [{"fig_id": "Figure 1", "description": "...", "bbox": [0.10, 0.26, 0.48, 0.40]}],
  "section_conclusions": [{"content": "...", "bbox": [0.10, 0.80, 0.90, 0.86]}],
  "entities": [{"name": "...", "type": "person|location|organization|method|model|dataset|metric|task|concept|other"}],
  "keywords": [{"term": "...", "type": "person|location|organization|method|model|dataset|metric|task|concept|other"}],
  "relations": [{"from": "...", "relation": "...", "to": "..."}]
}

Constraints:
1) If there is visible text, texts[] MUST NOT be empty.
2) If bbox is uncertain, still keep the content with approximate bbox.
3) Do not return all-empty arrays unless this page is truly blank.
4) Keep entities/keywords concise and grounded in visible content.
5) Avoid generic words as keywords; prefer specific technical noun phrases.
6) Return JSON only.
""".strip()

PARSE_PROMPT_TEXT_ONLY = """
Extract paragraph/line text from this page with high recall.
Return ONLY JSON:
{
  "texts": [{"content": "...", "bbox": [0.10, 0.12, 0.86, 0.18]}]
}

Rules:
1) If there is visible text, texts[] MUST NOT be empty.
2) Keep each item as one coherent line/paragraph segment.
3) bbox should be approximate normalized coordinates in [0,1].
4) If a bbox is uncertain, still provide approximate bbox.
5) Return JSON only.
""".strip()

MAX_PARSE_ATTEMPTS = 2

_ALLOWED_TYPES = {
    "person",
    "location",
    "organization",
    "method",
    "model",
    "dataset",
    "metric",
    "task",
    "concept",
    "other",
}

_TYPE_ALIASES = {
    "人物": "person",
    "作者": "person",
    "person": "person",
    "people": "person",
    "human": "person",
    "地点": "location",
    "位置": "location",
    "国家": "location",
    "城市": "location",
    "location": "location",
    "place": "location",
    "组织": "organization",
    "机构": "organization",
    "公司": "organization",
    "organization": "organization",
    "org": "organization",
    "方法": "method",
    "算法": "method",
    "method": "method",
    "algorithm": "method",
    "模型": "model",
    "model": "model",
    "数据集": "dataset",
    "语料": "dataset",
    "dataset": "dataset",
    "benchmark": "dataset",
    "指标": "metric",
    "评价指标": "metric",
    "metric": "metric",
    "任务": "task",
    "task": "task",
    "概念": "concept",
    "术语": "concept",
    "concept": "concept",
    "other": "other",
    "其它": "other",
    "其他": "other",
}


def render_pdf_to_images(pdf_path: str | Path, pages_dir: str | Path, zoom: float = 2.6) -> list[Path]:
    pages_output = Path(pages_dir)
    pages_output.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(str(pdf_path))
    total_pages = len(doc)
    print(f"    PDF pages: {total_pages}")

    image_paths: list[Path] = []
    matrix = fitz.Matrix(zoom, zoom)

    for page_idx, page in enumerate(doc, start=1):
        if page_idx % 10 == 0 or page_idx == total_pages:
            print(f"    Rendering page {page_idx}/{total_pages}...")
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        output_path = pages_output / f"page_{page_idx}.png"
        pix.save(output_path)
        image_paths.append(output_path)

    print(f"    Render complete: {len(image_paths)} images")
    return image_paths


def prepare_input_as_images(input_path: str | Path, pages_dir: str | Path) -> list[Path]:
    input_file = Path(input_path)
    suffix = input_file.suffix.lower()

    print(f"    Preparing input: {input_file.name} ({suffix})")

    if suffix == ".pdf":
        return render_pdf_to_images(input_file, pages_dir)

    if suffix in {".png", ".jpg", ".jpeg", ".webp"}:
        pages_output = Path(pages_dir)
        pages_output.mkdir(parents=True, exist_ok=True)
        target_path = pages_output / "page_1.png"
        shutil.copyfile(input_file, target_path)
        return [target_path]

    raise ValueError(f"Unsupported file type: {input_file.suffix}")


def _source_tag(source_name: str) -> str:
    stem = Path(source_name).stem
    safe = "".join(ch if ch.isalnum() else "_" for ch in stem)
    return safe.strip("_")[:40] or "doc"


def _make_node_id(source_name: str, page: int, item_type: str, index: int) -> str:
    return f"{_source_tag(source_name)}_page{page}_{item_type}_{index}"


def _make_cross_page_node_id(source_name: str, from_page: int, to_page: int) -> str:
    return f"{_source_tag(source_name)}_page{from_page}_to_page{to_page}_cross_text_0"


def _ensure_text(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _normalize_bbox(value: Any) -> list[float] | None:
    coords: list[float] | None = None

    if isinstance(value, (list, tuple)) and len(value) == 4:
        try:
            coords = [float(value[0]), float(value[1]), float(value[2]), float(value[3])]
        except (TypeError, ValueError):
            coords = None
    elif isinstance(value, dict):
        keys = ["x1", "y1", "x2", "y2"]
        if all(k in value for k in keys):
            try:
                coords = [float(value["x1"]), float(value["y1"]), float(value["x2"]), float(value["y2"])]
            except (TypeError, ValueError):
                coords = None

    if not coords:
        return None

    x1, y1, x2, y2 = coords
    x1 = max(0.0, min(1.0, x1))
    y1 = max(0.0, min(1.0, y1))
    x2 = max(0.0, min(1.0, x2))
    y2 = max(0.0, min(1.0, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _extract_content_bbox(item: Any, preferred_keys: list[str] | None = None) -> tuple[str, list[float] | None]:
    if preferred_keys is None:
        preferred_keys = ["content", "text", "markdown", "description"]

    if isinstance(item, str):
        return item, None
    if isinstance(item, dict):
        text_value = ""
        for key in preferred_keys:
            if key in item:
                text_value = _ensure_text(item.get(key, ""))
                if text_value:
                    break
        if not text_value:
            text_value = _ensure_text(item.get("value", ""))
        bbox = _normalize_bbox(item.get("bbox"))
        return text_value, bbox
    return "", None


def _normalize_term_type(value: Any) -> str:
    raw = _ensure_text(value).strip().lower()
    if not raw:
        return "other"
    if raw in _ALLOWED_TYPES:
        return raw
    return _TYPE_ALIASES.get(raw, "other")


def _normalize_entities(raw_entities: Any) -> list[dict[str, str]]:
    entities: list[dict[str, str]] = []
    seen = set()

    if not isinstance(raw_entities, list):
        return entities

    for item in raw_entities:
        if isinstance(item, str):
            name = item.strip()
            etype = "other"
        elif isinstance(item, dict):
            name = _ensure_text(item.get("name", "")).strip()
            etype = _normalize_term_type(item.get("type", ""))
        else:
            continue

        if not name:
            continue
        key = (name.casefold(), etype)
        if key in seen:
            continue
        seen.add(key)
        entities.append({"name": name, "type": etype})

    return entities


_KEYWORD_STOPWORDS = {
    "the",
    "a",
    "an",
    "this",
    "that",
    "these",
    "those",
    "and",
    "or",
    "but",
    "if",
    "then",
    "for",
    "of",
    "to",
    "in",
    "on",
    "with",
    "by",
    "from",
    "as",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "it",
    "its",
    "we",
    "our",
    "you",
    "your",
    "they",
    "their",
    "paper",
    "section",
    "figure",
    "table",
    "chapter",
    "introduction",
    "conclusion",
    "related work",
    "method",
    "results",
    "discussion",
    "本文",
    "我们",
    "你们",
    "他们",
    "本章",
    "该章",
    "本节",
    "该节",
    "图",
    "表",
    "章节",
    "实验结果",
    "结论",
}


def _clean_keyword_term(term: str) -> str:
    cleaned = re.sub(r"\s+", " ", term or "").strip()
    cleaned = cleaned.strip("，,。.;:：!?！？()（）[]【】{}<>\"'`")
    return cleaned


def _is_valid_keyword(term: str, keyword_type: str) -> bool:
    t = _clean_keyword_term(term)
    if not t:
        return False
    if len(t) < 2 or len(t) > 40:
        return False

    lower = t.casefold()
    if lower in _KEYWORD_STOPWORDS:
        return False

    # Drop pure numbers / codes / punctuation-like fragments.
    if re.fullmatch(r"[\d\W_]+", t):
        return False
    if re.fullmatch(r"[a-zA-Z]\d*", t):
        return False

    # For unknown type, be stricter to avoid noisy words.
    if keyword_type == "other":
        if len(re.findall(r"[A-Za-z\u4e00-\u9fff]", t)) < 3:
            return False

    # Avoid overly long free text phrases as keywords.
    if len(t.split()) > 5:
        return False

    return True


def _normalize_keywords(raw_keywords: Any) -> list[dict[str, str]]:
    keywords: list[dict[str, str]] = []
    seen = set()

    if isinstance(raw_keywords, list):
        for item in raw_keywords:
            if isinstance(item, str):
                term = item.strip()
                ktype = "concept"
            elif isinstance(item, dict):
                term = _ensure_text(item.get("term", "")).strip()
                if not term:
                    term = _ensure_text(item.get("name", "")).strip()
                ktype = _normalize_term_type(item.get("type", "concept"))
            else:
                continue

            term = _clean_keyword_term(term)
            if not term:
                continue
            if not _is_valid_keyword(term, ktype):
                continue
            token = term.casefold()
            if token in seen:
                continue
            seen.add(token)
            keywords.append({"term": term, "type": ktype})

    # Keep the page-level keyword set bounded and focused.
    # Prioritize typed semantic keywords over generic "other".
    keywords.sort(key=lambda x: 0 if x.get("type") != "other" else 1)
    return keywords[:12]


def _build_document_keywords(graph_pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = {}

    for page in graph_pages:
        page_num = int(page.get("page", 0))
        for kw in page.get("keywords", []):
            if not isinstance(kw, dict):
                continue
            term = _ensure_text(kw.get("term", "")).strip()
            if not term:
                continue
            token = term.casefold()
            entry = stats.setdefault(
                token,
                {
                    "term": term,
                    "type": _normalize_term_type(kw.get("type", "concept")),
                    "count": 0,
                    "pages": set(),
                },
            )
            entry["count"] += 1
            if page_num > 0:
                entry["pages"].add(page_num)

    ranked = sorted(
        stats.values(),
        key=lambda x: (int(x.get("count", 0)), len(x.get("pages", set()))),
        reverse=True,
    )

    result: list[dict[str, Any]] = []
    for item in ranked[:120]:
        result.append(
            {
                "term": item["term"],
                "type": item["type"],
                "count": int(item["count"]),
                "pages": sorted(item["pages"]),
            }
        )
    return result


def _structural_count(parsed: dict[str, Any]) -> int:
    return (
        len(parsed.get("texts", []))
        + len(parsed.get("tables", []))
        + len(parsed.get("figures", []))
        + len(parsed.get("section_conclusions", []))
    )


def _is_suspicious_parse(parsed: dict[str, Any]) -> bool:
    text_n = len(parsed.get("texts", []))
    struct_n = _structural_count(parsed)
    ent_n = len(parsed.get("entities", []))
    kw_n = len(parsed.get("keywords", []))
    rel_n = len(parsed.get("relations", []))
    signal_n = ent_n + kw_n

    # Case A: fully empty schema on a non-empty page is usually a model miss.
    if struct_n == 0 and ent_n == 0 and kw_n == 0 and rel_n == 0:
        return True

    # Case B: large entity list but zero textual structure is usually extraction drift.
    if struct_n == 0 and signal_n >= 12:
        return True

    # Case C: semantically rich page but no explicit text chunks.
    if text_n == 0 and signal_n >= 10:
        return True

    return False


def _merge_parsed_page(primary: dict[str, Any], fallback: dict[str, Any]) -> dict[str, Any]:
    merged = dict(primary)
    for key in ("texts", "tables", "figures", "section_conclusions", "entities", "keywords", "relations"):
        primary_items = primary.get(key, [])
        fallback_items = fallback.get(key, [])

        if not isinstance(primary_items, list):
            primary_items = []
        if not isinstance(fallback_items, list):
            fallback_items = []

        merged_items = list(primary_items)
        seen = {json.dumps(item, ensure_ascii=False, sort_keys=True) for item in merged_items}
        for item in fallback_items:
            marker = json.dumps(item, ensure_ascii=False, sort_keys=True)
            if marker not in seen:
                merged_items.append(item)
                seen.add(marker)
        merged[key] = merged_items
    return ensure_page_schema(merged)


def _parse_score(parsed: dict[str, Any]) -> tuple[int, int, int, int]:
    text_n = len(parsed.get("texts", []))
    struct_n = _structural_count(parsed)
    signal_n = len(parsed.get("entities", [])) + len(parsed.get("keywords", []))
    rel_n = len(parsed.get("relations", []))
    return (text_n, struct_n, signal_n, rel_n)


def _coerce_single_block_parse(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        return {}

    known_keys = {
        "texts",
        "tables",
        "figures",
        "section_conclusions",
        "entities",
        "keywords",
        "relations",
    }
    if any(key in data for key in known_keys):
        return data

    content, bbox = _extract_content_bbox(data, preferred_keys=["content", "text", "markdown", "description"])
    if not content:
        return data
    block: dict[str, Any] = {"content": content}
    if bbox:
        block["bbox"] = bbox
    return {"texts": [block]}


def _text_only_pass(vl_client: VLClient, image_path: Path) -> list[Any]:
    raw = vl_client.extract_structured_page(PARSE_PROMPT_TEXT_ONLY, image_path)
    data = _coerce_single_block_parse(extract_json_object(raw))
    texts = data.get("texts", [])
    return texts if isinstance(texts, list) else []


def _parse_page_with_retry(
    vl_client: VLClient,
    image_path: Path,
    page_num: int,
) -> dict[str, Any]:
    prompts = [PARSE_PROMPT, PARSE_PROMPT_REPAIR][:MAX_PARSE_ATTEMPTS]
    candidates: list[dict[str, Any]] = []

    for idx, prompt in enumerate(prompts, start=1):
        if idx > 1:
            print(
                "      Sparse parse on page "
                f"{page_num}, retrying extraction pass {idx}/{len(prompts)}..."
            )
        raw_text = vl_client.extract_structured_page(prompt, image_path)
        parsed = ensure_page_schema(_coerce_single_block_parse(extract_json_object(raw_text)))
        candidates.append(parsed)
        if not _is_suspicious_parse(parsed):
            return parsed

    best = max(candidates, key=_parse_score)
    if len(best.get("texts", [])) == 0:
        try:
            extra_texts = _text_only_pass(vl_client, image_path)
        except Exception:
            extra_texts = []
        if extra_texts:
            boosted = dict(best)
            boosted["texts"] = extra_texts
            best = ensure_page_schema(boosted)
            print(
                "      Injected text-only extraction on page "
                f"{page_num}: +{len(extra_texts)} text blocks"
            )
    print(
        "      Page "
        f"{page_num} remains sparse after retries; using best candidate "
        f"(score={_parse_score(best)})"
    )
    return best


def parse_images_to_documents(
    image_paths: list[Path],
    source_name: str,
    vl_client: VLClient,
) -> tuple[list[Document], dict[str, Any]]:
    documents: list[Document] = []
    graph_pages: list[dict[str, Any]] = []
    page_text_records: list[dict[str, Any]] = []

    total_pages = len(image_paths)
    print(f"    Parsing {total_pages} pages...")

    for page_num, image_path in enumerate(image_paths, start=1):
        print(f"    Parsing page {page_num}/{total_pages}: {image_path.name}")
        parsed = _parse_page_with_retry(vl_client, image_path, page_num)

        normalized_keywords = _normalize_keywords(parsed.get("keywords", []))

        page_node_ids: list[str] = []
        figure_id_map: dict[str, str] = {}
        conclusion_node_ids: list[str] = []
        text_node_ids: list[str] = []
        text_items: list[dict[str, Any]] = []

        for idx, text in enumerate(parsed["texts"]):
            node_id = _make_node_id(source_name, page_num, "text", idx)
            text_value, text_bbox = _extract_content_bbox(text, preferred_keys=["content", "text"])
            if not text_value:
                continue
            metadata = {
                "source": source_name,
                "page": page_num,
                "image_path": str(image_path),
                "type": "text",
                "node_id": node_id,
            }
            if text_bbox:
                metadata["bbox"] = text_bbox
            documents.append(Document(page_content=text_value, metadata=metadata))
            page_node_ids.append(node_id)
            text_node_ids.append(node_id)
            text_payload: dict[str, Any] = {
                "node_id": node_id,
                "content": text_value,
                "image_path": str(image_path),
            }
            if text_bbox:
                text_payload["bbox"] = text_bbox
            text_items.append(text_payload)

        for idx, table in enumerate(parsed["tables"]):
            node_id = _make_node_id(source_name, page_num, "table", idx)
            table_value, table_bbox = _extract_content_bbox(table, preferred_keys=["content", "markdown", "text"])
            if not table_value:
                continue
            metadata = {
                "source": source_name,
                "page": page_num,
                "image_path": str(image_path),
                "type": "table",
                "node_id": node_id,
            }
            if table_bbox:
                metadata["bbox"] = table_bbox
            documents.append(Document(page_content=table_value, metadata=metadata))
            page_node_ids.append(node_id)

        for idx, figure in enumerate(parsed["figures"]):
            fig = figure if isinstance(figure, dict) else {}
            fig_id = _ensure_text(fig.get("fig_id", f"figure_{idx + 1}")) or f"figure_{idx + 1}"
            description, fig_bbox = _extract_content_bbox(fig, preferred_keys=["description", "content", "text"])
            if not description:
                continue
            node_id = _make_node_id(source_name, page_num, "figure", idx)
            metadata = {
                "source": source_name,
                "page": page_num,
                "image_path": str(image_path),
                "type": "figure",
                "fig_id": fig_id,
                "node_id": node_id,
            }
            if fig_bbox:
                metadata["bbox"] = fig_bbox
            documents.append(Document(page_content=description, metadata=metadata))
            page_node_ids.append(node_id)
            figure_id_map[fig_id] = node_id

        for idx, conclusion in enumerate(parsed["section_conclusions"]):
            node_id = _make_node_id(source_name, page_num, "conclusion", idx)
            conclusion_value, conclusion_bbox = _extract_content_bbox(conclusion, preferred_keys=["content", "text"])
            if not conclusion_value:
                continue
            metadata = {
                "source": source_name,
                "page": page_num,
                "image_path": str(image_path),
                "type": "conclusion",
                "node_id": node_id,
            }
            if conclusion_bbox:
                metadata["bbox"] = conclusion_bbox
            documents.append(Document(page_content=conclusion_value, metadata=metadata))
            page_node_ids.append(node_id)
            conclusion_node_ids.append(node_id)

        graph_pages.append(
                {
                    "source": source_name,
                    "page": page_num,
                    "image_path": str(image_path),
                    # Entity nodes are disabled by design; keep empty for compatibility.
                    "entities": [],
                    "keywords": normalized_keywords,
                "relations": parsed["relations"],
                "section_conclusions": parsed["section_conclusions"],
                "figure_id_map": figure_id_map,
                "node_ids": page_node_ids,
                "text_node_ids": text_node_ids,
                "conclusion_node_ids": conclusion_node_ids,
            }
        )
        page_text_records.append({"page": page_num, "image_path": str(image_path), "texts": text_items})

    cross_page_links: list[dict[str, Any]] = []
    for idx in range(len(page_text_records) - 1):
        left = page_text_records[idx]
        right = page_text_records[idx + 1]
        left_texts = left.get("texts", [])
        right_texts = right.get("texts", [])
        if not left_texts or not right_texts:
            continue

        left_tail = left_texts[-1]
        right_head = right_texts[0]
        left_content = str(left_tail.get("content", "")).strip()
        right_content = str(right_head.get("content", "")).strip()
        if not left_content or not right_content:
            continue

        from_page = int(left.get("page", 0))
        to_page = int(right.get("page", 0))
        cross_node_id = _make_cross_page_node_id(source_name, from_page, to_page)
        merged_content = f"{left_content}\n{right_content}"

        documents.append(
            Document(
                page_content=merged_content,
                metadata={
                    "source": source_name,
                    "page": from_page,
                    "page_span": [from_page, to_page],
                    "image_path": str(left.get("image_path", "")),
                    "image_paths": [str(left.get("image_path", "")), str(right.get("image_path", ""))],
                    "type": "cross_page_text",
                    "node_id": cross_node_id,
                    "from_node_id": str(left_tail.get("node_id", "")),
                    "to_node_id": str(right_head.get("node_id", "")),
                },
            )
        )

        # Attach cross-page chunk to both pages for visualization and traceability.
        graph_pages[idx]["node_ids"].append(cross_node_id)
        graph_pages[idx + 1]["node_ids"].append(cross_node_id)

        cross_page_links.append(
            {
                "source": source_name,
                "from_page": from_page,
                "to_page": to_page,
                "from_node_id": str(left_tail.get("node_id", "")),
                "to_node_id": str(right_head.get("node_id", "")),
                "cross_node_id": cross_node_id,
            }
        )

    graph_data = {
        "source": source_name,
        "pages": graph_pages,
        "cross_page_links": cross_page_links,
        "document_keywords": _build_document_keywords(graph_pages),
    }

    text_count = len([d for d in documents if d.metadata.get("type") == "text"])
    table_count = len([d for d in documents if d.metadata.get("type") == "table"])
    figure_count = len([d for d in documents if d.metadata.get("type") == "figure"])
    conclusion_count = len([d for d in documents if d.metadata.get("type") == "conclusion"])
    cross_page_count = len([d for d in documents if d.metadata.get("type") == "cross_page_text"])
    keyword_total = sum(len(page.get("keywords", [])) for page in graph_pages)

    print("    Parse summary:")
    print(f"      text chunks: {text_count}")
    print(f"      tables: {table_count}")
    print(f"      figures: {figure_count}")
    print(f"      conclusions: {conclusion_count}")
    print(f"      cross-page merges: {cross_page_count}")
    print(f"      keywords: {keyword_total}")

    return documents, graph_data


def save_parsing_outputs(
    documents: list[Document],
    graph_data: dict[str, Any],
    parsed_dir: str | Path,
) -> tuple[Path, Path]:
    parsed_output = Path(parsed_dir)
    parsed_output.mkdir(parents=True, exist_ok=True)

    print(f"    Saving parsing outputs to: {parsed_output}")

    docs_payload = [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in documents]

    docs_path = parsed_output / "documents.json"
    graph_path = parsed_output / "graph_data.json"

    docs_path.write_text(json.dumps(docs_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    graph_path.write_text(json.dumps(graph_data, ensure_ascii=False, indent=2), encoding="utf-8")

    print("    Saved documents.json and graph_data.json")
    return docs_path, graph_path
