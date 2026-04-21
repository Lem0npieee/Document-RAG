from __future__ import annotations

import json
from typing import Any


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract the first valid JSON object from a model response string."""
    decoder = json.JSONDecoder()
    for start in (idx for idx, ch in enumerate(text) if ch == "{"):
        try:
            obj, _ = decoder.raw_decode(text[start:])
            if isinstance(obj, dict):
                print(f"      成功提取JSON对象，键: {list(obj.keys())}")
                return obj
        except json.JSONDecodeError:
            continue
    raise ValueError("No valid JSON object found in model response.")


def ensure_page_schema(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize parser output to a strict schema."""
    result = {
        "texts": raw.get("texts", []) if isinstance(raw.get("texts", []), list) else [],
        "tables": raw.get("tables", []) if isinstance(raw.get("tables", []), list) else [],
        "figures": raw.get("figures", []) if isinstance(raw.get("figures", []), list) else [],
        "section_conclusions": raw.get("section_conclusions", [])
        if isinstance(raw.get("section_conclusions", []), list)
        else [],
        "entities": raw.get("entities", []) if isinstance(raw.get("entities", []), list) else [],
        "relations": raw.get("relations", []) if isinstance(raw.get("relations", []), list) else [],
    }

    # 打印统计信息
    print(f"      规范化数据统计:")
    print(f"        文本段落: {len(result['texts'])} 个")
    print(f"        表格: {len(result['tables'])} 个")
    print(f"        图表: {len(result['figures'])} 个")
    print(f"        结论: {len(result['section_conclusions'])} 个")
    print(f"        实体: {len(result['entities'])} 个")
    print(f"        关系: {len(result['relations'])} 个")

    return result
