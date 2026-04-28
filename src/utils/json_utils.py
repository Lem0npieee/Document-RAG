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
                print(f"      Parsed JSON object keys: {list(obj.keys())}")
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
        "keywords": raw.get("keywords", []) if isinstance(raw.get("keywords", []), list) else [],
        "relations": raw.get("relations", []) if isinstance(raw.get("relations", []), list) else [],
    }

    print("      Normalized page schema stats:")
    print(f"        texts: {len(result['texts'])}")
    print(f"        tables: {len(result['tables'])}")
    print(f"        figures: {len(result['figures'])}")
    print(f"        section_conclusions: {len(result['section_conclusions'])}")
    print(f"        entities: {len(result['entities'])}")
    print(f"        keywords: {len(result['keywords'])}")
    print(f"        relations: {len(result['relations'])}")

    return result
