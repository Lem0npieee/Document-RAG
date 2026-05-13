from __future__ import annotations

import json
from typing import Any


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract the first valid JSON object from a model response string.

    When no valid JSON is found, wraps the raw text as a minimal text-only
    document so that parsing never fails completely. This is essential for
    smaller/quantized local models that may occasionally output non-JSON.
    """
    decoder = json.JSONDecoder()
    for start in (idx for idx, ch in enumerate(text) if ch == "{"):
        try:
            obj, _ = decoder.raw_decode(text[start:])
            if isinstance(obj, dict):
                print(f"      Parsed JSON object keys: {list(obj.keys())}")
                return obj
        except json.JSONDecodeError:
            continue

    # Fallback: wrap raw text into minimal text-only structure
    cleaned = text.strip()
    if cleaned:
        # Try to strip markdown code fences
        import re
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = cleaned.strip()

    if cleaned and len(cleaned) >= 4:
        print(f"      No JSON found, wrapping raw text ({len(cleaned)} chars) as fallback")
        return {"texts": [{"content": cleaned}]}

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
