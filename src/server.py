from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from build_multimodal_graphrag import build_knowledge_base
from src.config import get_settings
from src.rag.multimodal_graph_rag_chain import MultiModalGraphRAG


WEB_DIR = APP_ROOT / "web"
OUTPUTS_DIR = APP_ROOT / "outputs"
_doc_root_env = os.getenv("DOC_ROOT", "doc")
_doc_root_path = Path(_doc_root_env)
DOC_DIR = _doc_root_path if _doc_root_path.is_absolute() else APP_ROOT / _doc_root_path
UPLOAD_DIR = DOC_DIR
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Disable Flask built-in static route to avoid shadowing custom routes
# such as /outputs/<path:filename>.
app = Flask(__name__, static_folder=None)

_chain_cache: MultiModalGraphRAG | None = None
_chain_mtime: float | None = None
ALLOWED_UPLOAD_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".webp"}


def _safe_upload_name(raw_name: str) -> str:
    """Generate a filesystem-safe upload name even for non-ASCII filenames."""
    safe = secure_filename(raw_name)
    if safe:
        return safe

    suffix = Path(raw_name).suffix.lower()
    if suffix not in ALLOWED_UPLOAD_SUFFIXES:
        suffix = ""
    return f"upload_{uuid.uuid4().hex}{suffix}"


def _load_chain() -> MultiModalGraphRAG:
    global _chain_cache, _chain_mtime
    settings = get_settings()
    graph_path = settings.graph_dir / "graph.pkl"

    if not graph_path.exists():
        raise FileNotFoundError("graph.pkl not found. Please ingest a document first.")

    mtime = graph_path.stat().st_mtime
    if _chain_cache is None or _chain_mtime != mtime:
        _chain_cache = MultiModalGraphRAG(
            settings=settings,
            faiss_dir=settings.faiss_dir,
            graph_path=graph_path,
            pages_dir=settings.pages_dir,
        )
        _chain_mtime = mtime

    return _chain_cache


def _graph_data_path() -> Path:
    settings = get_settings()
    return settings.parsed_dir / "graph_data.json"


def _documents_path() -> Path:
    settings = get_settings()
    return settings.parsed_dir / "documents.json"


def _to_outputs_url(image_path: str) -> str:
    """Convert an image path to a web URL served by /outputs."""
    raw = str(image_path or "").replace("\\", "/").strip()
    if not raw:
        return ""

    lowered = raw.lower()
    marker = "outputs/"
    idx = lowered.find(marker)
    if idx != -1:
        relative = raw[idx + len(marker):].lstrip("/")
        return f"/outputs/{relative}"

    if raw.startswith("pages/"):
        return f"/outputs/{raw}"
    return ""


@app.get("/")
def index() -> Any:
    return send_from_directory(str(WEB_DIR), "index.html")


@app.get("/favicon.ico")
def favicon() -> Any:
    favicon_path = WEB_DIR / "favicon.ico"
    if favicon_path.exists():
        return send_from_directory(str(WEB_DIR), "favicon.ico")
    return ("", 204)


@app.get("/health")
def health() -> Any:
    return jsonify({"ok": True})


@app.get("/graph")
def graph_data() -> Any:
    path = _graph_data_path()
    if not path.exists():
        return jsonify({"documents": []})

    payload = json.loads(path.read_text(encoding="utf-8"))
    docs_path = _documents_path()
    node_map: dict[str, dict[str, Any]] = {}

    if docs_path.exists():
        docs = json.loads(docs_path.read_text(encoding="utf-8"))
        for item in docs:
            metadata = item.get("metadata", {}) if isinstance(item, dict) else {}
            node_id = str(metadata.get("node_id", "")).strip()
            if not node_id:
                continue

            image_path = str(metadata.get("image_path", "")).strip()
            node_map[node_id] = {
                "id": node_id,
                "type": str(metadata.get("type", "text")),
                "source": str(metadata.get("source", "")),
                "page": metadata.get("page"),
                "page_span": metadata.get("page_span", []),
                "fig_id": str(metadata.get("fig_id", "")),
                "content": str(item.get("page_content", "")),
                "bbox": metadata.get("bbox"),
                "image_path": image_path,
                "image_paths": metadata.get("image_paths", []),
                "image_url": _to_outputs_url(image_path),
            }

    documents = payload.get("documents", []) if isinstance(payload, dict) else []
    for doc in documents:
        pages = doc.get("pages", []) if isinstance(doc, dict) else []
        for page in pages:
            if not isinstance(page, dict):
                continue
            image_path = str(page.get("image_path", "")).strip()
            page["image_url"] = _to_outputs_url(image_path)

    if isinstance(payload, dict):
        payload["node_map"] = node_map
    else:
        payload = {"documents": [], "node_map": node_map}

    return jsonify(payload)


@app.post("/ingest")
def ingest() -> Any:
    if "file" not in request.files:
        return jsonify({"error": "file is required"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "empty filename"}), 400

    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_UPLOAD_SUFFIXES:
        return (
            jsonify({"error": "unsupported file type, only pdf/png/jpg/jpeg/webp are allowed"}),
            400,
        )

    filename = _safe_upload_name(file.filename)
    dest = UPLOAD_DIR / filename

    try:
        file.save(dest)
        result = build_knowledge_base(str(dest), force_rebuild=False)
        return jsonify({"ok": True, "result": result})
    except Exception as exc:
        return jsonify({"error": f"Ingest failed: {exc.__class__.__name__}: {exc}"}), 500


@app.post("/chat")
def chat() -> Any:
    payload = request.get_json(silent=True) or {}
    question = str(payload.get("question", "")).strip()
    if not question:
        return jsonify({"error": "question is required"}), 400

    try:
        chain = _load_chain()
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 400

    result = chain.ask(question)
    return jsonify(
        {
            "answer": result.answer,
            "pages": result.pages,
            "node_ids": result.node_ids,
            "relations": result.relations,
            "image_paths": result.image_paths,
        }
    )


@app.get("/outputs/<path:filename>")
def output_files(filename: str) -> Any:
    return send_from_directory(str(OUTPUTS_DIR), filename)


@app.get("/<path:filename>")
def static_files(filename: str) -> Any:
    return send_from_directory(str(WEB_DIR), filename)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=False)
