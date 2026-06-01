from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import re
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib import error, request

import dashscope
from dashscope import MultiModalConversation

from loader import PDFQASample, load_pdfqa_samples
from metrics import normalize_text


@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    image_tokens: int = 0

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "TokenUsage":
        usage = _find_usage_dict(payload)
        input_tokens = _first_int(usage, ("input_tokens", "prompt_tokens", "inputTokenCount"))
        output_tokens = _first_int(usage, ("output_tokens", "completion_tokens", "outputTokenCount"))
        total_tokens = _first_int(usage, ("total_tokens", "totalTokens", "totalTokenCount"))
        image_tokens = _first_int(usage, ("image_tokens", "imageTokens", "vision_tokens"))
        if not total_tokens and (input_tokens or output_tokens):
            total_tokens = input_tokens + output_tokens
        return cls(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            image_tokens=image_tokens,
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "image_tokens": self.image_tokens,
        }


def _default_paths() -> dict[str, Path]:
    eval_root = Path(__file__).resolve().parents[1]
    output_root = eval_root / "output" / "pdfqa"
    return {
        "annotations_root": eval_root / "input" / "pdfqa" / "annotations",
        "pdfs_root": eval_root / "input" / "pdfqa" / "pdfs",
        "kb_root": eval_root / "output" / "kb",
        "output_root": output_root,
        "records": output_root / "token_eval.jsonl",
        "metrics": output_root / "token_eval_metrics.json",
    }


def _configure_environment(kb_root: Path, doc_root: Path) -> None:
    os.environ["OUTPUT_ROOT"] = str(kb_root)
    os.environ["DOC_ROOT"] = str(doc_root)


def _image_to_data_uri(image_path: str | Path) -> str:
    path = Path(image_path)
    suffix = path.suffix.lower().replace(".", "") or "png"
    raw = path.read_bytes()
    encoded = base64.b64encode(raw).decode("utf-8")
    return f"data:image/{suffix};base64,{encoded}"


def _normalize_endpoint_base(value: str) -> str:
    base = str(value or "").strip().rstrip("/")
    if not base:
        return "https://dashscope.aliyuncs.com/compatible-mode/v1"
    if base.endswith("/chat/completions"):
        base = base[: -len("/chat/completions")]
    if base.endswith("/v1"):
        return base
    return f"{base}/compatible-mode/v1"


def _clean_model_text(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", str(text), flags=re.DOTALL | re.IGNORECASE)
    return cleaned.strip()


def _as_payload(response: Any) -> dict[str, Any]:
    try:
        payload = response.to_dict()
    except (AttributeError, KeyError):
        payload = response if isinstance(response, dict) else dict(response)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Unexpected response payload: {payload}")
    return payload


def _extract_dashscope_text(payload: dict[str, Any]) -> str:
    output = payload.get("output", {})
    choices = output.get("choices", []) if isinstance(output, dict) else []
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message", {})
        content = message.get("content", []) if isinstance(message, dict) else []
        if isinstance(content, str) and content.strip():
            return _clean_model_text(content)
        if isinstance(content, list):
            parts = [str(x.get("text", "")) for x in content if isinstance(x, dict) and x.get("text")]
            if parts:
                return _clean_model_text("\n".join(parts))
    raise RuntimeError(f"Unexpected DashScope response format: {payload}")


def _extract_openai_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices", [])
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message", {})
        content = message.get("content", "") if isinstance(message, dict) else ""
        if isinstance(content, str) and content.strip():
            return _clean_model_text(content)
        if isinstance(content, list):
            parts = [str(x.get("text", "")) for x in content if isinstance(x, dict) and x.get("text")]
            if parts:
                return _clean_model_text("\n".join(parts))
    output_text = payload.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return _clean_model_text(output_text)
    raise RuntimeError(f"Unexpected OpenAI-compatible response format: {payload}")


def _find_usage_dict(payload: dict[str, Any]) -> dict[str, Any]:
    if isinstance(payload.get("usage"), dict):
        return payload["usage"]
    output = payload.get("output")
    if isinstance(output, dict) and isinstance(output.get("usage"), dict):
        return output["usage"]
    return {}


def _first_int(payload: dict[str, Any], keys: tuple[str, ...]) -> int:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str) and value.isdigit():
            return int(value)
    return 0


class UsageTrackingVLClient:
    def __init__(self, settings: Any) -> None:
        self.settings = settings
        self.provider = settings.model_provider
        self.calls: list[dict[str, Any]] = []

    def _record(self, payload: dict[str, Any], answer: str, image_count: int, label: str) -> None:
        self.calls.append(
            {
                "label": label,
                "answer": answer,
                "image_count": image_count,
                "usage": TokenUsage.from_payload(payload).to_dict(),
                "raw_usage": _find_usage_dict(payload),
            }
        )

    def pop_last_call(self) -> dict[str, Any]:
        if not self.calls:
            return {"usage": TokenUsage().to_dict(), "raw_usage": {}, "answer": "", "image_count": 0}
        return self.calls.pop()

    def answer_question(self, prompt: str, image_paths: list[str | Path]) -> str:
        if self.provider != "dashscope":
            raise ValueError("run_token_eval.py only supports DashScope API for VLM calls.")
        return self._call_dashscope(prompt, image_paths, label="answer_question")

    def extract_structured_page(self, prompt: str, image_path: str | Path) -> str:
        return self.answer_question(prompt=prompt, image_paths=[image_path])

    def _call_dashscope(self, prompt: str, image_paths: list[str | Path], label: str) -> str:
        dashscope.api_key = self.settings.dashscope_api_key
        user_content: list[dict[str, str]] = []
        for image_path in image_paths:
            user_content.append({"image": _image_to_data_uri(image_path)})
        user_content.append({"text": prompt})

        max_attempts = 4
        for attempt in range(1, max_attempts + 1):
            try:
                response = MultiModalConversation.call(
                    model=self.settings.vl_model,
                    messages=[
                        {"role": "system", "content": [{"text": "You are a helpful multimodal assistant."}]},
                        {"role": "user", "content": user_content},
                    ],
                    result_format="message",
                )
                payload = _as_payload(response)
                answer = _extract_dashscope_text(payload)
                self._record(payload, answer, image_count=len(image_paths), label=label)
                return answer
            except Exception as exc:
                if attempt >= max_attempts or not _is_transient_error(exc):
                    raise
                time.sleep(min(6.0, 1.2 * attempt))
        raise RuntimeError("DashScope call failed after retries.")

class DocumentUploadClient:
    """Direct document-upload baseline using DashScope's OpenAI-compatible files API."""

    def __init__(self, api_key: str, model: str, base_url: str) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = _normalize_endpoint_base(base_url)
        self.file_ids: dict[str, str] = {}

    def upload(self, path: Path) -> str:
        resolved = str(path.resolve())
        if resolved in self.file_ids:
            return self.file_ids[resolved]

        boundary = f"----DocRAGTokenEval{int(time.time() * 1000)}"
        body = _multipart_body(
            boundary=boundary,
            fields={"purpose": "file-extract"},
            files={"file": path},
        )
        req = request.Request(
            f"{self.base_url}/files",
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": f"multipart/form-data; boundary={boundary}",
            },
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=300) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Document upload failed: HTTP {exc.code}: {raw[:500]}") from exc

        file_id = str(payload.get("id") or payload.get("file_id") or "").strip()
        if not file_id:
            raise RuntimeError(f"Document upload response did not include file id: {payload}")
        self.file_ids[resolved] = file_id
        return file_id

    def answer_question(self, question: str, doc_paths: list[Path]) -> tuple[str, dict[str, Any]]:
        file_ids = [self.upload(path) for path in doc_paths]
        messages = [{"role": "system", "content": "You are a helpful document question-answering assistant."}]
        for file_id in file_ids:
            messages.append({"role": "system", "content": f"fileid://{file_id}"})
        messages.append({"role": "user", "content": _short_answer_prompt(question)})

        body = json.dumps(
            {
                "model": self.model,
                "messages": messages,
                "temperature": 0,
            }
        ).encode("utf-8")
        req = request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=300) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Document QA call failed: HTTP {exc.code}: {raw[:500]}") from exc

        answer = _extract_openai_text(payload)
        return answer, {
            "file_ids": file_ids,
            "usage": TokenUsage.from_payload(payload).to_dict(),
            "raw_usage": _find_usage_dict(payload),
        }


def _multipart_body(boundary: str, fields: dict[str, str], files: dict[str, Path]) -> bytes:
    chunks: list[bytes] = []
    for name, value in fields.items():
        chunks.append(f"--{boundary}\r\n".encode("utf-8"))
        chunks.append(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"))
        chunks.append(str(value).encode("utf-8"))
        chunks.append(b"\r\n")

    for name, path in files.items():
        filename = path.name
        content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        chunks.append(f"--{boundary}\r\n".encode("utf-8"))
        chunks.append(
            (
                f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
                f"Content-Type: {content_type}\r\n\r\n"
            ).encode("utf-8")
        )
        chunks.append(path.read_bytes())
        chunks.append(b"\r\n")

    chunks.append(f"--{boundary}--\r\n".encode("utf-8"))
    return b"".join(chunks)


def _is_transient_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(
        marker in text
        for marker in (
            "ssl",
            "eof",
            "timed out",
            "timeout",
            "connection",
            "temporarily",
            "max retries exceeded",
            "remote end closed",
            "connection reset",
        )
    )


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _safe_write(path: Path, content: str, fallback_name: str) -> Path:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path
    except Exception:
        fallback = Path.cwd() / fallback_name
        fallback.write_text(content, encoding="utf-8")
        return fallback


def _append_jsonl(path: Path, item: dict[str, Any]) -> Path:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
        return path
    except Exception:
        fallback = Path.cwd() / path.name
        with fallback.open("a", encoding="utf-8") as f:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
        return fallback


def _load_existing(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            qid = str(item.get("question_id", "")).strip()
            if qid:
                rows[qid] = item
    return rows


def _all_pdf_docs(samples: list[PDFQASample]) -> list[Path]:
    docs: dict[str, Path] = {}
    for sample in samples:
        if sample.doc_path.exists():
            docs[str(sample.doc_path.resolve())] = sample.doc_path.resolve()
    return sorted(docs.values(), key=lambda p: p.name.lower())


def _baseline_doc_paths(sample: PDFQASample, all_docs: list[Path], scope: str) -> list[Path]:
    if scope == "all_docs":
        return all_docs
    if sample.doc_path.exists():
        return [sample.doc_path.resolve()]
    raise FileNotFoundError(f"PDF not found for {sample.source_hint}: {sample.doc_path}")


def _short_answer_prompt(question: str) -> str:
    return (
        "You are answering a document question. Use only the uploaded document pages.\n"
        "Return the shortest possible answer only.\n"
        "- For yes/no questions, output only yes or no. If the evidence is not explicit, answer no.\n"
        "- For numeric questions, output only the number or value.\n"
        "- For name/entity questions, output only the name.\n\n"
        f"Question: {question}\n\n"
        "Answer only:"
    )


def _usage_sum(records: list[dict[str, Any]], key: str) -> dict[str, int]:
    total = TokenUsage()
    for rec in records:
        usage = rec.get(key, {}).get("usage", {})
        total.input_tokens += int(usage.get("input_tokens", 0) or 0)
        total.output_tokens += int(usage.get("output_tokens", 0) or 0)
        total.total_tokens += int(usage.get("total_tokens", 0) or 0)
        total.image_tokens += int(usage.get("image_tokens", 0) or 0)
    return total.to_dict()


def _ratio_saved(docrag: int, full: int) -> float:
    if full <= 0:
        return 0.0
    return round(1.0 - (docrag / full), 6)


def _is_binary_answer(text: str) -> bool:
    return normalize_text(text) in {"yes", "no"}


def _keep_sample(sample: PDFQASample, profile: str) -> bool:
    if profile == "all":
        return True
    if not sample.answers:
        return False
    first = str(sample.answers[0] or "")
    if profile == "binary":
        return _is_binary_answer(first)
    if profile == "very_short":
        return len(first) <= 5
    if profile == "short":
        return len(first) <= 20
    return True


def _filter_samples_to_kb_docs(
    samples: list[PDFQASample],
    kb_root: Path,
) -> tuple[list[PDFQASample], list[PDFQASample]]:
    docs_json = kb_root / "parsed" / "documents.json"
    if not docs_json.exists():
        raise FileNotFoundError(f"--kb-docs-only requires KB documents file: {docs_json}")

    docs_list = json.loads(docs_json.read_text(encoding="utf-8"))
    kb_sources = {
        str(d.get("metadata", {}).get("source", ""))
        for d in docs_list
        if isinstance(d, dict)
    }
    kb_sources.discard("")
    if not kb_sources:
        raise ValueError(f"No KB source names found in {docs_json}")

    kept = [s for s in samples if s.doc_name in kb_sources]
    dropped = [s for s in samples if s.doc_name not in kb_sources]
    return kept, dropped


def parse_args() -> argparse.Namespace:
    defaults = _default_paths()
    parser = argparse.ArgumentParser(
        description="Compare VLM token usage between DocRAG and full-document upload."
    )
    parser.add_argument("--annotations-root", type=Path, default=defaults["annotations_root"])
    parser.add_argument("--pdfs-root", type=Path, default=defaults["pdfs_root"])
    parser.add_argument("--kb-root", type=Path, default=defaults["kb_root"])
    parser.add_argument("--output-root", type=Path, default=defaults["output_root"])
    parser.add_argument("--records-file", type=Path, default=defaults["records"])
    parser.add_argument("--metrics-file", type=Path, default=defaults["metrics"])
    parser.add_argument("--category", type=str, default="real", choices=["all", "real", "syn", "custom"])
    parser.add_argument("--qa-split", type=str, default="all", choices=["all", "raw", "vf", "cf"])
    parser.add_argument("--doc-name", type=str, default="", help="Only evaluate samples from this PDF file name or stem")
    parser.add_argument("--answer-profile", type=str, default="very_short", choices=["all", "binary", "very_short", "short"])
    parser.add_argument("--max-samples", type=int, default=20, help="0 means all samples")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--max-nodes", type=int, default=24)
    parser.add_argument("--mode", type=str, default="both", choices=["both", "docrag", "full_upload"])
    parser.add_argument("--full-upload-model", type=str, default="qwen-long-latest")
    parser.add_argument(
        "--full-upload-base-url",
        type=str,
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        help="OpenAI-compatible base URL used by the direct PDF upload baseline",
    )
    parser.add_argument(
        "--full-upload-scope",
        type=str,
        default="target_doc",
        choices=["target_doc", "all_docs"],
        help="target_doc uploads only the question's PDF; all_docs uploads every PDF in the selected sample set",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--strict-docs", action="store_true")
    parser.add_argument("--kb-docs-only", action="store_true", help="auto-exclude samples whose doc is not in KB")
    parser.add_argument("--log-every", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    _configure_environment(kb_root=args.kb_root, doc_root=args.pdfs_root)
    os.environ["DOCRAG_FORCE_DASHSCOPE_API"] = "1"

    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from src.config import get_settings
    from src.rag.multimodal_graph_rag_chain import MultiModalGraphRAG

    samples, warnings = load_pdfqa_samples(
        annotations_root=args.annotations_root,
        pdfs_root=args.pdfs_root,
        category=args.category,
        qa_split=args.qa_split,
        require_doc_exists=args.strict_docs,
        keep_unanswered=False,
        max_samples=0 if args.doc_name else args.max_samples,
    )
    if args.doc_name:
        requested = Path(args.doc_name).name.lower()
        requested_stem = Path(requested).stem.lower()
        samples = [
            s
            for s in samples
            if Path(s.source_hint).name.lower() == requested
            or Path(s.source_hint).stem.lower() == requested_stem
        ]
        if not samples:
            raise FileNotFoundError(f"No pdfQA samples matched --doc-name {args.doc_name!r}")
    samples = [s for s in samples if _keep_sample(s, args.answer_profile)]
    if args.doc_name and args.max_samples and args.max_samples > 0:
        samples = samples[: args.max_samples]
    kb_dropped_samples: list[PDFQASample] = []
    if args.kb_docs_only:
        before = len(samples)
        samples, kb_dropped_samples = _filter_samples_to_kb_docs(samples, args.kb_root)
        print(f"--kb-docs-only: dropped {before - len(samples)} samples (remaining {len(samples)})")
        for sample in kb_dropped_samples:
            print(f"  SKIP {sample.doc_name} - not in KB")

    settings = get_settings()
    settings.model_provider = "dashscope"
    settings.embedding_provider = "dashscope"
    settings.embedding_model = "text-embedding-v3"
    if settings.model_provider != "dashscope":
        raise ValueError("run_token_eval.py only supports DashScope API for VLM calls.")
    if settings.embedding_provider != "dashscope":
        raise ValueError("run_token_eval.py only supports DashScope API for embedding calls.")
    tracker = UsageTrackingVLClient(settings)
    upload_client = None
    all_docs = _all_pdf_docs(samples)
    if args.mode in {"both", "full_upload"}:
        if not settings.dashscope_api_key:
            raise ValueError("Direct document upload baseline requires DASHSCOPE_API_KEY.")
        upload_client = DocumentUploadClient(
            api_key=settings.dashscope_api_key,
            model=args.full_upload_model,
            base_url=args.full_upload_base_url,
        )

    chain = None
    if args.mode in {"both", "docrag"}:
        expected_faiss = args.kb_root / "faiss_index" / "index.faiss"
        expected_graph = args.kb_root / settings.graph_dirname / "graph.pkl"
        if not expected_faiss.exists() or not expected_graph.exists():
            raise FileNotFoundError(
                "KB artifacts not found. Build KB first. Missing: "
                f"{expected_faiss if not expected_faiss.exists() else expected_graph}"
            )
        chain = MultiModalGraphRAG(
            settings=settings,
            faiss_dir=settings.faiss_dir,
            graph_path=settings.graph_dir / "graph.pkl",
            pages_dir=settings.pages_dir,
        )
        chain.vl_client = tracker

    existing = _load_existing(args.records_file) if args.resume else {}
    records_written = args.records_file
    done = 0
    failed = 0

    print("============================================================")
    print("Token Evaluation")
    print(f"mode: {args.mode}")
    print(f"provider: {settings.model_provider}, model: {settings.active_vl_model}")
    if args.mode in {"both", "full_upload"}:
        print(f"full_upload_model: {args.full_upload_model}")
        print(f"full_upload_scope: {args.full_upload_scope}, pdf_docs: {len(all_docs)}")
    print(f"annotations_root: {args.annotations_root}")
    print(f"pdfs_root: {args.pdfs_root}")
    print(f"kb_root: {args.kb_root}")
    print(f"samples: {len(samples)}")
    if args.doc_name:
        print(f"doc_name: {args.doc_name}")
    print("============================================================")
    if warnings:
        print(f"warnings: {len(warnings)}")

    for idx, sample in enumerate(samples, start=1):
        if args.resume and sample.question_id in existing:
            done += 1
            continue

        record: dict[str, Any] = {
            "question_id": sample.question_id,
            "question": sample.question,
            "answers": sample.answers,
            "source_hint": sample.source_hint,
            "doc_name": sample.doc_name,
            "category": sample.category,
            "dataset": sample.dataset,
            "question_type": sample.question_type,
            "created_at": datetime.now().isoformat(),
            "docrag": None,
            "full_upload": None,
            "error": "",
        }

        try:
            if args.mode in {"both", "docrag"}:
                result = chain.ask_eval(
                    question=sample.question,
                    source_hint=sample.source_hint,
                    k=args.k,
                    max_nodes=args.max_nodes,
                )
                call = tracker.pop_last_call()
                record["docrag"] = {
                    "answer": result.answer,
                    "pages": result.pages,
                    "node_ids": result.node_ids,
                    "image_count": call.get("image_count", len(result.image_paths)),
                    "usage": call.get("usage", TokenUsage().to_dict()),
                    "raw_usage": call.get("raw_usage", {}),
                }

            if args.mode in {"both", "full_upload"}:
                doc_paths = _baseline_doc_paths(sample, all_docs=all_docs, scope=args.full_upload_scope)
                answer, call = upload_client.answer_question(
                    question=sample.question,
                    doc_paths=doc_paths,
                )
                record["full_upload"] = {
                    "answer": answer,
                    "doc_count": len(doc_paths),
                    "doc_names": [p.name for p in doc_paths],
                    "file_ids": call.get("file_ids", []),
                    "usage": call.get("usage", TokenUsage().to_dict()),
                    "raw_usage": call.get("raw_usage", {}),
                }

            if record["docrag"] and record["full_upload"]:
                d_usage = record["docrag"]["usage"]
                f_usage = record["full_upload"]["usage"]
                record["savings"] = {
                    "input_token_saved_ratio": _ratio_saved(d_usage.get("input_tokens", 0), f_usage.get("input_tokens", 0)),
                    "output_token_saved_ratio": _ratio_saved(d_usage.get("output_tokens", 0), f_usage.get("output_tokens", 0)),
                    "total_token_saved_ratio": _ratio_saved(d_usage.get("total_tokens", 0), f_usage.get("total_tokens", 0)),
                }

            records_written = _append_jsonl(records_written, record)
            existing[sample.question_id] = record
            done += 1
        except Exception as exc:
            failed += 1
            record["error"] = f"{exc.__class__.__name__}: {exc}"
            record["traceback"] = traceback.format_exc()
            records_written = _append_jsonl(records_written, record)
            existing[sample.question_id] = record

        if args.log_every > 0 and idx % args.log_every == 0:
            print(f"progress: {idx}/{len(samples)}, done={done}, failed={failed}")

    ordered_records = [existing[s.question_id] for s in samples if s.question_id in existing]
    docrag_records = [r for r in ordered_records if isinstance(r.get("docrag"), dict)]
    full_records = [r for r in ordered_records if isinstance(r.get("full_upload"), dict)]
    paired_records = [
        r
        for r in ordered_records
        if isinstance(r.get("docrag"), dict) and isinstance(r.get("full_upload"), dict)
    ]

    docrag_usage = _usage_sum(paired_records if paired_records else docrag_records, "docrag")
    full_usage = _usage_sum(paired_records if paired_records else full_records, "full_upload")

    summary = {
        "updated_at": datetime.now().isoformat(),
        "mode": args.mode,
        "provider": settings.model_provider,
        "model": settings.active_vl_model,
        "annotations_root": str(args.annotations_root),
        "pdfs_root": str(args.pdfs_root),
        "kb_root": str(args.kb_root),
        "records_file": str(records_written),
        "category": args.category,
        "qa_split": args.qa_split,
        "doc_name": args.doc_name,
        "answer_profile": args.answer_profile,
        "kb_docs_only": bool(args.kb_docs_only),
        "kb_docs_dropped_count": len(kb_dropped_samples),
        "kb_docs_dropped": sorted({s.doc_name for s in kb_dropped_samples}),
        "full_upload_model": args.full_upload_model if args.mode in {"both", "full_upload"} else "",
        "full_upload_scope": args.full_upload_scope if args.mode in {"both", "full_upload"} else "",
        "sample_count": len(samples),
        "done_count": done,
        "failed_count": failed,
        "paired_count": len(paired_records),
        "docrag_usage": docrag_usage,
        "full_upload_usage": full_usage,
        "savings": {
            "input_token_saved_ratio": _ratio_saved(docrag_usage["input_tokens"], full_usage["input_tokens"]),
            "output_token_saved_ratio": _ratio_saved(docrag_usage["output_tokens"], full_usage["output_tokens"]),
            "total_token_saved_ratio": _ratio_saved(docrag_usage["total_tokens"], full_usage["total_tokens"]),
            "input_token_saved_percent": round(_ratio_saved(docrag_usage["input_tokens"], full_usage["input_tokens"]) * 100, 2),
            "output_token_saved_percent": round(_ratio_saved(docrag_usage["output_tokens"], full_usage["output_tokens"]) * 100, 2),
            "total_token_saved_percent": round(_ratio_saved(docrag_usage["total_tokens"], full_usage["total_tokens"]) * 100, 2),
            "upload_token_saved_percent": round(_ratio_saved(docrag_usage["input_tokens"], full_usage["input_tokens"]) * 100, 2),
            "download_token_saved_percent": round(_ratio_saved(docrag_usage["output_tokens"], full_usage["output_tokens"]) * 100, 2),
        },
        "avg_uploaded_units": {
            "docrag_images": round(
                sum(int(r["docrag"].get("image_count", 0) or 0) for r in docrag_records) / max(len(docrag_records), 1),
                3,
            ),
            "full_upload_pdfs": round(
                sum(int(r["full_upload"].get("doc_count", 0) or 0) for r in full_records) / max(len(full_records), 1),
                3,
            ),
        },
    }

    metrics_written = _safe_write(
        args.metrics_file,
        json.dumps(summary, ensure_ascii=False, indent=2),
        fallback_name="token_eval_metrics_fallback.json",
    )

    print("============================================================")
    print("Token evaluation done.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"records_file: {records_written}")
    print(f"metrics_file: {metrics_written}")
    print("============================================================")


if __name__ == "__main__":
    main()
