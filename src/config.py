from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass
class Settings:
    dashscope_api_key: str
    model_provider: str = "dashscope"
    vl_model: str = "qwen3-vl-8b-instruct"
    embedding_provider: str = "dashscope"
    embedding_model: str = "text-embedding-v3"
    cliproxy_api_base_url: str = "http://127.0.0.1:8317"
    cliproxy_api_key: str = "sk-dummy"
    cliproxy_provider: str = "qclaw"
    cliproxy_vl_model: str = "modelroute"
    doc_root: Path = Path("doc")
    output_root: Path = Path("outputs")
    pages_dirname: str = "pages"
    parsed_dirname: str = "parsed"
    faiss_dirname: str = "faiss_index"
    graph_dirname: str = "doc_graph"

    @property
    def pages_dir(self) -> Path:
        return self.output_root / self.pages_dirname

    @property
    def parsed_dir(self) -> Path:
        return self.output_root / self.parsed_dirname

    @property
    def faiss_dir(self) -> Path:
        return self.output_root / self.faiss_dirname

    @property
    def graph_dir(self) -> Path:
        return self.output_root / self.graph_dirname

    @property
    def doc_dir(self) -> Path:
        return self.doc_root


    @property
    def active_vl_model(self) -> str:
        if self.model_provider == "cliproxyapi":
            return self.cliproxy_vl_model
        return self.vl_model


def _clean_env_value(value: str | None) -> str:
    if value is None:
        return ""
    return value.strip().strip("\"").strip("'")


def get_settings() -> Settings:
    load_dotenv()
    dashscope_api_key = _clean_env_value(os.getenv("DASHSCOPE_API_KEY", ""))

    model_provider = _clean_env_value(os.getenv("MODEL_PROVIDER", "dashscope")).lower()
    if model_provider not in {"dashscope", "cliproxyapi"}:
        raise ValueError("MODEL_PROVIDER must be either 'dashscope' or 'cliproxyapi'.")

    embedding_provider = _clean_env_value(os.getenv("EMBEDDING_PROVIDER", "dashscope")).lower()
    if embedding_provider not in {"dashscope", "local"}:
        raise ValueError("EMBEDDING_PROVIDER must be either 'dashscope' or 'local'.")

    # API key validation: VLM (model_provider) and embedding are independent
    needs_dashscope_key = (
        model_provider == "dashscope"  # VLM uses DashScope
        or embedding_provider == "dashscope"  # embedding uses DashScope
    )
    if needs_dashscope_key and (not dashscope_api_key or dashscope_api_key == "sk-你的真实密钥"):
        raise ValueError(
            "Missing DASHSCOPE_API_KEY. Required when MODEL_PROVIDER=dashscope "
            "or EMBEDDING_PROVIDER=dashscope. Create .env based on .env.example first."
        )

    embedding_model = _clean_env_value(os.getenv("EMBEDDING_MODEL", "text-embedding-v3"))
    if embedding_provider == "local" and embedding_model == "text-embedding-v3":
        embedding_model = "BAAI/bge-small-zh-v1.5"
        print(f"  EMBEDDING_PROVIDER=local, using default model: {embedding_model}")

    settings = Settings(
        dashscope_api_key=dashscope_api_key,
        model_provider=model_provider,
        vl_model=_clean_env_value(os.getenv("VL_MODEL", "qwen3-vl-8b-instruct")),
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        cliproxy_api_base_url=_clean_env_value(
            os.getenv("CLIPROXY_API_BASE_URL", "http://127.0.0.1:8317")
        ),
        cliproxy_api_key=_clean_env_value(os.getenv("CLIPROXY_API_KEY", "sk-dummy")),
        cliproxy_provider=_clean_env_value(os.getenv("CLIPROXY_PROVIDER", "qclaw")),
        cliproxy_vl_model=_clean_env_value(
            os.getenv("CLIPROXY_VL_MODEL", "modelroute")
        ),
        doc_root=Path(os.getenv("DOC_ROOT", "doc")),
        output_root=Path(os.getenv("OUTPUT_ROOT", "outputs")),
    )

    if settings.model_provider == "cliproxyapi":
        if not settings.cliproxy_api_base_url:
            raise ValueError("Missing CLIPROXY_API_BASE_URL when MODEL_PROVIDER=cliproxyapi.")
        if not settings.cliproxy_api_key:
            raise ValueError("Missing CLIPROXY_API_KEY when MODEL_PROVIDER=cliproxyapi.")
        if not settings.cliproxy_provider:
            raise ValueError("Missing CLIPROXY_PROVIDER when MODEL_PROVIDER=cliproxyapi.")
        if not settings.cliproxy_vl_model:
            raise ValueError("Missing CLIPROXY_VL_MODEL when MODEL_PROVIDER=cliproxyapi.")

    settings.doc_dir.mkdir(parents=True, exist_ok=True)
    settings.output_root.mkdir(parents=True, exist_ok=True)
    settings.pages_dir.mkdir(parents=True, exist_ok=True)
    settings.parsed_dir.mkdir(parents=True, exist_ok=True)
    settings.faiss_dir.mkdir(parents=True, exist_ok=True)
    settings.graph_dir.mkdir(parents=True, exist_ok=True)

    return settings
