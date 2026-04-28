from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass
class Settings:
    api_key: str
    vl_model: str = "qwen3-vl-8b-instruct"
    embedding_model: str = "text-embedding-v3"
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


def _clean_env_value(value: str) -> str:
    return value.strip().strip("\"").strip("'")


def get_settings() -> Settings:
    load_dotenv()
    api_key = _clean_env_value(os.getenv("DASHSCOPE_API_KEY", ""))
    if not api_key or api_key == "sk-你的真实密钥":
        raise ValueError(
            "Missing DASHSCOPE_API_KEY. Create .env based on .env.example first."
        )

    settings = Settings(
        api_key=api_key,
        vl_model=os.getenv("VL_MODEL", "qwen3-vl-8b-instruct"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-v3"),
        doc_root=Path(os.getenv("DOC_ROOT", "doc")),
        output_root=Path(os.getenv("OUTPUT_ROOT", "outputs")),
    )

    settings.doc_dir.mkdir(parents=True, exist_ok=True)
    settings.output_root.mkdir(parents=True, exist_ok=True)
    settings.pages_dir.mkdir(parents=True, exist_ok=True)
    settings.parsed_dir.mkdir(parents=True, exist_ok=True)
    settings.faiss_dir.mkdir(parents=True, exist_ok=True)
    settings.graph_dir.mkdir(parents=True, exist_ok=True)

    return settings
