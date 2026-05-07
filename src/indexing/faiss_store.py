from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


def _path_has_non_ascii(path: Path) -> bool:
    try:
        str(path).encode("ascii")
        return False
    except UnicodeEncodeError:
        return True


def build_faiss_index(
    documents: list[Document],
    api_key: str,
    embedding_model: str,
    output_dir: str | Path,
) -> Path:
    print(f"    正在初始化嵌入模型: {embedding_model}")
    embedding = DashScopeEmbeddings(
        model=embedding_model,
        dashscope_api_key=api_key,
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"    将重建FAISS索引并覆盖目录: {output_path}")
    print(f"    正在构建FAISS向量索引，文档数: {len(documents)}")

    # DashScope embedding endpoint has batch limits.
    batch_size = 10
    total_batches = (len(documents) + batch_size - 1) // batch_size
    print(f"    将 {len(documents)} 个文档分成 {total_batches} 批进行嵌入，每批 {batch_size} 个")

    vectorstore = None
    for i in range(0, len(documents), batch_size):
        batch_end = min(i + batch_size, len(documents))
        batch_docs = documents[i:batch_end]
        batch_num = i // batch_size + 1
        print(f"    处理第 {batch_num}/{total_batches} 批，文档 {i + 1} 到 {batch_end}")

        batch_vectorstore = FAISS.from_documents(batch_docs, embedding)
        if vectorstore is None:
            vectorstore = batch_vectorstore
        else:
            vectorstore.merge_from(batch_vectorstore)

    if vectorstore is None:
        raise RuntimeError("No documents provided to build FAISS index.")

    print(f"    保存FAISS索引到: {output_path}")
    if _path_has_non_ascii(output_path):
        # Windows FAISS wheels can fail on non-ASCII paths.
        # Save to an ASCII temp directory first, then copy artifacts back.
        with tempfile.TemporaryDirectory(prefix="faiss_save_") as tmp_dir:
            tmp_path = Path(tmp_dir) / "faiss_index"
            tmp_path.mkdir(parents=True, exist_ok=True)
            vectorstore.save_local(str(tmp_path))
            output_path.mkdir(parents=True, exist_ok=True)
            shutil.copy2(tmp_path / "index.faiss", output_path / "index.faiss")
            shutil.copy2(tmp_path / "index.pkl", output_path / "index.pkl")
    else:
        vectorstore.save_local(str(output_path))

    index = vectorstore.index
    if hasattr(index, "d"):
        print(f"    FAISS索引维度: {index.d}，文档数: {index.ntotal}")
    else:
        print("    FAISS索引已保存")

    return output_path


def load_faiss_index(
    api_key: str,
    embedding_model: str,
    index_dir: str | Path,
) -> FAISS:
    embedding = DashScopeEmbeddings(
        model=embedding_model,
        dashscope_api_key=api_key,
    )

    index_path = Path(index_dir)
    faiss_file = index_path / "index.faiss"
    pkl_file = index_path / "index.pkl"
    if not faiss_file.exists() or not pkl_file.exists():
        raise FileNotFoundError(
            f"FAISS index files missing under {index_path}. "
            f"Expected: {faiss_file} and {pkl_file}"
        )

    if _path_has_non_ascii(index_path):
        with tempfile.TemporaryDirectory(prefix="faiss_load_") as tmp_dir:
            tmp_path = Path(tmp_dir) / "faiss_index"
            tmp_path.mkdir(parents=True, exist_ok=True)
            shutil.copy2(faiss_file, tmp_path / "index.faiss")
            shutil.copy2(pkl_file, tmp_path / "index.pkl")
            return FAISS.load_local(
                folder_path=str(tmp_path),
                embeddings=embedding,
                allow_dangerous_deserialization=True,
            )

    return FAISS.load_local(
        folder_path=str(index_path),
        embeddings=embedding,
        allow_dangerous_deserialization=True,
    )

