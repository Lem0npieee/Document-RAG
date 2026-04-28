from __future__ import annotations

from pathlib import Path

from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document


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

    # DashScope嵌入API批次大小限制为10，需要分批处理
    batch_size = 10
    total_batches = (len(documents) + batch_size - 1) // batch_size

    print(f"    将 {len(documents)} 个文档分成 {total_batches} 批进行嵌入，每批 {batch_size} 个")

    # 分批构建向量库
    from langchain_community.vectorstores import FAISS

    vectorstore = None
    for i in range(0, len(documents), batch_size):
        batch_end = min(i + batch_size, len(documents))
        batch_docs = documents[i:batch_end]
        batch_num = i // batch_size + 1

        print(f"    处理第 {batch_num}/{total_batches} 批，文档 {i+1} 到 {batch_end}")
        batch_vectorstore = FAISS.from_documents(batch_docs, embedding)

        if vectorstore is None:
            vectorstore = batch_vectorstore
        else:
            vectorstore.merge_from(batch_vectorstore)

    print(f"    保存FAISS索引到: {output_path}")
    vectorstore.save_local(str(output_path))

    # 获取索引维度信息
    index = vectorstore.index
    if hasattr(index, 'd'):
        print(f"    FAISS索引维度: {index.d}，文档数: {index.ntotal}")
    else:
        print(f"    FAISS索引已保存")

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
    return FAISS.load_local(
        folder_path=str(index_dir),
        embeddings=embedding,
        allow_dangerous_deserialization=True,
    )
