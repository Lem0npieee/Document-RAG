# 预构建知识库（KB）共享指南

## 概述

本仓库已将预构建好的知识库（KB）通过 **Git LFS** 纳入版本管理，队友无需重新运行耗时的文档解析和索引构建流程，直接 clone 即可使用。

## KB 内容

| 目录 | 内容 | 大小 | 管理方式 |
|------|------|------|----------|
| `eval/output/kb/pages/` | 13 份文档的 770 张页面截图 (PNG) | 562.6 MB | Git LFS |
| `eval/output/kb/faiss_index/` | FAISS 向量索引 (`index.faiss` + `index.pkl`) | 13.4 MB | Git LFS |
| `eval/output/kb/doc_graph/` | NetworkX 知识图谱 (`graph.pkl`) | 8.2 MB | Git LFS |
| `eval/output/kb/parsed/` | 文档元数据 JSON（`documents.json` + `graph_data.json`） | 6.5 MB | 普通 Git |
| `eval/output/kb/build_meta.json` | 构建配置记录 | < 1 KB | 普通 Git |

**总计约 590 MB**，其中大文件由 Git LFS 管理。

### 包含的文档

13 份企业可持续发展/ESG 报告（英文 PDF）：

- Microsoft 2022 Environmental Sustainability Report (81 页)
- AT&T 2022 Sustainability Summary (33 页)
- AstraZeneca Sustainability Report 2023 (40 页)
- Boeing 2023 Sustainability Report (100 页)
- CT REIT 2022 ESG Report (34 页)
- Coca-Cola Business Sustainability Report 2022 (88 页)
- CostCo Climate Action Plan (15 页)
- JP Morgan Climate Report 2022 (36 页)
- Meta 2023 Sustainability Report (60 页)
- Veolia ESG Report 2023 (27 页)
- Walmart ESG Highlights 2023 (43 页)
- Westpac 2023 Climate Report (116 页)

## 队友上手步骤

### 前置条件

```bash
# macOS
brew install git-lfs

# Ubuntu / Debian
sudo apt install git-lfs

# Windows
# 下载安装包: https://git-lfs.com
```

### 1. 初始化 Git LFS

```bash
git lfs install
```

### 2. Clone 仓库（含 KB）

```bash
git clone <仓库地址>
cd Document-RAG
git lfs pull   # 拉取 LFS 管理的大文件
```

> 如果已经 clone 过但没有 KB 文件，只需执行：
> ```bash
> git lfs install
> git lfs pull
> ```

### 3. 验证 KB 完整性

```bash
# 检查关键文件是否存在
python3 -c "
import os
files = [
    'eval/output/kb/parsed/documents.json',
    'eval/output/kb/parsed/graph_data.json',
    'eval/output/kb/doc_graph/graph.pkl',
    'eval/output/kb/faiss_index/index.faiss',
    'eval/output/kb/faiss_index/index.pkl',
]
for f in files:
    ok = 'OK' if os.path.exists(f) else 'MISSING'
    print(f'  [{ok}] {f}')
"
```

预期输出全部为 `[OK]`。

### 4. 配置环境并启动

```bash
# 安装依赖（使用 uv）
uv sync

# 复制并编辑 .env（填入自己的 DashScope API Key）
cp .env.example .env

# 启动服务
uv run python src/server.py
```

浏览器打开 `http://localhost:8000/` 即可看到已加载的知识图谱。

### 5. 运行评测

KB 位于 `eval/output/kb/`，评测脚本可直接使用：

```bash
uv run python eval/code/run_pdfqa_eval.py --resume --strict-docs --category real
```

## 更新 KB（重新构建后提交）

如果需要更新 KB（例如添加新文档或调整解析参数）：

### 1. 重新构建

```bash
uv run python eval/code/build_pdfqa_kb.py --resume --strict-docs --category real
```

### 2. 提交更新

```bash
# Git LFS 会自动管理 .png / .pkl / .faiss 等大文件
git add eval/output/kb/
git commit -m "chore: 更新预构建知识库"

# 推送（含 LFS 文件）
git push
git lfs push origin main   # 确保 LFS 文件也推送到远程
```

## .gitignore 说明

```gitignore
# 仅忽略非 KB 的评测产物
eval/output/*
!eval/output/kb/

# 根目录的 outputs / faiss_index / doc_graph 也被忽略
# 避免本地测试产物被误提交
/outputs/
/faiss_index/
/doc_graph/
```

- KB 在 `eval/output/kb/` 下，已被 `.gitignore` 例外放行
- 其他评测产物（如 `eval/output/pdfqa/`）仍被忽略
- 根目录下的 `outputs/`、`faiss_index/`、`doc_graph/` 仅匹配根目录，不影响 KB

## .gitattributes 说明

```gitattributes
eval/output/kb/pages/**/*.png   filter=lfs diff=lfs merge=lfs -text
eval/output/kb/faiss_index/*    filter=lfs diff=lfs merge=lfs -text
eval/output/kb/doc_graph/*      filter=lfs diff=lfs merge=lfs -text
```

- 二进制文件（PNG / pickle / FAISS）由 Git LFS 管理，不存入 Git 历史
- JSON 文本文件（documents.json / graph_data.json）由普通 Git 管理，可正常 diff

## 常见问题

### Q: `git lfs pull` 很慢怎么办？

A: KB 约 590MB，首次下载需要一定时间。如果在国内网络较慢，可以：
- 使用代理：`git config --global http.proxy http://127.0.0.1:7890`
- 或直接找队友拷贝 `eval/output/kb/` 整个目录

### Q: 我不想用 DashScope API，能用本地模型吗？

A: 可以。编辑 `.env`：
```env
MODEL_PROVIDER=ollama
OLLAMA_API_BASE_URL=http://localhost:11434/v1
OLLAMA_VL_MODEL=minicpm-v:8b
EMBEDDING_PROVIDER=local
```
注意：本地 VLM 模型（如 minicpm-v:8b）问答质量较差，适合测试流程，不建议用于正式评测。

### Q: KB 能跨平台使用吗？

A: 可以。FAISS 索引、graph.pkl、JSON 文件都是跨平台的，macOS / Linux / Windows 通用。
