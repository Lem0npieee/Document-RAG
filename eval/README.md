# pdfQA 评估（仅后端）

此文件夹包含本项目针对 `pdfQA` 的完整评估流水线。
本评估仅评估当前后端链（`MultiModalGraphRAG`）。

## 文件夹结构

```text
eval/
  code/
    loader.py
    metrics.py
    build_pdfqa_kb.py
    run_pdfqa_eval.py
    run_token_eval.py
  input/
    pdfqa/
      annotations/        # pdfQA 注释（JSON 文件）
      pdfs/               # pdfQA 基准测试的 PDF 文件
  output/
    kb/                   # 评估用的 FAISS/图/页面 数据
    pdfqa/                # 预测 / 指标 / 错误
```

## 1）下载并放置数据

您需要两项数据：

1. `pdfQA-Annotations`
   - Hugging Face: `pdfqa/pdfQA-Annotations`
2. `pdfQA-Benchmark`（PDF 文件）
   - Hugging Face: `pdfqa/pdfQA-Benchmark`

官方下载脚本见：
- https://github.com/tobischimanski/pdfQA

将文件放置为：

```text
eval/input/pdfqa/annotations/real-pdfQA/.../*.json
eval/input/pdfqa/annotations/syn-pdfQA/.../*.json
eval/input/pdfqa/pdfs/real-pdfQA/.../*.pdf
eval/input/pdfqa/pdfs/syn-pdfQA/.../*.pdf
```

加载器会递归扫描，因此子文件夹可以嵌套。

## 2）构建评估知识库（KB）

```bash
python eval/code/build_pdfqa_kb.py --resume --strict-docs --category real
```

有用的参数：

- `--category all|real|syn`（默认：`real`）
- `--max-samples 200`（快速 smoke 运行）
- `--max-docs 50`
- `--force-rebuild`

该脚本将 KB 输出写入 `eval/output/kb`，通过设置：
- `OUTPUT_ROOT=eval/output/kb`
- `DOC_ROOT=eval/input/pdfqa/pdfs`

## 3）运行评估

```bash
python eval/code/run_pdfqa_eval.py --resume --strict-docs --kb-docs-only --category real --k 5 --max-nodes 24
```

输出文件：

- `eval/output/pdfqa/predictions.jsonl`
- `eval/output/pdfqa/metrics.json`
- `eval/output/pdfqa/errors_topk.jsonl`

如果当前 KB 只构建了部分 PDF，建议保留 `--kb-docs-only`。该参数会读取 `eval/output/kb/parsed/documents.json`，自动跳过标注集中存在、但当前知识库中不存在的文档样本，避免把“未构建 KB 的文档”统计为无效失败。运行时会输出类似：

```text
--kb-docs-only: dropped 2 samples (remaining 18)
```

## 4）指标

脚本会报告：

- `ANLS`
- `EM`
- `evidence_page_recall`（仅当注释中存在页面标签时）
- 按类别、数据集和问题类型分组的指标

## 5）运行 Token 成本评估

`run_token_eval.py` 用于比较两种问答方式的 token 消耗：

- `DocRAG`：使用已构建好的知识图谱检索相关上下文，再调用 API 回答。
- `Full upload`：每个问题直接把目标 PDF 文档上传给 API 回答。

运行前需要保证已经存在评估知识库：

```text
eval/output/kb
```

真实 `real-pdfQA / ClimRetrieve` 数据的标注文件通常不是 `*_rawQA.json` 命名，因此建议使用 `--qa-split all`：

```bash
python -B eval/code/run_token_eval.py --category real --qa-split all --kb-docs-only --answer-profile all --max-samples 20 --mode both --full-upload-scope target_doc
```

如果只想先做小样本测试，可以降低样本数：

```bash
python -B eval/code/run_token_eval.py --category real --qa-split all --kb-docs-only --answer-profile all --max-samples 5 --mode both --full-upload-scope target_doc
```

如果只想测试某一篇 PDF，可以指定文档名：

```bash
python -B eval/code/run_token_eval.py --category real --qa-split all --doc-name "2022 Microsoft Environmental Sustainability Report.pdf" --kb-docs-only --answer-profile all --max-samples 5 --mode both --full-upload-scope target_doc
```

`--kb-docs-only` 会读取 `eval/output/kb/parsed/documents.json`，自动跳过标注集中存在、但当前知识库中不存在的文档样本。例如当前 KB 未构建 HM Group 和 New Look 时，脚本会输出类似：

```text
--kb-docs-only: dropped 2 samples (remaining 18)
```

输出文件：

- `eval/output/pdfqa/token_eval.jsonl`
- `eval/output/pdfqa/token_eval_metrics.json`

主要关注指标：

- `docrag_usage.input_tokens`：DocRAG 方案上传给模型的输入 token。
- `full_upload_usage.input_tokens`：整篇 PDF 上传方案的输入 token。
- `savings.input_token_saved_percent`：输入 token 节省比例，也可理解为 upload token 节省比例。
- `savings.total_token_saved_percent`：总 token 节省比例。
- `done_count` / `failed_count`：成功和失败样本数。若 full-upload API 触发额度或限流，可能出现部分失败。
- `kb_docs_dropped_count` / `kb_docs_dropped`：被 `--kb-docs-only` 跳过的样本数量和文档名。

当前脚本默认使用 `.env` 中的 DashScope API 配置，并使用 API embedding，不使用本地 embedding 模型。

## 注意事项

- `source_hint` 始终设置为匹配的 PDF 基本文件名，以确保检索限制在目标来源内。
- 如果样本没有答案文本，默认会跳过该样本。
- 如果某个 JSON 注释指向缺失的 PDF 且设置了 `--strict-docs`，该样本会被跳过并记录为警告。
