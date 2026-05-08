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
python eval/code/run_pdfqa_eval.py --resume --strict-docs --category real --k 5 --max-nodes 24
```

输出文件：

- `eval/output/pdfqa/predictions.jsonl`
- `eval/output/pdfqa/metrics.json`
- `eval/output/pdfqa/errors_topk.jsonl`

## 4）指标

脚本会报告：

- `ANLS`
- `EM`
- `evidence_page_recall`（仅当注释中存在页面标签时）
- 按类别、数据集和问题类型分组的指标

## 注意事项

- `source_hint` 始终设置为匹配的 PDF 基本文件名，以确保检索限制在目标来源内。
- 如果样本没有答案文本，默认会跳过该样本。
- 如果某个 JSON 注释指向缺失的 PDF 且设置了 `--strict-docs`，该样本会被跳过并记录为警告。
