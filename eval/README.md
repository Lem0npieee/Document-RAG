# DocVQA Evaluation (Backend Only)

本目录用于**只评测当前模型**（`MultiModalGraphRAG`），不做前端改动，也不做横向对比。  
按你的要求，评测相关代码、输入、输出都放在 `eval/` 下。

## 目录约定

```text
eval/
  code/                    # 评测脚本
    loader.py
    metrics.py
    build_docvqa_kb.py
    run_docvqa_eval.py
  input/
    docvqa/
      val.json             # 标注文件（你放）
      images/              # 图片目录（你放）
  output/
    kb/                    # 评测用知识库产物（FAISS/graph/parsed/pages）
    docvqa/                # 评测结果（predictions/metrics/errors）
```

## 1) 准备输入数据

把 DocVQA 数据放到：

- `eval/input/docvqa/val.json`
- `eval/input/docvqa/images/`

`val.json` 每条样本至少应包含：

- `question`（问题）
- `answers`（答案列表，或 `answer`）
- `image`（图片名/相对路径）

## 2) 构建评测知识库（只跑后端）

```bash
python eval/code/build_docvqa_kb.py --resume --strict-images
```

说明：

- 脚本会自动把 `OUTPUT_ROOT` 指向 `eval/output/kb`，不会写到默认 `outputs/`。
- `--resume` 支持断点续跑。
- `--force-rebuild` 可强制重建单图入库。

## 3) 跑评测

```bash
python eval/code/run_docvqa_eval.py --resume --strict-images --k 5 --max-nodes 24
```

输出文件：

- `eval/output/docvqa/predictions.jsonl`
- `eval/output/docvqa/metrics.json`
- `eval/output/docvqa/errors_topk.jsonl`

## 可选参数

- `--max-samples 100`：先小样本试跑
- `--anls-threshold 0.5`：ANLS 阈值
- `--log-every 20`：进度打印间隔

