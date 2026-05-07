# pdfQA Evaluation (Backend Only)

This folder contains a complete evaluation pipeline for this project on `pdfQA`.
It evaluates the current backend chain (`MultiModalGraphRAG`) only.

## Folder Layout

```text
eval/
  code/
    loader.py
    metrics.py
    build_pdfqa_kb.py
    run_pdfqa_eval.py
  input/
    pdfqa/
      annotations/        # pdfQA-Annotations (JSON files)
      pdfs/               # pdfQA-Benchmark PDFs
  output/
    kb/                   # FAISS/graph/pages for evaluation
    pdfqa/                # predictions / metrics / errors
```

## 1) Download and place data

You need both:

1. `pdfQA-Annotations`  
   - Hugging Face: `pdfqa/pdfQA-Annotations`
2. `pdfQA-Benchmark` (PDF files)  
   - Hugging Face: `pdfqa/pdfQA-Benchmark`

Official download scripts are listed here:
- https://github.com/tobischimanski/pdfQA

Place files as:

```text
eval/input/pdfqa/annotations/real-pdfQA/.../*.json
eval/input/pdfqa/annotations/syn-pdfQA/.../*.json
eval/input/pdfqa/pdfs/real-pdfQA/.../*.pdf
eval/input/pdfqa/pdfs/syn-pdfQA/.../*.pdf
```

The loader scans recursively, so exact subfolders can be nested.

## 2) Build evaluation KB

```bash
python eval/code/build_pdfqa_kb.py --resume --strict-docs --category real
```

Useful args:

- `--category all|real|syn` (default: `real`)
- `--max-samples 200` (quick smoke run)
- `--max-docs 50`
- `--force-rebuild`

This script writes KB artifacts into `eval/output/kb` by setting:
- `OUTPUT_ROOT=eval/output/kb`
- `DOC_ROOT=eval/input/pdfqa/pdfs`

## 3) Run evaluation

```bash
python eval/code/run_pdfqa_eval.py --resume --strict-docs --category real --k 5 --max-nodes 24
```

Outputs:

- `eval/output/pdfqa/predictions.jsonl`
- `eval/output/pdfqa/metrics.json`
- `eval/output/pdfqa/errors_topk.jsonl`

## 4) Metrics

The script reports:

- `ANLS`
- `EM`
- `evidence_page_recall` (only when page labels exist in annotations)
- group metrics by category, dataset, and question type

## Notes

- `source_hint` is always set to the matched PDF basename, so retrieval stays within the target source.
- If a sample has no answer text, it is skipped by default.
- If a JSON annotation points to a missing PDF and `--strict-docs` is set, that sample is skipped and logged as a warning.

