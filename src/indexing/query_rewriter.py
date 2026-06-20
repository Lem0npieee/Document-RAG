"""Query rewriting: LLM rewrites long/colloquial queries for better retrieval.

When a query is long (>50 chars) or has many non-ASCII characters, a lightweight
LLM (e.g. qwen3.7-plus via DashScope compatible API) rewrites it into a concise,
keyword-rich search query. Original question is preserved for VLM generation.

Rewritten queries are cached to output/query_cache/rewrites.json.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

CACHE_DIRNAME = "query_cache"
CACHE_FILENAME = "rewrites.json"


class QueryRewriter:
    """Rewrite long or colloquial user questions for better retrieval quality."""

    MIN_CHARS_FOR_REWRITE = 50
    MIN_NON_ASCII_RATIO = 0.3
    MIN_WORDS_FOR_REWRITE = 8

    REWRITE_PROMPT = (
        "Rewrite the following user question into a concise, keyword-rich "
        "search query optimized for document retrieval. "
        "Extract the core entities, metrics, and concepts. "
        "Output ONLY the rewritten query — no explanation, no prefixes.\n\n"
        "Original question: {question}\n\n"
        "Rewritten query:"
    )

    def __init__(
        self,
        api_key: str,
        model: str = "qwen3.7-plus",
        api_base: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        output_root: Path | None = None,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._api_base = api_base
        self._cache: dict[str, str] = {}
        self._cache_path: Path | None = None
        if output_root is not None:
            cache_dir = output_root / CACHE_DIRNAME
            cache_dir.mkdir(parents=True, exist_ok=True)
            self._cache_path = cache_dir / CACHE_FILENAME
            self._load_cache()

    # ---- Public API --------------------------------------------------------

    def should_rewrite(self, question: str) -> bool:
        """Check whether the question would benefit from rewriting."""
        q = question.strip()
        if len(q) > self.MIN_CHARS_FOR_REWRITE:
            return True
        non_ascii = sum(1 for c in q if ord(c) > 127)
        if len(q) > 0 and non_ascii / len(q) > self.MIN_NON_ASCII_RATIO:
            return True
        words = q.split()
        if len(words) > self.MIN_WORDS_FOR_REWRITE:
            return True
        return False

    def rewrite(self, question: str) -> str:
        """Rewrite the question (with caching)."""
        if not self.should_rewrite(question):
            return question

        cache_key = self._cache_key(question)
        if cache_key in self._cache:
            print(f"  Query rewriting: cache hit")
            return self._cache[cache_key]

        rewritten = self._call_llm(question)
        self._cache[cache_key] = rewritten
        self._save_cache()
        return rewritten

    # ---- Internal ----------------------------------------------------------

    @staticmethod
    def _cache_key(question: str) -> str:
        return hashlib.sha256(question.strip().encode()).hexdigest()[:16]

    def _load_cache(self) -> None:
        if self._cache_path is None or not self._cache_path.exists():
            return
        try:
            self._cache = json.loads(self._cache_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            self._cache = {}

    def _save_cache(self) -> None:
        if self._cache_path is None:
            return
        try:
            self._cache_path.write_text(
                json.dumps(self._cache, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError:
            pass

    def _call_llm(self, question: str) -> str:
        import http.client
        import urllib.error
        import urllib.request

        prompt = self.REWRITE_PROMPT.format(question=question.strip())
        body = json.dumps(
            {
                "model": self._model,
                "messages": [
                    {"role": "system", "content": "You are a search query optimizer."},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": 80,
                "temperature": 0.1,
            }
        ).encode("utf-8")

        req = urllib.request.Request(
            f"{self._api_base}/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            print(f"  Query rewriting API error: {exc}")
            return question
        except urllib.error.URLError as exc:
            print(f"  Query rewriting network error: {exc}")
            return question

        choices = payload.get("choices", [])
        if not choices:
            return question

        content = choices[0].get("message", {}).get("content", "")
        rewritten = str(content).strip()

        if not rewritten or len(rewritten) < 3:
            return question

        print(f"  Query rewritten: {question[:60]}... → {rewritten[:80]}...")
        return rewritten
