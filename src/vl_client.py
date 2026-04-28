from __future__ import annotations

import base64
import time
from pathlib import Path
from typing import Any

import dashscope
from dashscope import MultiModalConversation


class DashScopeVLClient:
    """Thin wrapper around DashScope multimodal conversation API."""

    def __init__(self, api_key: str, model: str) -> None:
        self.api_key = api_key
        self.model = model

    def _to_data_uri(self, image_path: str | Path) -> str:
        path = Path(image_path)
        suffix = path.suffix.lower().replace(".", "") or "png"
        raw = path.read_bytes()
        encoded = base64.b64encode(raw).decode("utf-8")
        return f"data:image/{suffix};base64,{encoded}"

    def _extract_text(self, response: Any) -> str:
        payload = {}
        try:
            payload = response.to_dict()
        except (AttributeError, KeyError):
            if isinstance(response, dict):
                payload = response
            else:
                try:
                    payload = dict(response)
                except (TypeError, ValueError):
                    raise RuntimeError(f"Unexpected DashScope response format: {response}")

        choices = payload.get("output", {}).get("choices", [])
        for choice in choices:
            message = choice.get("message", {})
            content = message.get("content", [])
            if isinstance(content, str):
                print(f"      Extracted response text length: {len(content)}")
                return content
            if isinstance(content, list):
                parts: list[str] = []
                for item in content:
                    if isinstance(item, dict) and item.get("text"):
                        parts.append(str(item["text"]))
                if parts:
                    combined = "\n".join(parts)
                    print(f"      Extracted multipart response text length: {len(combined)}")
                    return combined

        raise RuntimeError(f"Unexpected DashScope response format: {payload}")

    def _is_transient_error(self, exc: Exception) -> bool:
        text = str(exc).lower()
        transient_markers = [
            "ssl",
            "eof",
            "timed out",
            "timeout",
            "connection",
            "temporarily",
            "max retries exceeded",
            "remote end closed",
            "connection reset",
        ]
        return any(marker in text for marker in transient_markers)

    def _call(self, prompt: str, image_paths: list[str | Path]) -> str:
        dashscope.api_key = self.api_key
        user_content: list[dict[str, str]] = []
        for image_path in image_paths:
            user_content.append({"image": self._to_data_uri(image_path)})
        user_content.append({"text": prompt})

        print(f"      Calling DashScope API, model: {self.model}, images: {len(image_paths)}")

        max_attempts = 4
        for attempt in range(1, max_attempts + 1):
            try:
                response = MultiModalConversation.call(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": [{"text": "You are a helpful multimodal assistant."}]},
                        {"role": "user", "content": user_content},
                    ],
                    result_format="message",
                )
                print("      API call succeeded, parsing response...")
                return self._extract_text(response)
            except Exception as exc:
                if attempt >= max_attempts or not self._is_transient_error(exc):
                    raise
                sleep_s = min(6.0, 1.2 * attempt)
                print(
                    "      API transient error, retrying "
                    f"{attempt}/{max_attempts} after {sleep_s:.1f}s: {exc}"
                )
                time.sleep(sleep_s)

        raise RuntimeError("DashScope call failed after retries.")

    def extract_structured_page(self, prompt: str, image_path: str | Path) -> str:
        return self._call(prompt=prompt, image_paths=[image_path])

    def answer_question(self, prompt: str, image_paths: list[str | Path]) -> str:
        return self._call(prompt=prompt, image_paths=image_paths)
