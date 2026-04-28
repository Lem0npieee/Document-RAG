from __future__ import annotations

import base64
import json
import re
import time
from pathlib import Path
from typing import Any, Protocol
from urllib import error, request

import dashscope
from dashscope import MultiModalConversation

from src.config import Settings


def _image_to_data_uri(image_path: str | Path) -> str:
    path = Path(image_path)
    suffix = path.suffix.lower().replace(".", "") or "png"
    raw = path.read_bytes()
    encoded = base64.b64encode(raw).decode("utf-8")
    return f"data:image/{suffix};base64,{encoded}"


def _clean_model_text(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", str(text), flags=re.DOTALL | re.IGNORECASE)
    return cleaned.strip()


class VLClient(Protocol):
    def extract_structured_page(self, prompt: str, image_path: str | Path) -> str:
        ...

    def answer_question(self, prompt: str, image_paths: list[str | Path]) -> str:
        ...


class CLIProxyAPIError(RuntimeError):
    def __init__(self, message: str, status_code: int | None = None, code: Any = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code


class DashScopeVLClient:
    """Thin wrapper around DashScope multimodal conversation API."""

    def __init__(self, api_key: str, model: str) -> None:
        self.api_key = api_key
        self.model = model

    def _to_data_uri(self, image_path: str | Path) -> str:
        return _image_to_data_uri(image_path)

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


class CLIProxyVLClient:
    """Minimal CLIProxyAPI client using provider-specific OpenAI-compatible routes."""

    def __init__(self, api_base_url: str, api_key: str, provider: str, model: str) -> None:
        self.api_base_url = api_base_url.rstrip("/")
        self.api_key = api_key
        self.provider = provider
        self.model = model

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _post_json(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(endpoint, data=body, headers=self._headers(), method="POST")
        try:
            with request.urlopen(req, timeout=180) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            try:
                error_payload = json.loads(raw)
            except json.JSONDecodeError:
                error_payload = {}
            err = error_payload.get("error", {}) if isinstance(error_payload, dict) else {}
            message = str(err.get("message") or raw or exc.reason)
            raise CLIProxyAPIError(message=message, status_code=exc.code, code=err.get("code")) from exc
        except error.URLError as exc:
            raise CLIProxyAPIError(f"CLIProxyAPI request failed: {exc.reason}") from exc

    def _extract_chat_text(self, payload: dict[str, Any]) -> str:
        choices = payload.get("choices", [])
        for choice in choices:
            message = choice.get("message", {})
            content = message.get("content", "")
            if isinstance(content, str) and content.strip():
                return _clean_model_text(content)
            if isinstance(content, list):
                parts: list[str] = []
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    text = item.get("text")
                    if isinstance(text, str) and text.strip():
                        parts.append(text)
                if parts:
                    return _clean_model_text("\n".join(parts))
        raise RuntimeError(f"Unexpected CLIProxy chat response format: {payload}")

    def _extract_responses_text(self, payload: dict[str, Any]) -> str:
        output_text = payload.get("output_text")
        if isinstance(output_text, str) and output_text.strip():
            return _clean_model_text(output_text)

        outputs = payload.get("output", [])
        for item in outputs:
            if not isinstance(item, dict):
                continue
            contents = item.get("content", [])
            parts: list[str] = []
            for content in contents:
                if not isinstance(content, dict):
                    continue
                text = content.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text)
            if parts:
                return _clean_model_text("\n".join(parts))
        raise RuntimeError(f"Unexpected CLIProxy responses format: {payload}")

    def _chat_endpoint(self) -> str:
        return f"{self.api_base_url}/api/provider/{self.provider}/v1/chat/completions"

    def _responses_endpoint(self) -> str:
        return f"{self.api_base_url}/api/provider/{self.provider}/v1/responses"

    def _chat_payload(self, prompt: str, image_paths: list[str | Path]) -> dict[str, Any]:
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for image_path in image_paths:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": _image_to_data_uri(image_path)},
                }
            )
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a helpful multimodal assistant."},
                {"role": "user", "content": content},
            ],
        }

    def _responses_payload(self, prompt: str, image_paths: list[str | Path]) -> dict[str, Any]:
        content: list[dict[str, Any]] = [{"type": "input_text", "text": prompt}]
        for image_path in image_paths:
            content.append(
                {
                    "type": "input_image",
                    "image_url": _image_to_data_uri(image_path),
                }
            )
        return {
            "model": self.model,
            "input": [
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": "You are a helpful multimodal assistant."}],
                },
                {
                    "role": "user",
                    "content": content,
                },
            ],
        }

    def _call_chat(self, prompt: str, image_paths: list[str | Path]) -> str:
        response = self._post_json(
            self._chat_endpoint(),
            self._chat_payload(prompt=prompt, image_paths=image_paths),
        )
        return self._extract_chat_text(response)

    def _call_responses(self, prompt: str, image_paths: list[str | Path]) -> str:
        response = self._post_json(
            self._responses_endpoint(),
            self._responses_payload(prompt=prompt, image_paths=image_paths),
        )
        return self._extract_responses_text(response)

    def _call(self, prompt: str, image_paths: list[str | Path]) -> str:
        try:
            return self._call_chat(prompt=prompt, image_paths=image_paths)
        except CLIProxyAPIError as exc:
            if image_paths and exc.code == 1400:
                return self._call_responses(prompt=prompt, image_paths=image_paths)
            raise

    def extract_structured_page(self, prompt: str, image_path: str | Path) -> str:
        return self._call(prompt=prompt, image_paths=[image_path])

    def answer_question(self, prompt: str, image_paths: list[str | Path]) -> str:
        return self._call(prompt=prompt, image_paths=image_paths)


def create_vl_client(settings: Settings) -> VLClient:
    if settings.model_provider == "cliproxyapi":
        return CLIProxyVLClient(
            api_base_url=settings.cliproxy_api_base_url,
            api_key=settings.cliproxy_api_key,
            provider=settings.cliproxy_provider,
            model=settings.cliproxy_vl_model,
        )

    return DashScopeVLClient(
        api_key=settings.dashscope_api_key,
        model=settings.vl_model,
    )
