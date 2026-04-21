from __future__ import annotations

import base64
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
        # 尝试调用 to_dict，但捕获可能发生的 KeyError（来自 __getattr__）
        try:
            payload = response.to_dict()
        except (AttributeError, KeyError):
            # 如果 to_dict 不存在或 __getattr__ 引发 KeyError
            if isinstance(response, dict):
                payload = response
            else:
                # 尝试将 response 视为字典
                try:
                    payload = dict(response)
                except (TypeError, ValueError):
                    raise RuntimeError(f"Unexpected DashScope response format: {response}")

        choices = payload.get("output", {}).get("choices", [])
        for choice in choices:
            message = choice.get("message", {})
            content = message.get("content", [])
            if isinstance(content, str):
                print(f"      提取到文本内容，长度: {len(content)} 字符")
                return content
            if isinstance(content, list):
                parts: list[str] = []
                for item in content:
                    if isinstance(item, dict) and item.get("text"):
                        parts.append(str(item["text"]))
                if parts:
                    combined = "\n".join(parts)
                    print(f"      提取到多部分文本内容，总长度: {len(combined)} 字符")
                    return combined

        raise RuntimeError(f"Unexpected DashScope response format: {payload}")

    def _call(self, prompt: str, image_paths: list[str | Path]) -> str:
        dashscope.api_key = self.api_key
        user_content: list[dict[str, str]] = []
        for image_path in image_paths:
            user_content.append({"image": self._to_data_uri(image_path)})
        user_content.append({"text": prompt})

        print(f"      调用DashScope API，模型: {self.model}, 图片数: {len(image_paths)}")
        response = MultiModalConversation.call(
            model=self.model,
            messages=[
                {"role": "system", "content": [{"text": "You are a helpful multimodal assistant."}]},
                {"role": "user", "content": user_content},
            ],
            result_format="message",
        )
        print(f"      API调用成功，正在提取响应内容...")
        return self._extract_text(response)

    def extract_structured_page(self, prompt: str, image_path: str | Path) -> str:
        return self._call(prompt=prompt, image_paths=[image_path])

    def answer_question(self, prompt: str, image_paths: list[str | Path]) -> str:
        return self._call(prompt=prompt, image_paths=image_paths)
