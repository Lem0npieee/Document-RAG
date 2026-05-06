from __future__ import annotations

import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import networkx as nx

from src.parsing.pipeline import _parse_page_with_retry
from src.rag.multimodal_graph_rag_chain import MultiModalGraphRAG
from src.server import _ingest_build_lock, app


class FakeVLClient:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls: list[str] = []

    def extract_structured_page(self, prompt: str, image_path: str | Path) -> str:
        self.calls.append(prompt)
        if not self.responses:
            raise AssertionError("No fake responses left.")
        return self.responses.pop(0)

    def answer_question(self, prompt: str, image_paths: list[str | Path]) -> str:
        raise NotImplementedError


class ParsingBudgetTests(unittest.TestCase):
    def test_parse_stops_on_first_success(self) -> None:
        client = FakeVLClient(
            ['{"texts":[{"content":"ok","bbox":[0.1,0.1,0.2,0.2]}],"tables":[],"figures":[],"section_conclusions":[],"entities":[],"keywords":[],"relations":[]}']
        )
        parsed = _parse_page_with_retry(client, Path("page_1.png"), 1)
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(parsed["texts"][0]["content"], "ok")

    def test_parse_uses_second_attempt_then_stops(self) -> None:
        client = FakeVLClient(
            [
                '{"texts":[],"tables":[],"figures":[],"section_conclusions":[],"entities":[],"keywords":[],"relations":[]}',
                '{"texts":[{"content":"retry ok","bbox":[0.1,0.1,0.2,0.2]}],"tables":[],"figures":[],"section_conclusions":[],"entities":[],"keywords":[],"relations":[]}',
            ]
        )
        parsed = _parse_page_with_retry(client, Path("page_1.png"), 1)
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(parsed["texts"][0]["content"], "retry ok")

    def test_parse_runs_text_only_once_after_two_sparse_attempts(self) -> None:
        client = FakeVLClient(
            [
                '{"texts":[],"tables":[],"figures":[],"section_conclusions":[],"entities":[],"keywords":[],"relations":[]}',
                '{"texts":[],"tables":[],"figures":[],"section_conclusions":[],"entities":[],"keywords":[],"relations":[]}',
                '{"texts":[{"content":"fallback text","bbox":[0.1,0.1,0.2,0.2]}]}',
            ]
        )
        parsed = _parse_page_with_retry(client, Path("page_1.png"), 1)
        self.assertEqual(len(client.calls), 3)
        self.assertEqual(parsed["texts"][0]["content"], "fallback text")

    def test_single_block_parse_does_not_trigger_text_only(self) -> None:
        client = FakeVLClient(['{"content":"single block","bbox":[0.1,0.1,0.2,0.2]}'])
        parsed = _parse_page_with_retry(client, Path("page_1.png"), 1)
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(parsed["texts"][0]["content"], "single block")


class RelationBudgetTests(unittest.TestCase):
    def test_relation_budget_prioritizes_non_same_page(self) -> None:
        chain = MultiModalGraphRAG.__new__(MultiModalGraphRAG)
        chain.MAX_RELATION_LINES = 4
        chain.MAX_SAME_PAGE_RELATION_LINES = 2
        chain.graph = nx.DiGraph()
        for node_id in ["a", "b", "c", "d"]:
            chain.graph.add_node(node_id)
        chain.graph.add_edge("a", "b", relation="same_page")
        chain.graph.add_edge("b", "a", relation="same_page")
        chain.graph.add_edge("a", "c", relation="supports")
        chain.graph.add_edge("c", "d", relation="refers_to")
        chain.graph.add_edge("d", "a", relation="uses")
        chain.graph.add_edge("b", "d", relation="same_page")

        lines = chain._collect_relations(["a", "b", "c", "d"])
        self.assertLessEqual(len(lines), 4)
        non_same_page = [line for line in lines if "--[same_page]-->" not in line]
        self.assertEqual(len(non_same_page), 3)
        self.assertIn("a --[supports]--> c", lines)
        self.assertIn("c --[refers_to]--> d", lines)
        self.assertIn("d --[uses]--> a", lines)


class IngestGuardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = app.test_client()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.upload_dir = Path(self.temp_dir.name) / "uploads"
        self.upload_dir.mkdir(parents=True, exist_ok=True)

    def _post_ingest(self, headers: dict[str, str] | None = None):
        data = {
            "file": (io.BytesIO(b"fake png bytes"), "demo.png"),
        }
        with patch("src.server.UPLOAD_DIR", self.upload_dir):
            return self.client.post(
                "/ingest",
                data=data,
                headers=headers or {},
                content_type="multipart/form-data",
            )

    def test_ingest_rejects_missing_token(self) -> None:
        with patch.dict(os.environ, {"INGEST_API_TOKEN": "secret"}, clear=False):
            response = self._post_ingest()
        self.assertEqual(response.status_code, 403)

    def test_ingest_accepts_correct_token(self) -> None:
        with patch.dict(os.environ, {"INGEST_API_TOKEN": "secret"}, clear=False):
            with patch("src.server.build_knowledge_base", return_value={"ok": True}):
                response = self._post_ingest(headers={"X-Ingest-Token": "secret"})
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["ok"])

    def test_ingest_rejects_when_build_is_in_progress(self) -> None:
        acquired = _ingest_build_lock.acquire(blocking=False)
        self.assertTrue(acquired)
        self.addCleanup(lambda: _ingest_build_lock.locked() and _ingest_build_lock.release())
        with patch.dict(os.environ, {}, clear=False):
            response = self._post_ingest()
        self.assertEqual(response.status_code, 409)


if __name__ == "__main__":
    unittest.main()
