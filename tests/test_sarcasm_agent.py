from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from sarcasm_agent import SarcasmAgent
from sarcasm_agent.schemas import SarcasmInput


class FakeSarcasmLLMClient:
    def __init__(self, result: dict) -> None:
        self.result = result
        self.last_payload: SarcasmInput | None = None

    def analyze(self, payload: SarcasmInput) -> dict:
        self.last_payload = payload
        return self.result


class SarcasmAgentTestCase(unittest.TestCase):
    def _payload(self, text: str) -> dict:
        return {
            "id": "msg_001",
            "user_id": "u_1001",
            "text": text,
            "source": "chat",
            "created_at": "2026-03-24T14:00:00",
        }

    def test_detect_sarcasm_from_llm_result(self) -> None:
        client = FakeSarcasmLLMClient(
            {
                "is_sarcasm": True,
                "surface_emotion": "开心",
                "true_emotion": "厌烦",
                "revised_intensity": 74,
                "confidence": 0.85,
                "reason": "表面正向词与负面场景形成反差，真实情绪偏厌烦。",
            }
        )
        agent = SarcasmAgent(client=client)

        result = agent.detect(self._payload("太好了，周末又能继续改需求了。"))

        self.assertTrue(result.is_sarcasm)
        self.assertEqual(result.surface_emotion, "开心")
        self.assertEqual(result.true_emotion, "厌烦")
        self.assertEqual(result.revised_intensity, 74)
        self.assertEqual(client.last_payload.text, "太好了，周末又能继续改需求了。")

    def test_detect_dict_returns_expected_shape(self) -> None:
        client = FakeSarcasmLLMClient(
            {
                "is_sarcasm": False,
                "surface_emotion": "中性",
                "true_emotion": "中性",
                "revised_intensity": 20,
                "confidence": 0.56,
                "reason": "没有明显反讽结构。",
            }
        )
        agent = SarcasmAgent(client=client)

        result = agent.detect_dict(self._payload("今天按计划推进。"))

        self.assertFalse(result["is_sarcasm"])
        self.assertEqual(result["true_emotion"], "中性")
        self.assertIn("reason", result)

    def test_build_messages_contains_schema_and_text(self) -> None:
        client = FakeSarcasmLLMClient(
            {
                "is_sarcasm": True,
                "surface_emotion": "开心",
                "true_emotion": "厌烦",
                "revised_intensity": 69,
                "confidence": 0.82,
                "reason": "夸张正向和负向事件冲突。",
            }
        )
        agent = SarcasmAgent(client=client)

        messages = agent.build_messages(self._payload("真棒，凌晨还在改。"))

        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")
        self.assertIn("真棒，凌晨还在改。", messages[1]["content"])
        self.assertIn("is_sarcasm", messages[0]["content"])

    def test_invalid_true_emotion_raises(self) -> None:
        client = FakeSarcasmLLMClient(
            {
                "is_sarcasm": True,
                "surface_emotion": "开心",
                "true_emotion": "疲倦",
                "revised_intensity": 70,
                "confidence": 0.8,
                "reason": "标签不在范围内。",
            }
        )
        agent = SarcasmAgent(client=client)

        with self.assertRaises(ValueError):
            agent.detect(self._payload("谢谢你让我周末也这么充实。"))

    def test_invalid_intensity_range_raises(self) -> None:
        client = FakeSarcasmLLMClient(
            {
                "is_sarcasm": True,
                "surface_emotion": "开心",
                "true_emotion": "厌烦",
                "revised_intensity": 101,
                "confidence": 0.9,
                "reason": "越界强度。",
            }
        )
        agent = SarcasmAgent(client=client)

        with self.assertRaises(ValueError):
            agent.detect(self._payload("太好了，又来活了。"))

    def test_invalid_is_sarcasm_type_raises(self) -> None:
        client = FakeSarcasmLLMClient(
            {
                "is_sarcasm": "true",
                "surface_emotion": "开心",
                "true_emotion": "厌烦",
                "revised_intensity": 70,
                "confidence": 0.78,
                "reason": "类型错误。",
            }
        )
        agent = SarcasmAgent(client=client)

        with self.assertRaises(ValueError):
            agent.detect(self._payload("真棒，又要加班了。"))


if __name__ == "__main__":
    unittest.main()
