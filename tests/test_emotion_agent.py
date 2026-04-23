from __future__ import annotations

import unittest
import sys
from pathlib import Path 

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from emotion_agent import EmotionAgent
from emotion_agent.schemas import EmotionInput


class FakeEmotionLLMClient:
    def __init__(self, result: dict) -> None:
        self.result = result
        self.last_payload: EmotionInput | None = None

    def analyze(self, payload: EmotionInput) -> dict:
        self.last_payload = payload
        return self.result


class EmotionAgentTestCase(unittest.TestCase):
    def _payload(self, text: str) -> dict:
        return {
            "id": "msg_001",
            "user_id": "u_1001",
            "text": text,
            "source": "chat",
            "created_at": "2026-03-24T14:00:00",
        }

    def test_analyze_surface_emotion_from_llm_result(self) -> None:
        client = FakeEmotionLLMClient(
            {
                "tokens": ["太好了", "周末", "又", "能", "继续", "改", "需求"],
                "emotion_words": ["太好了"],
                "degree_words": [],
                "negation_words": [],
                "contrast_words": [],
                "emotion": "开心",
                "intensity": 62,
                "confidence": 0.61,
                "reason": "文本表面存在明显正向表达“太好了”，情绪方向初步判为正向",
            }
        )
        agent = EmotionAgent(client=client)

        result = agent.emotionRe(self._payload("太好了，周末又能继续改需求了。"))

        self.assertEqual(result.emotion, "开心")
        self.assertEqual(result.intensity, 62)
        self.assertEqual(result.confidence, 0.61)
        self.assertEqual(result.emotion_words, ["太好了"])
        self.assertEqual(client.last_payload.text, "太好了，周末又能继续改需求了。")

    def test_analyze_dict_returns_expected_shape(self) -> None:
        client = FakeEmotionLLMClient(
            {
                "tokens": ["我", "现在", "特别", "焦虑"],
                "emotion_words": ["焦虑"],
                "degree_words": ["特别"],
                "negation_words": [],
                "contrast_words": [],
                "emotion": "焦虑",
                "intensity": 78,
                "confidence": 0.86,
                "reason": "文本直接表达焦虑，并由“特别”增强强度。",
            }
        )
        agent = EmotionAgent(client=client)

        result = agent.emotionRe_dict(self._payload("我现在特别焦虑"))

        self.assertEqual(result["emotion"], "焦虑")
        self.assertEqual(result["degree_words"], ["特别"])
        self.assertIn("reason", result)

    def test_build_messages_contains_schema_and_text(self) -> None:
        client = FakeEmotionLLMClient(
            {
                "tokens": [],
                "emotion_words": [],
                "degree_words": [],
                "negation_words": [],
                "contrast_words": [],
                "emotion": "中性",
                "intensity": 10,
                "confidence": 0.5,
                "reason": "无明显情绪词。",
            }
        )
        agent = EmotionAgent(client=client)

        messages = agent.build_messages(self._payload("今天正常开会"))

        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")
        self.assertIn("今天正常开会", messages[1]["content"])
        self.assertIn("emotion_words", messages[0]["content"])

    def test_invalid_emotion_raises(self) -> None:
        client = FakeEmotionLLMClient(
            {
                "tokens": ["累"],
                "emotion_words": ["累"],
                "degree_words": [],
                "negation_words": [],
                "contrast_words": [],
                "emotion": "疲惫",
                "intensity": 55,
                "confidence": 0.7,
                "reason": "第一版暂不支持该标签。",
            }
        )
        agent = EmotionAgent(client=client)

        with self.assertRaises(ValueError):
            agent.emotionRe(self._payload("我好累"))

    def test_invalid_score_range_raises(self) -> None:
        client = FakeEmotionLLMClient(
            {
                "tokens": ["烦"],
                "emotion_words": ["烦"],
                "degree_words": [],
                "negation_words": [],
                "contrast_words": [],
                "emotion": "厌烦",
                "intensity": 101,
                "confidence": 0.7,
                "reason": "强度越界。",
            }
        )
        agent = EmotionAgent(client=client)

        with self.assertRaises(ValueError):
            agent.emotionRe(self._payload("烦"))


if __name__ == "__main__":
    unittest.main()
