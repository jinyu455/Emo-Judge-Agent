from __future__ import annotations

import unittest
import sys
from pathlib import Path 

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from router_agent import RouterAgent
from router_agent.schemas import RouterInput

class FakeRouterLLMClient:
    def __init__(self, result: dict) -> None:
        self.result = result
        self.last_payload: RouterInput | None = None

    def classify(self, payload: RouterInput) -> dict:
        self.last_payload = payload
        return self.result


class RouterAgentTestCase(unittest.TestCase):
    def _payload(self, text: str) -> dict:
        return {
            "id": "msg_001",
            "user_id": "u_1001",
            "text": text,
            "source": "chat",
            "created_at": "2026-03-24T14:00:00",
        }

    def test_route_direct_from_llm_result(self) -> None:
        client = FakeRouterLLMClient(
            {
                "sample_type": "direct",
                "need_sarcasm_check": False,
                "need_mix_check": False,
                "routing_reason": "直接表达焦虑情绪。",
                "evidence": ["明显情绪词: 焦虑"],
            }
        )
        agent = RouterAgent(client=client)

        result = agent.route(self._payload("我现在特别焦虑"))

        self.assertEqual(result.sample_type, "direct")
        self.assertFalse(result.need_sarcasm_check)
        self.assertFalse(result.need_mix_check)
        self.assertEqual(client.last_payload.text, "我现在特别焦虑")

    def test_route_sarcasm_suspected_from_llm_result(self) -> None:
        client = FakeRouterLLMClient(
            {
                "sample_type": "sarcasm_suspected",
                "need_sarcasm_check": True,
                "need_mix_check": False,
                "routing_reason": "表面正向，场景负向，疑似反讽。",
                "evidence": ["太好了", "周末继续改需求"],
            }
        )
        agent = RouterAgent(client=client)

        result = agent.route(self._payload("太好了，周末又能继续改需求了。"))

        self.assertEqual(result.sample_type, "sarcasm_suspected")
        self.assertTrue(result.need_sarcasm_check)
        self.assertFalse(result.need_mix_check)

    def test_build_messages_contains_schema_and_text(self) -> None:
        client = FakeRouterLLMClient(
            {
                "sample_type": "mix",
                "need_sarcasm_check": False,
                "need_mix_check": True,
                "routing_reason": "包含转折和复合情绪。",
                "evidence": ["开心", "累", "但"],
            }
        )
        agent = RouterAgent(client=client)

        messages = agent.build_messages(self._payload("开心是开心，但也挺累"))

        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")
        self.assertIn("开心是开心，但也挺累", messages[1]["content"])
        self.assertIn("sample_type", messages[0]["content"])

    def test_invalid_sample_type_raises(self) -> None:
        client = FakeRouterLLMClient(
            {
                "sample_type": "unknown",
                "need_sarcasm_check": False,
                "need_mix_check": False,
                "routing_reason": "invalid",
                "evidence": [],
            }
        )
        agent = RouterAgent(client=client)

        with self.assertRaises(ValueError):
            agent.route(self._payload("test"))


if __name__ == "__main__":
    unittest.main()
