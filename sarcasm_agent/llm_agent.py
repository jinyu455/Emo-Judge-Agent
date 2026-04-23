from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any, Protocol

from .schemas import SarcasmInput, SarcasmResult


EMOTION_LABELS = {"开心", "悲伤", "愤怒", "焦虑", "厌烦", "中性","疲惫","失落","无奈",}

SYSTEM_PROMPT = """你是情绪识别系统中的 Sarcasm Agent。

你的任务是专门识别“反讽表达”，不要输出与任务无关的信息。
该模块通常在 Router 给出 need_sarcasm_check=true 时被调用。

你需要一次性完成：
1. 判断是否反讽 is_sarcasm
2. 给出句面情绪 surface_emotion（按表层词面判断）
3. 给出真实情绪 true_emotion（结合语境修正后）
4. 给出修正后的强度 revised_intensity
5. 给出置信度 confidence
6. 给出简短解释 reason

主情绪标签只能从以下 9 类中选择：
- 开心
- 悲伤
- 愤怒
- 焦虑
- 厌烦
- 中性
- 疲惫
- 失落
- 无奈


反讽判断重点：
- 正向词 + 负向事件
- 夸张赞美 + 抱怨语境
- 重复受害信号（如“又”）
- 负面场景（加班、改需求、被催、深夜开会等）

输出规则：
- surface_emotion / true_emotion 只能从上述标签中选择
- revised_intensity 是 0 到 100 的整数
- confidence 是 0 到 1 的小数
- reason 用一句中文解释，不超过 90 字

输出要求：
- 只返回 JSON
- 不要输出 markdown
- 字段必须完整
- 不要增加额外字段

输出格式：
{
  "is_sarcasm": true,
  "surface_emotion": "开心",
  "true_emotion": "厌烦",
  "revised_intensity": 74,
  "confidence": 0.85,
  "reason": "表面正向词与负面工作场景形成反差，真实情绪更偏厌烦"
}
"""


class SarcasmLLMClient(Protocol):
    def analyze(self, payload: SarcasmInput) -> dict[str, Any]:
        """Send payload to an LLM and return the parsed JSON result."""


class SarcasmAgent:
    """LLM-based sarcasm agent."""

    def __init__(self, client: SarcasmLLMClient) -> None:
        self.client = client

    def detect(self, payload: SarcasmInput | dict[str, Any]) -> SarcasmResult:
        item = payload if isinstance(payload, SarcasmInput) else SarcasmInput(**payload)
        raw_result = self.client.analyze(item)
        return self._build_result(raw_result)

    def detect_dict(self, payload: SarcasmInput | dict[str, Any]) -> dict[str, Any]:
        return self.detect(payload).to_dict()

    def _build_result(self, raw_result: dict[str, Any]) -> SarcasmResult:
        surface_emotion = str(raw_result.get("surface_emotion", "")).strip()
        true_emotion = str(raw_result.get("true_emotion", "")).strip()
        if surface_emotion not in EMOTION_LABELS:
            raise ValueError(f"Invalid surface_emotion from LLM: {surface_emotion!r}")
        if true_emotion not in EMOTION_LABELS:
            raise ValueError(f"Invalid true_emotion from LLM: {true_emotion!r}")

        revised_intensity = self._coerce_int(raw_result.get("revised_intensity"), "revised_intensity")
        if not 0 <= revised_intensity <= 100:
            raise ValueError(f"Invalid revised_intensity from LLM: {revised_intensity!r}")

        confidence = self._coerce_float(raw_result.get("confidence"), "confidence")
        if not 0 <= confidence <= 1:
            raise ValueError(f"Invalid confidence from LLM: {confidence!r}")

        return SarcasmResult(
            is_sarcasm=self._coerce_bool(raw_result.get("is_sarcasm"), "is_sarcasm"),
            surface_emotion=surface_emotion,
            true_emotion=true_emotion,
            revised_intensity=revised_intensity,
            confidence=confidence,
            reason=str(raw_result.get("reason", "")).strip(),
        )

    def _coerce_bool(self, value: Any, field_name: str) -> bool:
        if isinstance(value, bool):
            return value
        raise ValueError(f"Invalid {field_name} from LLM: {value!r}")

    def _coerce_int(self, value: Any, field_name: str) -> int:
        if isinstance(value, bool):
            raise ValueError(f"Invalid {field_name} from LLM: {value!r}")
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid {field_name} from LLM: {value!r}") from exc

    def _coerce_float(self, value: Any, field_name: str) -> float:
        if isinstance(value, bool):
            raise ValueError(f"Invalid {field_name} from LLM: {value!r}")
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid {field_name} from LLM: {value!r}") from exc

    def build_messages(self, payload: SarcasmInput | dict[str, Any]) -> list[dict[str, str]]:
        item = payload if isinstance(payload, SarcasmInput) else SarcasmInput(**payload)
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": self._build_user_prompt(item)},
        ]

    def _build_user_prompt(self, payload: SarcasmInput) -> str:
        return (
            "请判断下面这条消息是否反讽，并返回 JSON 结果。\n\n"
            f"{json.dumps(asdict(payload), ensure_ascii=False, indent=2)}"
        )
