from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any, Protocol

from .schemas import MixInput, MixResult


MIX_EMOTION_LABELS = {"开心", "悲伤", "愤怒", "焦虑", "厌烦", "中性","疲惫","失落","无奈",}

SYSTEM_PROMPT = """你是情绪识别系统中的 Mix Agent。

你的任务是处理“单标签难以表达”的复杂文本，重点关注：
1. 是否混合情绪 is_mixed
2. 主情绪 primary_emotion
3. 次情绪 secondary_emotion
4. 情绪比例 mix_ratio
5. 修正强度 revised_intensity
6. 置信度 confidence
7. 简短解释 reason

你需要重点识别：
- 转折结构（但、但是、不过、然而、就是、只是）
- 模糊低能量表达（提不起劲、说不上来、还好但空）
- 同句中的双向情绪（轻松但空、开心但累）

情绪标签建议从以下集合中选择：
- 开心
- 悲伤
- 愤怒
- 焦虑
- 厌烦
- 中性
- 疲惫
- 失落
- 无奈

输出规则：
- is_mixed 为布尔值
- primary_emotion / secondary_emotion 必须是单个标签
- mix_ratio 为对象，至少包含 primary_emotion 与 secondary_emotion 两个键
- mix_ratio 的值为 0 到 1 的小数，整体和接近 1（允许轻微浮动）
- revised_intensity 是 0 到 100 的整数
- confidence 是 0 到 1 的小数
- reason 用一句中文说明，不超过 100 字

输出要求：
- 只返回 JSON
- 不要输出 markdown
- 字段必须完整
- 不要增加额外字段

输出格式：
{
  "is_mixed": true,
  "primary_emotion": "疲惫",
  "secondary_emotion": "开心",
  "mix_ratio": {
    "疲惫": 0.58,
    "开心": 0.42
  },
  "revised_intensity": 57,
  "confidence": 0.79,
  "reason": "句子存在转折结构“但”，前半句偏正向，后半句突出疲惫感，属于混合情绪"
}
"""


class MixLLMClient(Protocol):
    def analyze(self, payload: MixInput) -> dict[str, Any]:
        """Send payload to an LLM and return the parsed JSON result."""


class MixAgent:
    """LLM-based mix emotion agent."""

    def __init__(self, client: MixLLMClient) -> None:
        self.client = client

    def mixRe(self, payload: MixInput | dict[str, Any]) -> MixResult:
        item = payload if isinstance(payload, MixInput) else MixInput(**payload)
        raw_result = self.client.analyze(item)
        return self._build_result(raw_result)

    def mixRe_dict(self, payload: MixInput | dict[str, Any]) -> dict[str, Any]:
        return self.mixRe(payload).to_dict()

    def _build_result(self, raw_result: dict[str, Any]) -> MixResult:
        primary_emotion = str(raw_result.get("primary_emotion", "")).strip()
        secondary_emotion = str(raw_result.get("secondary_emotion", "")).strip()

        if primary_emotion not in MIX_EMOTION_LABELS:
            raise ValueError(f"Invalid primary_emotion from LLM: {primary_emotion!r}")
        if secondary_emotion not in MIX_EMOTION_LABELS:
            raise ValueError(f"Invalid secondary_emotion from LLM: {secondary_emotion!r}")

        mix_ratio = self._coerce_mix_ratio(raw_result.get("mix_ratio"), primary_emotion, secondary_emotion)
        revised_intensity = self._coerce_int(raw_result.get("revised_intensity"), "revised_intensity")
        if not 0 <= revised_intensity <= 100:
            raise ValueError(f"Invalid revised_intensity from LLM: {revised_intensity!r}")

        confidence = self._coerce_float(raw_result.get("confidence"), "confidence")
        if not 0 <= confidence <= 1:
            raise ValueError(f"Invalid confidence from LLM: {confidence!r}")

        return MixResult(
            is_mixed=self._coerce_bool(raw_result.get("is_mixed"), "is_mixed"),
            primary_emotion=primary_emotion,
            secondary_emotion=secondary_emotion,
            mix_ratio=mix_ratio,
            revised_intensity=revised_intensity,
            confidence=confidence,
            reason=str(raw_result.get("reason", "")).strip(),
        )

    def _coerce_mix_ratio(
        self,
        value: Any,
        primary_emotion: str,
        secondary_emotion: str,
    ) -> dict[str, float]:
        if not isinstance(value, dict) or not value:
            raise ValueError("Invalid mix_ratio from LLM: expected non-empty dict")

        ratio: dict[str, float] = {}
        for emotion, amount in value.items():
            key = str(emotion).strip()
            if key not in MIX_EMOTION_LABELS:
                raise ValueError(f"Invalid mix_ratio emotion from LLM: {key!r}")
            ratio[key] = self._coerce_float(amount, "mix_ratio")
            if ratio[key] < 0 or ratio[key] > 1:
                raise ValueError(f"Invalid mix_ratio value from LLM: {ratio[key]!r}")

        if primary_emotion not in ratio or secondary_emotion not in ratio:
            raise ValueError("Invalid mix_ratio from LLM: missing primary/secondary emotion keys")

        total = sum(ratio.values())
        if not 0.95 <= total <= 1.05:
            raise ValueError(f"Invalid mix_ratio sum from LLM: {total!r}")

        return ratio

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

    def build_messages(self, payload: MixInput | dict[str, Any]) -> list[dict[str, str]]:
        item = payload if isinstance(payload, MixInput) else MixInput(**payload)
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": self._build_user_prompt(item)},
        ]

    def _build_user_prompt(self, payload: MixInput) -> str:
        return (
            "请判断下面这条消息是否属于混合情绪，并返回 JSON 结果。\n\n"
            f"{json.dumps(asdict(payload), ensure_ascii=False, indent=2)}"
        )
