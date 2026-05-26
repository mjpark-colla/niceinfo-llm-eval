"""LLM-as-Judge 호출. Task-specific prompt 적용 (G-Eval / MT-Bench / LogicKor).

- primary: gpt-4o, fallback: gpt-4o-mini
- task_type: "summarization" | "dialogue" | "reasoning"
- 출력 형식:
  - summarization: 4 차원 + 종합 (G-Eval)
  - dialogue/reasoning: Rating: [[N]] (MT-Bench 표준)
"""
import os
import re
import logging
from typing import Literal
from openai import AsyncOpenAI, NotFoundError, APIError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from eval.config import JudgeConfig
from eval.utils.prompts import (
    build_summarization_prompt,
    build_dialogue_prompt,
    build_reasoning_prompt,
    build_judge_prompt,  # legacy fallback
)
from eval.utils.cost import calc_openai_cost

log = logging.getLogger(__name__)

TaskType = Literal["summarization", "dialogue", "reasoning"]


class LLMJudge:
    """OpenAI 기반 judge. task-specific prompt 적용."""

    def __init__(self, config: JudgeConfig):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY 환경변수가 필요합니다.")
        self.client = AsyncOpenAI(api_key=api_key)
        self.config = config
        self._active_model = config.primary_model
        self._fallback_attempted = False

    @property
    def model(self) -> str:
        return self._active_model

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=20),
        retry=retry_if_exception_type(APIError),
    )
    async def _call(self, system: str, user: str, model: str) -> tuple[str, int, int]:
        """OpenAI Chat Completions 호출 → (raw_text, tokens_in, tokens_out)."""
        resp = await self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        return (
            resp.choices[0].message.content or "",
            resp.usage.prompt_tokens,
            resp.usage.completion_tokens,
        )

    async def _invoke(self, system: str, user: str) -> tuple[str, int, int]:
        """primary 호출, NotFound 시 fallback 자동 전환."""
        try:
            return await self._call(system, user, self._active_model)
        except NotFoundError:
            if not self._fallback_attempted:
                log.warning(
                    f"Judge {self._active_model} 사용 불가, "
                    f"{self.config.fallback_model}로 전환"
                )
                self._active_model = self.config.fallback_model
                self._fallback_attempted = True
            return await self._call(system, user, self._active_model)

    # ============================================================
    # Task-specific scoring entries
    # ============================================================
    async def score_summarization(
        self,
        passage: str,
        summary: str,
        reference: str | list[str] | None = None,
    ) -> dict:
        """G-Eval 요약 평가."""
        system, user = build_summarization_prompt(passage, summary, reference)
        raw, tin, tout = await self._invoke(system, user)
        score, details = _parse_summarization(raw)
        cost = calc_openai_cost(self._active_model, tin, tout)
        return {
            "score": score,
            "raw": raw,
            "judge_model": self._active_model,
            "tokens_in": tin,
            "tokens_out": tout,
            "cost_usd": cost,
            "details": details,
            "task_type": "summarization",
        }

    async def score_dialogue(
        self,
        question: str,
        answer: str,
        reference: str | list[str] | None = None,
        prev_question: str | None = None,
        prev_answer: str | None = None,
    ) -> dict:
        """MT-Bench 스타일 single/multi-turn 대화 평가."""
        system, user = build_dialogue_prompt(
            question, answer, reference, prev_question, prev_answer
        )
        raw, tin, tout = await self._invoke(system, user)
        score = _parse_rating(raw)
        cost = calc_openai_cost(self._active_model, tin, tout)
        return {
            "score": score,
            "raw": raw,
            "judge_model": self._active_model,
            "tokens_in": tin,
            "tokens_out": tout,
            "cost_usd": cost,
            "details": {},
            "task_type": "dialogue",
        }

    async def score_reasoning(
        self,
        question: str,
        answer: str,
        reference: str | list[str] | None = None,
        prev_question: str | None = None,
        prev_answer: str | None = None,
    ) -> dict:
        """LogicKor 스타일 추론 평가."""
        system, user = build_reasoning_prompt(
            question, answer, reference, prev_question, prev_answer
        )
        raw, tin, tout = await self._invoke(system, user)
        score = _parse_rating(raw)
        cost = calc_openai_cost(self._active_model, tin, tout)
        return {
            "score": score,
            "raw": raw,
            "judge_model": self._active_model,
            "tokens_in": tin,
            "tokens_out": tout,
            "cost_usd": cost,
            "details": {},
            "task_type": "reasoning",
        }

    # ============================================================
    # Legacy entry (기존 호출자 호환)
    # ============================================================
    async def score(
        self,
        question: str,
        answer: str,
        reference: str | list[str] | None = None,
        context: str | None = None,
    ) -> dict:
        """Legacy generic — task type 불명 시 dialogue로 처리."""
        system, user = build_judge_prompt(question, answer, reference, context)
        raw, tin, tout = await self._invoke(system, user)
        score = _parse_rating(raw, fallback_legacy=True)
        cost = calc_openai_cost(self._active_model, tin, tout)
        return {
            "score": score,
            "raw": raw,
            "judge_model": self._active_model,
            "tokens_in": tin,
            "tokens_out": tout,
            "cost_usd": cost,
            "details": {},
            "task_type": "legacy",
        }


# ============================================================
# 파싱 함수들
# ============================================================
_RATING_PATTERNS = [
    re.compile(r"Rating\s*[:：]\s*\[\[\s*(\d+(?:\.\d+)?)\s*\]\]", re.IGNORECASE),
    re.compile(r"\[\[\s*(\d+(?:\.\d+)?)\s*\]\]"),
    re.compile(r"점수\s*[:：]\s*(\d+(?:\.\d+)?)"),
    re.compile(r"Rating\s*[:：]\s*(\d+(?:\.\d+)?)", re.IGNORECASE),
]


def _parse_rating(raw: str, fallback_legacy: bool = False) -> float:
    """raw response에서 Rating 점수 추출.

    우선순위: 'Rating: [[N]]' (MT-Bench 표준) → '[[N]]' → '점수: N' → 'Rating: N'.
    실패 시 0.0 (caller가 error 감지 가능).
    """
    for pattern in _RATING_PATTERNS:
        m = pattern.search(raw)
        if m:
            try:
                score = float(m.group(1))
                return max(1.0, min(10.0, score))
            except ValueError:
                continue
    return 0.0


def _parse_summarization(raw: str) -> tuple[float, dict]:
    """G-Eval 요약 응답에서 4차원 점수 + 종합 추출.

    형식:
        사실 충실도: [1-5]
        핵심 정보 포함도: [1-5]
        일관성: [1-5]
        간결성: [1-5]
        종합: [1.0-10.0]
        이유: ...

    Returns:
        (종합 점수 0-10, 차원별 details dict)
    """
    dim_patterns = {
        "faithfulness": [r"사실\s*충실도", r"faithfulness"],
        "relevance": [r"핵심\s*정보\s*포함도", r"핵심\s*정보", r"relevance"],
        "coherence": [r"일관성", r"coherence"],
        "conciseness": [r"간결성", r"conciseness"],
    }

    details = {}
    for key, patterns in dim_patterns.items():
        for p in patterns:
            m = re.search(rf"{p}\s*[:：]\s*\[?\s*(\d+(?:\.\d+)?)\s*\]?", raw, re.IGNORECASE)
            if m:
                try:
                    val = float(m.group(1))
                    details[key] = max(1.0, min(5.0, val))
                    break
                except ValueError:
                    continue

    # 종합 추출 — judge가 "종합: [9.6]" 같이 대괄호 표기하는 케이스도 잡음
    overall = None
    m = re.search(r"종합\s*(?:점수)?\s*[:：]\s*\[?\s*(\d+(?:\.\d+)?)\s*\]?", raw)
    if m:
        try:
            overall = float(m.group(1))
            overall = max(1.0, min(10.0, overall))
        except ValueError:
            pass

    # 종합 미추출 시 4차원 평균 × 2로 계산
    if overall is None and len(details) == 4:
        overall = sum(details.values()) / 4 * 2
        overall = max(1.0, min(10.0, overall))
    elif overall is None and len(details) > 0:
        # 일부만 추출된 경우, 추출된 차원의 평균 × 2
        overall = sum(details.values()) / len(details) * 2
        overall = max(1.0, min(10.0, overall))
    elif overall is None:
        # 전부 실패
        overall = 0.0

    return overall, details
