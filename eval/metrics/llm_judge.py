"""LLM-as-Judge 호출 (OpenAI gpt-4o primary, gpt-4o-mini fallback). 비동기 버전."""
import os
import re
import logging
from openai import AsyncOpenAI, NotFoundError, APIError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from eval.config import JudgeConfig
from eval.utils.prompts import build_judge_prompt
from eval.utils.cost import calc_openai_cost

log = logging.getLogger(__name__)


class LLMJudge:
    """OpenAI 기반 judge. primary 실패 시 fallback 자동 전환."""

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

    async def score(
        self,
        question: str,
        answer: str,
        reference: str | list[str] | None = None,
        context: str | None = None,
    ) -> dict:
        """답변을 1~10점으로 채점 (async)."""
        system, user = build_judge_prompt(question, answer, reference, context)

        try:
            raw, tin, tout = await self._call(system, user, self._active_model)
        except NotFoundError:
            if not self._fallback_attempted:
                log.warning(
                    f"Judge {self._active_model} 사용 불가, "
                    f"{self.config.fallback_model}로 전환"
                )
                self._active_model = self.config.fallback_model
                self._fallback_attempted = True
            raw, tin, tout = await self._call(system, user, self._active_model)

        score = self._parse_score(raw)
        cost = calc_openai_cost(self._active_model, tin, tout)

        return {
            "score": score,
            "raw": raw,
            "judge_model": self._active_model,
            "tokens_in": tin,
            "tokens_out": tout,
            "cost_usd": cost,
        }

    @staticmethod
    def _parse_score(raw: str) -> float:
        """raw response에서 '점수: N' 추출. 실패 시 0.0."""
        m = re.search(r"점수\s*[:：]\s*(\d+(?:\.\d+)?)", raw)
        if m:
            score = float(m.group(1))
            return max(1.0, min(10.0, score))
        return 0.0
