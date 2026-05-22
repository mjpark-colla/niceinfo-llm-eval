"""평가 대상 모델 호출 추상화 (vLLM = OpenAI 호환 API). 비동기 버전."""
import os
import time
import logging
from openai import AsyncOpenAI, APIError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from eval.config import ModelConfig
from eval.utils.io import strip_thinking

log = logging.getLogger(__name__)


class TargetClient:
    """평가 대상 모델 호출. vLLM OpenAI 호환 API.

    비동기 호출 지원. 멀티턴 대화 지원.
    """

    def __init__(self, config: ModelConfig):
        self.config = config
        api_key = os.environ.get(config.api_key_env or "VLLM_API_KEY", "EMPTY")
        self.client = AsyncOpenAI(api_key=api_key, base_url=config.base_url)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=15),
        retry=retry_if_exception_type(APIError),
    )
    async def _call(self, messages: list[dict]) -> tuple[str, int, int, float]:
        """OpenAI 호환 호출 → (text, tokens_in, tokens_out, elapsed_sec)."""
        t0 = time.perf_counter()
        extra = {}
        if not self.config.enable_thinking:
            extra["chat_template_kwargs"] = {"enable_thinking": False}

        resp = await self.client.chat.completions.create(
            model=self.config.name,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            extra_body=extra if extra else None,
        )
        elapsed = time.perf_counter() - t0

        text = resp.choices[0].message.content or ""
        if self.config.strip_thinking:
            text = strip_thinking(text)

        return text, resp.usage.prompt_tokens, resp.usage.completion_tokens, elapsed

    async def chat_single(self, prompt: str) -> dict:
        """단일 turn 호출."""
        text, tin, tout, elapsed = await self._call([
            {"role": "user", "content": prompt}
        ])
        return {
            "text": text,
            "tokens_in": tin,
            "tokens_out": tout,
            "elapsed_sec": elapsed,
        }

    async def chat_multi_turn(self, prompts: list[str]) -> list[dict]:
        """멀티턴 호출. 각 turn마다 이전 대화 history 누적."""
        messages: list[dict] = []
        results = []
        for prompt in prompts:
            messages.append({"role": "user", "content": prompt})
            text, tin, tout, elapsed = await self._call(messages)
            messages.append({"role": "assistant", "content": text})
            results.append({
                "text": text,
                "tokens_in": tin,
                "tokens_out": tout,
                "elapsed_sec": elapsed,
            })
        return results
