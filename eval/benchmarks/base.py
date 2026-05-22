"""모든 벤치마크의 공통 인터페이스."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterable, Literal, Any


@dataclass
class Sample:
    """평가용 단일 샘플."""
    id: str
    prompt: str                            # 모델 입력 (system/user 분리는 벤치별 처리)
    reference: str | list[str] | None = None  # 정답 (있을 때, 자동 메트릭용)
    metadata: dict[str, Any] = field(default_factory=dict)
    # multi-turn 지원: 후속 prompt들
    follow_up_prompts: list[str] = field(default_factory=list)


@dataclass
class TurnResult:
    """한 turn의 평가 결과."""
    turn: int                              # 1, 2, ...
    prompt: str
    model_output: str
    score: float = 0.0
    metric_details: dict[str, Any] = field(default_factory=dict)
    judge_raw: str | None = None
    tokens_in: int = 0
    tokens_out: int = 0
    judge_tokens_in: int = 0
    judge_tokens_out: int = 0


@dataclass
class EvalResult:
    """한 sample의 전체 평가 결과 (멀티턴 포함)."""
    sample_id: str
    model: str                             # 평가 대상 display_name
    benchmark: str
    turns: list[TurnResult] = field(default_factory=list)
    score: float = 0.0                     # 샘플 종합 점수 (turn 평균)
    elapsed_sec: float = 0.0
    cost_usd: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "sample_id": self.sample_id,
            "model": self.model,
            "benchmark": self.benchmark,
            "score": self.score,
            "turns": [
                {
                    "turn": t.turn,
                    "prompt": t.prompt,
                    "model_output": t.model_output,
                    "score": t.score,
                    "metric_details": t.metric_details,
                    "judge_raw": t.judge_raw,
                    "tokens_in": t.tokens_in,
                    "tokens_out": t.tokens_out,
                    "judge_tokens_in": t.judge_tokens_in,
                    "judge_tokens_out": t.judge_tokens_out,
                }
                for t in self.turns
            ],
            "elapsed_sec": self.elapsed_sec,
            "cost_usd": self.cost_usd,
            "metadata": self.metadata,
        }


class Benchmark(ABC):
    """모든 벤치마크가 구현해야 하는 인터페이스."""
    name: str                              # "ko_mt_bench"
    metric_type: Literal["judge", "auto"]  # judge 또는 auto

    @abstractmethod
    def samples(self, limit: int | None = None, seed: int = 42) -> Iterable[Sample]:
        """평가용 샘플 yield. limit 만큼만 반환."""
        ...

    @abstractmethod
    async def evaluate_turn(
        self,
        sample: Sample,
        turn_idx: int,
        prompt: str,
        model_output: str,
        judge=None,
    ) -> TurnResult:
        """한 turn 평가하여 TurnResult 반환 (async)."""
        ...

    def is_multi_turn(self) -> bool:
        """멀티턴 벤치 여부."""
        return False
