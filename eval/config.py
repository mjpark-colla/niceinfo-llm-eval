"""평가 설정 (모델, judge, 벤치 가중치)."""
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class ModelConfig:
    name: str                              # HuggingFace 모델명
    display_name: str                      # 결과 jsonl에 기록할 이름
    quantization: str                      # "awq-4bit", "bf16", "fp8"
    serving: Literal["vllm", "openai"] = "vllm"
    base_url: str = "http://vllm:8000/v1"
    api_key_env: str = "VLLM_API_KEY"      # vLLM은 기본 EMPTY
    max_tokens: int = 2048
    temperature: float = 0.0
    enable_thinking: bool = False          # Qwen3 thinking 토글
    strip_thinking: bool = True            # <think>...</think> 제거


@dataclass
class JudgeConfig:
    # GPT-4 family가 LLM-as-Judge 사실상 표준 (Zheng et al. NeurIPS 2023).
    # gpt-4o가 현재 GPT-4 family 대표 — 한국어 평가에서도 KMMLU 등이 채택.
    # 근거: references/judge-model-choice.md
    primary_model: str = "gpt-4o"
    fallback_model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 500
    max_retries: int = 3


@dataclass
class BenchmarkSpec:
    name: str
    sample_limit: int | None = None        # None = 전체
    weight: float = 0.0                    # 종합 점수 가중치
    seed: int = 42


# 평가 대상 모델 (운영 후보군)
TARGET_MODELS = [
    ModelConfig(
        name="Qwen/Qwen3-32B-AWQ",
        display_name="Qwen3-32B-AWQ",
        quantization="awq-4bit",
    ),
    ModelConfig(
        name="Qwen/Qwen3-30B-A3B",
        display_name="Qwen3-30B-A3B-BF16",
        quantization="bf16",
    ),
    ModelConfig(
        name="Qwen/Qwen3.6-35B-A3B",
        display_name="Qwen3.6-35B-A3B-BF16",
        quantization="bf16",
    ),
    # 공식 FP8 양자화 — H100 NVL native 지원, 운영 환경 최적
    ModelConfig(
        name="Qwen/Qwen3.6-35B-A3B-FP8",
        display_name="Qwen3.6-35B-A3B-FP8",
        quantization="fp8",
    ),
]


# Judge 설정
JUDGE = JudgeConfig()


# 벤치마크별 설정 (CLAUDE.md 5-3-1 확정 내용)
BENCHMARKS = {
    "ko_mt_bench":  BenchmarkSpec("ko_mt_bench",  sample_limit=80,  weight=0.25),
    "logickor":     BenchmarkSpec("logickor",     sample_limit=42,  weight=0.15),
    "ko_ifeval":    BenchmarkSpec("ko_ifeval",    sample_limit=150, weight=0.20),
    "aihub_582":    BenchmarkSpec("aihub_582",    sample_limit=150, weight=0.20),
    "aihub_90":     BenchmarkSpec("aihub_90",     sample_limit=150, weight=0.20),
}


# 가중치 합 검증
assert abs(sum(b.weight for b in BENCHMARKS.values()) - 1.0) < 1e-6, \
    "Benchmark weights must sum to 1.0"


# 비용 (1M 토큰 USD, 2026-05 기준)
OPENAI_PRICING = {
    "gpt-5":         {"input": 5.00,  "output": 20.00},
    "gpt-5-mini":    {"input": 0.50,  "output": 2.00},
    "gpt-4o":        {"input": 5.00,  "output": 15.00},
    "gpt-4o-mini":   {"input": 0.15,  "output": 0.60},
}
