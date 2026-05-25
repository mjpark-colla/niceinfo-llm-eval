# LLM 평가·운영 핵심 개념

> 작성일: 2026-05-22
> 프로젝트: niceinfo
> 본 문서: 평가 진행 중 등장한 주요 개념·기술 정리

---

## 목차

1. [양자화 (Quantization)](#1-양자화-quantization)
2. [Context Length (max_model_len)](#2-context-length-max_model_len)
3. [vLLM 메모리 옵션](#3-vllm-메모리-옵션)
4. [Tokenizer](#4-tokenizer)
5. [Async / Concurrency](#5-async--concurrency)
6. [LLM-as-Judge](#6-llm-as-judge)
7. [BERTScore num_layers](#7-bertscore-num_layers)
8. [평가 vs 운영 환경 정합성](#8-평가-vs-운영-환경-정합성)

---

## 1. 양자화 (Quantization)

### 1.1 개념

신경망 모델의 **가중치(weight)** 를 더 적은 비트로 표현해서 메모리·속도 최적화.

```
양자화 안 함 (BF16): 가중치 1개 = 16-bit (2 bytes)
양자화 (4-bit):      가중치 1개 = 4-bit (0.5 bytes)
```

→ 4-bit 양자화 = 메모리 **1/4**.

### 1.2 비유: 사진 압축

| 방식 | 파일 크기 | 화질 |
|---|---|---|
| 원본 RAW (BF16) | 50MB | 100% |
| JPEG 고품질 (FP8) | 25MB | ~99% |
| JPEG 중품질 (AWQ 4bit) | 12MB | ~97% |
| JPEG 저품질 (INT4) | 8MB | ~93% |

압축 ↑ = 파일 작아짐 + 품질 약간 손실.

### 1.3 주요 양자화 방식

| 방식 | 비트 | 메모리 비율 | 품질 손실 | 본 프로젝트 |
|---|---|---|---|---|
| **BF16** | 16 | 100% | 0% (원본) | 30B/35B A3B 사용 |
| **FP8** | 8 | 50% | ~0.5% | H100 native, A100 emulation 느림 |
| **AWQ** (4bit) | 4 | 25% | ~1-2% | 32B-AWQ 사용 (공식) |
| **GPTQ** (4bit) | 4 | 25% | ~1-2% | AWQ와 동급 |
| **GGUF Q4** | 4 | 25% | ~2-3% | llama.cpp 전용 (GLM/Kimi) |
| **INT4** | 4 | 25% | ~3-5% | 권장도 낮음 |

### 1.4 동작 원리 차이

- **BF16/FP16**: 부동소수점 그대로. 양자화 X.
- **FP8**: 부동소수점이지만 8-bit. H100 GPU native 지원.
- **AWQ (Activation-aware Weight Quantization)**:
  - "중요한 가중치"는 정확히 저장, "덜 중요한"건 4-bit로 압축
  - **활성값(activation) 분포 기반** 양자화 결정 → 정확도 우수
- **GPTQ**: AWQ와 비슷한 4-bit 방식. 거의 동급.
- **GGUF**: llama.cpp 전용 양자화 포맷. CPU 추론도 가능.

### 1.5 양자화 손실이 큰 작업

| 작업 | 양자화 영향 |
|---|---|
| 일반 대화·요약 | 미미 (1-2%) |
| 일반 추론 | 미미 (1-3%) |
| **복잡 수학** | 약간 큼 (3-5%) |
| **긴 코드 생성** | 약간 큼 (3-5%) |
| 짧은 답변 | 미미 |

기업분석보고서는 **일반 생성·요약** 위주 → 양자화 영향 작음.

### 1.6 모델 단위 양자화 vs 런타임 양자화

- **공식/사전 양자화 모델**: HuggingFace에 미리 양자화된 모델 (예: `Qwen/Qwen3-32B-AWQ`)
- **런타임 양자화**: vLLM이 BF16 모델을 로드하면서 자동 양자화 (예: `--quantization bitsandbytes`)

→ Qwen MoE 모델 (30B-A3B, 35B-A3B)은 공식 AWQ 없음 (MoE는 양자화 어려움). 런타임 양자화 또는 BF16 직접 사용.

---

## 2. Context Length (max_model_len)

### 2.1 개념

LLM이 한 번에 처리할 수 있는 **input + output 토큰 총 수**.

```
max_model_len = input tokens + output tokens
              = prompt + 대화 history + 생성 응답
```

### 2.2 Native vs YaRN 확장

| 용어 | 의미 |
|---|---|
| **Native** | 모델이 처음 학습된 시퀀스 길이. 가장 안정적. |
| **YaRN (Yet another RoPE N)** | RoPE 기반 position encoding을 확장해 더 긴 context 지원. 약간 품질 손실. |

본 프로젝트 모델별:

| 모델 | Native | YaRN 확장 후 |
|---|---|---|
| Qwen3-30B-A3B | 32K | 131K |
| Qwen3-32B-AWQ | 32K | 131K |
| **Qwen3.6-35B-A3B** | **262K** ⭐ | 1M |

→ Qwen3.6은 처음부터 long context로 학습되어 우위.

### 2.3 KV cache 메모리 계산

context 길이가 늘어나면 **KV cache 메모리**가 추가로 필요.

```
KV cache per token = 2 (K,V) × num_layers × num_heads × head_dim × bytes_per_value
```

Qwen3 계열 추정 (layers~48, heads~32, head_dim~128, BF16=2bytes):
- per token: 2 × 48 × 32 × 128 × 2 = **~786 KB/token**

| Context | KV cache |
|---|---|
| 8K tokens | ~6.3 GB |
| **32K tokens** | **~25 GB** |
| 128K tokens | ~100 GB |
| 262K tokens | **~206 GB** ⚠️ |

→ 모델 weight + KV cache 합이 GPU 메모리 안에 들어와야.

### 2.4 본 프로젝트 적용

A100 80GB 환경에서 32K context 시:

| 모델 | Model | KV cache (32K, BF16) | 합계 | 가능 |
|---|---|---|---|---|
| Qwen3-32B-AWQ (4bit) | 19 | 25 | 44GB | ✅ |
| Qwen3-30B-A3B BF16 | 60 | 25 | 85GB | ⚠️ 빠듯 (gpu-util 0.95로 OK) |
| Qwen3.6-35B-A3B BF16 | 70 | 25 | 95GB | ⚠️ 빠듯 (gpu-util 0.95로 OK) |

---

## 3. vLLM 메모리 옵션

### 3.1 `--gpu-memory-utilization`

vLLM이 GPU 메모리 중 사용할 비율 (기본 0.9).

- 0.9 → 80GB × 0.9 = **72GB 사용**
- 0.95 → **76GB 사용**
- 1.0은 위험 (시스템 buffer 부족)

본 프로젝트는 **0.95**로 설정해 BF16 35B 32K 가능.

### 3.2 `--kv-cache-dtype`

KV cache 양자화로 메모리 절약.

| 값 | 메모리 비율 | 품질 손실 |
|---|---|---|
| `auto` (BF16) | 100% | 0% |
| `fp8` | 50% | <1% |
| `fp8_e5m2` | 50% | <1% |

A100은 FP8 native 미지원 → emulation 사용 (속도 약간 손실).

### 3.3 `--max-num-seqs`

동시 처리 가능한 sequence 수 제한. KV cache 메모리 영향.

- 기본 256~512
- 줄이면 메모리 절약 (단 처리량 ↓)

### 3.4 `--cpu-offload-gb`

모델 weight 일부를 CPU RAM으로. 메모리 절약하지만 **매우 느림**.

### 3.5 `--quantization`

런타임 동적 양자화.

| 값 | 효과 |
|---|---|
| `bitsandbytes` | 4-bit 동적 양자화 (AWQ 유사) |
| `fp8` | FP8 동적 양자화 |

### 3.6 OOM 시 변경 우선순위 (본 프로젝트 규칙)

1. **`--gpu-memory-utilization`** 조정 (0.9 → 0.95)
2. 그 외 모든 메모리 옵션 변경은 **반드시 사용자 의논 후 진행**

---

## 4. Tokenizer

### 4.1 개념

텍스트를 **토큰(token)** 단위로 분할. 모델은 토큰을 입력으로 받음.

```
"안녕하세요" → ["안녕", "하세요"] (2 tokens, 예시)
```

### 4.2 모델별 토크나이저 효율

같은 한국어 문장이라도 모델마다 토큰 수 다름.

| 토크나이저 | 한국어 1 char당 token (대략) |
|---|---|
| 영어 기준 BPE | ~1.0 |
| GPT 시리즈 | ~0.8 |
| **Qwen tokenizer** | **~0.5-0.7** (한국어 잘 처리) |

### 4.3 본 프로젝트 실측

AIHub 90 특허전체 실제 측정:
```
43,356 char → 30,488 tokens
비율: 30,488 / 43,356 = 0.70
```

→ 한국어 + 영어 혼합 특허는 char 1당 **~0.7 token**.

### 4.4 한국어 BERTScore용 모델

`klue/roberta-large` — KLUE 벤치마크용으로 학습된 한국어 RoBERTa.
- BERTScore 임베딩 추출용
- 24 layers, num_layers=17 권장 (별도 문서 참조)

---

## 5. Async / Concurrency

### 5.1 vLLM Continuous Batching

vLLM은 **동시 요청 자동 batch 처리**:
- 한 GPU에서 여러 sequence를 동시 진행
- GPU 활용도 향상
- throughput 증가

### 5.2 평가 코드 동시 호출

```python
import asyncio

sem = asyncio.Semaphore(concurrency)

async def process(sample):
    async with sem:
        # vLLM 호출 + judge 호출
        ...

await asyncio.gather(*(process(s) for s in samples))
```

### 5.3 성능 향상 (본 프로젝트)

| 방식 | sample당 시간 | 향상 |
|---|---|---|
| 순차 호출 (concurrency=1) | ~14-23초 | 1x |
| **동시 8개** | **~2.2초** | **~6-10x** |

→ **7시간 → 약 1.5시간** 단축.

### 5.4 주의사항

- OpenAI API rate limit (Tier별 제한)
- 너무 큰 concurrency → 메모리 부담
- 본 프로젝트는 **concurrency=8** 사용

---

## 6. LLM-as-Judge

상세: `judge-model-choice.md` 참조.

### 핵심 요약

- **GPT-4 family가 LLM-as-Judge 사실상 표준** (Zheng et al. NeurIPS 2023)
- 인간 평가와 80%+ 일치 (인간-인간 일치율 수준)
- 본 프로젝트: **gpt-4o primary, gpt-4o-mini fallback**
- 적용 벤치: Ko-MT-Bench, LogicKor

### Judge bias 회피

- Self-preference: 후보 model family와 다른 family judge
- Verbosity: 길이 제한 prompt
- Position: pairwise 시 swap

---

## 7. BERTScore num_layers

상세: `bertscore-num-layers.md` 참조.

### 핵심 요약

- BERT 모델은 여러 layer가 다른 의미를 표현
- 의미 유사도엔 **중상위 layer (약 70% 깊이)** 가 최적
- `klue/roberta-large` (24 layers) → **num_layers=17**

### 본 프로젝트 fix

```python
# Before: KeyError 발생
P, R, F = bert_score(..., model_type="klue/roberta-large", lang="ko")

# After: 명시적 num_layers
P, R, F = bert_score(..., model_type="klue/roberta-large", num_layers=17, lang="ko")
```

---

## 8. 평가 vs 운영 환경 정합성

### 8.1 핵심 원칙

> **평가의 max_length는 모델의 능력이 아니라 "운영 환경의 max_length"를 따라가야 한다.**

운영 8K → 평가 8K
운영 32K → 평가 32K

### 8.2 본 프로젝트 적용

| 환경 | GPU | 사용 양자화 | Max context |
|---|---|---|---|
| 평가 (A100 80GB) | A100 SXM4 | BF16 + AWQ | 32K |
| 운영 (H100 NVL 94GB) | H100 NVL | BF16/AWQ (FP8 가능) | 32K |

→ 평가-운영 양자화·context 일관 (정합).

### 8.3 운영 데이터 input 길이 측정 권장

평가 결과의 신뢰도는 운영 input 길이가 평가 sample 분포와 일치할 때 보장.

본인 use case (기업분석보고서) 운영 input 길이 측정 후:
- 평가 sample의 90th percentile (11,904 tokens) 안에 들어옴 → 평가 신뢰
- 30K+ token 빈도 ↑ → 추가 검증 또는 운영 max_length 32K 필수

---

## 9. 본 프로젝트 결정 요약

| 항목 | 결정 |
|---|---|
| Qwen3-32B 양자화 | AWQ 4bit (공식) |
| Qwen3-30B-A3B 양자화 | BF16 (공식 MoE, AWQ 없음) |
| Qwen3.6-35B-A3B 양자화 | BF16 (공식 MoE, AWQ 없음) |
| max_model_len | **32K 통일** (모델 native 활용) |
| gpu-memory-utilization | **0.95** (메모리 한도까지) |
| KV cache dtype | auto (BF16, FP8 변경 시 사용자 의논) |
| Judge | gpt-4o (gpt-4o-mini fallback) |
| Concurrency | 8 |
| BERTScore | klue/roberta-large, num_layers=17 |

---

## 10. 참고 문헌

- **Quantization 일반**:
  - Frantar et al., GPTQ (2022): https://arxiv.org/abs/2210.17323
  - Lin et al., AWQ (2023): https://arxiv.org/abs/2306.00978
  - NVIDIA FP8 정보: https://docs.nvidia.com/deeplearning/transformer-engine/

- **Context Length·YaRN**:
  - YaRN paper: https://arxiv.org/abs/2309.00071

- **LLM-as-Judge**:
  - Zheng et al. 2023, MT-Bench: https://arxiv.org/abs/2306.05685

- **BERTScore**:
  - Zhang et al. 2020, BERTScore: https://arxiv.org/abs/1904.09675

- **vLLM**:
  - vLLM docs: https://docs.vllm.ai/

---

## 11. 관련 본 프로젝트 문서

- `references/llm-models.md` — 5개 후보 모델 상세 사양
- `references/evaluation-datasets.md` — 평가 데이터셋 명세
- `references/judge-model-choice.md` — Judge 모델 선택 근거
- `references/bertscore-num-layers.md` — BERTScore num_layers 설명
- `CLAUDE.md` — 프로젝트 전체 평가 방향
