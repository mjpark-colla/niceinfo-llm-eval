# LLM 모델 테스트 후보 정리

> 작성일: 2026-05-19
> 프로젝트: niceinfo
> 대상 환경: H100 NVL 94GB × 1장 (+ 임베딩/리랭커 동시 운용)

---

## 테스트 모델 리스트

| # | 모델 | HuggingFace | 용도 |
|---|---|---|---|
| 1 | Qwen3-30B-A3B | `Qwen/Qwen3-30B-A3B` | vLLM 서빙 (경량 MoE) |
| 2 | Qwen3-32B-AWQ | `Qwen/Qwen3-32B-AWQ` | vLLM 서빙 (Dense 4bit) |
| 3 | Qwen3.6-35B-A3B | `Qwen/Qwen3.6-35B-A3B` | vLLM 서빙 (멀티모달 메인 후보) |
| 4 | GLM-5.1 | `unsloth/GLM-5.1-GGUF` (UD-IQ2_M) | llama.cpp 로컬 실행 |
| 5 | Kimi K2.5 | `unsloth/Kimi-K2.5-GGUF` (UD-TQ1_0) | llama.cpp 로컬 실행 |

---

## 1. Qwen3-30B-A3B (Alibaba, 2025)

| 항목 | 값 |
|---|---|
| 아키텍처 | MoE |
| 총 / 활성 파라미터 | 30.5B / 3.3B |
| 전문가 구성 | 128 experts, top-8 라우팅 |
| 레이어 / 어텐션 | 48 layers, GQA (Q 32 / KV 4) |
| 컨텍스트 | 32K → 131K (YaRN) |
| 모달리티 | 텍스트 |
| Thinking 모드 | O (토글 가능) |
| 라이선스 | Apache 2.0 |

**특징**
- 작은 활성 파라미터(3.3B)로 큰 모델급 성능을 내는 효율형 MoE
- thinking / non-thinking 모드 전환으로 추론·대화 용도 모두 커버
- 100+ 언어 지원, 텍스트 전용
- 단일 H100 NVL에 충분히 들어가는 가벼운 vLLM 서빙 후보

---

## 2. Qwen3-32B-AWQ (Alibaba, 2025)

| 항목 | 값 |
|---|---|
| 아키텍처 | Dense (MoE 아님) |
| 총 파라미터 | 32.8B (전부 활성) |
| 양자화 | AWQ 4-bit (weights 4bit / activations 16bit) |
| 레이어 / 어텐션 | 64 layers, GQA (Q 64 / KV 8) |
| 컨텍스트 | 32K → 131K (YaRN) |
| 모달리티 | 텍스트 |
| 라이선스 | Apache 2.0 |

**특징**
- Dense 구조라 MoE 대비 **출력 결정성·재현성이 좋은 편** (전문가 라우팅 변동성 없음)
- FP16 대비 메모리 약 1/3, 추론 속도 최대 3배
- NVIDIA Tensor Core(awq_marlin) 최적화
- 약 19GB VRAM으로 단일 GPU에 가장 작게 들어감

---

## 3. Qwen3.6-35B-A3B (Alibaba, 2026.04) ⭐ 메인 후보

| 항목 | 값 |
|---|---|
| 아키텍처 | Hybrid Sparse MoE (Gated DeltaNet + Gated Attention) |
| 총 / 활성 파라미터 | 35B / 3B |
| 전문가 구성 | 256 experts (8 routed + 1 shared) |
| 컨텍스트 | **262K → 1M (YaRN)** |
| 모달리티 | **텍스트 + 이미지 + 비디오** |
| Thinking 모드 | O (Thinking preservation: 멀티턴 reasoning trace 유지) |
| 라이선스 | Apache 2.0 |

**특징**
- 후보 5종 중 **가장 최신·가장 능력치 우위**
- 멀티모달 입력 지원 (텍스트+이미지+비디오)
- 262K 네이티브 컨텍스트, YaRN으로 1M까지 확장 → RAG·긴 문서 친화적
- Function calling, structured output 강화 / agentic coding 특화
- FP8(35GB) 또는 AWQ 4-bit(20GB)로 단일 H100 NVL에 여유롭게 서빙 가능

---

## 4. GLM-5.1 (Zhipu AI / z.ai, 2026.04)

| 항목 | 값 |
|---|---|
| 아키텍처 | MoE |
| 총 / 활성 파라미터 | **754B / 40B** |
| 전문가 구성 | 256 routed + 1 shared, top-8 라우팅 |
| 어텐션 | MLA + DeepSeek Sparse Attention + MTP head |
| 컨텍스트 | 203K |
| 모달리티 | 텍스트 |
| Thinking 모드 | O (기본 ON) |
| 라이선스 | MIT |
| 테스트 양자화 | `unsloth/GLM-5.1-GGUF` · `GLM-5.1-UD-IQ2_M.gguf` (~236GB) |

**특징**
- **Agentic engineering 플래그십** — 코딩·터미널·리포지토리 자동화 특화
- SWE-Bench Pro SoTA, NL2Repo·Terminal-Bench 2.0에서 강세
- 단일 H100 NVL **vLLM 서빙 불가** (양자화해도 200GB+)
- llama.cpp + 시스템 RAM 256GB 조합으로만 단일 GPU 환경에서 실행 가능
- 공식 속도 수치 미공개 (체감 느림 예상)

---

## 5. Kimi K2.5 (Moonshot AI, 2026)

| 항목 | 값 |
|---|---|
| 아키텍처 | MoE + 멀티모달 |
| 총 / 활성 파라미터 | **1T / 32B** |
| 전문가 구성 | **384 experts**, MLA |
| 컨텍스트 | 262K (입·출력 동일) |
| 모달리티 | **텍스트 + 이미지** |
| 네이티브 양자화 | **INT4 (원본부터)** → 양자화 손실 적음 |
| 운영 모드 | Instant / Thinking / Agent / **Agent Swarm** |
| 테스트 양자화 | `unsloth/Kimi-K2.5-GGUF` · `Kimi-K2.5-UD-TQ1_0.gguf` (~240GB) |

**특징**
- **Agent Swarm**: 최대 100 sub-agent 병렬, 1,500 tool call까지 오케스트레이션
- 비전 기반 agentic tool use, cross-modal reasoning 강점
- 원본부터 INT4로 릴리스 → 추가 양자화에도 품질 손실 적음
- 단일 H100 NVL **vLLM 서빙 불가**
- llama.cpp + 24GB GPU + 256GB RAM에서 **~10 tok/s 공식 검증** (H100 NVL이면 더 빠를 가능성)

---

## 한눈에 비교

| 모델 | 타입 | 총 / 활성 | 컨텍스트 | 모달리티 | 단일 H100 NVL 94GB |
|---|---|---|---|---|---|
| Qwen3-30B-A3B | MoE | 30.5B / 3.3B | 32K→131K | 텍스트 | ✅ vLLM 가능 |
| Qwen3-32B-AWQ | Dense (4bit) | 32.8B | 32K→131K | 텍스트 | ✅ vLLM 가능 |
| Qwen3.6-35B-A3B | Hybrid MoE | 35B / 3B | 262K→1M | 텍스트/이미지/영상 | ✅ vLLM 가능 |
| **Qwen3.6-35B-A3B-FP8** | Hybrid MoE (FP8) | 35B / 3B | 262K→1M | 텍스트/이미지/영상 | ✅ **권장** (메모리 1/2, H100 native) |
| GLM-5.1 | MoE | 754B / 40B | 203K | 텍스트 | ❌ vLLM 불가, ⚠️ llama.cpp만 |
| Kimi K2.5 | MoE | 1T / 32B | 262K | 텍스트/이미지 | ❌ vLLM 불가, ⚠️ llama.cpp만 |

### Qwen3.6-35B-A3B 변형 선택 가이드

| 변형 | 메모리 | A100 | H100 | 평가/운영 |
|---|---|---|---|---|
| BF16 (원본) | ~70GB | ⚠️ 빠듯 | ✅ | 평가용 |
| **FP8 (공식)** | **~35GB** | ⚠️ emulation | ✅ **native** | **운영 권장** |

---

## 참고 링크

- [Qwen/Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B)
- [Qwen/Qwen3-32B-AWQ](https://huggingface.co/Qwen/Qwen3-32B-AWQ)
- [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
- [zai-org/GLM-5.1](https://huggingface.co/zai-org/GLM-5.1) · [unsloth/GLM-5.1-GGUF](https://huggingface.co/unsloth/GLM-5.1-GGUF)
- [moonshotai/Kimi-K2.5](https://huggingface.co/moonshotai/Kimi-K2.5) · [unsloth/Kimi-K2.5-GGUF](https://huggingface.co/unsloth/Kimi-K2.5-GGUF)
