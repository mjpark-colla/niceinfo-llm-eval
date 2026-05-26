# Phase 1 한국어 텍스트 생성 능력 평가 리포트

> **프로젝트**: niceinfo — 금융 도메인 LLM 평가
> **평가일**: 2026-05-22
> **작성자**: PolarPulse
> **상태**: Phase 1 (한국어 텍스트 생성 능력 비교) 완료

---

## 1. 요약 (Executive Summary)

운영 환경(H100 NVL 94GB, 폐쇄망)에서 후보가 될 수 있는 **Qwen 4종 모델**의 한국어 텍스트 생성 능력을 5개 공개 벤치마크(572 sample)로 비교 평가했다. 결론:

- **운영 권장 모델**: **Qwen3.6-35B-A3B (FP8)** — weighted total 7.25
  - BF16 원본(7.26) 대비 양자화 손실 **0.14%** (실질적 무손실)
  - 메모리 ~35GB (BF16의 절반), H100 NVL native 지원
- **양자화 손실 검증**: BF16↔FP8 점수 차 0.01점 → 운영 결정 기준("1-2% 이내") 충족
- **벤치별 강점 분기**:
  - 추론·대화 → 35B 모델군 우세
  - 요약·instruction following → 30B-A3B 우세
- **차후 산출물(target task) 확정 시** 이 결과를 기반선으로 도메인 평가(Phase 2) 추가 예정

---

## 2. 평가 설계

### 2-1. 평가 대상

| 모델 | 양자화 | 메모리 (VRAM) | 운영 후보 등급 |
|---|---|---|---|
| Qwen3-30B-A3B | BF16 | ~60GB | 후보 |
| Qwen3-32B-AWQ | AWQ 4-bit | ~19GB | 후보 (최소 메모리) |
| Qwen3.6-35B-A3B | BF16 | ~70GB | 평가 기준선 |
| **Qwen3.6-35B-A3B-FP8** | **FP8** | **~35GB** | **운영 메인** ⭐ |

### 2-2. 평가 환경

- **하드웨어**: GCP a2-ultragpu-1g, NVIDIA A100 80GB × 1, RAM 170GB
- **추론 엔진**: vLLM (vllm-openai latest), max-model-len 32K, gpu-memory-utilization 0.95
- **Judge LLM**: OpenAI `gpt-4o` (NeurIPS 2023 LLM-as-Judge 표준)
- **샘플링**: temperature 0.0 (deterministic), thinking mode strip
- **동시성**: async + Semaphore, concurrency=8

### 2-3. 벤치마크 구성 (총 572 sample)

| 벤치 | 측정 측면 | 샘플 수 | 가중치 | 지표 |
|---|---|---|---|---|
| Ko-MT-Bench | 멀티턴 대화·유창성 | 80 | 25% | LLM-as-Judge (1~10) |
| LogicKor | 다영역 추론·생성 | 42 | 15% | LLM-as-Judge (1~10) |
| Ko-IFEval | Instruction following | 150 | 20% | Strict accuracy → 1~10 환산 |
| AI Hub 582 | 한국어 요약 | 150 | 20% | ROUGE-L + BERTScore-ko |
| AI Hub 90 | 논문·특허 요약 | 150 | 20% | ROUGE-L + BERTScore-ko |

종합 점수 = 가중평균.

> ⚠️ 벤치 간 점수 스케일이 다릅니다 (Ko-MT/LogicKor 1~10, AI Hub 0~10 환산). 모델 간 **상대 비교**는 유효하나, 벤치 간 절대값 비교는 무의미합니다.

---

## 3. 평가 결과

### 3-1. 종합 점수 (가중평균)

| 순위 | 모델 | weighted total | BF16 35B 대비 |
|---|---|---|---|
| 🥇 | **Qwen3.6-35B-A3B-BF16** | **7.26** | (기준) |
| 🥈 | **Qwen3.6-35B-A3B-FP8** | **7.25** | **−0.14%** |
| 🥉 | Qwen3-32B-AWQ | 7.18 | −1.0% |
| 4 | Qwen3-30B-A3B-BF16 | 7.14 | −1.6% |

### 3-2. 벤치별 점수

| 모델 | Ko-MT-Bench | LogicKor | Ko-IFEval | AIHub 582 | AIHub 90 |
|---|---|---|---|---|---|
| Qwen3.6-35B-A3B-BF16 | 8.80 | 8.94 | 9.49 | 4.54 | 4.55 |
| **Qwen3.6-35B-A3B-FP8** | **8.85** ⬆ | **9.06** ⬆ | 9.33 ⬇ | 4.54 | 4.55 |
| Qwen3-32B-AWQ | 8.74 | 8.73 | 9.24 | 4.61 | 4.60 |
| Qwen3-30B-A3B-BF16 | 8.33 | 8.37 | **9.55** ⭐ | **4.73** ⭐ | **4.76** ⭐ |

⭐ 벤치 1위 / ⬆ BF16 대비 향상 / ⬇ BF16 대비 하락

---

## 4. 핵심 분석

### 4-1. BF16 vs FP8 양자화 손실 (운영 결정 핵심)

| 벤치 | BF16 | FP8 | 절대차 | 상대차 |
|---|---|---|---|---|
| Ko-MT-Bench | 8.80 | 8.85 | +0.05 | +0.6% |
| LogicKor | 8.94 | 9.06 | +0.12 | +1.3% |
| Ko-IFEval | 9.49 | 9.33 | −0.16 | −1.7% |
| AIHub 582 | 4.54 | 4.54 | 0.00 | 0.0% |
| AIHub 90 | 4.55 | 4.55 | 0.00 | 0.0% |
| **Weighted total** | **7.26** | **7.25** | **−0.01** | **−0.14%** |

**해석**:
- FP8 양자화는 종합 점수에서 **0.14% 손실** — 통계적·실용적으로 무의미한 수준
- Ko-MT-Bench / LogicKor에선 오히려 FP8가 **소폭 우세** (모델 가중치 분포에 따른 결정성 차이로 추정, judge noise일 가능성도 있음)
- Ko-IFEval에서 FP8가 1.7% 하락 — 형식 엄격 준수 항목에서 약간 영향, 운영상 수용 가능 범위

**운영 결정**: ✅ **FP8 채택**
- 메모리 절반 (70GB → 35GB) → 임베딩·리랭커 동시 운용 여유
- H100 NVL native 지원 → 추론 속도 우위
- 손실 < 1% → "양자화 손실 1-2% 이내면 FP8 사용" 기준 충족

### 4-2. 4 모델 비교 — 벤치별 강점 분기

종합 점수는 35B-FP8/BF16이 우세하지만, **벤치별로 강점이 다릅니다**:

#### 추론·대화 (Ko-MT-Bench, LogicKor)
- **35B 모델군 (BF16/FP8) 우세** — 멀티턴 대화·논리 추론에서 30B-A3B 대비 0.5점 이상 차이
- 35B의 256 experts (8 routed + 1 shared) 구조 + 262K 컨텍스트 강점이 반영된 결과로 추정

#### Instruction Following (Ko-IFEval)
- **Qwen3-30B-A3B (9.55) 1위**, 35B 모델군(9.33~9.49)보다 우수
- 형식 엄격 준수에서 작은 모델이 더 안정적 — 추론 분기 적어 instruction 흐트러짐 적은 것으로 추정

#### 요약 (AIHub 582 / 90)
- **Qwen3-30B-A3B (4.73, 4.76) 1위**, 35B 모델군(4.54, 4.55)보다 명확히 우수
- 30B의 핵심 추출·재구성 능력이 강함 — 장문 요약 산출물의 경우 30B가 더 적합할 수 있음

### 4-3. 산출물 형태별 권장 모델

target task가 확정되지 않은 현 시점에서, 산출물 형태별 권장:

| 산출물 후보 | 권장 모델 | 근거 |
|---|---|---|
| 기업분석보고서 (장문, 사실+해석+추천) | **Qwen3.6-35B-A3B-FP8** | 추론·대화·instruction 균형, 262K 컨텍스트 |
| 요약 중심 (재무 요약, 시장 요약) | **Qwen3-30B-A3B** | 요약 성능 1위, 메모리 60GB 적정 |
| 형식 엄격 산출물 (양식 보고서) | **Qwen3-30B-A3B** | Ko-IFEval 1위 |
| 멀티턴 대화/QA | **Qwen3.6-35B-A3B-FP8** | Ko-MT-Bench 1위, 메모리 효율 |
| 메모리 최소화 우선 | Qwen3-32B-AWQ | 19GB만 사용, 종합 점수 3위 (Phase 1 기준 충분) |

---

## 5. 운영 권장사항

### 5-1. 1순위 후보: **Qwen3.6-35B-A3B-FP8**

근거:
- 종합 점수 2위 (1위 BF16과 차이 0.14%)
- 메모리 효율 (35GB) → 임베딩·리랭커 + LLM 동시 운용 가능
- H100 NVL native FP8 지원 → 추론 속도 우위
- 262K 네이티브 컨텍스트 → RAG·긴 문서 친화

### 5-2. 후보 시나리오

- **Target task가 요약·instruction 중심**으로 확정되면 → **Qwen3-30B-A3B** 재평가 필요
- **메모리 압박이 더 심해지면** → **Qwen3-32B-AWQ** (19GB) 폴백 가능, 종합 점수 -1% 손실 감수

### 5-3. 후속 검증 필요 사항

1. **운영 양자화 sanity check** — 폐쇄망 반입 후 실제 서비스 양자화로 본 점수 재현 확인
2. **운영 지표 측정** — TTFT, TPS, VRAM 점유, 동시 임베딩·리랭커 부하 (Phase 2-A 예정)
3. **도메인 평가** — target task 확정 후 금융 지식·보고서 형식 평가 추가 (Phase 2)
4. **자체 평가셋** — DART 30건 또는 산출물 형태에 맞는 30~50건 골든셋 (Phase 2-B)

---

## 6. 한계 및 주의 사항

### 6-1. Phase 1 평가의 한계

- **데이터 오염 가능성**: 최신 모델이 공개 벤치 학습 데이터에 포함되었을 가능성 있음 → 후속 자체 평가가 변별력 보강에 중요
- **공개 벤치 ≠ 운영 품질**: FP8/AWQ4 양자화 환경에서 다시 검증 필요
- **target task 미확정**: 본 결과는 "한국어 텍스트 생성 능력 일반"에 한정. 도메인 적합도는 별도 평가 필요
- **벤치 가중치**: 5개 벤치 가중치(0.25/0.15/0.20/0.20/0.20)는 4개 평가 측면 균등 반영 의도 — target task에 따라 재설계 가능

### 6-2. Judge LLM 관련

- GPT-4o 사용 (NeurIPS 2023 표준), self-preference bias 회피 위해 Qwen family 모델과 다른 family 사용
- Verbosity bias / position bias 인지 — 길이 제어 prompt 사용, pairwise는 적용하지 않음
- 1~10 점수의 noise는 ±0.2점 수준일 수 있음 (재현 실험 미수행)

### 6-3. 미평가 모델

GLM-5.1 / Kimi-K2.5는 본 Phase 1에서 미평가 (GCP 인스턴스 RAM 부족으로 보류). 운영 가능성 낮으나, **품질 ceiling reference**로 후속 평가 시 고려 가능.

---

## 7. 평가 비용

| 항목 | 비용 |
|---|---|
| GCP a2-ultragpu-1g (A100 80GB) | ~$50 |
| OpenAI gpt-4o Judge API | ~$20 |
| 기타 (이미지·다운로드) | ~$5 |
| **합계** | **~$75** |

---

## 8. 다음 단계 (Phase 2 진행 시)

1. **Target task 확정** (고객사 협의)
2. **운영 지표 측정** — TTFT, TPS, VRAM, 동시 부하
3. **도메인 평가 도입** — KFinEval-Pilot, Allganize 등
4. **자체 평가셋 설계** — DART 기반 30~50건 골든셋
5. **운영 양자화 재측정** — 폐쇄망 환경 sanity check
6. **고객사 PoC 데이터 협의** (NDA 필요)

---

## 9. 결과 자료 위치

- **집계**: `/Users/minji/Documents/PolarPulse/niceinfo/results/summary.json`
- **모델별 raw 응답·judge 점수**: `/Users/minji/Documents/PolarPulse/niceinfo/results/<모델명>/*.jsonl`
- **GitHub (코드)**: https://github.com/mjpark-colla/niceinfo-llm-eval
- **본 리포트**: `/Users/minji/Documents/PolarPulse/niceinfo/reports/phase1-korean-text-generation-evaluation.md`

---

*작성: 2026-05-22 / Phase 1 완료 시점*
