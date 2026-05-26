# LLM 모델 평가 결과 보고서 (Phase 1)

> **프로젝트**: niceinfo — 금융 도메인 한국어 LLM 시스템
> **작성**: PolarPulse · 2026-05-25
> **버전**: 초안 v1
> **상태**: Phase 1 (한국어 텍스트 생성 능력) 평가 완료

---

## 1. 평가 목적

금융 도메인에서 기업 정보(재무·산업·공시·시장)를 입력으로 받아 한국어 텍스트 산출물을 자동 생성하는 LLM 기반 시스템 구축에 앞서, 운영 환경(H100 NVL 94GB × 1, 폐쇄망)에서 사용 가능한 **Qwen 4종 후보 모델의 한국어 텍스트 생성 능력을 객관적으로 비교**한다.

- Target task(보고서·요약·QA 등)가 확정되지 않은 상태이므로, 모든 후보 산출물에 공통으로 요구되는 **한국어 생성 능력 일반**부터 평가
- Target task 확정 시 즉시 도메인 평가(금융 지식, 보고서 형식 등)로 확장 가능한 구조 확보

---

## 2. 평가 모델 목록 / 특징

| 모델 | 양자화 | VRAM | 아키텍처 | 특징 |
|---|---|---|---|---|
| **Qwen3-30B-A3B** | BF16 (16-bit) | ~60 GB | MoE (활성 3.3B) | 100+ 언어, MoE 효율 구조 |
| **Qwen3-32B-AWQ** | AWQ (4-bit) | ~19 GB | Dense | 출력 결정성·재현성 강함, 메모리 최소 |
| **Qwen3.6-35B-A3B (BF16)** | BF16 (16-bit) | ~70 GB | Hybrid MoE | 멀티모달, 262K 컨텍스트, agentic |
| **Qwen3.6-35B-A3B (FP8)** | FP8 (8-bit) | ~35 GB | Hybrid MoE | 공식 FP8, H100 NVL native 지원 |

- 모두 Apache 2.0 라이선스 (상업 이용 가능)
- 모두 단일 H100 NVL 94GB에서 vLLM 서빙 가능
- 양자화 정밀도 차이가 있어 점수 비교 시 명시 (FP8/AWQ vs BF16)

---

## 3. 평가 데이터셋 목록

| 벤치마크 | 측정 측면 | 샘플 수 | 출처 |
|---|---|---|---|
| Ko-MT-Bench | 멀티턴 대화·유창성 | 80 (8 카테고리 × 10문제) | davidkim205/ko-bench |
| LogicKor | 다영역 추론·생성 | 42 (6 카테고리 × 7문제) | maywell/LogicKor |
| Ko-IFEval | Instruction following (형식 준수) | 150 (subset) | allganize/IFEval-Ko |
| AIHub 582 | 한국어 요약 (일반) | 150 (도메인 균등 샘플링) | AI Hub 한국어 요약 |
| AIHub 90 | 한국어 요약 (논문·특허) | 150 (카테고리 균등) | AI Hub 논문·특허 |

→ **총 572 sample / 모델당**. 4 모델 = **2,288 평가 case**.

---

## 4. 평가 데이터셋 특징

### Ko-MT-Bench (멀티턴 대화)
- 멀티턴(2 turn) 대화 능력 측정
- 글쓰기·수학·추론·STEM·인문 등 8 카테고리
- LLM-as-Judge 평가 표준 벤치마크 (한국어 MT-Bench)

### LogicKor (추론)
- 추론·수학·글쓰기·코딩·이해·문법 6 카테고리
- 멀티턴(2 turn), 참고 정답 제공
- GPT-4/Claude 점수가 anchor로 활용 가능

### Ko-IFEval (Instruction Following)
- IFEval의 한국어 버전
- 형식 엄격 준수 자동 측정 (단어 수·키워드·JSON 등)
- LLM-Judge 불필요, 룰 기반 채점

### AIHub 582 (한국어 요약·일반)
- 뉴스·보고서·도서·법령 등 일반 한국어 텍스트
- 평균 길이 1-3K 토큰
- 사실 충실도·핵심 정보 포함도 평가 핵심

### AIHub 90 (논문·특허 요약)
- 학술 논문 / 특허 명세서 (긴 입력, 최대 30K 토큰)
- 도메인 전문 어휘 다수
- 장문 요약 능력 측정

---

## 5. 평가 방법

### 추론 환경
- **GPU**: GCP A100 80GB × 1 (평가용; 운영은 H100 NVL 94GB)
- **추론 엔진**: vLLM (최신 안정 버전, max-model-len 32K)
- **동시 처리**: async + Semaphore concurrency=8
- **샘플링**: temperature 0.0 (deterministic, 재현 가능)
- **thinking 모드**: 답변 텍스트만 평가 (thinking trace 제거)

### LLM-as-Judge (생성형 평가)
- **Judge 모델**: OpenAI `gpt-4o` (학술 표준, Zheng et al. NeurIPS 2023)
- **Fallback**: `gpt-4o-mini` (gpt-4o 장애 시 자동)
- **Task별 prompt 분리** (v2 재설계, 2026-05-25):
  - 요약: **G-Eval 4차원 rubric** (Faithfulness / Relevance / Coherence / Conciseness)
  - 대화: MT-Bench 표준 single-v1 format (`Rating: [[N]]`)
  - 추론: LogicKor 스타일 (reference 비교 강조)
- **Bias 회피**: self-preference (다른 family judge 사용), verbosity (길이 명시적 회피), position (pairwise 미사용)

### 자동 측정 (보조)
- **요약**: ROUGE-1/2/L + BERTScore-Ko (klue/roberta-large, num_layers=17) — G-Eval 보조
- **Instruction following**: Strict / Loose accuracy 자동 룰

### 평가 단계
- **v1** (2026-05-22): 자체 generic Judge prompt → 변별력 부족 확인
- **v2** (2026-05-25): G-Eval / MT-Bench / LogicKor 표준 prompt로 **재평가**
  - 모델 답변(model_output) 그대로 보존, Judge prompt만 새 버전으로 GPT-4o 재호출
  - 변수 통제 = 순수 Judge prompt 효과만 측정

---

## 6. 평가 기준

### 6-1. 측정 차원
한국어 텍스트 생성 능력을 4개 측면으로 분해:
1. **한국어 유창성·격식** — 문법·어휘·자연스러움
2. **장문 구조화 능력** — 논리 흐름·구성·완결성
3. **Instruction following** — 요청 형식·요구사항 준수
4. **요약·재구성** — 입력 텍스트 핵심 추출·재생성

### 6-2. 종합 가중치 (현재)
```
Ko-MT-Bench    25%   (멀티턴 대화·유창성)
LogicKor       15%   (추론, 샘플 신뢰도 보정)
Ko-IFEval      20%   (instruction following)
AIHub 582      20%   (요약 일반)
AIHub 90       20%   (요약 장문)
```
- target task 미확정이므로 **균형 가중치** 채택
- raw per-bench 점수는 보존되어 **target 확정 후 가중치 재집계 가능** (재평가 불필요)

### 6-3. 시나리오별 가중치 (target task별 권장)
| 시나리오 | 가중치 분포 (5 벤치 순서대로) |
|---|---|
| 균등 | 20/20/20/20/20 |
| 현재 (기본) | 25/15/20/20/20 |
| 요약 중심 | 10/10/20/30/30 |
| 추론·형식 중심 ⚠️ | 20/25/25/15/15 |
| 대화 QA | 35/25/25/7/8 |

⚠️ "추론·형식 중심"은 **직접 평가 데이터셋 없음** (보고서 같은 장문 생성). 추론·instruction·요약 능력의 가중 평균으로 **proxy 추정**. 정확한 보고서 평가는 Phase 2-B에서 자체 평가셋(DART 등)으로 보강 예정.

### 6-4. 평가 지표 요약
| 벤치 | 메인 지표 | 보조 지표 | 점수 범위 |
|---|---|---|---|
| Ko-MT-Bench | LLM-Judge MT-Bench 표준 | - | 1-10 |
| LogicKor | LLM-Judge LogicKor 스타일 | - | 1-10 |
| Ko-IFEval | Strict accuracy % | Loose accuracy | 0-10 환산 |
| AIHub 582 / 90 | G-Eval 4차원 → 종합 | ROUGE-L, BERTScore | 1-10 |

---

## 7. 최종 결론

### 7-1. 종합 순위 (현재 가중치 기준)

| 순위 | 모델 | 종합 점수 | 메모리 |
|---|---|---|---|
| 🥇 | **Qwen3-32B-AWQ** | **7.962** | 19 GB |
| 🥈 | **Qwen3-30B-A3B-BF16** | **7.932** | 60 GB |
| 🥉 | Qwen3.6-35B-A3B-FP8 | 7.865 | 35 GB |
| 4 | Qwen3.6-35B-A3B-BF16 | 7.800 | 70 GB |

### 7-2. Target task별 권장 모델

| Target task 후보 | 🥇 권장 모델 | 근거 |
|---|---|---|
| 요약 중심 (재무·시장 요약 등) | **Qwen3-30B-A3B-BF16** | AIHub 90 1위, faithfulness 1위 |
| 추론·형식 중심 (보고서 proxy) | **Qwen3-32B-AWQ** | 종합 1위, 메모리 19GB |
| 대화·QA (멀티턴 대화) | **Qwen3.6-35B-A3B-FP8** | Ko-MT-Bench 1위 |
| 균등 평가 | **Qwen3-32B-AWQ** | 모든 측면 균형 + 메모리 최소 |

### 7-3. 핵심 의사결정 사항

1. **Qwen3.6-35B-A3B-BF16(70GB)는 운영 채택 비추천**
   - 어떤 시나리오에서도 1위가 아님
   - 메모리 70GB 사용 대비 효율 최악
   - 동일 모델 FP8(35GB)이 약간 더 좋음 → **FP8가 BF16 완전 대체 가능**

2. **양자화 손실 미미** (FP8 vs BF16): +0.83% — 운영 결정 기준(1-2% 이내) 충족
3. **작은 모델이 강함** — 운영 효율 관점에서 매우 긍정적
   - 32B-AWQ (19GB) 또는 30B-A3B (60GB)가 35B 모델보다 우수
   - MoE 활성 파라미터 3-3.3B의 효과 확인
4. **데이터셋 크기·가중치 영향 격리** — raw per-bench 점수 보존으로 target 확정 후 즉시 재집계 가능

### 7-4. 평가 변별력 검증
- v1 Judge prompt(자체 generic)에서 모든 모델 점수가 7-9점에 압축 → 변별력 부족
- v2 재설계(G-Eval / MT-Bench / LogicKor 표준) 후 **요약 평가 변별력 3.0~5.3배 증가**
- 학술 표준 prompt 채택 → 외부 비교·재현성 확보

---

## 8. 이후 확인 필요 사항

### 8-1. 즉시 필요 (Phase 2 진입 전)
- **Target task 형태 확정** — 고객사 협의 필요 (기업분석보고서 / 요약 / QA / 대화 등)
- 산출물 길이·구조·형식 명세
- 사용자 시나리오 (배치 생성 vs 실시간 대화 등)

### 8-2. Phase 2-A — 도메인 평가 추가 (Target 확정 직후)
- 금융 도메인 지식 평가 (KFinEval-Pilot, Allganize 금융 LLM 리더보드)
- 컴플라이언스·안전성 평가 (KFinEval-Pilot toxicity 등)
- 가중치 재설계 (한국어 생성 vs 금융 지식 비율)

### 8-3. Phase 2-B — 자체 평가셋 (Target task 맞춤)
- **DART 자체 평가셋 30-50건** (보고서 형태 확정 시)
  - 예: 회사개요·재무·해석·리스크·정보부족거부 5 카테고리
- 도메인 전문가 reference + LLM-Judge rubric 결합
- 정확한 산출물 형태별 평가

### 8-4. 운영 지표 측정 — 미완료
- **TTFT (Time To First Token)** — 사용자 응답 시간 결정적 지표
- **TPS (Throughput)** — 분당 처리 가능 요청 수
- **VRAM 사용량** — 동시 임베딩·리랭커 부하 시 영향
- **Cold start 시간** — 모델 로딩 latency
- 본 평가의 속도 분석은 A100 환경의 **대략적 상대 비교**만 가능 (운영 SLA 적용 불가)

### 8-5. Phase 3 — 운영 환경 검증 (폐쇄망 H100 NVL)
- **폐쇄망 반입 후 운영 양자화 sanity check** — 클라우드 점수가 폐쇄망에서 재현되는지
- **동시 부하 측정** — LLM + 임베딩 + 리랭커 동시 운용 시 latency·메모리 경합
- 운영 SLA 합의 (응답 시간 보장 등)
- 고객사 실 데이터 기반 최종 검증 (NDA 협의 필요)

### 8-6. 평가 신뢰도 보강 (선택)
- **Self-preference bias 검증** — gpt-4o가 OpenAI 스타일 답변을 선호하는지 Claude judge로 재평가
- **Inter-annotator agreement** — 인간 평가 30건과 LLM-Judge 점수 상관도 (Cohen's kappa)
- **데이터 오염 검증** — Qwen 학습 데이터에 공개 벤치 포함 여부

### 8-7. 미평가 후보 모델 (참고)
- **GLM-5.1** (754B / 40B active) — 평가 환경 RAM 부족으로 보류, 운영 가능성 낮음
- **Kimi-K2.5** (1T / 32B active) — 동일

---

## 부록: 평가 비용·시간

| 항목 | 시간 | 비용 |
|---|---|---|
| Phase 1 v1 평가 (GCP A100 + Judge) | ~10시간 | ~$70 |
| Phase 1 v2 재평가 (rejudge, Mac 로컬) | ~11분 | ~$30 |
| 디스크 유지 (3일) | - | ~$10 |
| **총 Phase 1 비용** | | **~$110** |

## 부록: 평가 자료 위치
- **메인 리포트(상세)**: `reports/phase1-korean-text-generation-evaluation.md`
- **비교 분석(구 vs 신)**: `reports/comparison.md`
- **속도 분석**: `reports/speed.md`
- **인터랙티브 대시보드**: `reports/index_v2.html` (가중치 슬라이더)
- **GitHub 코드**: https://github.com/mjpark-colla/niceinfo-llm-eval
- **본 보고서 (이해관계자용 초안)**: `reports/stakeholder-summary.md`

---

*문의: mjpark@polarpulse.ai*
