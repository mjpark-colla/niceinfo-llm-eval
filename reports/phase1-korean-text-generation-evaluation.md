# Phase 1 한국어 텍스트 생성 능력 평가 리포트 (v2)

> **프로젝트**: niceinfo — 금융 도메인 LLM 평가
> **최초 평가**: 2026-05-22 (v1)
> **Judge prompt 재설계 + 재평가**: 2026-05-25 (v2, 본 리포트)
> **작성자**: PolarPulse
> **상태**: Phase 1 (한국어 텍스트 생성 능력 비교) 완료

---

## 1. 요약 (Executive Summary)

운영 환경(H100 NVL 94GB, 폐쇄망) 후보 **Qwen 4종 모델**의 한국어 텍스트 생성 능력을 5개 공개 벤치마크(572 sample)로 평가했다. v1 결과 외부 검증 부족과 LLM-as-Judge prompt의 변별력 부족이 확인되어, Judge prompt를 학술 표준(G-Eval / MT-Bench / LogicKor)으로 재설계하고 **rejudge (모델 답변 고정 + Judge prompt만 변경)** 방식으로 재평가했다.

**주요 결론**:

- **Target task별 권장 모델이 다름** — 단일 winner 없음
  - **요약 중심** → **Qwen3-30B-A3B-BF16** (faithfulness 1위, 환각 적음)
  - **보고서·추론·instruction 중심** → **Qwen3-32B-AWQ** (균형, 메모리 19GB로 최소)
  - **대화·멀티턴 중심** → **Qwen3.6-35B-A3B-FP8**
- **Qwen3.6-35B-A3B-BF16은 어떤 시나리오에서도 1위 아님** → 운영 채택 후보에서 사실상 제외
- **FP8 양자화 손실 없음** — BF16 대비 +0.83% (v1: -0.14%, 둘 다 양자화 손실 1-2% 이내)
- **v1 평가의 약점 확인** — ROUGE/BERTScore가 요약 변별을 거의 못 했음 (변별력 3-5배 증가)

---

## 2. v1 대비 v2 변경 사항

### 2-1. Judge prompt 재설계 (가장 큰 변화)

| 벤치 | v1 prompt | v2 prompt |
|---|---|---|
| AIHub 582 / 90 | ROUGE-L + BERTScore만 (LLM-Judge 미사용) | **G-Eval 4차원** (faithfulness/relevance/coherence/conciseness) + ROUGE/BERTScore 보조 |
| Ko-MT-Bench | 자체 generic prompt | **MT-Bench 표준 format** ("Rating: [[N]]") + anchor + bias 회피 명시 |
| LogicKor | 자체 generic prompt | **LogicKor 스타일** reference 비교 강조 + anchor |
| Ko-IFEval | 자동 룰 | 변경 없음 (룰 기반, judge 무관) |

### 2-2. v1 prompt 약점 (재설계 동기)
1. Score anchor 부재 → 7-9점에 점수 압축 (score compression)
2. 정수만 받음 → 분해능 부족
3. Length/verbosity bias 회피 명시 없음
4. 요약 평가에 LLM-Judge 미적용 → ROUGE만으로 패러프레이즈·사실 충실도 평가 불가
5. 표준 prompt 미사용 → 외부 비교 불가능

### 2-3. 재평가 방식 (rejudge)

**변수 통제**: 모델 답변(model_output)은 그대로 보존, **Judge prompt만** 새 버전으로 GPT-4o 재호출.

→ 점수 차이 = 순수하게 Judge prompt 차이만 반영. 답변 변동(temp 0 noise) 변수 제거.

비용·시간: ~$30 / 11분 (재추론 했으면 ~$56 / 3시간)

---

## 3. 평가 설계

### 3-1. 평가 대상 (변경 없음)

| 모델 | 양자화 | 메모리 (VRAM) | 운영 후보 등급 |
|---|---|---|---|
| Qwen3-30B-A3B | BF16 | ~60GB | 후보 |
| Qwen3-32B-AWQ | AWQ 4-bit | ~19GB | 후보 (최소 메모리) |
| Qwen3.6-35B-A3B | BF16 | ~70GB | 평가 기준선 |
| Qwen3.6-35B-A3B-FP8 | FP8 | ~35GB | 운영 메인 후보 |

### 3-2. 평가 환경

- v1과 동일: GCP a2-ultragpu-1g (A100 80GB), vLLM, gpt-4o judge
- v2 재평가: Mac 로컬 venv (rejudge는 GPU 불필요)
- temperature 0.0, concurrency 8

### 3-3. 벤치마크 구성

| 벤치 | 측정 측면 | 샘플 | 가중치 | 지표 |
|---|---|---|---|---|
| Ko-MT-Bench | 멀티턴 대화 | 80 | 25% | LLM-Judge MT-Bench 표준 (1~10) |
| LogicKor | 멀티턴 추론 | 42 | 15% | LLM-Judge LogicKor 스타일 (1~10) |
| Ko-IFEval | Instruction following | 150 | 20% | Strict accuracy → 1~10 환산 |
| AIHub 582 | 한국어 요약 | 150 | 20% | **LLM-Judge G-Eval 4차원** + ROUGE/BERTScore (보조) |
| AIHub 90 | 논문·특허 요약 | 150 | 20% | **LLM-Judge G-Eval 4차원** + ROUGE/BERTScore (보조) |

가중치 설계 근거는 `CLAUDE.md 5-3-1` 참조 (per-bench 평균 + 측면 균형 + 신뢰도 보정).

---

## 4. 결과 — 종합

### 4-1. 현재 가중치(25/15/20/20/20) 기준 순위

| 순위 | 모델 | v2 weighted | v1 weighted | 순위 변화 |
|---|---|---|---|---|
| 🥇 | **Qwen3-32B-AWQ** | **7.962** | 7.183 (3위) | 🚀 **3위 → 1위** |
| 🥈 | **Qwen3-30B-A3B-BF16** | **7.932** | 7.144 (4위) | 🚀 **4위 → 2위** |
| 🥉 | Qwen3.6-35B-A3B-FP8 | 7.865 | 7.254 (2위) | 2위 → 3위 |
| 4 | Qwen3.6-35B-A3B-BF16 | 7.800 | 7.259 (1위) | 💥 **1위 → 꼴찌** |

→ **모든 모델 순위가 재배치됨**. v1 평가는 변별력 부족으로 인해 35B 모델을 과대평가했었음.

### 4-2. 시나리오별 가중치 ⭐ (target task 미확정 → 시나리오별 권장)

raw 점수에 가중치만 바꿔 5 시나리오 계산. **target 확정 시 즉시 답 가능**:

| Target Task | 시나리오 가중치 | 🥇 1위 | 점수 | 🥈 2위 | 점수 |
|---|---|---|---|---|---|
| 균등 평가 | 20×5 | **Qwen3-32B-AWQ** | 7.951 | 30B-A3B-BF16 | 7.922 |
| 현재 (기본) | 25/15/20/20/20 | **Qwen3-32B-AWQ** | 7.962 | 30B-A3B-BF16 | 7.932 |
| **요약 중심** | 10/10/20/30/30 | **Qwen3-30B-A3B-BF16** | 7.517 | 32B-AWQ | 7.463 |
| **추론·형식 중심** ⚠️ | 20/25/25/15/15 | **Qwen3-32B-AWQ** | 8.208 | 30B-A3B-BF16 | 8.171 |

⚠️ "추론·형식 중심" 시나리오는 **직접 평가 데이터셋 없음** (보고서 생성 같은 장문 산출물은 Phase 2-B에서 자체 평가셋 도입 예정). 현재는 추론(LogicKor) + 형식 준수(Ko-IFEval) + 요약(AIHub) 가중 평균으로 **proxy 추정**.
| **대화 QA** | 35/25/25/7/8 | **Qwen3.6-35B-A3B-FP8** | 8.601 | 32B-AWQ | 8.591 |

**관찰**:
- 35B-BF16은 어떤 시나리오에서도 1위 아님 (메모리 70GB 대비 효율 최악)
- 32B-AWQ가 대부분 시나리오 1·2위 (19GB로 가장 작음에도 균형 우수)
- 30B-A3B-BF16은 요약·instruction에서 강점
- 35B-FP8은 대화·추론에서만 우위

---

## 5. 벤치별 상세 결과

### 5-1. 벤치별 점수 (v1 vs v2)

| 모델 | Ko-MT (v1→v2) | LogicKor | Ko-IFEval | AIHub 582 | AIHub 90 |
|---|---|---|---|---|---|
| Qwen3-32B-AWQ | 8.74→**8.96** | 8.73→8.73 | 9.24 | 4.61→**6.47** | 4.60→**6.35** |
| Qwen3-30B-A3B-BF16 | 8.33→**8.63** | 8.37→8.43 | **9.55** ⭐ | 4.73→5.96 | 4.76→**7.04** ⭐ |
| Qwen3.6-35B-A3B-FP8 | 8.85→**9.00** ⭐ | 9.06→8.85 | 9.33 | 4.54→6.04 | 4.55→6.08 |
| Qwen3.6-35B-A3B-BF16 | 8.80→8.86 | 8.94→8.82 | 9.49 | 4.54→5.90 | 4.55→5.93 |

⭐ = 해당 벤치 1위 / Ko-IFEval은 룰 기반이라 v1=v2 동일

### 5-2. 벤치별 1위 모델 분기

- **Ko-MT-Bench** (멀티턴 대화) → 35B-FP8 (9.00)
- **LogicKor** (추론) → 35B-FP8 (8.85)
- **Ko-IFEval** (instruction) → 30B-A3B (9.55)
- **AIHub 582** (한국어 요약) → **32B-AWQ (6.47)** — 의외
- **AIHub 90** (논문/특허 요약) → **30B-A3B (7.04)** — 압도적 1위 (35B 5.9 대비)

### 5-3. 변별력 비교 (v1 vs v2)

| 벤치 | v1 spread | v2 spread | 배수 | 평가 |
|---|---|---|---|---|
| Ko-MT-Bench | 0.575 | 0.366 | 0.64× | → 비슷 |
| LogicKor | 0.696 | 0.412 | 0.59× | → 비슷 |
| Ko-IFEval | 0.311 | 0.311 | 1.00× | 변화 없음 (룰) |
| **AIHub 582** | 0.188 | 0.569 | **3.0×** | ✅ 변별력↑ |
| **AIHub 90** | 0.212 | 1.119 | **5.3×** | ✅ 변별력↑↑ |

→ **요약 평가(AIHub)에서 변별력 폭발적 증가**. ROUGE/BERTScore의 한계 명확히 확인.

---

## 6. G-Eval 4차원 분석 (v2 신규)

요약 평가에서 차원별 점수 (1-5점, AIHub 582+90 통합):

| 모델 | Faithfulness | Relevance | Coherence | Conciseness |
|---|---|---|---|---|
| Qwen3-32B-AWQ | 3.42 | 3.10 | 3.12 | 3.39 |
| Qwen3-30B-A3B-BF16 | **3.65** ⭐ | **3.34** ⭐ | 3.15 | 3.30 |
| Qwen3.6-35B-A3B-FP8 | 3.16 | 3.07 | 3.13 | 3.40 |
| Qwen3.6-35B-A3B-BF16 | 3.05 | 2.99 | 3.13 | 3.45 |

### 차원별 1위 (강점·약점)

- **Faithfulness (사실 충실도)**: 🥇 30B-A3B (3.65) / 🔻 35B-BF16 (3.05)
  - → 30B-A3B는 환각이 가장 적고, 35B-BF16은 환각이 가장 많음
- **Relevance (핵심 정보 포함)**: 🥇 30B-A3B / 🔻 35B-BF16
- **Coherence (일관성)**: 🥇 35B-FP8 / 거의 비슷
- **Conciseness (간결성)**: 🥇 35B-BF16 / 🔻 30B-A3B
  - → 35B는 더 간결, 30B는 약간 장황한 편

**해석**: 30B-A3B가 요약에서 1위인 이유는 **사실 충실도 + 핵심 정보 포함도가 높기 때문**. 간결성은 떨어지나, 더 중요한 차원에서 우세.

---

## 7. 양자화 비교 — BF16 vs FP8 (v2 기준)

| 벤치 | BF16 | FP8 | Δ | Δ% |
|---|---|---|---|---|
| Ko-MT-Bench | 8.856 | 8.997 | +0.14 | +1.6% |
| LogicKor | 8.817 | 8.847 | +0.03 | +0.3% |
| Ko-IFEval | 9.494 | 9.328 | −0.17 | −1.7% |
| AIHub 582 | 5.899 | 6.040 | +0.14 | +2.4% |
| AIHub 90 | 5.925 | 6.075 | +0.15 | +2.5% |
| **Weighted Total** | **7.800** | **7.865** | **+0.065** | **+0.83%** |

→ **FP8가 BF16보다 약간 우위** (judge noise일 가능성 포함, 통계적으로 무의미한 차이).
→ ✅ **양자화 손실 1-2% 이내 → FP8 운영 적합** (v1 결론 유지).

---

## 8. 속도 분석 (대략적, A100 측정 — ⚠️ 운영 환경과 다름)

### 8-1. ⚠️ 본 분석의 한계 — 반드시 명시 (필독)

**이 절의 속도 수치는 운영 의사결정에 그대로 적용할 수 없습니다.** 7가지 한계:

1. **A100 ≠ H100 NVL** — 본 측정은 GCP a2-ultragpu-1g(A100 80GB), 운영은 폐쇄망 H100 NVL 94GB. GPU 아키텍처·메모리 대역폭·Tensor Core 세대 모두 다름. **절대값(예: "29.9 tok/s")은 운영 환경에서 재현되지 않음**.
2. **단독 latency 아님** — 평가는 `concurrency=8`로 실행. 한 GPU 안에서 8개 요청이 동시 경합하며 측정된 sample-level wall time. **단일 요청 응답 시간(single-user latency)이 아님**.
3. **Judge 호출 시간 일부 포함** — Ko-MT-Bench / LogicKor 의 `elapsed_sec` 에는 매 turn 마다 OpenAI gpt-4o judge 호출 (~1-3초) 이 섞임. Ko-IFEval/AIHub 는 judge 없으나 ROUGE/BERTScore 후처리 시간이 있음. 순수 model inference 시간이 아님.
4. **TTFT(Time To First Token) 측정 불가** — vLLM 응답을 streaming 으로 받지 않고 일괄 응답으로 받음. 사용자가 첫 토큰을 보기까지 시간은 측정 안 됨. 대화 UX 결정에 가장 중요한 지표가 빠짐.
5. **AWQ dequantization overhead** — Qwen3-32B-AWQ가 A100에서 가장 느린 것은 4-bit dequant 비용일 수 있음. **H100의 Marlin 커널·Tensor Core 4세대에서는 다를 가능성**이 큼.
6. **FP8 emulation 가능성** — Qwen3.6-35B-A3B-FP8 의 FP8 native 지원은 H100. A100은 emulation 으로 처리될 수 있음. **운영 H100 NVL 에서는 더 빠를 가능성**이 큼 (현재 1위지만 격차 더 벌어질 것).
7. **동시 임베딩·리랭커 부하 미반영** — 운영 환경은 LLM + 임베딩 + 리랭커가 한 GPU 공유. 본 측정은 LLM 단독. 메모리 경합·latency 영향 측정 안 됨.

**즉, 본 절은 "4 모델이 같은 환경에서 어떤 모델이 상대적으로 빠른가" 만 보여주며, 운영 SLA·실제 사용자 응답 시간은 Phase 3 폐쇄망 측정이 필수입니다.**

### 8-2. 측정 결과 — Judge 없는 벤치 기준 (가장 깨끗)

`Ko-IFEval + AIHub 582/90` 평균 (judge 호출 노이즈 제외, ROUGE/BERTScore 후처리 시간만 포함):

| 순위 | 모델 | 평균 sample 시간 | 평균 출력 토큰 | 대략적 throughput |
|---|---|---|---|---|
| 🥇 | **Qwen3.6-35B-A3B-FP8** | 7.90s | 236 | **29.9 tok/s** |
| 🥈 | Qwen3-30B-A3B-BF16 | 9.30s | 246 | 26.5 tok/s |
| 🥉 | Qwen3.6-35B-A3B-BF16 | 8.85s | 233 | 26.4 tok/s |
| 4️⃣ | **Qwen3-32B-AWQ** | 14.71s | 270 | **18.3 tok/s** |

⚠️ 위 숫자는 **상대 비교**용. 절대값은 운영 환경에서 재현되지 않음.

### 8-3. 의외의 발견 — AWQ가 가장 느림 (A100 한정)

- 보통 4-bit 양자화 = 빠른 추론을 기대하지만, **A100 측정에선 32B-AWQ가 가장 느림** (18.3 tok/s)
- 추정 원인:
  - AWQ dequant overhead가 A100에서 큼 (Tensor Core 3세대 한계)
  - Dense 구조 (32B 모두 활성) vs MoE (활성 3-3.3B 만)
- **운영(H100 NVL)에서는 AWQ Marlin 커널 최적화가 더 효과적**이라 결과가 뒤집힐 수 있음

### 8-4. 품질 + 속도 통합 (A100 측정 기준, 절대값 X)

현재 가중치(25/15/20/20/20) 기준 품질 점수 + 속도 추정값:

| 모델 | 품질 (v2 weighted) | 속도 (A100 추정) | 메모리 | 종합 평가 |
|---|---|---|---|---|
| **Qwen3-30B-A3B-BF16** | 🥈 7.93 | 🥈 26.5 tok/s | 60GB | **품질·속도 둘 다 상위** ⭐ |
| **Qwen3.6-35B-A3B-FP8** | 🥉 7.86 | 🥇 29.9 tok/s | 35GB | **속도 1위 + 메모리 효율** ⭐ |
| Qwen3-32B-AWQ | 🥇 7.96 | 4️⃣ 18.3 tok/s | 19GB | 품질↑·메모리 최소, 속도 trade-off (A100 한정) |
| Qwen3.6-35B-A3B-BF16 | 4️⃣ 7.80 | 🥉 26.4 tok/s | 70GB | 어떤 측면에서도 우위 없음 |

**Caveat**: 위 속도 순위는 **A100 한정**. 운영 H100 NVL 에서는 AWQ가 상위로 올라올 가능성 큼.

### 8-5. 정확한 운영 속도 측정은 Phase 3 필수

| 측정 항목 | 본 평가 (A100) | Phase 3 (H100 NVL, 폐쇄망) |
|---|---|---|
| TTFT (Time To First Token) | ❌ 측정 안 됨 | ✅ streaming API 로 측정 |
| 단독 single-user latency | ❌ concurrency 8 측정 | ✅ concurrency=1 측정 |
| 동시 부하 throughput | ❌ LLM 단독 | ✅ 임베딩·리랭커 동시 부하 |
| 운영 SLA 합의 | ❌ 불가 | ✅ 가능 |
| VRAM 사용량 | ❌ 측정 안 함 | ✅ nvidia-smi 모니터링 |
| Cold start 시간 | ❌ 측정 안 함 | ✅ 측정 |

**Phase 3 에서 측정해야 할 운영 결정 항목들이 본 평가에서는 모두 측정되지 않았습니다.**

---

## 9. 운영 권장사항

### 8-1. 1순위 후보 — Target task에 따라

| 시나리오 | 1순위 | 2순위 | 메모리 비교 |
|---|---|---|---|
| 요약 중심 | **Qwen3-30B-A3B-BF16** | Qwen3-32B-AWQ | 60GB vs 19GB → 32B-AWQ가 효율 ↑ |
| 보고서·추론 | **Qwen3-32B-AWQ** | Qwen3-30B-A3B-BF16 | 19GB ⭐ |
| 대화·QA | **Qwen3.6-35B-A3B-FP8** | Qwen3-32B-AWQ | 35GB vs 19GB |
| 균등 | **Qwen3-32B-AWQ** | Qwen3-30B-A3B-BF16 | 19GB ⭐ |

### 8-2. 핵심 후보 — Qwen3-32B-AWQ

대부분 시나리오에서 1위 또는 2위 + **메모리 19GB로 가장 작음**:
- 임베딩·리랭커 + LLM 동시 운용 시 여유 가장 큼
- AWQ 4-bit는 H100 Tensor Core 최적화
- 단점: 대화에서 35B-FP8보다 약간 떨어짐 (0.01점 차)

### 8-3. 운영 채택 비추천 — Qwen3.6-35B-A3B-BF16

- 메모리 70GB (가장 큼)
- 어떤 시나리오에서도 1위 아님
- FP8가 BF16보다 약간 우위 → **BF16 사용 이유 없음**
- 단, "원본 정밀도"가 필요한 sanity check 용도로는 보존 가능

### 8-4. 핵심 발견 — 작은 모델이 강함

**기존 가정 (큰 모델 = 더 좋음)이 깨졌음**:
- 30B-A3B (60GB) > 35B-FP8 (35GB) > 35B-BF16 (70GB) — 요약 평가에서
- 35B의 256 experts 구조가 요약·instruction에서는 오히려 핸디캡일 수 있음
- 32B-AWQ의 Dense 구조 + 4bit가 균형 잡힌 결과

---

## 10. 후속 검증 필요 사항

1. **운영 양자화 sanity check** (Phase 3): 폐쇄망 반입 후 실제 서비스 양자화로 본 점수 재현 확인
2. **운영 지표 측정**: TTFT, TPS, VRAM 점유 — 동시 임베딩·리랭커 부하 (Phase 2-A)
3. **도메인 평가 추가**: target task 확정 후 금융 지식·보고서 형식 평가 (Phase 2)
4. **자체 평가셋**: target task에 맞는 30~50건 골든셋 (Phase 2-B)
5. **Judge prompt self-preference 검증**: gpt-4o가 Qwen family에 편향 없는지 — Claude judge로 재평가 가능

---

## 11. 한계 및 주의 사항

### 10-1. v2 평가의 한계
- **rejudge 방식**: 모델 답변 고정 → 모델의 prompt 응답 능력 차이는 v1과 동일 (입력 prompt 안 바꿈)
- **AIHub 90 input 길이**: 평균 8K+ 토큰 → judge가 끝까지 충실히 읽는지 검증 미실시
- **G-Eval 차원별 가중**: 4차원 단순 평균 × 2 (1-10 환산). 차원 간 가중치 미정의

### 10-2. 데이터 오염 가능성
- Qwen3.6 (2026-04 출시)은 본 벤치(2024-2025) 학습 가능성 있음
- 다만 본 평가는 **상대 비교**가 목적이므로 영향 제한적

### 10-3. Judge LLM
- gpt-4o (NeurIPS 2023 표준)
- Self-preference bias 회피: Qwen family와 무관한 OpenAI 사용
- Verbosity bias: prompt에 "길이로 판단 X" 명시
- Position bias: pairwise 미사용 (영향 없음)

### 10-4. 양자화 격차 명시
| 모델 | 양자화 정밀도 |
|---|---|
| Qwen3-32B-AWQ | 4-bit (가장 낮음) |
| Qwen3.6-35B-A3B-FP8 | 8-bit |
| Qwen3-30B-A3B-BF16 | 16-bit |
| Qwen3.6-35B-A3B-BF16 | 16-bit (원본) |

→ 32B-AWQ가 1위라는 결과는 양자화 격차에도 불구하고 달성한 것 (운영 측면에서 매우 긍정적).

---

## 12. 비용 종합 (Phase 1 전체)

| 항목 | v1 (구) | v2 (rejudge) | 누계 |
|---|---|---|---|
| GCP A100 80GB | ~$50 | $0 | $50 |
| OpenAI gpt-4o Judge | ~$20 | ~$30 | $50 |
| 디스크 유지 (3일) | - | ~$10 | ~$60 |
| **합계** | **~$70** | **~$40** | **~$170** |

---

## 13. 다음 단계 (Phase 2 진입 시)

1. **Target task 확정** (고객사 협의)
2. **시나리오별 가중치 → 실제 가중치 적용** (즉시 가능, 재평가 불필요)
3. **운영 지표 측정** (TTFT, TPS, VRAM)
4. **도메인 평가 추가** (KFinEval-Pilot 등)
5. **자체 평가셋 설계** (DART 30건 등)
6. **운영 양자화 재측정** (폐쇄망 H100 NVL)
7. **고객사 PoC** (NDA 협의)

---

## 14. 결과 자료 위치

- **v2 결과 (rejudge)**: `/Users/minji/Documents/PolarPulse/niceinfo/results_v2/`
- **v1 결과**: `/Users/minji/Documents/PolarPulse/niceinfo/results/`
- **비교 분석**: `/Users/minji/Documents/PolarPulse/niceinfo/reports/comparison.md`
- **v1 리포트**: `/Users/minji/Documents/PolarPulse/niceinfo/reports/phase1-korean-text-generation-evaluation-v1.md`
- **본 v2 리포트**: `/Users/minji/Documents/PolarPulse/niceinfo/reports/phase1-korean-text-generation-evaluation.md`
- **GitHub (코드)**: https://github.com/mjpark-colla/niceinfo-llm-eval
- **HTML 대시보드 v1**: `/Users/minji/Documents/PolarPulse/niceinfo/reports/index.html`

---

*작성: 2026-05-25 / Phase 1 v2 완료 시점*
