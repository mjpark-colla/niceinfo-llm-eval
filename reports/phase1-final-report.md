# Phase 1 최종 평가 리포트 — 한국어 텍스트 생성 능력 비교

> **작성**: 2026-05-26
> **프로젝트**: niceinfo (PolarPulse)
> **버전**: Phase 1 final (G-Eval parsing 버그 수정 + verify572 cross-validation 후)

---

## 0. 요약 (TL;DR)

**4 모델 한국어 텍스트 생성 능력 평가 결과** (572 sample × 4 모델 × 2 회 측정 = 4,576 평가 case):

| 순위 | 모델 | 종합 점수 | 권장 운영 |
|---|---|---|---|
| 🥇 | **Qwen3.6-35B-A3B-BF16** | **9.18** | 품질 1위, 그러나 메모리 70 GB |
| 🥈 | **Qwen3.6-35B-A3B-FP8** | **9.15** | **운영 메인 추천 — 품질·메모리·속도 균형** |
| 🥉 | Qwen3-32B-AWQ | 9.10 | 메모리 최소 (19GB), AWQ 4bit |
| 4️⃣ | Qwen3-30B-A3B-BF16 | 9.00 | Ko-IFEval 강점 (instruction 1위) |

**핵심 결론**:
- **Qwen3.6-35B-A3B-FP8** 운영 메인 권장 — BF16 대비 품질 -0.3% (사실상 동률) + 메모리 50% + 속도 13% 우위 + H100 NVL native FP8 지원
- 1-2위 차이 0.04 점 (judge noise 범위 내) → BF16/FP8 사실상 동률
- v2 와 cloud 재평가 (verify572) 에서 4 모델 **순위 100% 일치** → robust 검증 완료

---

## 1. 평가 개요

### 1-1. 목적

기업 정보 → 한국어 텍스트 산출물 자동 생성 LLM 운영 결정 전, 4 후보 모델의 **한국어 텍스트 생성 능력** 비교.

### 1-2. 평가 환경

| 항목 | 값 |
|---|---|
| Cloud GPU | A100 80GB (GCP us-central1-a a2-ultragpu-1g) |
| 운영 GPU (참고) | H100 NVL 94GB (폐쇄망) |
| vLLM | vllm-openai latest |
| max-model-len | 32,768 |
| gpu-memory-utilization | 0.95 |
| 동시 처리 | async + Semaphore(8) |
| 샘플링 | temperature 0.0 (deterministic) |
| Judge | OpenAI gpt-4o (fallback gpt-4o-mini) |

### 1-3. 평가 데이터셋

| 벤치 | 측정 측면 | 샘플 수 | 채점 | 가중치 |
|---|---|---|---|---|
| Ko-MT-Bench | 대화·유창성 | 80 (멀티턴 × 2) | LLM-Judge (MT-Bench format) | 0.25 |
| LogicKor | 추론·생성 | 42 (멀티턴 × 2) | LLM-Judge (LogicKor 스타일) | 0.15 |
| Ko-IFEval | Instruction 준수 | 150 | 룰 기반 (자동) | 0.20 |
| AIHub 582 | 일반 요약 | 150 | LLM-Judge (G-Eval 4차원) | 0.20 |
| AIHub 90 | 논문/특허 요약 | 150 | LLM-Judge (G-Eval 4차원) | 0.20 |
| **합계** | — | **572 / 모델** | — | 1.00 |

→ 총 **2,288 평가 case × 2 회 (v2 + verify572)** = 4,576 sample.

---

## 2. 평가 단계 이력

| 단계 | 일자 | 내용 |
|---|---|---|
| v1 | 2026-05-22 | 자체 generic Judge prompt → 변별력 부족 확인 |
| v2 | 2026-05-25 | G-Eval / MT-Bench / LogicKor 표준 prompt 도입, rejudge |
| **버그 발견·수정** | **2026-05-26** | **G-Eval `"종합: [N]"` 대괄호 형식 regex 미지원 → score=0 다수. 수정 후 reparse** |
| verify572 | 2026-05-26 | cloud 에서 4 모델 전체 재평가, v2 와 cross-validation |

### 2-1. 버그 경위

**원인**: `eval/metrics/llm_judge.py` 의 score 추출 regex 가 prompt 가 요청한 출력 형식 (`종합: [9.6]`) 의 대괄호를 처리 못 함.

```python
# 옛 regex (버그)
re.search(r"종합\s*[:：]\s*(\d+(?:\.\d+)?)", raw)  # 대괄호 미지원

# 새 regex (수정)
re.search(r"종합\s*(?:점수)?\s*[:：]\s*\[?\s*(\d+(?:\.\d+)?)\s*\]?", raw)
```

**영향**:
- AIHub 582·90 각 sample 의 39-57건 (모델당) 이 `score=0` 으로 잘못 저장
- 이전 보고된 v2 결과 (AWQ 1위 7.96) 가 완전히 잘못된 결과
- judge_raw 는 보존되어 있어 **재추론 없이 regex 수정 후 reparse 만으로 100% 복구**

**검증**:
- Mac 에서 `results_v2/` 재파싱 → 0점 sample → 9점대 평균 복구
- 20 sample 무작위 검증: 100% 일치 (judge_raw 의 실제 점수 ↔ 저장된 score)

---

## 3. 최종 결과

### 3-1. 종합 순위

| 순위 | 모델 | v2 (재파싱) | verify572 | 평균 | 메모리 | 속도 (A100) |
|---|---|---|---|---|---|---|
| 🥇 | **Qwen3.6-35B-A3B-BF16** | 9.239 | 9.114 | **9.18** | 70 GB | 145 tok/s |
| 🥈 | **Qwen3.6-35B-A3B-FP8** | 9.230 | 9.077 | **9.15** | 35 GB | **164 tok/s** |
| 🥉 | Qwen3-32B-AWQ | 9.174 | 9.022 | 9.10 | 19 GB | 67 tok/s |
| 4️⃣ | Qwen3-30B-A3B-BF16 | 9.088 | 8.911 | 9.00 | 60 GB | 132 tok/s |

### 3-2. 벤치별 점수 (v2 / verify 평균)

| 모델 | Ko-MT | LogicKor | Ko-IFEval | AIHub 582 | AIHub 90 |
|---|---|---|---|---|---|
| Qwen3.6-35B-A3B-BF16 | 8.83 | 8.70 | **9.43** | **9.36** | **9.53** |
| Qwen3.6-35B-A3B-FP8 | **8.88** | 8.62 | 9.34 | 9.33 | **9.53** |
| Qwen3-32B-AWQ | 8.83 | 8.58 | 9.24 | 9.30 | 9.48 |
| Qwen3-30B-A3B-BF16 | 8.52 | 8.20 | **9.57** | 9.20 | 9.44 |

**관찰**:
- **35B-A3B-BF16** — 5 벤치 중 **3 개에서 1위** (Ko-IFEval, AIHub 582, AIHub 90)
- **35B-A3B-FP8** — Ko-MT-Bench 1위 (대화 강점)
- **30B-A3B-BF16** — Ko-IFEval (룰 기반·noise 없음) 1위 (instruction 준수)
- **AIHub (요약) 양쪽 모두 35B-A3B-BF16 / FP8 1, 2위** → 최신 모델 우위

### 3-3. G-Eval 4차원 (요약, AIHub 582+90 통합, 1-5 점)

| 모델 | Faithfulness | Relevance | Coherence | Conciseness |
|---|---|---|---|---|
| **Qwen3.6-35B-A3B-BF16** | **4.95** | **4.03** | **4.95** | **4.86** |
| Qwen3.6-35B-A3B-FP8 | 4.93 | 4.01 | 4.95 | 4.84 |
| Qwen3-32B-AWQ | 4.93 | 4.01 | 4.89 | 4.80 |
| Qwen3-30B-A3B-BF16 | 4.91 | 3.99 | 4.85 | 4.79 |

**관찰**:
- **35B-BF16 이 4 차원 모두 1위 또는 공동 1위** — 요약 품질의 모든 측면에서 우수
- **Relevance (4.0 수준)** — 모든 모델 공통 약점. 핵심 정보 포함도가 다른 차원 대비 낮음
- Faithfulness 4.9+ — 환각 거의 없음 (운영 안전성 OK)

### 3-4. 양자화 비교 (BF16 vs FP8 — 동일 모델 Qwen3.6-35B-A3B)

| 지표 | BF16 | FP8 | Δ |
|---|---|---|---|
| 종합 점수 (v2/verify 평균) | 9.18 | 9.15 | **-0.3%** |
| 메모리 (weights) | 70 GB | 35 GB | **-50%** |
| 메모리 (vLLM util 0.95) | 75 GiB | 75 GiB | 0 |
| 속도 (A100, tok/s) | 145 | **164** | **+13%** |
| TTFT | 0.31 s | 0.32 s | 거의 동일 |
| H100 NVL native FP8 | ❌ | **✅** | H100 가속 효과 추가 기대 |

→ **FP8 양자화 손실 1% 이내** 충분 (운영 기준 1-2% 이내). **품질 손실 무시 가능 + 메모리 절반 + 속도 우위** → FP8 채택이 합리적.

### 3-5. v2 ↔ verify572 robust 검증

| 모델 | 평균 차이 (verify - v2) | 최대 벤치 변동 | 순위 변동 |
|---|---|---|---|
| Qwen3.6-35B-A3B-BF16 | -0.125 | LogicKor -0.232 | 1위 → 1위 |
| Qwen3.6-35B-A3B-FP8 | -0.153 | LogicKor -0.458 | 2위 → 2위 |
| Qwen3-32B-AWQ | -0.152 | LogicKor -0.302 | 3위 → 3위 |
| Qwen3-30B-A3B-BF16 | -0.177 | LogicKor -0.462 | 4위 → 4위 |

**관찰**:
- 4 모델 모두 verify 에서 일관되게 0.1-0.2 점 낮음 → **systematic judge noise** (절대값 흔들림, 상대 순위 그대로)
- **상대 순위 100% 일치** → 결론 robust
- Ko-IFEval (룰 기반) 만 차이 ±0.03 (거의 동일) → 룰 기반 채점이 가장 reproducibility 높음
- LogicKor 변동이 가장 큼 (-0.46) — 42 sample 작은 데이터셋 한계

---

## 4. 시나리오별 가중치 분석

target task 미확정 상태에서 가능한 운영 시나리오별 1위 모델 (v2 기준):

| 시나리오 | 가중치 (Ko-MT/LogicKor/IFEval/AIHub582/90) | 🥇 1위 | 점수 |
|---|---|---|---|
| 균등 | 20/20/20/20/20 | Qwen3.6-35B-A3B-BF16 | 9.27 |
| **현재 (기본)** | **25/15/20/20/20** | **Qwen3.6-35B-A3B-BF16** | **9.24** |
| 요약 중심 | 10/10/20/30/30 | Qwen3.6-35B-A3B-BF16 | 9.36 |
| 추론·형식 중심 | 20/25/25/15/15 | Qwen3.6-35B-A3B-BF16 | 9.16 |
| 대화 QA | 35/25/25/7/8 | Qwen3.6-35B-A3B-FP8 | 9.12 |

→ **5 시나리오 중 4 시나리오에서 35B-BF16 1위, 1 시나리오에서 FP8 1위** — Qwen3.6-35B-A3B 계열이 모든 운영 시나리오에서 강함.

---

## 5. 운영 권장

### 5-1. 메인 추천: **Qwen3.6-35B-A3B-FP8**

**근거**:
1. **품질**: BF16 대비 -0.3% (사실상 동률, judge noise 범위 내)
2. **메모리**: 35 GB (BF16 70 GB 대비 절반) — H100 NVL 94GB 에서 임베딩·리랭커 동시 운용 여유 확보
3. **속도**: 164 tok/s (BF16 145 대비 +13%, AWQ 67 대비 +145%)
4. **H100 NVL native**: H100 NVL 의 FP8 native 가속 → 추가 속도 이득 기대 (Phase 3 측정 예정)
5. **양자화 안정성**: 공식 FP8 모델 — Qwen 팀이 직접 변환·검증

### 5-2. 백업 옵션

| 시나리오 | 백업 모델 | 근거 |
|---|---|---|
| Faithfulness (환각 최소화) 최우선 | **Qwen3.6-35B-A3B-BF16** | G-Eval Faithfulness 1위 4.95, FP8 와 0.02 차이 |
| 메모리 극단 제약 (19GB) | **Qwen3-32B-AWQ** | 메모리 19GB, 종합 3위 (9.10) |
| Instruction following 최우선 | **Qwen3-30B-A3B-BF16** | Ko-IFEval 1위 9.57 (룰 기반) |

### 5-3. 운영 비추천

- **Qwen3-32B (BF16, 무양자화)** — 속도 24 tok/s (FP8 대비 -85%), 메모리 65 GB → 운영 부적합

### 5-4. 결정 시 주의

- **점수 spread 0.18** — 1-2위 동률 (0.04 점 차이), 1-4위 전체도 가까움. judge noise 범위 내
- **공개 benchmark contamination 가능성** — Phase 2-B 자체 평가셋 (DART) 으로 보강 필수
- **운영 환경 (H100 NVL) ≠ 평가 환경 (A100)** — Phase 3 폐쇄망 sanity check 필수

---

## 6. 이후 단계

### 6-1. 즉시 (Phase 2 진입 전)

- [ ] **Target task 형태 확정** (고객사 협의) — 기업분석보고서 / 요약 / QA / 대화 등
- [ ] 산출물 길이·구조·형식 명세
- [ ] 사용자 시나리오 (배치 vs 실시간 대화)

### 6-2. Phase 2-A — 도메인 평가 추가

- [ ] 금융 도메인 지식 평가 (KFinEval-Pilot, Allganize 금융 LLM 리더보드)
- [ ] 컴플라이언스·안전성 평가 (KFinEval-Pilot toxicity)
- [ ] 가중치 재설계 (한국어 생성 vs 금융 지식 비율)

### 6-3. Phase 2-B — 자체 평가셋

- [ ] DART 자체 평가셋 30-50건 (보고서 형태 확정 시)
- [ ] 도메인 전문가 reference + LLM-Judge rubric 결합

### 6-4. Phase 3 — 운영 환경 검증 (폐쇄망 H100 NVL)

- [ ] 폐쇄망 반입 후 운영 양자화 sanity check
- [ ] 동시 부하 측정 (LLM + 임베딩 + 리랭커)
- [ ] 운영 SLA 합의 (TTFT, TPS, 동시 처리량)
- [ ] 고객사 실 데이터 검증 (NDA 협의 필요)

---

## 7. 부속 자료

| 자료 | 위치 |
|---|---|
| 본 리포트 | `reports/phase1-final-report.md` |
| HTML 대시보드 | `reports/index_v2.html` — 가중치 시뮬레이션·답변 비교·G-Eval 레이더 |
| 평가 raw data (v2, 재파싱) | `results_v2/<model>/<bench>.jsonl` |
| 평가 raw data (verify572) | `results_verify572/<model>/<bench>.jsonl` |
| 백업 (원본 buggy) | `results_v2/<model>/<bench>.jsonl.bak` |
| 모델 사양 | `references/llm-models.md` |
| 데이터셋 명세 | `references/evaluation-datasets.md` |
| Judge model 선택 근거 | `references/judge-model-choice.md` |
| 속도 실측 결과 | `references/speed-benchmark.md` |
| Confluence (회사 공유) | [260519 나이스평가정보 LLM 모델 비교](https://polarpulse.atlassian.net/wiki/spaces/P/pages/79527939) |
| GitHub | [mjpark-colla/niceinfo-llm-eval](https://github.com/mjpark-colla/niceinfo-llm-eval) |

---

## 8. 학술 근거

- **MT-Bench** — Zheng et al., *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*, NeurIPS 2023 (arXiv 2306.05685)
- **IFEval** — Zhou et al., *Instruction-Following Evaluation for Large Language Models*, 2023 (arXiv 2311.07911)
- **G-Eval** — Liu et al., *G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment*, 2023 (arXiv 2303.16634)
- **AWQ** — Lin et al., *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration*, 2023 (arXiv 2306.00978)
- **vLLM** — Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention*, SOSP 2023

---

_작성: 2026-05-26 · 작성자: mjpark@polarpulse.ai (with Claude) · 검토 필요_
