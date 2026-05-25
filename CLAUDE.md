# niceinfo — 금융 도메인 LLM 평가 프로젝트

> 작성일: 2026-05-20
> 소속: PolarPulse
> 상태: 모델 비교 단계 (운영 전) — **target task 재확정 진행 중**

---

## 1. 프로젝트 목적

금융 도메인에서 기업 정보(재무, 산업, 공시, 시장)를 입력으로 받아 한국어 텍스트 산출물을 자동 생성하는 LLM 기반 시스템 구축.

**Target task 상태 (2026-05-20 기준)**:
- 초기 가정은 **기업분석보고서 자동 생성** (사실+해석+추천 혼합 장문 보고서)
- 2026-05-20 고객사 소통 결과 **기업분석보고서가 아닐 가능성** 발생
- 구체 산출물 형태가 재확정될 때까지 평가 범위는 **"한국어 텍스트 생성 능력 일반"** 으로 한정
- Target task 확정 후 도메인 특화 평가(금융 지식·보고서 형식) 추가 예정

## 2. 현재 단계 — 한국어 텍스트 생성 능력 모델 비교

Target task가 재확정 중이므로, 모든 후보 산출물에 공통으로 요구되는 **한국어 텍스트 생성 능력**부터 비교한다. 본 단계 목표:
- 5개 후보 모델의 한국어 생성 능력 객관 비교
- 운영군(Qwen)과 참고군(GLM/Kimi)을 분리 평가
- 공개 데이터셋 기반 간소 평가 (자체 데이터 생성 부담 최소)
- Target task 확정 시 도메인 평가를 후속으로 확장 가능한 구조 유지

## 3. 평가 대상 모델

상세는 `references/llm-models.md` 참조. 요약:

| 모델 | 운영 모드 | 분류 |
|---|---|---|
| Qwen/Qwen3-30B-A3B | vLLM | 운영 후보 |
| Qwen/Qwen3-32B-AWQ | vLLM (AWQ 4bit) | 운영 후보 |
| Qwen/Qwen3.6-35B-A3B | vLLM (FP8) | **운영 메인 후보** |
| zai-org/GLM-5.1 → `unsloth/GLM-5.1-GGUF UD-IQ2_M` | llama.cpp | **참고용** (운영 가능성 낮음) |
| moonshotai/Kimi-K2.5 → `unsloth/Kimi-K2.5-GGUF UD-TQ1_0` | llama.cpp | **참고용** (운영 가능성 낮음) |

**중요**: 모델군별 양자화 정밀도가 다름 (Qwen=FP8/AWQ4 vs GLM=IQ2 vs Kimi=TQ1). 점수 비교 시 명시 필요.

## 4. 인프라

### 운영 환경 (폐쇄망)
- GPU: H100 NVL 94GB × 1장
- RAM: 256GB+ (GGUF mmap 오프로드용)
- 동시 운용: LLM + 임베딩 + 리랭커
- **외부 API/네트워크 접근 전면 불가**

### 평가 환경 (GCP 클라우드)
- 권장 인스턴스: **a3-highgpu-2g** (H100 SXM5 80GB × 2, RAM 468GB)
  - 운영(NVL 94GB)과 GPU 14GB 차이, RAM 여유로 GLM/Kimi GGUF 수용 가능
  - vLLM은 1 GPU만 사용하도록 제한해 운영 환경 시뮬레이션
- 추론 엔진: Qwen → vLLM, GLM/Kimi → llama.cpp
- 평가 결과는 폐쇄망에 반입 후 우승자 sanity check 권장

## 5. 평가 방향성 (현재 합의된 사항)

### 5-1. 기본 원칙
- Target task 미확정 상태이므로 평가 범위는 **"한국어 텍스트 생성 능력"으로 한정**
- 도메인 특화 평가(금융 지식, 보고서 형식, DART 자체 평가)는 **target task 확정 후 후속 단계**로 이동
- **GLM/Kimi는 참고용 평가** — 운영 결정엔 미반영, quality ceiling reference로 활용
- 자체 데이터 생성 부담은 최소화 (당장은 공개 데이터셋만)

### 5-2. 현재 단계의 평가 차원

한국어 텍스트 생성 능력을 분해:

- **한국어 유창성·격식** (Fluency & Style): 문법·어휘·톤
- **장문 구조화 능력** (Structured long-form): 일관성·구성·논리 흐름
- **Instruction following**: 요청 형식·요구사항 준수
- **요약·재구성** (Summarization): 입력 텍스트의 핵심 추출과 재생성

> 위 4개는 어떤 산출물 형태(보고서/요약/QA/대화)가 최종 확정되더라도 **공통으로 요구되는 기본기**.

### 5-3. Phase 1 — 한국어 텍스트 생성 능력 비교 (현재 단계)

**역할**: 5개 후보 모델의 한국어 생성 능력 객관 비교.

> 후보 데이터셋의 상세 명세(태스크·수량·예시·사용 시 참고)는 `references/evaluation-datasets.md` 참조.

#### 5-3-1. 평가 실행 설계 (2026-05-21 확정)

**평가 대상**: Qwen 4개 (운영 후보군) — GLM/Kimi는 RAM 부족으로 보류

| 모델 | 양자화 | VRAM | 비고 |
|---|---|---|---|
| Qwen/Qwen3-32B-AWQ | AWQ 4-bit | ~19GB | 공식 AWQ |
| Qwen/Qwen3-30B-A3B | BF16 | ~60GB | 공식 AWQ 없음 (MoE) |
| Qwen/Qwen3.6-35B-A3B | BF16 | ~70GB | 공식 AWQ 없음 (MoE) |
| **Qwen/Qwen3.6-35B-A3B-FP8** | **FP8** | **~35GB** | **공식 FP8, H100 NVL native, 운영 최적** |

→ Qwen3.6-35B-A3B는 BF16과 FP8 둘 다 평가 (운영 환경 결정용).

**샘플 수 (모델당 총 572건)**:

| 벤치 | 샘플 수 | 비고 |
|---|---|---|
| Ko-MT-Bench | 80 (전체) | 8 카테고리 × 10문제, 멀티턴 |
| LogicKor | 42 (전체) | 6 카테고리 × 7문제, 멀티턴 |
| Ko-IFEval | 150 | 자동 룰 기반 (subset) |
| AI Hub 582 | 150 | 한국어 요약 (도메인 균등) |
| AI Hub 90 | 150 | 논문/특허 요약 (카테고리 균등) |

→ 통계 신뢰도 ±10% 이내 목표. 샘플 수는 **벤치 내부 신뢰도**만 결정, 종합 가중치는 별도.

**종합 점수 가중치**:

```
종합 점수 = 0.25 × Ko-MT-Bench
        + 0.15 × LogicKor
        + 0.20 × Ko-IFEval
        + 0.20 × AI Hub 582
        + 0.20 × AI Hub 90
```

**가중치 설계 근거 (2026-05-25 명문화)**:

1. **데이터셋 크기 영향 제거**:
   - per-benchmark 평균 → weighted sum 방식 채택
   - LogicKor 42 sample vs AIHub 150 sample 처럼 샘플 수가 다르더라도, 각 벤치 점수를 먼저 평균낸 뒤 가중치를 곱하므로 **샘플 수가 큰 벤치가 자동 가중되는 일이 없음**
   - "샘플 수 = 벤치 내부 신뢰도" / "종합 가중치 = 측면 중요도" 로 의도 분리

2. **개별 가중치 의도**:
   - **Ko-MT-Bench 25%**: 한국어 LLM 평가의 표준 reference 벤치. 멀티턴 + 8 카테고리 다양성으로 일반 능력의 대표 지표
   - **LogicKor 15%**: 42 sample로 가장 작은 데이터셋 → 통계적 신뢰도(분산) 보정 차원에서 낮게
   - **Ko-IFEval / AIHub 582 / AIHub 90 각 20%**: 측면별 (instruction/요약-일반/요약-장문) 균형 분포

3. **사후 가변성 (decoupling 설계)**:
   - raw per-benchmark 점수가 `results/summary.json` 의 `per_benchmark` 에 보존되므로, **가중치는 평가 후에도 자유롭게 변경·재집계 가능** (재평가 불필요)
   - target task 확정 시 그에 맞는 가중치로 즉시 재집계
   - 예: 요약 중심이면 AIHub 가중치↑, 추론·형식 중심이면 LogicKor·Ko-IFEval 우선
   - ⚠️ 보고서 생성 같은 장문 산출물은 직접 평가 데이터셋 부재 → 가중치 시뮬레이션은 proxy 추정. 본격 평가는 Phase 2-B 자체 평가셋에서.

4. **현재 가중치의 의미**:
   - target task **미확정 상태에서의 균형 평가**
   - "어느 task에도 절대적 최적은 아니지만, 모든 측면 점수를 일정 비율 반영"
   - target 확정 후 시나리오별 가중치 표(균등 / 요약 중심 / 보고서 / QA) 를 함께 제시

**Judge LLM** (근거: `references/judge-model-choice.md`):
- Primary: **`gpt-4o`** (LLM-as-Judge 사실상 표준, Zheng et al. NeurIPS 2023)
- Fallback: `gpt-4o-mini` (gpt-4o 실패 시 자동)
- 호출 방식: async + Semaphore (concurrency 8)
- 적용 벤치 (v2 — 2026-05-25 재설계):
  - **Ko-MT-Bench**: MT-Bench 표준 single-v1 / multi-turn prompt, "Rating: [[N]]" format
  - **LogicKor**: LogicKor 스타일 (reference 비교 강조), "Rating: [[N]]"
  - **AIHub 582 / 90**: **G-Eval 4차원 rubric** (faithfulness/relevance/coherence/conciseness → 종합 1-10)
  - ROUGE/BERTScore는 metric_details에 보조로 보존 (reference-based 표준 비교용)
- 미적용 벤치: Ko-IFEval (자동 룰)
- gpt-5 미선택 이유: reasoning model이라 API 호환성·응답 속도·평가용 검증 부족
- 출력 형식: 점수 + 4 차원 details + 이유 (jsonl 의 `metric_details`, `judge_raw` 에 보존)

**기타 설정**:
- Thinking mode: `<think>...</think>` 제거 후 평가 (사용자에게 보이는 답변 기준)
- Temperature: 0.0 (deterministic)
- Resume: 활성화 (중단 후 재시작 시 이미 평가한 sample skip)

**예상 비용·시간**:
- GCP 인스턴스: ~$55 (10시간 이내)
- OpenAI Judge API: ~$15-25 (366회 호출)
- 총: **~$70-80**

#### 5-3-2. Judge prompt 재설계 (2026-05-25, v2)

**동기**: v1 평가 결과 점수가 7-9점에 압축되어 변별력 부족이 명확. 특히 AIHub 요약 평가가 ROUGE/BERTScore만 사용 → 모델 변별 거의 못함 (spread 0.18~0.21).

**재설계 내용**:
1. **AIHub 582/90**: ROUGE/BERTScore → **G-Eval 4차원 + ROUGE/BERTScore 보조** (Liu et al. 2023)
   - Faithfulness (사실 충실도, 환각 검출)
   - Relevance (핵심 정보 포함도)
   - Coherence (논리 일관성)
   - Conciseness (간결성)
2. **Ko-MT-Bench**: 자체 generic → **MT-Bench 표준 single-v1 format** (lmsys/FastChat 기반)
3. **LogicKor**: 자체 generic → **LogicKor 스타일** (reference 비교 강조)
4. **공통 개선**:
   - Score anchor 명시 (1-2, 3-4, 5-6, 7-8, 9-10 각 단계 의미)
   - "Rating: [[N]]" 표준 format
   - 점수 분해능 0.5 단위
   - Length/verbosity bias 회피 명시
   - 멀티턴에서 실제 이전 답변을 judge에 전달 (기존 "(생략)" 처리 개선)

**재평가 방식 (rejudge)**:
- 모델 답변(`turns[].model_output`)은 그대로 보존
- Judge prompt만 새 버전으로 GPT-4o 재호출
- 변수 통제 → 순수 prompt 효과만 측정
- 구현: `eval/rejudge.py` (인퍼런스 재사용 모드)

**v2 결과 요약** (현재 가중치 25/15/20/20/20 기준):
- 🥇 **Qwen3-32B-AWQ**: 7.962 (v1 3위 → 1위)
- 🥈 **Qwen3-30B-A3B-BF16**: 7.932 (v1 4위 → 2위)
- 🥉 Qwen3.6-35B-A3B-FP8: 7.865 (v1 2위 → 3위)
- 4️⃣ Qwen3.6-35B-A3B-BF16: 7.800 (v1 1위 → 꼴찌)

→ **모든 모델 순위 재배치**. 자세한 시나리오별 결과는 `reports/phase1-korean-text-generation-evaluation.md` v2 참조.

**변별력 변화 (AIHub)**:
- AIHub 582: spread 0.188 → 0.569 (**3.0배 증가**)
- AIHub 90: spread 0.212 → 1.119 (**5.3배 증가**)

**v2 비용·시간**:
- rejudge (Mac venv): GPU $0, OpenAI ~$30, 11분
- 사용 명령: `python -m eval.rejudge --input-dir results --output-dir results_v2`

```
[Main — 한국어 텍스트 생성 능력] 가중치 100%
  · 생성형 공개 벤치만 사용 (NLU/MCQ 제외)
  · 후보 벤치 풀 (권장):
    - AI Hub 논문 요약 / 한국어 요약   (장문 요약 직접 측정)
    - Ko-MT-Bench                       (멀티턴 + 유창성, LLM-as-Judge)
    - KITE                              (한국어 특화 instruction following)
    - Ko-IFEval                         (형식 준수)
    - LogicKor (maywell/LogicKor on HF) (다영역 추론·생성, GPT-4/Claude anchor 비교 가능)
    - Open Ko-LLM Leaderboard2 / Horangi (벤치 모음 프레임워크)
  · 4가지 측면 (유창성·장문 구조화·instruction·요약) 모두 커버하도록 2~4개 조합 선택

  · LogicKor 사용 시 주의:
    - 2024-10 GitHub archived (read-only). 데이터·viewer는 HuggingFace에서 접근 가능
    - 메인테이너의 "상위권 변별력 약함" 지적은 GPT-4/Claude/Gemini급 폐쇄형 모델 대상.
      본 케이스(30~40B 오픈 모델)에선 변별력 있을 가능성 큼
    - GPT-4/Claude 등 공개 점수가 anchor로 활용 가능 — 고객 제시 시 상대 위치 표현에 유용
    - 42 prompt 소규모라 단독 결정 근거로는 부족 — 다른 벤치와 종합 사용
    - 데이터 오염 가능성 인지 (2024년 이후 출시 모델들이 학습 데이터로 포함했을 수 있음)
```

**Phase 1에서 사용하지 않는 것**:
- NLU 전용 벤치 (KLUE 등) — 생성 능력 측정 못함
- MCQ 단독 벤치 (KMMLU, Ko-MMLU, HRM8K, CLIcK 등) — 객관식이지 생성이 아님
- 금융 도메인 벤치 (KFinEval-Pilot, Allganize 등) — target task 확정 후 도입
- DART 기반 자체 평가 — target task 확정 후 도입

**랭킹/결정 방식**:
- 절대 임계값 X — 상대 비교 위주
- 모델 5개를 카테고리별 점수 + 종합 점수로 정렬
- 목적은 "선정"이 아닌 **한국어 생성 능력 프로파일링** (target task 확정 시 도메인 평가 추가하기 위한 베이스라인 확보)

### 5-4. 후속 단계 — Target task 확정 후

Target task가 확정되면 다음을 도입:

**Phase 2-A (Target 확정 직후 — 도메인 평가 추가)**:
- 금융 도메인 지식 평가 (KFinEval-Pilot, Allganize 금융 LLM 리더보드 등)
- 컴플라이언스/안전성 (KFinEval-Pilot toxicity 등)
- 가중치 재설계: 한국어 생성 vs 금융 지식 비율

**Phase 2-B (자체 평가)**:
- 확정된 target task에 맞는 자체 평가셋 설계
- 보고서 생성이면 DART 30건 (5 카테고리 × 6건: 회사개요·재무·해석·리스크·정보부족거부)
- 다른 산출물이면 그에 맞는 30~50건 골든셋
- 도메인 전문가 reference 페어 + LLM-as-Judge 결합

**Phase 3 (운영 검증)**:
- 폐쇄망 반입 후 운영 양자화로 sanity check
- 고객사 데이터 기반 최종 검증 (NDA 협의 필요)
- 운영 SLA 합의 및 PoC

### 5-5. 평가 시 공통 주의 사항
- **데이터 오염(contamination)**: 최신 모델이 공개 벤치 학습 가능성. 변별력 떨어질 수 있음 → 후속 단계의 자체 평가가 중요한 이유
- **공개 벤치 점수 ≠ 운영 품질**: full precision 기준 점수가 운영 양자화에서 무너질 수 있음 → 우승 후보는 운영 양자화로 재측정 필요
- **MCQ 점수 ≠ 생성 능력**: 메인 평가에서 MCQ-only 벤치 사용 금지
- **LLM-as-Judge self-preference bias**: 후보 모델을 judge로 사용 금지
- **양자화 격차**: Qwen FP8/AWQ4 vs GLM IQ2 vs Kimi TQ1 — 점수 비교 시 명시

### 5-6. 평가 지표 (Metrics)

#### 5-6-1. 지표 카테고리

| 카테고리 | 지표 | 측정 대상 | 장점 | 단점 |
|---|---|---|---|---|
| Reference 기반 | ROUGE-1/2/L | N-gram 표면 겹침 | 표준·자동·빠름 | 의미 못 잡음, 패러프레이즈 약함 |
| Reference 기반 | BERTScore (Ko) | 문맥 임베딩 의미 유사도 | 패러프레이즈 강함 | 임베딩 모델 의존 |
| Reference 기반 | BLEU | N-gram precision | 표준 | 생성형엔 부적합 |
| Judge 기반 | LLM-as-Judge (1~10) | 종합 품질·유창성 | 인간 판단과 상관↑ | 비용·judge 편향 |
| Judge 기반 | Pairwise comparison | 모델 간 선호도 | 노이즈 강함, 변별력↑ | 결과 해석 복잡 |
| Judge 기반 | Rubric scoring | 항목별 정량 | 다차원 진단 | rubric 설계 중요 |
| Compliance | Strict / Loose accuracy | 형식 준수율 | 자동 채점 | 품질엔 무관 |
| 운영 | TTFT, TPS, VRAM, 생성시간 | 운영 부하·비용 | 객관 측정 | 환경 의존 |

#### 5-6-2. 벤치별 사용 지표 (v2 — 2026-05-25 재설계 반영)

| 벤치 | 메인 지표 | 보조 지표 | 점수 범위 |
|---|---|---|---|
| AI Hub 논문/한국어 요약 | **LLM-Judge G-Eval 4차원** | ROUGE-1/2/L, BERTScore-Ko | 1~10 (rubric 4 × 1-5 → 환산) |
| Ko-MT-Bench | LLM-as-Judge MT-Bench 표준 (1~10), 8 카테고리 평균 | - | 1~10 |
| LogicKor | LLM-as-Judge LogicKor 스타일 (1~10), 6 카테고리 평균 | - | 1~10 |
| KITE | Rubric score (항목별 1~5) | - | 1~5 |
| Ko-IFEval | Strict accuracy %, Loose accuracy % | - | 0~100% → 0~10 환산 |

→ **벤치 표준 지표를 그대로 사용**. 임의로 새 지표 만들지 않음.
→ 각 벤치의 데이터 구조·수량·예시는 `references/evaluation-datasets.md` 참조.

#### 5-6-3. 종합 점수 산출 방식

**단일 종합 점수 만들지 않음.** 정보 손실이 크기 때문.

권장:
- **측면별 점수표** (Main) — 각 벤치 점수를 그대로 표로 제시
- **순위 기반 보조 집계** (Sub) — 벤치별 순위 평균
- **운영 지표 별도 표** — 품질과 분리해 봄

예시:
```
[품질 표]
| 모델            | Ko-MT (1~10) | LogicKor (1~10) | ROUGE-L | IFEval Strict |
| Qwen3-30B-A3B  | 8.2          | 7.8             | 0.42    | 85%           |
| Qwen3-32B-AWQ  | 8.0          | 7.6             | 0.45    | 87%           |
| ...

[운영 표]
| 모델           | TTFT(ms) | TPS  | VRAM(GB) | 5K토큰생성(s) |
```

#### 5-6-4. LLM-as-Judge 운용
- Judge 모델: 클라우드 평가에선 **외부 강한 모델 사용 가능** (Claude 4.x, GPT-4o 등)
- self-preference bias 회피: 후보 5개와 무관한 family 사용
- position bias 회피: pairwise 비교 시 A/B 순서 swap
- verbosity bias 인지: 길이 normalize 또는 length-controlled prompting

#### 5-6-5. 지표 사용 시 함정
- **ROUGE 한계**: 패러프레이즈 잘하는 모델 손해 → BERTScore와 같이 봄
- **Judge verbosity bias**: 긴 답변에 후한 점수 → 길이 제한
- **Judge position bias**: 첫번째 후보 선호 → swap
- **Judge self-preference**: 같은 family 선호 → 외부 family judge
- **BERTScore 임베딩 의존**: 임베딩 모델 고정 (예: `klue/roberta-large` 또는 `BAAI/bge-m3`)
- **데이터 오염 의심 시**: 해당 벤치 가중치↓ 또는 제외

## 6. 핵심 제약 사항

| 제약 | 영향 |
|---|---|
| 폐쇄망 운영 | 외부 호스팅 API 전면 불가. 모든 모델·도구·평가 데이터 온프레미스 |
| 실 고객 데이터 외부 반출 금지 | 클라우드 평가는 공개 데이터(DART, 공개 벤치)만 사용 |
| 단일 H100 NVL 94GB | GLM/Kimi는 llama.cpp+RAM 오프로드만 가능, 운영 시 느림 |
| 양자화 격차 | Qwen FP8/AWQ4 vs GLM IQ2 vs Kimi TQ1 — 직접 latency·quality 비교 부적절 |
| 도메인 전문가 자원 제한 | 대규모 reference 작성 불가, 가능한 한 자동 채점·proxy 데이터 활용 |

## 7. 결정 보류 사항

### 7-1. 고객사 의존 (가장 우선)
- **Target task 형태 확정** — 기업분석보고서 / 요약 / QA / 대화 / 기타
- 산출물 길이·구조·형식 명세
- 사용자 시나리오 (배치 생성 vs 실시간 대화 등)

### 7-2. Phase 1 (현재 단계)
- 한국어 생성 벤치 최종 셋 선택
  - 후보: AI Hub 논문 요약, AI Hub 한국어 요약, Ko-MT-Bench, KITE, Ko-IFEval, LogicKor, Open Ko-LLM Leaderboard2, Horangi
  - 후보별 상세 명세: `references/evaluation-datasets.md`
  - 4개 측면(유창성·장문·instruction·요약) 커버 기준으로 2~4개 조합
- LLM-as-Judge 모델 선정 (클라우드 평가니까 외부 강한 모델 사용 가능)
- 평가 자동화 도구 (lm-evaluation-harness 등) 환경 세팅

### 7-3. 후속 단계 (Target 확정 후)
- 도메인 평가 벤치 셋 (KFinEval-Pilot / Allganize / KMMLU subset 등 중 선택)
- 자체 평가셋 도입 형태 (DART 30건 또는 다른 형태)
- 도메인 전문가 자원 확보 일정
- 운영 양자화 재측정 시점

## 8. 작업 디렉터리 구조

```
niceinfo/                              # Mac 로컬 (설계 문서 + eval 미러)
├── CLAUDE.md                          # 본 문서
├── eval/                              # 평가 코드 (cloud와 sync, GitHub 원본은 cloud)
├── data/                              # AI Hub 원본 zip + 샘플 jsonl
├── sample_aihub.py                    # 샘플링 스크립트
└── references/
    ├── llm-models.md                  # 5개 모델 상세 사양·비교표
    ├── evaluation-datasets.md         # Phase 1 평가 데이터셋 후보 명세
    └── judge-model-choice.md          # GPT-4o judge 선택 근거

GCP 인스턴스 niceinfo-eval-a100:/home/minji/niceinfo-eval/
├── (위 eval/, data/ 포함)
├── docker-compose.yml                 # vLLM + eval 컨테이너
├── Dockerfile.eval
├── requirements.txt
├── setup.sh                           # 신규 인스턴스 재구축용
└── results/                           # 평가 결과 jsonl 저장

GitHub: https://github.com/mjpark-colla/niceinfo-llm-eval (cloud → push)
```

## 9. 진행 상태 (2026-05-25 기준)

### 완료된 작업
- ✅ GCP a2-ultragpu-1g 인스턴스 (us-central1-a, A100 80GB)
- ✅ Docker + NVIDIA Container Toolkit, vLLM `vllm-openai:latest`
- ✅ 4개 Qwen 모델 다운로드 + 본격 평가 (572 sample × 4 모델)
- ✅ AI Hub 샘플 (582: 150, 90: 150) + Ko-MT-Bench (80) + LogicKor (42) + Ko-IFEval (150)
- ✅ 평가 코드 (`eval/` 20+ 파일)
- ✅ GitHub repo: mjpark-colla/niceinfo-llm-eval
- ✅ **Phase 1 v1 평가 완료** (2026-05-22, 구 generic Judge prompt)
- ✅ **Phase 1 v2 평가 완료** (2026-05-25, **Judge prompt 재설계 + rejudge**)
  - G-Eval (요약), MT-Bench 표준 (대화), LogicKor 스타일 (추론)
  - 모델 답변 고정, Judge만 GPT-4o 재호출 (Mac venv 환경, GPU $0)
  - 결과 `results_v2/`, 비교 `reports/comparison.md`, 메인 리포트 v2
- ✅ HTML 대시보드 v1 (`reports/index.html`)
- ✅ FP8 양자화 평가 (BF16 대비 +0.83%, 손실 1-2% 이내 유지)

### v2 핵심 결과 (자세한 건 메인 리포트 참조)
- 신 1위: **Qwen3-32B-AWQ** (균등·기본·보고서 시나리오)
- 요약 중심 시나리오 1위: **Qwen3-30B-A3B-BF16** (faithfulness 1위)
- 대화 시나리오 1위: **Qwen3.6-35B-A3B-FP8**
- **Qwen3.6-35B-A3B-BF16**: 어떤 시나리오에서도 1위 아님 → 운영 채택 비추천
- AIHub 변별력 3-5배 증가 (Judge prompt 재설계 효과 확인)

### 미완료 작업
- ⏳ HTML 대시보드 v2 (가중치 슬라이더 추가)
- ⏳ Cloud GitHub commit (v2 코드: prompts.py, llm_judge.py, benchmarks/*, rejudge.py, compare.py)
- ⏳ 운영 지표 측정 (TTFT, TPS, VRAM) — Phase 2-A
- ⏳ GLM/Kimi 참고 평가 — RAM 부족으로 보류
- ⏳ Target task 확정 (고객사 소통)

### 다음 세션 재개 가이드

#### 작업 종류별 환경

**A. Judge prompt 추가 변경 + rejudge만** (GPU 불필요, Mac 로컬)
```bash
cd /Users/minji/Documents/PolarPulse/niceinfo
source .venv/bin/activate  # venv 활성화
# .env에 OPENAI_API_KEY 있음
.venv/bin/python -m eval.rejudge --input-dir results --output-dir results_v3
.venv/bin/python -m eval.compare  # 비교 분석
```

**B. 새 모델·새 벤치 평가 (재추론 필요)** — GCP 인스턴스 필요

1. **인스턴스 start** (Local SSD 부착으로 인해 `--discard-local-ssd=false` 필수):
   ```bash
   gcloud compute instances start niceinfo-eval-a100 --zone=us-central1-a
   # stop 시: --discard-local-ssd=false 필수
   gcloud compute instances stop niceinfo-eval-a100 --zone=us-central1-a --discard-local-ssd=false
   ```
   ⚠️ us-central1-a STOCKOUT 가능성 — 자원 부족 시 us-central1-c 또는 다른 region 이전 고려

2. **VS Code Remote-SSH 재연결** (IP 새로 발급되므로):
   ```bash
   gcloud compute config-ssh
   ```
   → VS Code → Cmd+Shift+P → "Remote-SSH: Connect to Host"

3. **컨테이너 상태 확인**:
   ```bash
   cd ~/niceinfo-eval
   docker compose ps     # 이전 컨테이너가 살아있을 수 있음
   docker compose up -d vllm eval  # 필요 시 재시작
   ```

4. **이어서 평가 진행** (옵션 둘 중 하나):
   - **옵션 A** (권장): 다른 벤치 mini-sanity 먼저
     ```bash
     # 4개 벤치 각 3 sample
     for B in logickor ko_ifeval aihub_582 aihub_90; do
       docker compose exec -T eval python -m eval.run \
         --model Qwen3-32B-AWQ --benchmark $B --limit 3 \
         --results-dir /app/results/sanity
     done
     ```
   - **옵션 B**: 바로 본격 평가
     ```bash
     docker compose exec -T eval python -m eval.run \
       --model Qwen3-32B-AWQ --results-dir /app/results
     ```

5. **나머지 2개 모델 다운로드 + 평가**:
   - Qwen3-30B-A3B (BF16, ~60GB)
   - Qwen3.6-35B-A3B (BF16, ~70GB)
   - vLLM 컨테이너 재시작 (MODEL_NAME 환경변수 변경)

### 비용·시간 예상
- 본격 평가: GPU ~$45 (8h) + Judge ~$20 = **~$65**
- 다음 세션 시간: 6-8시간 (모델 로딩·평가 포함)

### ⚠️ 보안 주의사항
- 작업 종료 시 **반드시 인스턴스 stop** (시간당 $5.5)
- 토큰·키 채팅에 노출되지 않도록 주의 (.env 사용)
- 다음 평가 후 GitHub PAT와 OpenAI key 모두 revoke 권장

## 9. 관련 메모리

`/Users/minji/.claude/projects/-Users-minji-Documents-PolarPulse-niceinfo/memory/`:
- `user_polarpulse.md` — 사용자 프로필
- `project_niceinfo_tech_eval.md` — 프로젝트 컨텍스트
- `project_closed_network.md` — 폐쇄망 제약
