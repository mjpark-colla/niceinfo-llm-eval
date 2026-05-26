# LLM-as-Judge 프롬프트 정리 (데이터셋별)

> **버전**: v2 (2026-05-25 재설계 후 사용 중)
> **Judge 모델**: OpenAI `gpt-4o` (primary) / `gpt-4o-mini` (fallback)
> **출처**: `eval/utils/prompts.py`

---

## 데이터셋 ↔ 프롬프트 매핑

| 데이터셋 | Judge 사용 여부 | 적용 프롬프트 | 점수 형식 |
|---|---|---|---|
| Ko-MT-Bench | ✅ LLM-Judge | **Dialogue** (MT-Bench 표준 single-v1 / multi-turn) | `Rating: [[N]]` (1-10, 0.5 단위) |
| LogicKor | ✅ LLM-Judge | **Reasoning** (LogicKor 스타일, reference 비교 강조) | `Rating: [[N]]` (1-10, 0.5 단위) |
| AIHub 582 | ✅ LLM-Judge | **Summarization** (G-Eval 4차원 rubric) | 4차원 × 1-5점 + 종합 1-10 |
| AIHub 90 | ✅ LLM-Judge | **Summarization** (G-Eval 4차원 rubric) | 4차원 × 1-5점 + 종합 1-10 |
| Ko-IFEval | ❌ Judge 없음 | (자동 룰 채점) | Strict / Loose accuracy |

---

## 0. 공통 System Prompt 베이스 (SYSTEM_BASE)

모든 task-specific system prompt 앞에 공통으로 붙는 베이스:

```
당신은 공정하고 엄정한 한국어 LLM 답변 평가관입니다.
주관적 인상이 아닌 명시된 기준에 따라 일관되게 채점하세요.

다음 편향을 의식적으로 회피하세요:
- 답변의 길이·형식·구조가 화려하다고 점수를 후하게 주지 않기 (verbosity bias)
- 답변이 자신감 있게 쓰였더라도 사실 오류 또는 무관함이 있으면 감점
- 한국어 표현이 매끄러우면 좋지만, 정확성·관련성이 우선
```

→ 모든 prompt 에 적용. 길이·형식 bias 회피, 정확성 우선 명시.

---

## 1. Summarization (G-Eval) — AIHub 582 / AIHub 90

### 1-1. System Prompt

```
[SYSTEM_BASE 공통] +

당신의 임무는 원문 대비 모델이 생성한 요약의 품질을 4개 차원으로 평가하는 것입니다.
```

### 1-2. User Prompt 템플릿

```
다음은 원문과 AI 모델이 생성한 요약입니다. 4개 차원으로 평가하세요.

[원문]
{passage}

[모델 요약]
{summary}

[참고 요약 (있을 경우 비교 가능)]
- {reference}

---

## 평가 차원 (각 1-5점)

### 1. 사실 충실도 (Faithfulness) — 원문에 없는 내용·잘못된 사실 없음
- 1: 다수의 환각 또는 명백한 오류
- 2: 명백한 사실 오류 1-2개
- 3: 사소한 부정확 (수치·고유명사 등)
- 4: 거의 정확, 매우 사소한 오류
- 5: 완벽히 정확, 환각 없음

### 2. 핵심 정보 포함도 (Relevance) — 원문의 중요 정보를 빠뜨리지 않음
- 1: 핵심 정보 대부분 누락
- 2: 핵심 정보 일부 누락
- 3: 핵심 일부만 포함
- 4: 핵심 대부분 포함
- 5: 모든 핵심 정보 포함

### 3. 일관성 (Coherence) — 논리적 흐름·문장 연결 자연스러움
- 1: 문장 단절, 모순, 비논리적
- 2: 흐름이 부자연스러움
- 3: 평균적
- 4: 매끄러운 흐름
- 5: 매우 매끄럽고 잘 구성됨

### 4. 간결성 (Conciseness) — 불필요한 중언부언·장황함 없음
- 1: 매우 장황, 반복 많음
- 2: 다소 장황
- 3: 적정과 장황 사이
- 4: 적정 길이, 간결함
- 5: 매우 간결하고 정제됨

---

## 채점 절차 (중요)
먼저 차원별로 단계별 사고(Chain-of-Thought)를 한 후 점수를 매기세요.
점수만 던지지 말고 근거를 짧게 명시하세요.

## 출력 형식 (반드시 준수, 다른 텍스트 금지)
사실 충실도: [1-5]
핵심 정보 포함도: [1-5]
일관성: [1-5]
간결성: [1-5]
종합: [4개 평균 × 2, 1.0~10.0]
이유: [2-4문장 요약]
```

### 1-3. 학술 근거
- **G-Eval**: Liu et al. 2023, *"G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment"*
- 인간 평가와 Spearman 0.5+ 상관
- 요약 평가 학술 표준 (SummEval 등)

### 1-4. 출력 예시 (실제)
```
사실 충실도: 5
핵심 정보 포함도: 4
일관성: 5
간결성: 5
종합: 9.6
이유: 모델 요약은 원문에서 언급된 최저임금 가이드라인 설정과 제도 개선의
       필요성을 정확히 반영하고 있습니다. 박영선 위원의 발언 내용이 일부
       누락되어 있어 핵심 정보 포함도에서 약간의 감점이 있었습니다.
```

---

## 2. Dialogue (MT-Bench 표준) — Ko-MT-Bench

### 2-1. System Prompt

```
[SYSTEM_BASE 공통] +

당신의 임무는 사용자 질문에 대한 AI 답변의 품질을 1-10점으로 평가하는 것입니다.
점수는 반드시 "Rating: [[N]]" 형식으로 명시하세요. N은 1-10 (0.5 단위 가능).
```

### 2-2. User Prompt — Single-Turn (Turn 1)

```
다음 사용자 질문에 대한 AI 답변의 품질을 평가하세요.

[질문]
{question}

[답변]
{answer}

[참고 답변]
{reference}

---

## 평가 요소
- 유용성: 질문 의도에 충실히 답했는가
- 정확성: 사실 오류 없음
- 관련성: 무관한 내용 없음
- 깊이: 표면적이지 않고 충분한 정보
- 완결성: 답변이 마무리되어 있음
- 한국어 품질: 문법·어휘·격식 적절

## 점수 기준 (1-10, 0.5 단위 가능)
- 1-2: 답변 거부 또는 완전히 잘못된 답변
- 3-4: 심각한 오류 또는 질문과 무관
- 5-6: 평균, 일부 오류·부족함
- 7-8: 양호, 사소한 결함
- 9-10: 우수, 거의 완벽

## 출력 형식
먼저 평가 이유를 2-4문장으로 제시한 후, 마지막 줄에 "Rating: [[N]]" 형식으로 점수.
N은 1-10 사이 숫자 (예: 7, 7.5, 8).
```

### 2-3. User Prompt — Multi-Turn (Turn 2)

```
다음은 사용자와 AI의 멀티턴 대화입니다. 두 번째 AI 답변의 품질을 평가하세요.
이전 대화 맥락과의 일관성도 고려하세요.

[첫 번째 질문]
{question1}

[첫 번째 답변]
{answer1}

[두 번째 질문]
{question2}

[두 번째 답변]
{answer2}

[참고 답변]
{reference}

---

## 평가 요소
- 유용성: 두 번째 질문 의도에 충실
- 맥락 일관성: 첫 번째 답변과 모순되지 않음
- 정확성: 사실 오류 없음
- 깊이·완결성·한국어 품질

## 점수 기준 (1-10, 0.5 단위 가능)
- 1-2: 답변 거부 또는 완전 잘못, 또는 이전 답변과 모순
- 3-4: 심각한 오류
- 5-6: 평균
- 7-8: 양호, 사소한 결함
- 9-10: 우수

## 출력 형식
평가 이유 2-4문장 후, 마지막 줄에 "Rating: [[N]]".
```

### 2-4. 학술 근거
- **MT-Bench**: Zheng et al. NeurIPS 2023, *"Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena"*
- LLM-Judge 학술 표준 format (`[[N]]`)
- lmsys/FastChat 의 `single-v1` / `single-v1-multi-turn` 한국어 적용

### 2-5. 멀티턴 처리 (v2 개선)
- **v1**: 이전 답변을 `(생략)` 처리 → 일관성 평가 부정확
- **v2**: 실제 이전 답변을 judge prompt 에 전달 → 정확한 맥락 평가

---

## 3. Reasoning (LogicKor 스타일) — LogicKor

### 3-1. System Prompt

```
[SYSTEM_BASE 공통] +

당신의 임무는 한국어 추론·논리 문제에 대한 AI 답변을 평가하는 것입니다.
참고 정답이 주어진 경우 그것과 의미적으로 일치하는지 우선 확인하세요.
점수는 반드시 "Rating: [[N]]" 형식으로 명시하세요. N은 1-10 (0.5 단위 가능).
```

### 3-2. User Prompt — Single-Turn

```
다음 한국어 추론·논리 문제에 대한 AI 답변을 평가하세요.

[질문]
{question}

[답변]
{answer}

[참고 답변]
{reference}

---

## 평가 요소 (참고 정답이 있으면 우선 비교)
- 정답성: 참고 정답과 의미적으로 일치하는가 (참고 정답 부재 시 자체 판단)
- 추론 과정: 논리 과정이 명확·타당한가
- 정확성: 계산·사실 오류 없음
- 한국어 품질·완결성

## 점수 기준 (1-10, 0.5 단위 가능)
- 1-2: 완전히 틀린 답
- 3-4: 부분적 정답, 추론 흐릿
- 5-6: 평균
- 7-8: 정답 + 명확한 추론
- 9-10: 완벽한 정답 + 깔끔한 추론

## 출력 형식
평가 이유 2-4문장 후, 마지막 줄에 "Rating: [[N]]".
```

### 3-3. User Prompt — Multi-Turn

```
다음은 추론·논리 문제의 멀티턴 대화입니다. 두 번째 답변을 평가하세요.

[첫 번째 질문]
{question1}

[첫 번째 답변]
{answer1}

[두 번째 질문]
{question2}

[두 번째 답변]
{answer2}

[참고 답변]
{reference}

---

## 평가 요소
- 정답성 (참고 정답 우선)
- 추론 일관성: 첫 번째와 모순 없음
- 정확성·한국어 품질

## 점수 기준 (1-10, 0.5 단위)
- 1-2: 완전 틀림 또는 이전과 모순
- 3-4: 부분 정답
- 5-6: 평균
- 7-8: 정답 + 명확
- 9-10: 완벽

## 출력 형식
평가 이유 2-4문장 후, 마지막 줄에 "Rating: [[N]]".
```

### 3-4. 학술 근거
- maywell/LogicKor (HuggingFace) 스타일
- reference 비교 강조 (한국어 추론 벤치마크의 표준 방식)

---

## 4. Ko-IFEval — Judge 없음, 자동 룰

### 4-1. 채점 방식

LLM-Judge 사용 안 함. **자동 룰 기반 채점**:
- **Strict accuracy**: 모든 instruction 룰을 엄격히 만족했는가
- **Loose accuracy**: 어느 정도 허용 범위 내 만족했는가
- 종합 점수 = `(strict + loose) / 2 × 10` (0-10 환산)

### 4-2. 채점 룰 예시

| Instruction ID | 룰 |
|---|---|
| `length:num_words` | 정확한 단어 수 (strict) / ±10% (loose) |
| `keywords:include` | 모든 키워드 포함 (strict) / 대소문자 무시 (loose) |
| `format:json` | JSON 파싱 가능 (strict) / `{}` 포함 (loose) |
| `punctuation:no` | 문장부호 없음 |
| `detectable_format:number_highlighted_sections` | `*highlighted*` 패턴 N회 |

→ 구현: `eval/benchmarks/ko_ifeval.py::KoIFEval._check_instruction`

### 4-3. Judge 미사용 이유
- 객관 룰로 자동 채점 가능 → Judge 호출 비용 절감
- IFEval 원본도 자동 룰 채점 표준
- LLM-Judge가 룰 판정에서 노이즈 추가 가능성 (역효과)

---

## 5. 공통 설계 원칙 (v2)

| 원칙 | 적용 |
|---|---|
| **Score anchor 명시** | 1-2, 3-4, 5-6, 7-8, 9-10 각 단계 의미 정의 (모든 prompt) |
| **점수 분해능 0.5 단위** | 정수 → 실수 (변별력↑) |
| **Verbosity bias 회피** | "길이·형식이 화려하다고 점수 후하게 X" 명시 |
| **CoT (Chain-of-Thought)** | "단계별 사고 후 점수 매겨라" (요약 평가) |
| **Rubric 분리** | 요약은 4차원 → 종합 (다차원 진단 가능) |
| **표준 format** | MT-Bench `Rating: [[N]]` 형식 (외부 비교 가능) |
| **멀티턴 실제 답변 전달** | v1 `(생략)` → v2 실제 이전 답변 (정확한 맥락 평가) |

---

## 6. v1 → v2 변경 사항 (참고)

v1 (2026-05-22): 자체 generic prompt — 모든 벤치 같은 prompt 사용. anchor 없음, 정수만, bias 회피 명시 없음.
v2 (2026-05-25): 위 task-specific prompt 4종으로 분리 + 표준 prompt 채택.

**효과**: AIHub 변별력 3.0배~5.3배 증가, 외부 학술 비교 가능, 차원별 분석 가능.

---

## 7. 자료 위치

- 코드: [`eval/utils/prompts.py`](https://github.com/mjpark-colla/niceinfo-llm-eval/blob/main/eval/utils/prompts.py)
- Judge 호출: [`eval/metrics/llm_judge.py`](https://github.com/mjpark-colla/niceinfo-llm-eval/blob/main/eval/metrics/llm_judge.py)
- 실제 평가 raw 응답: `results_v2/<모델>/<벤치>.jsonl` 의 `turns[].judge_raw`

---

*작성: 2026-05-25 / Phase 1 v2 / 문의: mjpark@polarpulse.ai*
