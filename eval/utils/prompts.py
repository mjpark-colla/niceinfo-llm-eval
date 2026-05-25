"""Task-specific LLM-as-Judge prompt 빌더.

벤치마크 task 특성에 맞춰 4가지 분리:
- summarization (G-Eval, 요약 평가용 4차원 rubric)
- multiturn_dialogue (MT-Bench 표준 single-v1 변형, 한국어)
- multiturn_reasoning (LogicKor 스타일, reference 비교 강조)
- generic (legacy, 호환용)

설계 원칙 (CLAUDE.md 5-3-1 보강):
- Score anchor 명시 (1-10 각 단계 의미)
- Length/verbosity bias 회피 명시
- 점수 분해능 0.5 단위 허용
- Rubric 차원 분리 (요약은 4차원 → 종합)
- 표준 MT-Bench / LogicKor format 따름
"""
from typing import Literal


# ============================================================
# 공통 System Prompt 베이스
# ============================================================
SYSTEM_BASE = """당신은 공정하고 엄정한 한국어 LLM 답변 평가관입니다.
주관적 인상이 아닌 명시된 기준에 따라 일관되게 채점하세요.

다음 편향을 의식적으로 회피하세요:
- 답변의 길이·형식·구조가 화려하다고 점수를 후하게 주지 않기 (verbosity bias)
- 답변이 자신감 있게 쓰였더라도 사실 오류 또는 무관함이 있으면 감점
- 한국어 표현이 매끄러우면 좋지만, 정확성·관련성이 우선
"""

SUMMARIZATION_SYSTEM = SYSTEM_BASE + """
당신의 임무는 원문 대비 모델이 생성한 요약의 품질을 4개 차원으로 평가하는 것입니다."""

DIALOGUE_SYSTEM = SYSTEM_BASE + """
당신의 임무는 사용자 질문에 대한 AI 답변의 품질을 1-10점으로 평가하는 것입니다.
점수는 반드시 "Rating: [[N]]" 형식으로 명시하세요. N은 1-10 (0.5 단위 가능)."""

REASONING_SYSTEM = SYSTEM_BASE + """
당신의 임무는 한국어 추론·논리 문제에 대한 AI 답변을 평가하는 것입니다.
참고 정답이 주어진 경우 그것과 의미적으로 일치하는지 우선 확인하세요.
점수는 반드시 "Rating: [[N]]" 형식으로 명시하세요. N은 1-10 (0.5 단위 가능)."""


# ============================================================
# 1. G-Eval Summarization (AIHub 582 / 90)
#    참고: Liu et al. 2023, "G-Eval: NLG Evaluation using GPT-4"
# ============================================================
SUMMARIZATION_USER = """다음은 원문과 AI 모델이 생성한 요약입니다. 4개 차원으로 평가하세요.

[원문]
{passage}

[모델 요약]
{summary}
{reference_block}
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
"""


# ============================================================
# 2. MT-Bench Single-Turn (Ko-MT-Bench turn 1)
#    참고: lmsys/FastChat single-v1 prompt, 한국어 적용
# ============================================================
DIALOGUE_SINGLE_USER = """다음 사용자 질문에 대한 AI 답변의 품질을 평가하세요.

[질문]
{question}

[답변]
{answer}
{reference_block}
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
"""


# ============================================================
# 3. MT-Bench Multi-Turn (Ko-MT-Bench turn 2)
# ============================================================
DIALOGUE_MULTITURN_USER = """다음은 사용자와 AI의 멀티턴 대화입니다. 두 번째 AI 답변의 품질을 평가하세요.
이전 대화 맥락과의 일관성도 고려하세요.

[첫 번째 질문]
{question1}

[첫 번째 답변]
{answer1}

[두 번째 질문]
{question2}

[두 번째 답변]
{answer2}
{reference_block}
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
"""


# ============================================================
# 4. LogicKor (추론·논리 특화, 단일턴/멀티턴 공용)
#    참고: maywell/LogicKor 평가 방식 — reference 비교 강조
# ============================================================
REASONING_SINGLE_USER = """다음 한국어 추론·논리 문제에 대한 AI 답변을 평가하세요.

[질문]
{question}

[답변]
{answer}
{reference_block}
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
"""

REASONING_MULTITURN_USER = """다음은 추론·논리 문제의 멀티턴 대화입니다. 두 번째 답변을 평가하세요.

[첫 번째 질문]
{question1}

[첫 번째 답변]
{answer1}

[두 번째 질문]
{question2}

[두 번째 답변]
{answer2}
{reference_block}
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
"""


# ============================================================
# Builder 함수들
# ============================================================
TaskType = Literal["summarization", "dialogue", "reasoning"]


def _format_reference_block(reference: str | list[str] | None) -> str:
    if not reference:
        return ""
    if isinstance(reference, list):
        # 빈 문자열·None 제거
        refs = [str(r) for r in reference if r]
        if not refs:
            return ""
        if len(refs) == 1:
            return f"\n[참고 답변]\n{refs[0]}\n"
        ref_text = "\n".join(f"- {r}" for r in refs)
        return f"\n[참고 답변 (여러 개 가능)]\n{ref_text}\n"
    return f"\n[참고 답변]\n{reference}\n"


def build_summarization_prompt(
    passage: str,
    summary: str,
    reference: str | list[str] | None = None,
) -> tuple[str, str]:
    """G-Eval 요약 평가 prompt → (system, user)."""
    reference_block = ""
    if reference:
        refs = reference if isinstance(reference, list) else [reference]
        refs = [r for r in refs if r]
        if refs:
            ref_text = "\n".join(f"- {r}" for r in refs)
            reference_block = f"\n[참고 요약 (있을 경우 비교 가능)]\n{ref_text}\n"

    user = SUMMARIZATION_USER.format(
        passage=passage,
        summary=summary,
        reference_block=reference_block,
    )
    return SUMMARIZATION_SYSTEM, user


def build_dialogue_prompt(
    question: str,
    answer: str,
    reference: str | list[str] | None = None,
    prev_question: str | None = None,
    prev_answer: str | None = None,
) -> tuple[str, str]:
    """MT-Bench 스타일 single/multi-turn → (system, user)."""
    reference_block = _format_reference_block(reference)

    if prev_question is not None and prev_answer is not None:
        user = DIALOGUE_MULTITURN_USER.format(
            question1=prev_question,
            answer1=prev_answer,
            question2=question,
            answer2=answer,
            reference_block=reference_block,
        )
    else:
        user = DIALOGUE_SINGLE_USER.format(
            question=question,
            answer=answer,
            reference_block=reference_block,
        )
    return DIALOGUE_SYSTEM, user


def build_reasoning_prompt(
    question: str,
    answer: str,
    reference: str | list[str] | None = None,
    prev_question: str | None = None,
    prev_answer: str | None = None,
) -> tuple[str, str]:
    """LogicKor 스타일 추론 평가 → (system, user)."""
    reference_block = _format_reference_block(reference)

    if prev_question is not None and prev_answer is not None:
        user = REASONING_MULTITURN_USER.format(
            question1=prev_question,
            answer1=prev_answer,
            question2=question,
            answer2=answer,
            reference_block=reference_block,
        )
    else:
        user = REASONING_SINGLE_USER.format(
            question=question,
            answer=answer,
            reference_block=reference_block,
        )
    return REASONING_SYSTEM, user


# ============================================================
# Legacy (호환용) — 기존 코드에서 build_judge_prompt 호출 시 안전 fallback
# ============================================================
def build_judge_prompt(
    question: str,
    answer: str,
    reference: str | list[str] | None = None,
    context: str | None = None,
) -> tuple[str, str]:
    """Legacy generic prompt. 가능하면 task-specific 사용 권장."""
    # context가 있으면 멀티턴 추론으로 매핑
    if context:
        return build_reasoning_prompt(
            question=question, answer=answer, reference=reference,
            prev_question=context, prev_answer="(이전 답변 생략)",
        )
    return build_dialogue_prompt(
        question=question, answer=answer, reference=reference,
    )
