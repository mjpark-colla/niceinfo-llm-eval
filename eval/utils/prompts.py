"""Judge LLM 평가 prompt 템플릿."""

JUDGE_SYSTEM = "당신은 한국어 LLM 답변을 엄정하게 평가하는 심사관입니다. 점수와 이유를 정확한 형식으로 답변하세요."


JUDGE_USER_SINGLE_TURN = """아래는 사용자 질문과 AI 모델의 답변입니다. 답변을 평가해주세요.

[질문]
{question}

[답변]
{answer}
{reference_block}
[평가 기준]
1. 정확성: 사실 오류가 없는가
2. 유용성: 질문의 의도에 충실한가
3. 한국어 품질: 문법, 어휘, 자연스러움, 격식이 적절한가
4. 완결성: 답변이 논리적이고 완성되어 있는가

다음 형식으로만 답변하세요. 다른 텍스트는 추가하지 마세요.

점수: [1-10 사이의 정수]
이유: [한두 문장의 평가 근거]
"""


JUDGE_USER_MULTI_TURN = """아래는 사용자와 AI 모델의 대화입니다. 마지막 AI 답변을 평가해주세요.

[대화 맥락]
{context}

[마지막 질문]
{question}

[마지막 답변]
{answer}
{reference_block}
[평가 기준]
1. 정확성: 사실 오류가 없는가
2. 맥락 일관성: 이전 대화와 잘 이어지는가
3. 한국어 품질: 문법, 어휘, 자연스러움, 격식이 적절한가
4. 완결성: 답변이 논리적이고 완성되어 있는가

다음 형식으로만 답변하세요. 다른 텍스트는 추가하지 마세요.

점수: [1-10 사이의 정수]
이유: [한두 문장의 평가 근거]
"""


def build_judge_prompt(
    question: str,
    answer: str,
    reference: str | list[str] | None = None,
    context: str | None = None,
) -> tuple[str, str]:
    """(system, user) prompt 튜플 반환."""
    if isinstance(reference, list):
        ref_text = "\n".join(f"- {r}" for r in reference)
        reference_block = f"\n[참고 정답 (여러 개 가능)]\n{ref_text}\n"
    elif reference:
        reference_block = f"\n[참고 정답]\n{reference}\n"
    else:
        reference_block = ""

    if context:
        user = JUDGE_USER_MULTI_TURN.format(
            context=context,
            question=question,
            answer=answer,
            reference_block=reference_block,
        )
    else:
        user = JUDGE_USER_SINGLE_TURN.format(
            question=question,
            answer=answer,
            reference_block=reference_block,
        )

    return JUDGE_SYSTEM, user
