# Judge LLM 모델 선택 근거

> 작성일: 2026-05-21
> 프로젝트: niceinfo
> 결정: **GPT-4o** 를 LLM-as-Judge primary 모델로 선택

---

## 결정 요약

| 항목 | 값 |
|---|---|
| Primary judge | **gpt-4o** |
| Fallback judge | gpt-4o-mini |
| 적용 벤치 | Ko-MT-Bench, LogicKor |
| 호출 패턴 | 점수(1~10) + 이유 (순차 호출) |

---

## 1. LLM-as-a-Judge의 학술적 표준

### 1.1 시작점: NeurIPS 2023 seminal paper

**"Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena"** (Zheng et al., 2023)
- arxiv: https://arxiv.org/abs/2306.05685
- NeurIPS 2023 spotlight 채택, 6,500+ 인용
- LLM-as-Judge를 학계에 공식 도입한 시작점

### 1.2 핵심 결과

- **GPT-4 judge가 인간 평가자와 80%+ 일치율** 달성
- 이는 인간 평가자끼리의 일치율(human-human agreement)과 **동일한 수준**
- 통계적으로 "GPT-4 judge ≈ 인간 평가자 1명"으로 해석 가능
- MT-Bench(80문제) 및 Chatbot Arena(LMSYS) 평가의 사실상 표준

### 1.3 이후 영향

이 논문 발표 이후 거의 모든 LLM 평가 논문/리더보드가 GPT-4 family 를 judge로 채택. LLM-as-Judge 분야의 de facto standard 정착.

---

## 2. 한국어 평가에서의 채택 현황

| 벤치/평가 | Judge 모델 | 비고 |
|---|---|---|
| **Ko-MT-Bench** (davidkim205/ko-bench) | GPT-4 | 원조 MT-Bench 방식 그대로 |
| **LogicKor** (maywell) | GPT-4 | 6 카테고리 1~10 점 |
| **KUDGE** | GPT-4 / GPT-4o | 한국어 최초 LLM-as-Judge meta-eval 벤치마크 |
| **KMMLU 데이터 구축·평가** | **gpt-4o** | annotation, parsing에 사용 |
| **HRM8K, KMMLU-Redux/Pro** | GPT-4 family | 표준 답습 |

→ 한국어 학술 벤치마크에서도 GPT-4 family 가 사실상 표준.

---

## 3. 왜 gpt-4o인가 (gpt-4, gpt-4-turbo 대신)

| 변형 | 출시 | 상태 |
|---|---|---|
| gpt-4 (legacy) | 2023-03 | deprecated 진행 중 |
| gpt-4-turbo | 2023-11 | 중간 세대 |
| **gpt-4o** | 2024-05 | **현재 GPT-4 family 대표** |

### 비교

| 측면 | gpt-4 (legacy) | gpt-4o |
|---|---|---|
| 가격 (input/output) | $30 / $60 | $5 / $20 |
| 속도 | 느림 | ~2배 빠름 |
| 한국어 평가 성능 | 동급 | 동급 또는 약간 우위 |
| 평가 신뢰도 (MT-Bench 기준) | 표준 | **표준 갱신** |

→ 2024년 이후 학술 논문·리더보드 대부분 **gpt-4 → gpt-4o 전환**.

---

## 4. 왜 gpt-5 / o1 / o3 (reasoning model)을 안 쓰는가

| 사유 | 설명 |
|---|---|
| **API 호환성 차이** | `max_tokens` → `max_completion_tokens`, temperature 제약 등 다른 API contract |
| **응답 속도** | reasoning 시간으로 3~10배 느림 (judge call 366회 × 30초 = 3시간+) |
| **평가용 오버킬** | 단순 1~10점 채점은 GPT-4o 수준이면 충분 — reasoning model의 강점(복잡 추론)이 평가에 필요 없음 |
| **검증 부족** | gpt-5 기반 LLM-as-Judge 연구 아직 적음. 평가 신뢰도가 기존 GPT-4 family 만큼 학술적으로 검증되지 않음 |

→ 표준화·검증된 gpt-4o가 평가용으로 더 적합.

---

## 5. 본 프로젝트 적용

### 5.1 사용 시나리오

```python
# Ko-MT-Bench 평가 시 (각 turn마다)
score = judge.score(
    question=turn_prompt,
    answer=qwen_model_output,
    reference=ko_bench_refer,  # 있을 때
)
# → {"score": 8.0, "raw": "점수: 8\n이유: ...", "judge_model": "gpt-4o", ...}
```

### 5.2 비용 추정 (gpt-4o 기준)

| 벤치 | 호출 수 | 평균 토큰 (in/out) | 비용 |
|---|---|---|---|
| Ko-MT-Bench (80×2턴×3모델) | 480 | ~3K / ~0.3K | ~$10 |
| LogicKor (42×2턴×3모델) | 252 | ~3K / ~0.3K | ~$5 |
| **총** | **732** | — | **~$15** |

### 5.3 평가 시간

- gpt-4o 응답 평균: ~2초
- 732 calls × 2초 = ~25분 (순차 호출)

---

## 6. 주의사항

### Judge bias 회피
- **Self-preference bias**: 평가 대상이 Qwen/GLM/Kimi family라 gpt-4o family와 무관 → 영향 적음
- **Position bias**: pairwise 비교 안 함 (1~10 absolute score) → 해당 없음
- **Verbosity bias**: 평가 prompt에 "한두 문장 이유"로 제한 → 완화

### Fallback 전략
- gpt-4o 호출 실패 시 자동 gpt-4o-mini로 전환 (코드 구현됨)
- gpt-4o-mini는 신뢰도 약간 낮지만 비용 ~10x 저렴

---

## 7. 참고 문헌

- **Zheng et al. 2023** — Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena, NeurIPS 2023
  - https://arxiv.org/abs/2306.05685
- **Amphora (HuggingFace blog)** — Navigating Korean LLM Research #2: Evaluation Tools
  - https://huggingface.co/blog/amphora/navigating-ko-llm-research-2
- **KMMLU 논문** — https://arxiv.org/pdf/2402.11548
- **KMMLU-Redux to Pro** — https://arxiv.org/pdf/2507.08924
- **OpenAI API 문서** — Model parameter compatibility
  - https://platform.openai.com/docs/models/gpt-4o
- **LMSYS Chatbot Arena** — https://chat.lmsys.org/

---

## 8. 향후 재검토 트리거

다음 상황 발생 시 judge 모델 재검토:

1. **gpt-5 기반 평가 신뢰도 검증 논문** 다수 발표 시 → 전환 검토
2. **gpt-4o deprecation** 공지 시 → 후속 모델로 이전
3. **본 프로젝트 평가 결과가 인간 평가와 크게 어긋날 때** → judge 모델 또는 prompt 재설계
