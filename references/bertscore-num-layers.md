# BERTScore의 `num_layers` 파라미터 이해하기

> 작성일: 2026-05-22
> 배경: AIHub 요약 평가에서 BERTScore가 0.0으로 계산되던 버그 분석 및 수정 기록

---

## 한 줄 요약

> BERT 모델은 여러 층(layer)으로 쌓여 있고, **각 층마다 다른 종류의 의미를 표현**한다.
> BERTScore는 그중 **의미 유사도 측정에 가장 적합한 "중상위 층"**(약 70% 깊이)을 사용해야 한다.
> 한국어 모델(klue/roberta-large)은 라이브러리에 자동 매핑되어 있지 않아서, **17번 층을 직접 지정해야 한다.**

---

## 1. BERT 모델의 구조

`klue/roberta-large` 같은 모델은 **24개의 transformer layer**가 순서대로 쌓인 구조입니다.

입력 문장(한국어)은 이 24층을 **순서대로 통과**하면서 점점 더 추상적인 표현으로 변환됩니다.

```
입력: "안녕하세요"
   ↓
[Layer 1]   글자·형태소 분리, 위치 정보
   ↓
[Layer 2~6]   기본 패턴 인식
   ↓
[Layer 7~12]   문법, 구문 구조
   ↓
[Layer 13~19]   ★ 의미적 추상화 (이게 BERTScore에 최적)
   ↓
[Layer 20~24]   pre-training task에 특화된 출력
   ↓
출력: 임베딩 벡터
```

**중요**: 한국어 전용 layer 같은 건 없습니다. 모든 layer가 한국어를 다룹니다.

---

## 2. 왜 17번 층이 좋은가

BERTScore 논문(Zhang et al., ICLR 2020)이 실험으로 증명한 것:

> "Pre-training 끝난 모델의 **중상위 layer**에서 추출한 임베딩이 **인간의 의미 판단과 가장 잘 일치한다.**"

이유:
- **하위 layer (1~6)**: 단어 형태에 가까운 표현. 같은 의미인데 표현 다른 두 문장을 다르게 판단함.
- **중상위 layer (13~19)** ⭐: **의미 자체**를 잘 잡음. "행복하다" ≈ "기쁘다" 같은 paraphrase 판단 잘함.
- **최상위 layer (20~24)**: pre-training의 masking task에 특화. 의미 유사도엔 너무 specialized.

24-layer 모델의 경우 **17번 layer (약 70% 깊이)** 가 일반적으로 최적으로 알려져 있습니다.

### 비유: 회사 보고서 작성 단계

| Layer 깊이 | 회사 비유 | 역할 |
|---|---|---|
| 1~6 | 신입사원 | 받아쓰기, 글자 인식 |
| 7~12 | 대리·과장 | 문장 다듬기, 구조 정리 |
| **13~19** | **부장·이사** | **핵심 의미 추출** ⭐ |
| 20~24 | CEO | 특정 보고 양식에 fit |

→ BERTScore는 "**부장급 추상화**" 가 필요한 작업.

---

## 3. 우리가 겪은 문제

### 증상
AIHub 요약 평가에서 BERTScore가 항상 `0.0`으로 계산됨. 결과적으로 점수가 비정상적으로 낮음.

```json
{
  "metric_details": {
    "rouge1": 0.43,
    "rouge2": 0.18,
    "rougeL": 0.16,
    "bertscore_f1": 0.0    ← ⚠️
  }
}
```

### 원인 추적

`bert_score` 파이썬 라이브러리 내부:

```python
# bert_score/score.py
num_layers = model2layers[model_type]  # ← KeyError 발생
```

`model2layers` 사전에 등록된 모델 (영어 표준만 있음):
```python
model2layers = {
    "bert-base-uncased": 9,
    "bert-large-uncased": 18,
    "roberta-base": 10,
    "roberta-large": 17,
    "xlm-roberta-base": 9,
    "xlm-roberta-large": 17,
    # ... 한국어 모델 없음
}
```

`klue/roberta-large`는 등록되어 있지 않으므로 **KeyError**.

우리 코드에 있는 `try-except` 블록이 KeyError를 잡아서 BERTScore를 `0.0`으로 fallback:

```python
try:
    bert_f1 = compute_bertscore_single(model_output, references)
except Exception as e:
    bert_f1 = 0.0    ← 여기로 빠짐
```

→ 점수에 큰 손실 발생. ROUGE-L(0.16) + BERTScore(0.0) / 2 × 10 = **0.82점**

### 해결

`bert_score` 호출 시 **`num_layers=17`** 명시:

```python
P, R, F = bert_score(
    predictions,
    references,
    model_type="klue/roberta-large",
    num_layers=17,             # ← 추가
    lang="ko",
)
```

왜 17:
- `klue/roberta-large`는 영어 `roberta-large`와 **같은 24-layer 구조**
- 영어 roberta-large의 표준값(17)이 그대로 적용 가능
- BERTScore 논문에서 다양한 24-layer 모델에 대해 검증된 값

---

## 4. 수정 후 기대 효과

### 점수 변화 예시

Before:
- ROUGE-L: 0.165
- BERTScore: 0.0 (오류)
- combined = (0.165 + 0.0) / 2 × 10 = **0.82** / 10

After:
- ROUGE-L: 0.165
- BERTScore: ~0.72 (정상)
- combined = (0.165 + 0.72) / 2 × 10 = **~4.4** / 10

→ 점수가 실제 모델 능력 반영하도록 정상화.

### 한국어 요약 평가에서 일반적 BERTScore 범위

| 품질 | BERTScore F1 (klue/roberta-large) |
|---|---|
| 매우 좋음 | 0.85+ |
| 좋음 | 0.75~0.85 |
| 보통 | 0.65~0.75 |
| 미흡 | <0.65 |

(영어 BERTScore보다 한국어 BERTScore가 절대값이 약간 낮은 경향 — 모델 차이)

---

## 5. 다른 한국어 모델 사용 시

다른 한국어 BERT 모델을 BERTScore에 사용할 때 `num_layers` 참고값:

| 모델 | 총 layer | BERTScore 권장 `num_layers` |
|---|---|---|
| `klue/roberta-large` | 24 | **17** |
| `klue/roberta-base` | 12 | 8~9 |
| `monologg/koelectra-base-v3-discriminator` | 12 | 8 |
| `BAAI/bge-m3` (multilingual) | 24 | 17 |
| `intfloat/multilingual-e5-large` | 24 | 17 |

일반 원칙: **모델 깊이의 약 70% 지점** 사용. (정확한 최적값은 실험으로 결정)

---

## 6. 참고 문헌

- **Zhang et al., 2020** — BERTScore: Evaluating Text Generation with BERT
  - https://arxiv.org/abs/1904.09675 (ICLR 2020)
- **bert-score GitHub**: https://github.com/Tiiiger/bert_score
  - `model2layers` 정의: `bert_score/utils.py`
- **klue/roberta-large**: https://huggingface.co/klue/roberta-large

---

## 7. 코드 변경 위치

`eval/metrics/bertscore_korean.py`:
- `DEFAULT_NUM_LAYERS = 17` 상수 추가
- `compute_bertscore()` 함수에 `num_layers` 파라미터 추가
- 내부 `bert_score()` 호출 두 군데에 `num_layers=num_layers` 전달

수정 commit: 2026-05-22
