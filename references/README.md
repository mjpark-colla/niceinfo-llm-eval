# references/ — 참고 문서 인덱스

> 프로젝트 진행하면서 만든 참고·근거·정리 문서들의 모음.
> 새 문서 추가 시 이 인덱스에도 한 줄 추가해주세요.

---

## 문서 목록

### 1. [llm-models.md](./llm-models.md)
**평가 대상 5개 LLM 모델의 상세 사양·비교표**
- Qwen3-30B-A3B, Qwen3-32B-AWQ, Qwen3.6-35B-A3B, GLM-5.1, Kimi-K2.5
- 각 모델의 파라미터·아키텍처·컨텍스트·라이선스
- 단일 H100 NVL 환경에서의 실행 가능성

### 2. [evaluation-datasets.md](./evaluation-datasets.md)
**Phase 1 평가 데이터셋 후보 명세**
- Ko-MT-Bench, LogicKor, Ko-IFEval, AI Hub 582/90 등
- 각 데이터셋의 태스크·수량·예시·사용 시 참고사항
- 4 측면(유창성·장문 구조화·instruction·요약) 커버 권장 조합

### 3. [judge-model-choice.md](./judge-model-choice.md)
**LLM-as-Judge로 GPT-4o 선택한 근거**
- NeurIPS 2023 seminal paper (Zheng et al., MT-Bench)
- GPT-4 family가 사실상 표준인 이유
- gpt-5 (reasoning model) 미선택 사유
- 비용·신뢰도·bias 회피 분석

### 4. [bertscore-num-layers.md](./bertscore-num-layers.md)
**BERTScore의 `num_layers` 파라미터 이해**
- BERT 모델 layer 구조와 의미 표현
- 왜 17번 layer가 의미 유사도에 최적인지
- klue/roberta-large 사용 시 num_layers=17 명시 필요
- AIHub 요약 평가에서 BERTScore=0.0 버그 분석·수정 기록

### 5. [llm-eval-key-concepts.md](./llm-eval-key-concepts.md) ⭐ 종합
**LLM 평가·운영 핵심 개념 종합 정리**
- 양자화 (BF16/FP8/AWQ/GPTQ/GGUF)
- Context Length, YaRN, KV cache 메모리 계산
- vLLM 메모리 옵션 (gpu-memory-utilization 등)
- Tokenizer 효율 (한국어 char ↔ token)
- Async/Concurrency 평가
- 평가 vs 운영 환경 정합성
- 본 프로젝트 결정 요약

### 6. [speed-benchmark.md](./speed-benchmark.md) ⚡ 실측
**5 모델 속도·메모리 실측 결과 (2026-05-26, A100 80GB)**
- TTFT, Decode TPS, VRAM 점유 매트릭스
- Qwen3-32B 양자화(AWQ) vs 무양자화(BF16) 비교
- 속도 결정 요인 3가지 (MoE 구조 / 양자화 / FP8)
- "19GB 점유"의 진짜 의미 (weights vs 운영 점유)
- v1 추정 vs v2 실측 비교

---

## 문서 작성 가이드

새 문서 추가 시:

1. **파일명**: kebab-case, `.md` 확장자
2. **상단 메타데이터**: 작성일, 프로젝트, 본 문서 한 줄 설명
3. **본 인덱스에 등록**: 위 목록에 한 줄 추가
4. **관련 문서 링크**: 다른 references/ 문서나 CLAUDE.md 적극 링크

### 어떤 문서를 references/에 두는가

| 적합 | 부적합 |
|---|---|
| 의사결정의 근거 (왜 X 선택?) | 진행 상태·todo |
| 외부 개념 정리 (양자화, BERTScore 등) | 코드 구현 세부 |
| 데이터셋·모델 명세 | 일시적 디버그 메모 |
| 학술 문헌 참조·인용 | 매우 자주 변하는 정보 |

---

## 관련 문서 (references/ 밖)

- [`../CLAUDE.md`](../CLAUDE.md) — 프로젝트 전체 평가 방향성·진행 상태
- [`../eval/`](../eval/) — 평가 코드 (cloud에서 운영)
