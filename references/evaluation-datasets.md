# Phase 1 평가 데이터셋 후보

> 작성일: 2026-05-20
> 프로젝트: niceinfo
> 단계: Phase 1 — 한국어 텍스트 생성 능력 비교
> 본 문서: 사용 후보 공개 데이터셋의 상세 명세

---

## 개요

Phase 1에서 사용할 후보 공개 평가 데이터셋의 명세서. 각 항목은 다음을 포함:
- 데이터 설명
- 평가 태스크 / 측정 능력
- 데이터 수량·구성
- 데이터 예시
- 사용 시 참고사항

평가 4 측면(유창성·장문 구조화·instruction following·요약)을 모두 커버하도록 2~4개 조합 선택 예정.

---

## 1. Ko-MT-Bench (davidkim205/ko-bench)

### 데이터 설명
- 영문 MT-Bench의 한국어 번역·로컬라이즈 버전
- 한국어·한국 문화 맥락 반영 (예: 일론 머스크 → 천송이/유재석, 통화 단위·이름·회사명 한국화)
- 멀티턴 instruction following + 유창성 평가의 한국어 표준

### 평가 태스크
- **다영역 멀티턴 대화 능력**: 한 질문 → 답변 → 후속 질문 → 답변
- **종합 한국어 생성 능력**: 유창성, 추론, 글쓰기, 코딩 등 8 측면

### 데이터 수량·구성
- **80 questions** (8 카테고리 × 10 questions)
- **카테고리**:
  1. Coding (코딩)
  2. Extraction (추출)
  3. Humanities (인문학)
  4. Math (수학)
  5. Reasoning (추론)
  6. Roleplay (역할극)
  7. STEM (과학·기술·공학·수학)
  8. Writing (글쓰기)
- 각 question은 multi-turn (보통 2 turns)

### 평가 방식
- **LLM-as-Judge** (GPT-4 표준), 1~10점 채점
- 카테고리별 평균 + 전체 평균

### 데이터 예시 (구조)
```json
{
  "category": "writing",
  "turns": [
    "어머니의 환갑잔치 초청장을 작성해 주세요. 격식 있게.",
    "위 초청장의 톤을 조금 더 캐주얼하게 바꿔 주세요."
  ]
}
```

### 사용 시 참고
- HuggingFace: `davidkim205/ko-bench`
- 데이터 오염 가능성: 2024년 이후 모델은 학습 데이터에 포함했을 가능성
- Judge 모델 비용 고려 (80 questions × 2 turns × 5 모델 = 800 judge calls)

---

## 2. LogicKor (maywell/LogicKor)

### 데이터 설명
- 한국어 LLM 다영역 추론·생성 능력 평가 벤치마크
- 작성자: Jeonghwan Park (2024)
- GitHub `instructkr/LogicKor`는 2024-10-17 archived
- 데이터·코드는 HuggingFace에서 그대로 접근 가능

### 평가 태스크
- **다영역 추론·생성**: 추론, 수학, 글쓰기, 코딩, 이해, 문법
- 멀티턴 형식

### 데이터 수량·구성
- **42 questions** (6 카테고리 × 7 questions)
- **카테고리**:
  1. 추론 (Reasoning)
  2. 수학 (Math)
  3. 글쓰기 (Writing)
  4. 코딩 (Coding)
  5. 이해 (Understanding)
  6. 문법 (Grammar)
- 각 question은 multi-turn (2 turns) + reference answer 포함

### 평가 방식
- **LLM-as-Judge** (GPT-4 표준), 1~10점
- Reference answer 활용 가능 (참조 기반 채점)

### 데이터 예시 (구조)
```json
{
  "id": 1,
  "category": "추론(Reasoning)",
  "questions": [
    "1번 질문 (1턴)",
    "2번 질문 (2턴, 1번 답변 기반)"
  ],
  "references": [
    "1번 모범 답안",
    "2번 모범 답안"
  ]
}
```

### 사용 시 참고
- HuggingFace: `maywell/LogicKor`, DOI: `10.57967/hf/2440`
- **메인테이너의 "상위권 변별력 약함" 주의**: GPT-4·Claude·Gemini급 saturate. 단 30~40B 오픈 모델군은 변별력 있을 가능성↑
- **Anchor 비교 가능**: GPT-4/Claude 등 공개 점수가 있어 본 모델의 상대 위치 표현 유리
- 42 prompt는 소규모 → 단독 결정 근거로는 불충분, 다른 벤치와 종합 사용
- Dataset viewer가 Arrow validation 에러로 깨져 있을 수 있음. 다운로드는 정상

---

## 3. KITE (Korean Instruction-following Task Evaluation)

### 데이터 설명
- 한국어 특화 instruction-following 벤치마크
- 2025년 발표 (arxiv 2510.15558)
- 한국어 고유 특성(경어법·이중 수 체계·형태소 풍부함) 평가에 초점
- Open-ended 형식으로 기존 MCQ 위주 한국어 벤치의 한계 보완

### 평가 태스크
- **일반 + 한국어 특화 instruction following**
- Open-ended 생성 (객관식 X)
- 한국어 고유 요구사항 처리:
  - 경어법 (존댓말·반말·격식체) 적절성
  - 이중 수 체계 (고유어/한자어 수사)
  - 형태소·접속 표현

### 데이터 수량·구성
- ⚠️ 정확한 수량은 논문 직접 확인 필요 (검색 결과 미상세)
- 일반 instruction + 한국어 특화 instruction 두 갈래 구성으로 추정

### 평가 방식
- **자동 메트릭 + Human assessment 결합**
- Rubric 기반 채점

### 데이터 예시 (구조 추정)
```
Instruction: "다음 내용을 회사 임원께 보내는 공식 이메일로,
              경어를 사용해 5문장으로 작성하세요: [내용]"
Constraints: [경어 사용, 5문장 분량, 격식체]
```

### 사용 시 참고
- arxiv: 2510.15558
- 최신 벤치라 데이터 오염 가능성 낮음 (장점)
- 데이터셋 공개 형태·HuggingFace 미러 여부 사전 확인 필요
- Human assessment 부분은 자동화 불가 — Auto 메트릭 부분만 사용 가능 여부 확인 필요

---

## 4. Ko-IFEval

### 데이터 설명
- Google의 영문 IFEval (Instruction-Following Evaluation, 2023)의 한국어 버전
- Verifiable instruction-following 평가의 표준 벤치
- Open Ko-LLM Leaderboard 2의 구성 요소

### 평가 태스크
- **형식·제약 준수 능력**: 단어 수, 형식, 키워드, 대소문자 등
- 25개 instruction type, 각 prompt에 1~3개 verifiable 제약 부여

### 데이터 수량·구성
- 영문 원본 IFEval: **541 prompts**
- Ko-IFEval: 영문 번역 + 한국어 적응 (정확 수량 확인 필요)
- 25 instruction types (예: 단어 수 N개 이하, JSON 형식, 특정 키워드 포함 등)

### 평가 방식
- **4가지 자동 채점 메트릭**:
  1. Prompt-level **strict** accuracy: 한 prompt의 모든 제약 정확 준수율
  2. Instruction-level **strict** accuracy: 개별 제약 정확 준수율
  3. Prompt-level **loose** accuracy: 느슨한 기준 prompt 준수율
  4. Instruction-level **loose** accuracy: 느슨한 기준 instruction 준수율
- 자동 채점 (judge LLM 불필요, 가장 저비용)

### 데이터 예시
```
Prompt: "프랑스 혁명에 대해 한국어로 200~250 단어 사이로 설명하세요.
        반드시 '루이 16세'와 '바스티유'라는 단어를 포함해야 합니다."
Constraints:
  - length: 200~250 words
  - keywords: ["루이 16세", "바스티유"]
```

### 사용 시 참고
- **가장 저비용 벤치** (judge LLM 불필요)
- 형식 준수만 측정, 내용 품질은 미반영 → 다른 벤치와 보완 필수
- Open Ko-LLM Leaderboard 2 통해 결과 비교 가능

---

## 5. AI Hub 한국어 문서 요약 (dataSetSn=582)

### 데이터 설명
- 한국지능정보사회진흥원(NIA) AI Hub 제공 한국어 요약 데이터셋
- 다양한 도메인의 한국어 원문 + 전문 요약 페어
- 추출 요약(extractive) + 생성 요약(abstractive) 둘 다 포함

### 평가 태스크
- **장문 한국어 → 요약 생성**
- 보고서 작성과 가장 가까운 공개 task

### 데이터 수량·구성
- **원문**: 400,000건
  - 신문 기사: 300,000건
  - 기고문: 60,000건
  - 잡지 기사: 10,000건
  - 판결문: 30,000건
- **요약 데이터**: 800,000건 (추출 요약 400K + 생성 요약 400K)

### 평가 방식 (권장)
- **자동 메트릭**:
  - ROUGE-1, ROUGE-2, ROUGE-L
  - BERTScore (Ko, e.g., `klue/roberta-large` 또는 `BAAI/bge-m3` 임베딩)
- Reference 페어 있음 → 자동 채점 가능

### 데이터 예시 (구조)
```json
{
  "passage": "원문 기사 (수백 ~ 수천 단어)",
  "extractive_summary": ["원문에서 추출한 핵심 문장 3개"],
  "abstractive_summary": "원문을 재생성한 새로운 요약문"
}
```

### 사용 시 참고
- **다운로드 사전 승인 필요** (AI Hub 신청 → 며칠 소요 가능)
- 압축 분할 파일, Linux/WSL 환경 권장
- 평가 시 80만 건 전부 X. **샘플링(500~2,000건)으로 충분**
- 도메인 (뉴스/기고/잡지/판결) 선택 가능 — 보고서 톤에 가까운 도메인 우선 사용 권장

---

## 6. AI Hub 논문 요약 데이터셋 (dataSetSn=90)

### 데이터 설명
- AI Hub 학술 논문 요약 데이터셋
- 학술 톤·구조 → 보고서 톤과 유사도 매우 높음
- 한국어 + 영문 학술 자료 혼합

### 평가 태스크
- **학술 장문 → 요약·섹션 요약 생성**
- 구조화된 장문 생성 능력 평가에 가장 적합

### 데이터 수량·구성
- 논문: **180,000건** + 전체 요약 180,000건
- 섹션별 요약: 180,000건
- 특허 명세서: 170,000건 + 전체 요약 + 섹션 요약
- **총 약 700,000건 요약 텍스트**

### 평가 방식 (권장)
- AI Hub 한국어 요약과 동일: ROUGE + BERTScore-Ko
- 섹션별 요약 데이터를 활용하면 **구조화 능력**도 평가 가능

### 데이터 예시 (구조 추정)
```json
{
  "title": "논문 제목",
  "abstract_full": "원문 초록",
  "full_summary": "전체 요약",
  "section_summaries": {
    "introduction": "...",
    "methods": "...",
    "results": "...",
    "conclusion": "..."
  }
}
```

### 사용 시 참고
- **다운로드 사전 승인 필요**
- 보고서 작성 task와의 톤 매칭이 가장 좋아 **Phase 1 권장도 매우 높음**
- 평가 샘플 500~2,000건 권장
- 논문/특허 도메인 → 기업분석보고서 톤보다 학술적이지만, 구조화·인용·재생성 능력 평가엔 적합

---

## 7. Open Ko-LLM Leaderboard 2

### 데이터 설명
- Upstage·NIA 등이 운영하는 한국어 LLM 종합 평가 리더보드
- Ko-H5 (구버전) 후속, 2024-10 발표 (arxiv 2410.12445)
- 실용성·한국어 고유성 강조

### 평가 태스크 (구성 벤치)
- **번역 벤치** (영문 표준 → 한국어):
  - Ko-IFEval (instruction following)
  - Ko-GPQA Diamond (고난도 MCQ)
  - Ko-WinoGrande (상식 MCQ)
  - Ko-GSM8K (수학 추론)
  - Ko-EQ-Bench (감정·소셜 인지)
- **Native 한국어 4종 (스크래치)**:
  - KorNAT-Knowledge (한국 지식)
  - KorNAT-Social-Value (사회적 가치)
  - Ko-Harmlessness (안전성)
  - Ko-Helpfulness (실용성)

### 데이터 수량·구성
- 벤치별로 다름 — 위 각 항목 개별 확인 필요
- 총 9개 구성 벤치 종합 점수 제공

### 평가 방식
- 벤치별 표준 지표 (MCQ는 정답률, IFEval은 strict/loose, etc.)
- 일부 벤치는 자동, 일부는 LLM-as-Judge

### 사용 시 참고
- **공식 리더보드 점수 그대로 사용은 비추천**: 본인 환경·양자화로 직접 평가
- 구성 벤치 중 Phase 1 메인 적합: **Ko-IFEval** (위 4번 항목과 동일)
- MCQ 벤치(Ko-GPQA, Ko-WinoGrande)는 Phase 1 메인에서 제외 (생성 평가 아님)
- KorNAT·Ko-Harmlessness 등은 컴플라이언스 보조 평가용으로 검토 가능

---

## 8. Horangi (한국어 LLM 벤치마크 평가 프레임워크)

### 데이터 설명
- wandb 주관 오픈소스 한국어 LLM 평가 프레임워크
- 2축 평가: GLP(General Language Performance) + ALT(Alignment)

### 평가 태스크
- **GLP**: 한국어 일반 언어 능력
- **ALT**: 가치 정렬·안전성

### 데이터 수량·구성
- 여러 sub-benchmark 포함 (프레임워크라 단일 벤치 아님)
- wandb 통합으로 결과 시각화·비교 용이

### 평가 방식
- 프레임워크 내 통합 지표
- LLM-as-Judge 일부 포함

### 사용 시 참고
- GitHub: `wandb/llm-leaderboard-korean`
- 프레임워크 전체 도입은 본 프로젝트에 과함 — **개별 벤치 발췌 사용** 권장
- 평가 결과 시각화·재현성 측면에서 wandb 사용 시 강점

---

## 한 눈에 비교

| 벤치 | 데이터 형태 | 수량 | 평가 측면 | 채점 방식 | Phase 1 권장도 |
|---|---|---|---|---|---|
| **Ko-MT-Bench** | 멀티턴 대화 | 80 Q × 2 turn | 종합 생성·유창성 | LLM-as-Judge (1~10) | ★★★★★ |
| **LogicKor** | 멀티턴 추론 | 42 Q × 2 turn | 추론·생성 | LLM-as-Judge (1~10) | ★★★★ (Anchor 비교용) |
| **KITE** | Open-ended instruction | TBD | 한국어 특화 instruction | Auto + Human | ★★★★ (최신, 오염↓) |
| **Ko-IFEval** | 형식 제약 prompt | ~541개 | 형식 준수 | 자동 (Strict/Loose) | ★★★★★ (저비용·객관) |
| **AI Hub 한국어 요약** | 원문 + 요약 페어 | 80만 페어 | 장문 요약 | ROUGE + BERTScore | ★★★★★ (target 근접) |
| **AI Hub 논문 요약** | 학술 원문 + 요약 | 70만 요약 | 학술 장문 구조화 | ROUGE + BERTScore | ★★★★★ (보고서 톤 유사) |
| **Open Ko-LLM LB2** | 벤치 모음 | 9개 구성 | 종합 | 벤치별 표준 | ★★★ (Ko-IFEval만 발췌 권장) |
| **Horangi** | 평가 프레임워크 | — | 종합 | 프레임워크 내 통합 | ★★ (선택 사용) |

---

## Phase 1 최소 조합 권장안 (4 측면 커버)

| 측면 | 추천 벤치 |
|---|---|
| **유창성·종합 생성** | Ko-MT-Bench + LogicKor (anchor 비교 보조) |
| **장문 구조화** | AI Hub 논문 요약 (샘플 500~1,000건) |
| **Instruction following** | Ko-IFEval + KITE (가용 시) |
| **요약** | AI Hub 한국어 요약 (샘플 500~1,000건) |

**총 4~5개 벤치**, 측면별 1~2개 보강.
운영 부담·judge 비용 고려해 최종 2~4개로 압축 가능.

---

## 추가 참고

- **AI Hub 데이터셋 신청**: 본 평가 진행 결정 시 즉시 신청 (승인 며칠 소요)
- **HuggingFace 데이터셋 캐싱**: 폐쇄망 이전 가능성 고려해 로컬 캐싱 권장
- **데이터 오염 모니터링**: 의외로 점수가 saturate되면 해당 벤치는 가중치↓ 또는 제외
- **벤치별 표준 평가 코드**: 가능하면 원저자 제공 코드 그대로 사용 (재현성)

## 관련 문서

- `CLAUDE.md` — 프로젝트 전체 평가 방향성
- `references/llm-models.md` — 평가 대상 모델 명세
