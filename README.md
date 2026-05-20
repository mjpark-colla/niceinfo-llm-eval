# niceinfo-eval — 한국어 LLM 평가 프로젝트

PolarPulse niceinfo 프로젝트의 LLM 모델 평가 환경.

## 빠른 시작 (새 인스턴스에서)

```bash
git clone <this-repo>
cd niceinfo-eval
bash setup.sh                # Docker + NVIDIA toolkit 설치
cp .env.example .env         # 후 .env 편집해서 토큰 입력

# vLLM 서버 시작
MODEL_NAME=Qwen/Qwen3-32B-AWQ docker compose up -d vllm

# 평가 클라이언트 빌드·실행
docker compose build eval
docker compose run --rm eval bash
```

## 구조

```
niceinfo-eval/
├── setup.sh              # 새 인스턴스 환경 셋업
├── docker-compose.yml    # vLLM + eval 서비스
├── Dockerfile.eval       # 평가 클라이언트 이미지
├── requirements.txt      # 평가 클라이언트 Python deps
├── .env.example          # 환경 변수 템플릿
├── eval/                 # 평가 코드 (Git 추적)
├── models/               # HF 모델 캐시 (호스트, .gitignore)
├── data/                 # 평가 데이터셋 (호스트, .gitignore)
└── results/              # 평가 결과 (호스트, .gitignore, GCS 동기화 권장)
```

## 평가 대상 모델

- Qwen/Qwen3-30B-A3B (BF16)
- Qwen/Qwen3-32B-AWQ (AWQ 4bit)
- Qwen/Qwen3.6-35B-A3B (BF16)

GLM-5.1, Kimi-K2.5는 RAM 부족(170GB < 240GB)으로 본 인스턴스에서 평가 불가. 별도 환경 필요.

## 평가 벤치마크

- Ko-MT-Bench (LLM-as-Judge)
- LogicKor (LLM-as-Judge)
- Ko-IFEval (자동)
- AI Hub 요약 (ROUGE + BERTScore)
