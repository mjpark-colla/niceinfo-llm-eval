#!/bin/bash
# 3 Qwen 모델 자동 순차 평가
# 사용: bash scripts/run_eval_all.sh
set -e

cd "$(dirname "$0")/.."
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

# (HF_name, display_name) 쌍
MODELS=(
  "Qwen/Qwen3-32B-AWQ|Qwen3-32B-AWQ"
  "Qwen/Qwen3-30B-A3B|Qwen3-30B-A3B-BF16"
  "Qwen/Qwen3.6-35B-A3B|Qwen3.6-35B-A3B-BF16"
)

OVERALL_LOG="$LOG_DIR/run_all_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$OVERALL_LOG") 2>&1

echo "============================================================"
echo "  3 Qwen 모델 자동 평가 시작: $(date)"
echo "============================================================"

for entry in "${MODELS[@]}"; do
  HF_NAME="${entry%%|*}"
  DISPLAY="${entry##*|}"

  echo ""
  echo "============================================================"
  echo "[$(date +%H:%M:%S)] >>> 모델 시작: $DISPLAY ($HF_NAME)"
  echo "============================================================"

  # 현재 vLLM에 로드된 모델 확인
  CURRENT=$(sg docker -c "docker compose exec -T vllm curl -s http://localhost:8000/v1/models 2>/dev/null | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d[\"data\"][0][\"id\"])' 2>/dev/null" || echo "none")
  echo "현재 vLLM 모델: $CURRENT"

  if [ "$CURRENT" != "$HF_NAME" ]; then
    echo "[$(date +%H:%M:%S)] vLLM 모델 전환: $CURRENT → $HF_NAME"
    sg docker -c "docker compose stop vllm"
    sg docker -c "MODEL_NAME=\"$HF_NAME\" docker compose up -d vllm"

    echo "[$(date +%H:%M:%S)] vLLM ready 대기 중..."
    until sg docker -c "docker compose exec -T vllm curl -s -o /dev/null -w \"%{http_code}\" http://localhost:8000/v1/models" 2>/dev/null | grep -q 200; do
      sleep 15
      echo "  ...대기 중 ($(date +%H:%M:%S))"
    done
    sleep 5
    echo "[$(date +%H:%M:%S)] ✅ vLLM ready: $HF_NAME"
  else
    echo "vLLM 이미 $HF_NAME 서비스 중 — 재시작 생략"
  fi

  # 평가 실행
  echo "[$(date +%H:%M:%S)] 평가 실행 시작: $DISPLAY"
  sg docker -c "docker compose exec -T eval python -m eval.run --model \"$DISPLAY\" --results-dir /app/results" 2>&1 | tee "$LOG_DIR/eval-$DISPLAY.log" | grep -v "HTTP Request" | grep -v "WARNING"

  echo ""
  echo "[$(date +%H:%M:%S)] ✅ 완료: $DISPLAY"
done

echo ""
echo "============================================================"
echo "[$(date +%H:%M:%S)] 🎉 3 모델 평가 모두 완료"
echo "============================================================"

# 최종 집계
echo ""
echo "[최종 집계 실행]"
sg docker -c "docker compose exec -T eval python -m eval.run --aggregate-only --results-dir /app/results"

echo ""
echo "[$(date +%H:%M:%S)] All done!"
