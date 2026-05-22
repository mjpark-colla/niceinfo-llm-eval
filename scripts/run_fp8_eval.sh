#!/bin/bash
# Qwen3.6-35B-A3B-FP8 평가 wrapper
# 1) 다운로드 완료 대기 → 2) vLLM 전환 → 3) 평가 실행
set -e

cd "$(dirname "$0")/.."
LOG=logs/fp8_eval_$(date +%Y%m%d_%H%M%S).log
exec > >(tee -a "$LOG") 2>&1

echo "============================================================"
echo "[$(date +%H:%M:%S)] FP8 평가 wrapper 시작"
echo "============================================================"

echo ""
echo "[1] FP8 모델 다운로드 완료 대기..."
while sg docker -c "docker compose exec -T vllm bash -c \"ps aux | grep \\\"hf download\\\" | grep -v grep\"" 2>/dev/null | grep -q .; do
  sleep 30
  echo "  ...다운로드 진행 중 ($(date +%H:%M:%S))"
done
echo "[$(date +%H:%M:%S)] ✅ 다운로드 완료"

echo ""
echo "[2] vLLM 모델 전환 (BF16 → FP8)"
sg docker -c "docker compose stop vllm"
sed -i "s|^MODEL_NAME=.*|MODEL_NAME=Qwen/Qwen3.6-35B-A3B-FP8|" .env
sg docker -c "docker compose up -d vllm"

echo ""
echo "[3] vLLM ready 대기..."
until sg docker -c "docker compose exec -T vllm curl -s -o /dev/null -w \"%{http_code}\" http://localhost:8000/v1/models" 2>/dev/null | grep -q 200; do
  sleep 15
  echo "  ...vLLM 로딩 ($(date +%H:%M:%S))"
done
echo "[$(date +%H:%M:%S)] ✅ vLLM ready: Qwen/Qwen3.6-35B-A3B-FP8"

echo ""
echo "[4] FP8 평가 실행 (5 벤치, concurrency 8)"
sg docker -c "docker compose exec -T eval python -m eval.run --model Qwen3.6-35B-A3B-FP8 --results-dir /app/results --concurrency 8" 2>&1 | grep -v "HTTP Request" | grep -v "WARNING" | tail -100

echo ""
echo "============================================================"
echo "[$(date +%H:%M:%S)] 🎉 FP8 평가 완료"
echo "============================================================"

echo ""
echo "[자동 집계 — 전체 모델 비교]"
sg docker -c "docker compose exec -T eval python -m eval.run --aggregate-only --results-dir /app/results" 2>&1 | tail -30
