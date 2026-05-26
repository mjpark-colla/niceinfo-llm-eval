# 모델별 속도·메모리 실측 결과

> **측정일**: 2026-05-26
> **환경**: GCP a2-ultragpu-1g (NVIDIA A100 80GB × 1)
> **vLLM 옵션**: `--gpu-memory-utilization 0.95 --max-model-len 32768 --dtype auto` (Phase 1 v1/v2 환경 동일)
> **측정 도구**: `eval/speed_benchmark.py` (streaming API, VRAM polling 0.3초 간격)

---

## 1. 측정 요약 (medium prompt 기준)

| 순위 | 모델 | 구조 | 양자화 | **TPS** | VRAM 점유 | TTFT |
|---|---|---|---|---|---|---|
| 🥇 | **Qwen3.6-35B-A3B-FP8** | MoE (활성 3B) | FP8 | **164.2** | 75.2 GiB | 0.32s |
| 🥈 | Qwen3.6-35B-A3B (BF16) | MoE (활성 3B) | BF16 | 144.8 | 75.2 GiB | 0.31s |
| 🥉 | Qwen3-30B-A3B | MoE (활성 3.3B) | BF16 | 132.1 | 75.8 GiB | 0.04s |
| 4 | Qwen3-32B-AWQ | Dense 32.8B | AWQ 4-bit | 66.9 | 74.9 GiB | 0.04s |
| 5 | Qwen3-32B (BF16) | Dense 32.8B | BF16 | 23.8 | 75.1 GiB | 0.06s |

---

## 2. 측정 환경 상세

### 하드웨어
- GPU: NVIDIA A100-SXM4-80GB × 1 (us-central1-a)
- A100 메모리 bandwidth: 2,039 GB/s
- vCPU: 12, RAM: 170GB
- Disk: pd-ssd 1500GB + Local SSD 375GB

### vLLM 설정
```bash
--model <model_path>
--max-model-len 32768
--gpu-memory-utilization 0.95
--dtype auto
# (Phase 1 v1/v2 docker-compose.yml 동일)
```

### 측정 방식
- **streaming API** (OpenAI client)
- **TTFT** (Time To First Token): 첫 토큰 chunk 도착 시각
- **Decode TPS**: 첫 토큰 ~ 마지막 토큰 사이 평균 속도
- **VRAM**: 0.3초 간격 background polling, peak 값 기록
- **concurrency**: 1 (단독 sequence)
- **temperature**: 0.0 (deterministic)
- **max_tokens**: 512

### Prompt 길이별
- **short**: 14~72 자 (3 sample)
- **medium**: 40~74 자 (3 sample) ← **종합 대표 길이**
- **long**: 245 자 (1 sample)

---

## 3. 상세 측정 결과

### 3-1. Qwen3.6-35B-A3B-FP8 🥇

| 길이 | TTFT (s) | Decode TPS | Total (s) | Tokens out | VRAM peak (MiB) |
|---|---|---|---|---|---|
| short | 1.152 | 162.5 | 2.96 | 255 | 76,970 |
| medium | 0.324 | 164.2 | 3.62 | 512 | 76,970 |
| long | 0.076 | 164.2 | 3.29 | 512 | 76,970 |

→ **최고 속도**. FP8 양자화 + MoE 활성 3B 조합.

### 3-2. Qwen3.6-35B-A3B (BF16) 🥈

| 길이 | TTFT (s) | Decode TPS | Total (s) | Tokens out | VRAM peak (MiB) |
|---|---|---|---|---|---|
| short | 1.137 | 144.3 | 3.07 | 240 | 77,024 |
| medium | 0.308 | 144.8 | 3.98 | 512 | 77,024 |
| long | 0.072 | 144.9 | 3.63 | 512 | 77,024 |

→ FP8보다 약 13% 느림. BF16은 dequant 없으나 메모리 bandwidth 2배 사용.

### 3-3. Qwen3-30B-A3B 🥉

| 길이 | TTFT (s) | Decode TPS | Total (s) | Tokens out | VRAM peak (MiB) |
|---|---|---|---|---|---|
| short | 0.316 | 134.0 | 2.52 | 275 | 77,572 |
| medium | 0.039 | 132.1 | 3.85 | 490 | 77,572 |
| long | 0.049 | 133.3 | 3.94 | 509 | 77,572 |

→ MoE 활성 3.3B (약간 큼). FP8 모델 대비 약 80% 속도.

### 3-4. Qwen3-32B-AWQ

| 길이 | TTFT (s) | Decode TPS | Total (s) | Tokens out | VRAM peak (MiB) |
|---|---|---|---|---|---|
| short | 0.336 | 67.3 | 4.07 | 236 | 76,696 |
| medium | 0.041 | 66.9 | 7.80 | 503 | 76,696 |
| long | 0.099 | 67.6 | 7.91 | 510 | 76,696 |

→ Dense + AWQ 4-bit. AWQ가 BF16 Dense보다 약 **2.8배 빠름** (메모리 bandwidth 절감 효과).

### 3-5. Qwen3-32B (BF16, 양자화 X)

| 길이 | TTFT (s) | Decode TPS | Total (s) | Tokens out | VRAM peak (MiB) |
|---|---|---|---|---|---|
| short | 0.346 | 23.9 | 11.92 | 270 | 76,934 |
| medium | 0.062 | 23.8 | 21.51 | 506 | 76,934 |
| long | 0.091 | 24.0 | 21.51 | 510 | 76,934 |

→ **가장 느림**. Dense + 무양자화. 메모리 bandwidth bottleneck.

---

## 4. 핵심 발견

### 4-1. 메모리 점유 — 5 모델 거의 동일 (75GB)

```
모델별 VRAM 점유: 74.9 ~ 75.8 GiB
이유: vLLM이 --gpu-memory-utilization 0.95 × A100 80GB = 76GB
       시작 즉시 다 예약 (KV cache 영역으로)
```

→ **모델 weights (19~62GB) 차이는 KV cache 영역으로 흡수**. 운영 환경에서 모든 모델이 거의 풀 GPU 사용.

### 4-2. 속도 결정 요인 — 3가지 (영향 큰 순서)

| 요인 | 영향 | 예시 |
|---|---|---|
| **1. 구조 (MoE vs Dense)** | **~6배** | Dense 32B BF16 24 → MoE 활성 3B BF16 145 |
| **2. 양자화 (메모리 bandwidth)** | **~3배** | Dense BF16 24 → Dense AWQ 4bit 67 |
| **3. FP8 Tensor Core** | ~1.13배 | MoE BF16 145 → MoE FP8 164 |

### 4-3. 양자화 = 메모리 + 속도 둘 다 줄임

**핵심**: A100에서 추론은 **메모리 bandwidth bottleneck** (계산보다 weight 읽기가 느림). 양자화로 weights 크기 줄이면:
- 메모리: 4-bit는 BF16의 1/4
- Bandwidth 사용: 4-bit는 1/4 → forward pass 시간 4배 단축 (이론적)
- 실제: dequant overhead 등으로 약 2.8배 향상

```
A100 메모리 bandwidth: 2 TB/s
BF16 60GB weights → forward 1번에 30ms
AWQ 15GB weights → forward 1번에 7.5ms + dequant
```

### 4-4. AWQ 의외의 발견

이전 추측: "AWQ는 메모리만 줄임, 속도는 안 줄어듦"
**실측**: AWQ는 Dense BF16보다 **2.8배 빠름** (메모리 bandwidth 절감 효과)

다만:
- MoE 모델 (132~164 tok/s) > AWQ Dense (67 tok/s)
- **MoE 활성 파라미터 효과 > 양자화 효과**

---

## 5. 운영 시사점

### 5-1. 시나리오별 최적 모델

| 시나리오 | 추천 모델 | 근거 |
|---|---|---|
| **속도 최우선** | **Qwen3.6-35B-A3B-FP8** | 164 tok/s, MoE + FP8 |
| **메모리 절감** | **Qwen3-32B-AWQ** | weights 19GB만 사용 (util 낮춰 다른 워크로드 동시 운영 가능) |
| **속도 + 메모리 균형** | **Qwen3.6-35B-A3B-FP8** | weights 35GB + 속도 1위 |
| ❌ **운영 비추천** | Qwen3-32B (BF16) | Dense + 무양자화 = 24 tok/s, 가장 느림 |

### 5-2. "메모리 점유 19GB"의 진짜 의미

`references/llm-models.md` 의 모델별 메모리 표기 (예: "Qwen3-32B-AWQ ~19GB")는 **모델 weights만의 이론값**:

| 표기 | 실제 (vLLM util 0.95) |
|---|---|
| Qwen3-32B-AWQ "19GB" | 75 GiB 점유 (KV cache 포함) |
| Qwen3.6-35B-FP8 "35GB" | 75 GiB 점유 |
| Qwen3-30B-A3B "60GB" | 76 GiB 점유 |

→ vLLM의 `--gpu-memory-utilization 0.95` 설정 시 **시작 즉시 76GB 다 예약**. weights 크기에 따라 KV cache 영역만 다름 (동시 처리 가능 sequence 수에 영향).

→ "19GB 점유"는 weights 의미, 운영 실제 GPU 사용은 75GB.

---

## 6. 한계 및 주의

### 6-1. 측정 한계

| 항목 | 본 측정 | 운영 환경 차이 |
|---|---|---|
| GPU | A100 80GB (us-central1-a) | H100 NVL 94GB (폐쇄망) |
| 동시 처리 | concurrency 1 | 실제는 concurrency 1+ |
| context 사용 | 짧은 prompt (~100 tokens) | 긴 input·동시 sequence 가능 |
| Network | 단일 인스턴스 | 임베딩·리랭커 동시 부하 |

→ **A100 결과는 운영 환경 (H100 NVL)에서 다를 수 있음**. H100은 FP8 native 지원 → FP8 모델 가속 효과 더 큼 예상.

### 6-2. 측정 신뢰도

- ✅ Phase 1 v1/v2 동일 vLLM 옵션 (재현성 확보)
- ✅ 5 모델 동일 prompt set 측정
- ✅ VRAM polling 정확 (background 0.3초 간격)
- ⚠️ Prompt 7개로 적음 (단, deterministic temperature 0이라 noise 작음)
- ⚠️ Cold start latency 미측정

### 6-3. 가능한 후속 측정

- **H100 NVL 폐쇄망 sanity check** (Phase 3)
- **concurrency 변화** (동시 1, 2, 4, 8 sample 처리량)
- **장문 input 영향** (8K, 32K context 채우기)
- **VRAM utilization 변화** (0.50, 0.70, 0.95 비교)

---

## 7. 자료 위치

- **측정 결과 raw jsonl**: `results_speed/<모델>_ops.jsonl` (Mac 로컬, cloud에서 다운로드 후)
- **측정 코드**: `eval/speed_benchmark.py`
- **wrapper**: `eval/run_speed_all.sh`, `measure_qwen3_32b.sh`
- **본 보고서**: `references/speed-benchmark.md`

---

## 8. v1 추정 (Phase 1 jsonl 기반) vs v2 실측 차이

이전 `reports/speed.md`의 추정값은 Phase 1 평가 jsonl 의 `elapsed_sec` / `tokens_out` 기반 → 실측과 차이 큼:

| 모델 | v1 추정 (jsonl, concurrency 8) | **v2 실측 (concurrency 1, streaming)** |
|---|---|---|
| Qwen3.6-35B-A3B-FP8 | 29.9 tok/s | **164.2 tok/s** |
| Qwen3-30B-A3B | 26.5 tok/s | 132.1 tok/s |
| Qwen3.6-35B-A3B (BF16) | 26.4 tok/s | 144.8 tok/s |
| Qwen3-32B-AWQ | 18.3 tok/s | 66.9 tok/s |

→ **v1 추정은 부정확** (concurrency 8 환경 + judge 시간 포함). 본 v2 실측이 정확.

---

*작성: 2026-05-26 / Phase 1 v2 보강 / 문의: mjpark@polarpulse.ai*
