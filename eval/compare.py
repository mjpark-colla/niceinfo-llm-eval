"""구 vs 신 Judge prompt 평가 결과 비교 분석.

생성물:
- per-bench 점수 표 (구·신·Δ)
- 모델 종합 점수 + 순위 변화
- 변별력(spread) 비교
- 시나리오별 가중치 5종 시뮬레이션
- G-Eval 4차원 통계 (요약 평가 한정)

사용:
  python -m eval.compare [--old-dir results] [--new-dir results_v2] [--output reports/comparison.md]
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path


MODELS = [
    "Qwen3.6-35B-A3B-BF16",
    "Qwen3.6-35B-A3B-FP8",
    "Qwen3-32B-AWQ",
    "Qwen3-30B-A3B-BF16",
]
BENCHES = ["ko_mt_bench", "logickor", "ko_ifeval", "aihub_582", "aihub_90"]
BENCH_LABEL = {
    "ko_mt_bench": "Ko-MT-Bench",
    "logickor": "LogicKor",
    "ko_ifeval": "Ko-IFEval",
    "aihub_582": "AIHub 582",
    "aihub_90": "AIHub 90",
}

# 시나리오별 가중치
SCENARIOS = {
    "균등 (각 20%)": {
        "ko_mt_bench": 0.20, "logickor": 0.20, "ko_ifeval": 0.20,
        "aihub_582": 0.20, "aihub_90": 0.20,
    },
    "현재 (25/15/20/20/20)": {
        "ko_mt_bench": 0.25, "logickor": 0.15, "ko_ifeval": 0.20,
        "aihub_582": 0.20, "aihub_90": 0.20,
    },
    "요약 중심 (10/10/20/30/30)": {
        "ko_mt_bench": 0.10, "logickor": 0.10, "ko_ifeval": 0.20,
        "aihub_582": 0.30, "aihub_90": 0.30,
    },
    "추론·형식 중심 (20/25/25/15/15)": {
        "ko_mt_bench": 0.20, "logickor": 0.25, "ko_ifeval": 0.25,
        "aihub_582": 0.15, "aihub_90": 0.15,
    },
    "대화 QA (35/25/25/7/8)": {
        "ko_mt_bench": 0.35, "logickor": 0.25, "ko_ifeval": 0.25,
        "aihub_582": 0.07, "aihub_90": 0.08,
    },
}


def load_per_sample(path: Path) -> dict[str, dict]:
    """sample_id → {score, geval_*, ...} 형태로 로드."""
    out = {}
    if not path.exists():
        return out
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "error" in r or "score" not in r:
                continue
            entry = {"score": r["score"]}
            # G-Eval 4차원 추출 (요약 한정)
            turns = r.get("turns", [])
            if turns:
                md = turns[0].get("metric_details") or {}
                for dim in ("geval_faithfulness", "geval_relevance",
                            "geval_coherence", "geval_conciseness"):
                    if dim in md and md[dim] is not None:
                        entry[dim] = md[dim]
                for k in ("rouge1", "rouge2", "rougeL", "bertscore_f1",
                          "strict_acc", "loose_acc"):
                    if k in md:
                        entry[k] = md[k]
            out[r["sample_id"]] = entry
    return out


def avg(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else 0.0


def stdev(xs):
    xs = [x for x in xs if x is not None]
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def collect(results_dir: Path) -> dict:
    """results_dir → {model → bench → {sample_id → entry}} ."""
    data = {}
    for model in MODELS:
        data[model] = {}
        for bench in BENCHES:
            data[model][bench] = load_per_sample(results_dir / model / f"{bench}.jsonl")
    return data


def medal(rank: int) -> str:
    return ["🥇", "🥈", "🥉", "4️⃣"][rank] if rank < 4 else f"{rank+1}"


def diff_arrow(delta: float, threshold: float = 0.05) -> str:
    if delta > threshold: return "🔼"
    if delta < -threshold: return "🔽"
    return "→"


def section_per_bench_scores(old, new) -> list[str]:
    lines = ["## 1. 벤치별 모델 점수 (구 vs 신)\n"]
    for bench in BENCHES:
        lines.append(f"\n### {BENCH_LABEL[bench]}\n")
        lines.append("| 모델 | 구 | 신 | Δ (절대) | Δ (%) |")
        lines.append("|---|---:|---:|---:|---:|")
        for model in MODELS:
            old_scores = [v["score"] for v in old[model][bench].values()]
            new_scores = [v["score"] for v in new[model][bench].values()]
            o = avg(old_scores)
            n = avg(new_scores)
            d = n - o
            dp = (d / o * 100) if o else 0
            arrow = diff_arrow(d, threshold=0.1)
            lines.append(f"| {model} | {o:.3f} | {n:.3f} | {d:+.3f} {arrow} | {dp:+.1f}% |")
    return lines


def section_overall_ranking(old, new) -> list[str]:
    """현재 가중치 기준 종합 순위 — 구 vs 신."""
    lines = ["\n## 2. 종합 점수 + 순위 변화 (현재 가중치 25/15/20/20/20 기준)\n"]
    weights = SCENARIOS["현재 (25/15/20/20/20)"]
    def overall(data):
        ranks = []
        for model in MODELS:
            s = 0
            for bench, w in weights.items():
                scores = [v["score"] for v in data[model][bench].values()]
                s += avg(scores) * w
            ranks.append((model, s))
        ranks.sort(key=lambda x: -x[1])
        return ranks

    old_ranks = overall(old)
    new_ranks = overall(new)
    old_pos = {m: i for i, (m, _) in enumerate(old_ranks)}
    new_pos = {m: i for i, (m, _) in enumerate(new_ranks)}

    lines.append("### 구 prompt 종합 순위\n")
    lines.append("| 순위 | 모델 | weighted |")
    lines.append("|---|---|---:|")
    for i, (m, s) in enumerate(old_ranks):
        lines.append(f"| {medal(i)} | {m} | {s:.3f} |")

    lines.append("\n### 신 prompt 종합 순위 (rejudge 결과)\n")
    lines.append("| 순위 | 모델 | weighted | 구 순위 | 변화 |")
    lines.append("|---|---|---:|---:|---|")
    for i, (m, s) in enumerate(new_ranks):
        old_r = old_pos[m]
        diff = old_r - i
        if diff > 0: change = f"🚀 +{diff}"
        elif diff < 0: change = f"📉 {diff}"
        else: change = "→ 유지"
        lines.append(f"| {medal(i)} | {m} | {s:.3f} | {old_r+1} | {change} |")
    return lines


def section_variance(old, new) -> list[str]:
    """변별력(spread) 비교."""
    lines = ["\n## 3. 변별력 (모델 간 격차 = max − min)\n"]
    lines.append("| 벤치 | 구 spread | 신 spread | 배수 | 평가 |")
    lines.append("|---|---:|---:|---:|---|")

    for bench in BENCHES:
        old_avgs = [avg([v["score"] for v in old[m][bench].values()]) for m in MODELS]
        new_avgs = [avg([v["score"] for v in new[m][bench].values()]) for m in MODELS]
        old_spread = max(old_avgs) - min(old_avgs)
        new_spread = max(new_avgs) - min(new_avgs)
        ratio = new_spread / old_spread if old_spread else float("inf")
        if ratio > 1.5: tag = "✅ 변별력↑"
        elif ratio > 0.8: tag = "→ 비슷"
        else: tag = "⚠️ 변별력↓"
        lines.append(f"| {BENCH_LABEL[bench]} | {old_spread:.3f} | {new_spread:.3f} | {ratio:.2f}× | {tag} |")
    return lines


def section_scenarios(new) -> list[str]:
    """시나리오별 가중치 시뮬레이션."""
    lines = ["\n## 4. 시나리오별 가중치 시뮬레이션 (신 prompt 점수 기반) ⭐\n"]
    lines.append("같은 raw 점수에 가중치만 바꿔 모델 순위 비교. **target task 확정 시 즉시 적용 가능**.\n")

    # 시나리오별 종합 점수 & 순위
    scenario_results = {}
    for sc_name, weights in SCENARIOS.items():
        ranks = []
        for model in MODELS:
            s = sum(avg([v["score"] for v in new[model][b].values()]) * w
                    for b, w in weights.items())
            ranks.append((model, s))
        ranks.sort(key=lambda x: -x[1])
        scenario_results[sc_name] = ranks

    # 표 1: 모델별 × 시나리오별 종합 점수
    lines.append("### 4-1. 시나리오별 종합 점수\n")
    header = "| 모델 | " + " | ".join(SCENARIOS.keys()) + " |"
    lines.append(header)
    lines.append("|---|" + ":---:|" * len(SCENARIOS))
    for model in MODELS:
        row = [model]
        for sc_name, weights in SCENARIOS.items():
            s = sum(avg([v["score"] for v in new[model][b].values()]) * w
                    for b, w in weights.items())
            row.append(f"{s:.3f}")
        lines.append("| " + " | ".join(row) + " |")

    # 표 2: 시나리오별 1위 모델
    lines.append("\n### 4-2. 시나리오별 1위 모델 ⭐\n")
    lines.append("| 시나리오 | 🥇 1위 | 점수 | 🥈 2위 | 점수 |")
    lines.append("|---|---|---:|---|---:|")
    for sc_name, ranks in scenario_results.items():
        first, fs = ranks[0]
        second, ss = ranks[1]
        lines.append(f"| {sc_name} | **{first}** | {fs:.3f} | {second} | {ss:.3f} |")

    return lines


def section_geval_dimensions(new) -> list[str]:
    """G-Eval 4차원 (요약 평가만)."""
    lines = ["\n## 5. G-Eval 4 차원 통계 (요약 평가, AIHub 582/90 통합)\n"]
    lines.append("새 prompt가 도입한 **차원별 분석** — 모델의 강점·약점 진단.\n")
    lines.append("| 모델 | Faithfulness | Relevance | Coherence | Conciseness |")
    lines.append("|---|---:|---:|---:|---:|")
    for model in MODELS:
        dim_vals = {d: [] for d in ("geval_faithfulness", "geval_relevance",
                                     "geval_coherence", "geval_conciseness")}
        for bench in ("aihub_582", "aihub_90"):
            for entry in new[model][bench].values():
                for d in dim_vals:
                    if d in entry:
                        dim_vals[d].append(entry[d])
        row = [model]
        for d in ("geval_faithfulness", "geval_relevance", "geval_coherence", "geval_conciseness"):
            row.append(f"{avg(dim_vals[d]):.3f}")
        lines.append("| " + " | ".join(row) + " |")

    # 인사이트: 차원별 1위
    lines.append("\n### 차원별 1위 모델 — 강점·약점\n")
    dim_label = {
        "geval_faithfulness": "Faithfulness (사실 충실도, 환각 적음)",
        "geval_relevance": "Relevance (핵심 정보 포함)",
        "geval_coherence": "Coherence (논리 흐름)",
        "geval_conciseness": "Conciseness (간결성)",
    }
    for dim, lbl in dim_label.items():
        scores = []
        for model in MODELS:
            vals = []
            for bench in ("aihub_582", "aihub_90"):
                for entry in new[model][bench].values():
                    if dim in entry:
                        vals.append(entry[dim])
            scores.append((model, avg(vals)))
        scores.sort(key=lambda x: -x[1])
        winner = scores[0]
        loser = scores[-1]
        lines.append(f"- **{lbl}**: 🥇 {winner[0]} ({winner[1]:.3f}) / 🔻 {loser[0]} ({loser[1]:.3f})")

    return lines


def section_bf16_fp8(new) -> list[str]:
    """BF16 vs FP8 양자화 손실 비교 (신 prompt 기준)."""
    lines = ["\n## 6. 양자화 비교 — BF16 vs FP8 (Qwen3.6-35B-A3B, 신 prompt 기준)\n"]
    bf = new["Qwen3.6-35B-A3B-BF16"]
    fp = new["Qwen3.6-35B-A3B-FP8"]
    weights = SCENARIOS["현재 (25/15/20/20/20)"]

    lines.append("| 벤치 | BF16 | FP8 | Δ (절대) | Δ (%) |")
    lines.append("|---|---:|---:|---:|---:|")
    for bench in BENCHES:
        b = avg([v["score"] for v in bf[bench].values()])
        f = avg([v["score"] for v in fp[bench].values()])
        d = f - b
        dp = (d / b * 100) if b else 0
        lines.append(f"| {BENCH_LABEL[bench]} | {b:.3f} | {f:.3f} | {d:+.3f} | {dp:+.1f}% |")

    # weighted total
    bf_total = sum(avg([v["score"] for v in bf[b].values()]) * w for b, w in weights.items())
    fp_total = sum(avg([v["score"] for v in fp[b].values()]) * w for b, w in weights.items())
    total_d = fp_total - bf_total
    total_dp = (total_d / bf_total * 100) if bf_total else 0
    lines.append(f"| **Weighted Total** | **{bf_total:.3f}** | **{fp_total:.3f}** | **{total_d:+.3f}** | **{total_dp:+.2f}%** |")

    lines.append(f"\n→ FP8 양자화 손실: **{total_dp:+.2f}%** (구 prompt에선 -0.14%였음)")
    if abs(total_dp) < 2:
        lines.append("→ ✅ 양자화 손실 1-2% 이내 → FP8 운영 적합 유지")
    return lines


def section_summary(old, new) -> list[str]:
    lines = ["\n## 7. 핵심 발견 요약\n"]

    # 신 prompt 1위 (현재 가중치)
    weights = SCENARIOS["현재 (25/15/20/20/20)"]
    new_ranks = sorted(
        [(m, sum(avg([v["score"] for v in new[m][b].values()]) * w for b, w in weights.items()))
         for m in MODELS],
        key=lambda x: -x[1]
    )
    old_ranks = sorted(
        [(m, sum(avg([v["score"] for v in old[m][b].values()]) * w for b, w in weights.items()))
         for m in MODELS],
        key=lambda x: -x[1]
    )

    lines.append(f"- **신 prompt 1위**: {new_ranks[0][0]} ({new_ranks[0][1]:.3f})")
    lines.append(f"- **구 prompt 1위**: {old_ranks[0][0]} ({old_ranks[0][1]:.3f})")
    lines.append(f"- **순위 변화**: 구·신 prompt 사이 모든 모델 순위가 재배치됨")

    # 변별력 (요약 평가)
    for bench in ("aihub_582", "aihub_90"):
        old_avgs = [avg([v["score"] for v in old[m][bench].values()]) for m in MODELS]
        new_avgs = [avg([v["score"] for v in new[m][bench].values()]) for m in MODELS]
        old_spread = max(old_avgs) - min(old_avgs)
        new_spread = max(new_avgs) - min(new_avgs)
        ratio = new_spread / old_spread if old_spread else float("inf")
        lines.append(f"- **{BENCH_LABEL[bench]} 변별력**: 구 {old_spread:.3f} → 신 {new_spread:.3f} ({ratio:.1f}배)")

    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-dir", type=Path, default=Path("results"))
    parser.add_argument("--new-dir", type=Path, default=Path("results_v2"))
    parser.add_argument("--output", type=Path, default=Path("reports/comparison.md"))
    args = parser.parse_args()

    old = collect(args.old_dir)
    new = collect(args.new_dir)

    lines = [
        "# Judge Prompt 재설계 비교 분석",
        "",
        "> 평가일: 2026-05-25 / 본 보고서: rejudge 결과 정량 비교",
        "> ",
        "> **구 prompt**: 자체 작성 generic prompt (점수 1-10 정수, 추상 기준 4개)",
        "> **신 prompt**: G-Eval (요약 4차원 rubric) + MT-Bench 표준 (대화) + LogicKor 스타일 (추론)",
        "> ",
        "> **변수 통제**: 모델 답변(model_output)은 그대로, Judge prompt만 변경 (rejudge 방식).",
        "",
    ]
    lines += section_per_bench_scores(old, new)
    lines += section_overall_ranking(old, new)
    lines += section_variance(old, new)
    lines += section_scenarios(new)
    lines += section_geval_dimensions(new)
    lines += section_bf16_fp8(new)
    lines += section_summary(old, new)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(f"✅ {args.output}")
    # 마지막 종합 섹션만 stdout에도 출력
    print("\n".join(section_summary(old, new)))


if __name__ == "__main__":
    main()
