"""기존 평가 jsonl 에서 속도 지표 (대략적) 추출.

⚠️ 한계 (사용 전 필독):
- elapsed_sec는 sample 전체 시간 = model inference + judge 호출 + 후처리
- concurrency 8 환경에서 측정 → 단독 latency 아님 (GPU 경합 포함)
- A100 80GB 측정값 ≠ 운영 H100 NVL 94GB (절대값 운영 재현 X)
- TTFT (Time To First Token) 측정 불가 — streaming 아니라서
- 정확한 운영 지표는 Phase 3 폐쇄망 측정 필수

용도:
- 4 모델 동등 조건 **상대 비교** ("32B-AWQ가 35B-BF16보다 빠른가?")
- 답변 길이 / 처리량 대략적 추정

사용:
  python -m eval.speed [--results-dir results] [--output reports/speed.md]
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path


MODELS_ORDER = [
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
# judge 호출 유무 — 시간 해석에 중요
BENCH_HAS_JUDGE = {
    "ko_mt_bench": True,    # 매 turn judge 호출 (가장 노이즈 큼)
    "logickor": True,
    "ko_ifeval": False,     # 룰 기반, judge 없음 → 가장 깨끗
    "aihub_582": False,     # ROUGE/BERTScore만 (BERTScore가 약간 무거움)
    "aihub_90": False,
}


def load_samples(path: Path) -> list[dict]:
    items = []
    if not path.exists():
        return items
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "error" in r:
                continue
            items.append(r)
    return items


def collect_stats(results_dir: Path) -> dict:
    """{model: {bench: {stats}}}"""
    out = {}
    for model in MODELS_ORDER:
        out[model] = {}
        for bench in BENCHES:
            samples = load_samples(results_dir / model / f"{bench}.jsonl")
            if not samples:
                continue
            elapsed = []
            tokens_in_all = []
            tokens_out_all = []
            for s in samples:
                if "elapsed_sec" in s:
                    elapsed.append(s["elapsed_sec"])
                turns = s.get("turns") or []
                tin = sum(t.get("tokens_in", 0) for t in turns)
                tout = sum(t.get("tokens_out", 0) for t in turns)
                tokens_in_all.append(tin)
                tokens_out_all.append(tout)

            out[model][bench] = {
                "n": len(samples),
                "elapsed_avg": sum(elapsed) / len(elapsed) if elapsed else 0,
                "elapsed_p50": sorted(elapsed)[len(elapsed)//2] if elapsed else 0,
                "elapsed_p90": sorted(elapsed)[int(len(elapsed)*0.9)] if elapsed else 0,
                "tokens_in_avg": sum(tokens_in_all) / len(tokens_in_all) if tokens_in_all else 0,
                "tokens_out_avg": sum(tokens_out_all) / len(tokens_out_all) if tokens_out_all else 0,
                "approx_tps": (sum(tokens_out_all) / sum(elapsed)) if sum(elapsed) else 0,
            }
    return out


def section_intro() -> list[str]:
    return [
        "# 속도 (운영 지표) 대략적 분석 — v1 평가 jsonl 기반",
        "",
        "> 생성: 2026-05-25 / 출처: `results/<모델>/<벤치>.jsonl` 의 `elapsed_sec`, `turns[].tokens_*`",
        "",
        "## ⚠️ 본 분석의 한계 (반드시 명시)",
        "",
        "- **단독 latency 아님**: concurrency 8 환경에서 측정. GPU 경합 포함된 sample-level wall time.",
        "- **Judge 호출 시간 포함**: Ko-MT-Bench·LogicKor 는 매 turn OpenAI API 호출 시간 포함.",
        "- **TTFT 측정 불가**: streaming 아닌 일괄 응답이라 첫 토큰 시간 모름.",
        "- **A100 ≠ H100 NVL**: 평가 환경(A100 80GB) ≠ 운영 환경(H100 NVL 94GB). 절대값은 운영 재현 X.",
        "- **동등 조건 상대 비교만 유효**: 4 모델 모두 같은 vLLM·concurrency·sampling 설정 → 모델 간 순위는 의미 있음.",
        "",
        "**가장 깨끗한 비교**: Ko-IFEval (룰 기반, judge 호출 없음) + AIHub (BERTScore만, judge 없음).",
        "",
    ]


def section_per_model_summary(stats: dict) -> list[str]:
    """모델별 종합 요약."""
    lines = ["## 1. 모델별 종합 — judge 호출 없는 벤치 기준 (가장 깨끗)\n"]
    lines.append("Ko-IFEval + AIHub 582/90 평균 (judge 노이즈 제외):\n")
    lines.append("| 모델 | 평균 sample 시간(s) | 평균 출력 토큰 | 대략적 throughput (tok/s) | 평균 입력 토큰 |")
    lines.append("|---|---:|---:|---:|---:|")
    rows = []
    for model in MODELS_ORDER:
        e_total, t_total_out, t_total_in, n = 0, 0, 0, 0
        for bench in ("ko_ifeval", "aihub_582", "aihub_90"):
            s = stats.get(model, {}).get(bench)
            if not s:
                continue
            e_total += s["elapsed_avg"] * s["n"]
            t_total_out += s["tokens_out_avg"] * s["n"]
            t_total_in += s["tokens_in_avg"] * s["n"]
            n += s["n"]
        if n == 0:
            continue
        e_avg = e_total / n
        t_out_avg = t_total_out / n
        t_in_avg = t_total_in / n
        tps = t_out_avg / e_avg if e_avg else 0
        rows.append((model, e_avg, t_out_avg, tps, t_in_avg))

    # 가장 빠른 모델 (throughput 기준) 표시
    rows.sort(key=lambda x: -x[3])
    for i, (model, e_avg, t_out_avg, tps, t_in_avg) in enumerate(rows):
        medal = ["🥇", "🥈", "🥉", "4️⃣"][i] if i < 4 else " "
        lines.append(f"| {medal} {model} | {e_avg:.2f} | {t_out_avg:.0f} | **{tps:.1f}** | {t_in_avg:.0f} |")

    return lines


def section_per_bench(stats: dict) -> list[str]:
    """벤치별 모델 비교."""
    lines = ["\n## 2. 벤치별 모델 비교\n"]
    for bench in BENCHES:
        judge_note = " (⚠️ judge 호출 시간 포함)" if BENCH_HAS_JUDGE[bench] else ""
        lines.append(f"\n### {BENCH_LABEL[bench]}{judge_note}\n")
        lines.append("| 모델 | sample 평균(s) | p50 | p90 | 평균 출력 토큰 | 대략적 tok/s |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        rows = []
        for model in MODELS_ORDER:
            s = stats.get(model, {}).get(bench)
            if not s:
                continue
            rows.append((model, s))
        # 가장 빠른 sample 시간으로 정렬
        rows.sort(key=lambda x: x[1]["elapsed_avg"])
        for model, s in rows:
            lines.append(
                f"| {model} | {s['elapsed_avg']:.2f} | {s['elapsed_p50']:.2f} | "
                f"{s['elapsed_p90']:.2f} | {s['tokens_out_avg']:.0f} | {s['approx_tps']:.1f} |"
            )
    return lines


def section_takeaways(stats: dict) -> list[str]:
    """핵심 발견."""
    lines = ["\n## 3. 핵심 발견 (정직한 해석)\n"]

    # 가장 빠른/느린 모델 (judge 없는 벤치 기준)
    speeds = []
    for model in MODELS_ORDER:
        e_total, t_total_out, n = 0, 0, 0
        for bench in ("ko_ifeval", "aihub_582", "aihub_90"):
            s = stats.get(model, {}).get(bench)
            if not s:
                continue
            e_total += s["elapsed_avg"] * s["n"]
            t_total_out += s["tokens_out_avg"] * s["n"]
            n += s["n"]
        if n == 0:
            continue
        tps = (t_total_out / e_total) if e_total else 0
        speeds.append((model, tps))
    speeds.sort(key=lambda x: -x[1])

    if len(speeds) >= 2:
        fastest = speeds[0]
        slowest = speeds[-1]
        ratio = fastest[1] / slowest[1] if slowest[1] else 0
        lines.append(f"- **가장 빠른 모델**: {fastest[0]} ({fastest[1]:.1f} tok/s)")
        lines.append(f"- **가장 느린 모델**: {slowest[0]} ({slowest[1]:.1f} tok/s)")
        lines.append(f"- **속도 격차**: {ratio:.2f}배")

    # 답변 길이 추세
    lines.append("\n### 답변 길이 추세 (모델별 평균 출력 토큰)")
    lines.append("| 모델 | 평균 출력 토큰 (전체 벤치) |")
    lines.append("|---|---:|")
    for model in MODELS_ORDER:
        outs = []
        for bench in BENCHES:
            s = stats.get(model, {}).get(bench)
            if s:
                outs.append(s["tokens_out_avg"])
        if outs:
            lines.append(f"| {model} | {sum(outs)/len(outs):.0f} |")

    lines.append("\n### 운영 결정 시 고려사항")
    lines.append("- **답변 길이가 긴 모델일수록 sample 시간 길어짐** (당연) — 사용자에게 보이는 응답 시간 직접 영향")
    lines.append("- **AWQ 4-bit 모델**(32B-AWQ)이 메모리 적고 빠를 가능성 ↑ — Tensor Core 최적화")
    lines.append("- **FP8(35B-FP8)**: H100 native라 운영(H100 NVL)에선 더 빠를 것 — A100 측정값은 emulation 가능성")
    lines.append("- **BF16(30B-A3B, 35B-BF16)**: 메모리 큰 만큼 inference 부담")
    return lines


def section_next_steps() -> list[str]:
    return [
        "\n## 4. 다음 단계 (정확 측정)",
        "",
        "| 단계 | 환경 | 측정 내용 | 시점 |",
        "|---|---|---|---|",
        "| 클라우드 sanity (선택) | A100, concurrency 1, streaming | TTFT, TPS, VRAM | 필요 시 |",
        "| 운영 환경 측정 (필수) | H100 NVL 94GB, 폐쇄망 | 운영 SLA용 | Phase 3 |",
        "| 동시 부하 측정 | H100 NVL + 임베딩 + 리랭커 | 메모리 경합, latency 영향 | Phase 3 |",
        "",
        "**클라우드 sanity는 보류 권장**: A100 절대값이 운영(H100 NVL)에서 재현되지 않으므로 가치 제한적.",
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, default=Path("reports/speed.md"))
    args = parser.parse_args()

    stats = collect_stats(args.results_dir)

    lines = []
    lines += section_intro()
    lines += section_per_model_summary(stats)
    lines += section_per_bench(stats)
    lines += section_takeaways(stats)
    lines += section_next_steps()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(f"✅ {args.output}")

    # stdout에 핵심 발견만
    print()
    print("\n".join(section_takeaways(stats)[:15]))


if __name__ == "__main__":
    main()
