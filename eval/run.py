"""평가 실행 entrypoint (비동기·동시 호출 지원).

사용 예:
  # 단일 모델, 단일 벤치, sanity
  python -m eval.run --model Qwen3-32B-AWQ --benchmark ko_mt_bench --limit 5

  # 전체 모델·벤치, 동시 호출
  python -m eval.run --concurrency 8

  # 집계만
  python -m eval.run --aggregate-only
"""
import argparse
import asyncio
import logging
import time
from pathlib import Path

from eval.config import TARGET_MODELS, JUDGE, BENCHMARKS, ModelConfig
from eval.models import TargetClient
from eval.metrics.llm_judge import LLMJudge
from eval.benchmarks.base import Benchmark, Sample, EvalResult, TurnResult
from eval.benchmarks.ko_mt_bench import KoMTBench
from eval.benchmarks.logickor import LogicKor
from eval.benchmarks.ko_ifeval import KoIFEval
from eval.benchmarks.aihub import AIHub582, AIHub90
from eval.utils.io import append_jsonl, load_completed_ids, write_json
from eval.utils.cost import calc_openai_cost


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("eval.run")


BENCHMARK_CLASSES: dict[str, type[Benchmark]] = {
    "ko_mt_bench": KoMTBench,
    "logickor": LogicKor,
    "ko_ifeval": KoIFEval,
    "aihub_582": AIHub582,
    "aihub_90": AIHub90,
}


def get_target_model(name: str) -> ModelConfig:
    for m in TARGET_MODELS:
        if m.display_name == name or m.name == name:
            return m
    raise ValueError(f"Unknown model: {name}. Available: {[m.display_name for m in TARGET_MODELS]}")


async def evaluate_sample(
    benchmark: Benchmark,
    sample: Sample,
    client: TargetClient,
    judge: LLMJudge | None,
    model_display: str,
) -> EvalResult:
    """단일 sample 평가 (비동기). 멀티턴 포함."""
    t0 = time.perf_counter()
    result = EvalResult(
        sample_id=sample.id,
        model=model_display,
        benchmark=benchmark.name,
        metadata=sample.metadata,
    )

    all_prompts = [sample.prompt] + sample.follow_up_prompts

    if benchmark.is_multi_turn() and len(all_prompts) > 1:
        outputs = await client.chat_multi_turn(all_prompts)
    else:
        outputs = [await client.chat_single(all_prompts[0])]

    for i, (prompt, out) in enumerate(zip(all_prompts, outputs), start=1):
        prev_turn = result.turns[-1] if result.turns else None
        turn_result = await benchmark.evaluate_turn(
            sample, turn_idx=i, prompt=prompt, model_output=out["text"], judge=judge,
            prev_turn=prev_turn,
        )
        turn_result.tokens_in = out["tokens_in"]
        turn_result.tokens_out = out["tokens_out"]
        result.turns.append(turn_result)
        if turn_result.judge_tokens_in or turn_result.judge_tokens_out:
            judge_model = turn_result.metric_details.get("judge_model") if turn_result.metric_details else None
            if judge_model:
                result.cost_usd += calc_openai_cost(
                    judge_model, turn_result.judge_tokens_in, turn_result.judge_tokens_out
                )

    if result.turns:
        result.score = sum(t.score for t in result.turns) / len(result.turns)

    result.elapsed_sec = time.perf_counter() - t0
    return result


async def run_one(
    model_cfg: ModelConfig,
    benchmark: Benchmark,
    limit: int | None,
    seed: int,
    results_dir: Path,
    judge: LLMJudge | None,
    concurrency: int,
):
    """단일 (모델, 벤치) 조합 평가, 동시 호출."""
    out_path = results_dir / model_cfg.display_name / f"{benchmark.name}.jsonl"
    done_ids = load_completed_ids(out_path)
    log.info(f"[{model_cfg.display_name} × {benchmark.name}] 이미 평가된 {len(done_ids)}건 skip, concurrency={concurrency}")

    client = TargetClient(model_cfg)
    samples_to_eval = [
        s for s in benchmark.samples(limit=limit, seed=seed)
        if s.id not in done_ids
    ]
    total = len(samples_to_eval)
    if total == 0:
        log.info(f"  → 평가할 새 sample 없음")
        return

    sem = asyncio.Semaphore(concurrency)
    completed = {"n": 0, "errors": 0}
    lock = asyncio.Lock()  # file write 직렬화 (atomic 보장)

    async def process(sample: Sample):
        async with sem:
            try:
                res = await evaluate_sample(benchmark, sample, client, judge, model_cfg.display_name)
                async with lock:
                    append_jsonl(res.to_dict(), out_path)
                completed["n"] += 1
            except Exception as e:
                completed["errors"] += 1
                log.warning(f"sample {sample.id} 실패: {type(e).__name__}: {e}")
                async with lock:
                    append_jsonl({
                        "sample_id": sample.id,
                        "model": model_cfg.display_name,
                        "benchmark": benchmark.name,
                        "error": f"{type(e).__name__}: {e}",
                    }, out_path)
            # 진행률 로그
            n = completed["n"] + completed["errors"]
            if n % 10 == 0 or n == total:
                log.info(f"  ... {n}/{total} ({100*n/total:.0f}%), errors={completed['errors']}")

    await asyncio.gather(*(process(s) for s in samples_to_eval))

    log.info(f"[{model_cfg.display_name} × {benchmark.name}] 완료. 성공 {completed['n']}건, 실패 {completed['errors']}건")


def aggregate(results_dir: Path) -> dict:
    """모델별 종합 점수 계산 (벤치별 평균 × 가중치)."""
    import json
    summary: dict = {}
    for model_dir in results_dir.iterdir():
        if not model_dir.is_dir() or model_dir.name == "sanity":
            continue
        model_name = model_dir.name
        per_bench: dict[str, float] = {}
        for bench_file in model_dir.glob("*.jsonl"):
            bench_name = bench_file.stem
            scores = []
            with open(bench_file) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if "score" in r and "error" not in r:
                        scores.append(r["score"])
            if scores:
                per_bench[bench_name] = sum(scores) / len(scores)

        weighted = sum(
            per_bench.get(b, 0.0) * BENCHMARKS[b].weight
            for b in BENCHMARKS
            if b in per_bench
        )
        summary[model_name] = {
            "per_benchmark": per_bench,
            "weighted_total": weighted,
            "n_benchmarks": len(per_bench),
        }
    return summary


async def main_async(args):
    if args.model:
        models = [get_target_model(m) for m in args.model]
    else:
        models = TARGET_MODELS

    bench_names = args.benchmark or list(BENCHMARKS.keys())

    judge = None
    needs_judge = any(BENCHMARK_CLASSES[n]().metric_type == "judge" for n in bench_names)
    if needs_judge:
        judge = LLMJudge(JUDGE)
        log.info(f"Judge LLM 준비됨: {judge.model}")

    log.info(f"평가 시작: {len(models)} 모델 × {len(bench_names)} 벤치, concurrency={args.concurrency}")

    for model_cfg in models:
        log.info(f"\n>>> 모델: {model_cfg.display_name}")
        for bench_name in bench_names:
            benchmark = BENCHMARK_CLASSES[bench_name]()
            spec = BENCHMARKS[bench_name]
            limit = args.limit or spec.sample_limit
            log.info(f"  --- 벤치: {bench_name} (limit={limit})")
            await run_one(
                model_cfg=model_cfg,
                benchmark=benchmark,
                limit=limit,
                seed=spec.seed,
                results_dir=args.results_dir,
                judge=judge if benchmark.metric_type == "judge" else None,
                concurrency=args.concurrency,
            )

    log.info("\n=== 자동 집계 ===")
    summary = aggregate(args.results_dir)
    for model, info in summary.items():
        log.info(f"\n[{model}]")
        for b, s in info["per_benchmark"].items():
            log.info(f"  {b}: {s:.2f}")
        log.info(f"  → weighted total: {info['weighted_total']:.2f}")
    write_json(summary, args.results_dir / "summary.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", nargs="+", default=None,
                        help="모델 display_name (생략 시 전체)")
    parser.add_argument("--benchmark", nargs="+", default=None,
                        help="벤치 이름 (생략 시 전체)")
    parser.add_argument("--limit", type=int, default=None,
                        help="벤치별 샘플 수 override")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-dir", type=Path, default=Path("/app/results"))
    parser.add_argument("--concurrency", type=int, default=8,
                        help="동시 평가 수 (vLLM + judge). 기본 8")
    parser.add_argument("--aggregate-only", action="store_true",
                        help="기존 결과로 집계만 수행")
    args = parser.parse_args()

    if args.aggregate_only:
        summary = aggregate(args.results_dir)
        log.info("=== 종합 점수 ===")
        for model, info in summary.items():
            log.info(f"\n[{model}]")
            for b, s in info["per_benchmark"].items():
                log.info(f"  {b}: {s:.2f}")
            log.info(f"  → weighted total: {info['weighted_total']:.2f}")
        write_json(summary, args.results_dir / "summary.json")
        log.info(f"\nsummary.json 저장됨")
        return

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
