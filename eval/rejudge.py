"""Rejudge: 기존 평가 결과의 모델 답변을 그대로 두고 Judge prompt만 새로 적용.

목적:
- vLLM/GPU 없이 OpenAI API 호출만으로 새 Judge prompt 평가
- 답변 변수를 통제하여 순수 prompt 효과만 측정
- 비용·시간 절감 (~$39, ~1시간 vs 재추론 ~$56, ~3시간)

사용:
  # 전체 모델 × 전체 벤치 rejudge
  python -m eval.rejudge \
      --input-dir /app/results \
      --output-dir /app/results_v2 \
      --concurrency 8

  # 특정 모델만
  python -m eval.rejudge --input-dir /app/results --output-dir /app/results_v2 \
      --model Qwen3.6-35B-A3B-FP8

  # mini sanity (5 sample만)
  python -m eval.rejudge --input-dir /app/results --output-dir /app/results_v2 \
      --limit 5

설계:
- 입력 jsonl 의 turns[].prompt + turns[].model_output 사용
- 벤치별 task_type 매핑:
    aihub_582 / aihub_90  → summarization (G-Eval)
    ko_mt_bench           → dialogue (MT-Bench)
    logickor              → reasoning (LogicKor 스타일)
    ko_ifeval             → SKIP (룰 기반, judge 무관)
- 결과 jsonl 스키마는 기존과 호환 (HTML 대시보드 그대로 사용 가능)
"""
import argparse
import asyncio
import json
import logging
import re
import time
from pathlib import Path

from eval.config import JUDGE
from eval.metrics.llm_judge import LLMJudge
from eval.utils.io import append_jsonl, load_completed_ids, write_json
from eval.utils.cost import calc_openai_cost


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("eval.rejudge")


# 벤치 → task_type 매핑
BENCH_TASK = {
    "aihub_582": "summarization",
    "aihub_90": "summarization",
    "ko_mt_bench": "dialogue",
    "logickor": "reasoning",
    # ko_ifeval: 룰 기반, judge 호출 안 함 (스크립트가 skip)
}


def extract_passage(prompt: str) -> str:
    """AIHub summarization prompt에서 [원문] 영역 추출."""
    m = re.search(r"\[원문\]\s*\n(.*?)\n\s*\[요약\]", prompt, re.DOTALL)
    if m:
        return m.group(1).strip()
    return prompt


async def rejudge_summary_turn(
    judge: LLMJudge,
    turn: dict,
    sample_metadata: dict,
    sample_reference: list | None,
) -> dict:
    """요약 turn rejudge (G-Eval)."""
    passage = extract_passage(turn["prompt"])
    summary = turn["model_output"]
    result = await judge.score_summarization(
        passage=passage, summary=summary, reference=sample_reference
    )
    return _build_new_turn(turn, result)


async def rejudge_dialogue_turn(
    judge: LLMJudge,
    turn: dict,
    sample_metadata: dict,
    sample_reference: list | None,
    prev_turn: dict | None = None,
) -> dict:
    """대화 turn rejudge (MT-Bench)."""
    ref = None
    if isinstance(sample_reference, list) and len(sample_reference) >= turn["turn"]:
        ref_for_turn = sample_reference[turn["turn"] - 1]
        ref = ref_for_turn if ref_for_turn else None

    prev_q = prev_turn["prompt"] if prev_turn else None
    prev_a = prev_turn["model_output"] if prev_turn else None

    result = await judge.score_dialogue(
        question=turn["prompt"], answer=turn["model_output"],
        reference=ref, prev_question=prev_q, prev_answer=prev_a,
    )
    return _build_new_turn(turn, result)


async def rejudge_reasoning_turn(
    judge: LLMJudge,
    turn: dict,
    sample_metadata: dict,
    sample_reference: list | None,
    prev_turn: dict | None = None,
) -> dict:
    """추론 turn rejudge (LogicKor 스타일)."""
    ref = None
    if isinstance(sample_reference, list) and len(sample_reference) >= turn["turn"]:
        ref_for_turn = sample_reference[turn["turn"] - 1]
        ref = ref_for_turn if ref_for_turn else None

    prev_q = prev_turn["prompt"] if prev_turn else None
    prev_a = prev_turn["model_output"] if prev_turn else None

    result = await judge.score_reasoning(
        question=turn["prompt"], answer=turn["model_output"],
        reference=ref, prev_question=prev_q, prev_answer=prev_a,
    )
    return _build_new_turn(turn, result)


def _build_new_turn(old_turn: dict, judge_result: dict) -> dict:
    """기존 turn 데이터에 새 judge 결과 덮어쓰기. 답변(model_output)은 보존."""
    details = judge_result.get("details", {}) or {}
    metric_details = {
        "judge_model": judge_result["judge_model"],
        "task_type": judge_result.get("task_type"),
    }
    # G-Eval 4차원 점수
    if "faithfulness" in details:
        metric_details["geval_faithfulness"] = details["faithfulness"]
    if "relevance" in details:
        metric_details["geval_relevance"] = details["relevance"]
    if "coherence" in details:
        metric_details["geval_coherence"] = details["coherence"]
    if "conciseness" in details:
        metric_details["geval_conciseness"] = details["conciseness"]

    # 기존 보조 metric 보존 (ROUGE/BERTScore 등)
    old_details = old_turn.get("metric_details") or {}
    for key in ("rouge1", "rouge2", "rougeL", "bertscore_f1",
                "strict_acc", "loose_acc", "instruction_ids"):
        if key in old_details:
            metric_details[key] = old_details[key]

    new_turn = {
        "turn": old_turn["turn"],
        "prompt": old_turn["prompt"],
        "model_output": old_turn["model_output"],
        "score": judge_result["score"],
        "metric_details": metric_details,
        "judge_raw": judge_result["raw"],
        "tokens_in": old_turn.get("tokens_in", 0),
        "tokens_out": old_turn.get("tokens_out", 0),
        "judge_tokens_in": judge_result["tokens_in"],
        "judge_tokens_out": judge_result["tokens_out"],
    }
    return new_turn


async def rejudge_sample(
    judge: LLMJudge,
    record: dict,
    task_type: str,
) -> dict:
    """단일 sample rejudge. 모든 turn 처리, 종합 점수 재계산."""
    t0 = time.perf_counter()

    # error 레코드는 그대로 통과
    if "error" in record or not record.get("turns"):
        return record

    sample_metadata = record.get("metadata", {})
    # reference 는 jsonl 에 없음 — 벤치 데이터에 있음. 일단 None.
    sample_reference = None

    new_turns = []
    cost = 0.0
    for i, turn in enumerate(record["turns"]):
        prev_turn = new_turns[-1] if new_turns else None
        if task_type == "summarization":
            new_turn = await rejudge_summary_turn(
                judge, turn, sample_metadata, sample_reference
            )
        elif task_type == "dialogue":
            new_turn = await rejudge_dialogue_turn(
                judge, turn, sample_metadata, sample_reference, prev_turn=prev_turn
            )
        elif task_type == "reasoning":
            new_turn = await rejudge_reasoning_turn(
                judge, turn, sample_metadata, sample_reference, prev_turn=prev_turn
            )
        else:
            raise ValueError(f"Unknown task_type: {task_type}")
        new_turns.append(new_turn)
        cost += calc_openai_cost(
            new_turn["metric_details"]["judge_model"],
            new_turn["judge_tokens_in"],
            new_turn["judge_tokens_out"],
        )

    # 새 sample 종합 점수 = turn 평균
    sample_score = sum(t["score"] for t in new_turns) / len(new_turns) if new_turns else 0.0

    new_record = {
        "sample_id": record["sample_id"],
        "model": record["model"],
        "benchmark": record["benchmark"],
        "score": sample_score,
        "turns": new_turns,
        "elapsed_sec": time.perf_counter() - t0,
        "cost_usd": cost,
        "metadata": sample_metadata,
    }
    return new_record


def load_jsonl(path: Path) -> list[dict]:
    items = []
    if not path.exists():
        return items
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return items


async def rejudge_one_file(
    judge: LLMJudge,
    input_path: Path,
    output_path: Path,
    task_type: str,
    limit: int | None,
    concurrency: int,
):
    """단일 (모델, 벤치) jsonl rejudge."""
    records = load_jsonl(input_path)
    if not records:
        log.warning(f"  {input_path.name}: 비어있음, skip")
        return

    # resume: 이미 처리된 sample skip
    done_ids = load_completed_ids(output_path)
    to_process = [r for r in records if r.get("sample_id") not in done_ids]
    if limit is not None:
        to_process = to_process[:limit]

    total = len(to_process)
    if total == 0:
        log.info(f"  {input_path.name}: 모두 완료됨")
        return

    log.info(f"  {input_path.name}: {total}건 rejudge 시작 (skip {len(done_ids)})")

    sem = asyncio.Semaphore(concurrency)
    completed = {"n": 0, "errors": 0}
    lock = asyncio.Lock()

    async def process(record):
        async with sem:
            try:
                new_record = await rejudge_sample(judge, record, task_type)
                async with lock:
                    append_jsonl(new_record, output_path)
                completed["n"] += 1
            except Exception as e:
                completed["errors"] += 1
                log.warning(f"    sample {record.get('sample_id')}: {type(e).__name__}: {e}")
                async with lock:
                    append_jsonl({
                        "sample_id": record.get("sample_id"),
                        "model": record.get("model"),
                        "benchmark": record.get("benchmark"),
                        "error": f"{type(e).__name__}: {e}",
                    }, output_path)
            n = completed["n"] + completed["errors"]
            if n % 10 == 0 or n == total:
                log.info(f"    {n}/{total} ({100*n/total:.0f}%) errors={completed['errors']}")

    await asyncio.gather(*(process(r) for r in to_process))
    log.info(f"  {input_path.name}: 완료 {completed['n']}건, 실패 {completed['errors']}건")


async def main_async(args):
    judge = LLMJudge(JUDGE)
    log.info(f"Judge: {judge.model}")

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    # 모델 목록
    models = [args.model] if args.model else [
        d.name for d in input_dir.iterdir()
        if d.is_dir() and d.name not in ("sanity", "concurrency_test")
    ]
    benches = args.benchmark or list(BENCH_TASK.keys())

    log.info(f"Rejudge 시작: {len(models)} 모델 × {len(benches)} 벤치, "
             f"concurrency={args.concurrency}, limit={args.limit}")

    for model in models:
        log.info(f"\n>>> 모델: {model}")
        for bench in benches:
            if bench not in BENCH_TASK:
                log.info(f"  {bench}: task_type 매핑 없음, skip")
                continue
            input_path = input_dir / model / f"{bench}.jsonl"
            output_path = output_dir / model / f"{bench}.jsonl"
            if not input_path.exists():
                log.warning(f"  {input_path} 없음, skip")
                continue
            output_path.parent.mkdir(parents=True, exist_ok=True)
            await rejudge_one_file(
                judge=judge,
                input_path=input_path,
                output_path=output_path,
                task_type=BENCH_TASK[bench],
                limit=args.limit,
                concurrency=args.concurrency,
            )

    # Ko-IFEval은 룰 기반이라 그대로 복사
    if not args.benchmark or "ko_ifeval" in (args.benchmark or []):
        log.info("\n>>> Ko-IFEval: 룰 기반, 기존 결과 복사")
        for model in models:
            src = input_dir / model / "ko_ifeval.jsonl"
            dst = output_dir / model / "ko_ifeval.jsonl"
            if src.exists() and not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(src.read_bytes())
                log.info(f"  {model}: ko_ifeval.jsonl 복사 완료")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("/app/results"),
                        help="기존 평가 결과 디렉터리")
    parser.add_argument("--output-dir", type=Path, default=Path("/app/results_v2"),
                        help="rejudge 결과 출력 디렉터리")
    parser.add_argument("--model", default=None,
                        help="특정 모델만 처리 (display_name)")
    parser.add_argument("--benchmark", nargs="+", default=None,
                        help="특정 벤치만 처리")
    parser.add_argument("--limit", type=int, default=None,
                        help="모델·벤치당 sample 제한 (sanity용)")
    parser.add_argument("--concurrency", type=int, default=8,
                        help="동시 judge 호출 수")
    args = parser.parse_args()

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
