"""Ko-MT-Bench (davidkim205/ko-bench).

- 80 questions, 8 categories
- multi-turn (2 turns)
- LLM-as-Judge (1~10 score)
"""
import random
from typing import Iterable

from datasets import load_dataset

from eval.benchmarks.base import Benchmark, Sample, TurnResult


class KoMTBench(Benchmark):
    name = "ko_mt_bench"
    metric_type = "judge"

    HF_ID = "davidkim205/ko-bench"

    def samples(self, limit: int | None = None, seed: int = 42) -> Iterable[Sample]:
        ds = load_dataset(self.HF_ID, split="train")
        # 80 questions 표준. limit 적용
        items = list(ds)
        if limit is not None and limit < len(items):
            random.Random(seed).shuffle(items)
            items = items[:limit]

        for item in items:
            qid = item.get("question_id") or item.get("id") or item.get("idx")
            category = item.get("category", "unknown")
            # ko-bench actual schema: pairs = [{"prompt": ..., "refer": ...}, ...]
            pairs = item.get("pairs") or []

            if not pairs:
                continue

            prompts = [p.get("prompt", "") for p in pairs if p.get("prompt")]
            refers = [p.get("refer", "") for p in pairs]
            has_ref = any(r.strip() for r in refers if r)
            reference = refers if has_ref else None

            if not prompts:
                continue

            yield Sample(
                id=f"ko_mt-{qid}",
                prompt=prompts[0],
                reference=reference,
                follow_up_prompts=prompts[1:],
                metadata={"category": category, "question_id": qid},
            )

    def is_multi_turn(self) -> bool:
        return True

    async def evaluate_turn(
        self,
        sample: Sample,
        turn_idx: int,
        prompt: str,
        model_output: str,
        judge=None,
    ) -> TurnResult:
        if judge is None:
            raise ValueError("Ko-MT-Bench는 judge가 필요합니다.")

        # 1턴은 context 없음, 2턴 이상은 이전 turn 누적
        context = None
        if turn_idx > 1:
            context = f"턴 1 질문: {sample.prompt}\n(턴 1 답변 생략)"

        ref = None
        if sample.reference:
            ref = sample.reference if isinstance(sample.reference, list) else \
                  ([sample.reference[turn_idx - 1]] if isinstance(sample.reference, list) and len(sample.reference) >= turn_idx else None)

        result = await judge.score(
            question=prompt,
            answer=model_output,
            reference=ref,
            context=context,
        )

        return TurnResult(
            turn=turn_idx,
            prompt=prompt,
            model_output=model_output,
            score=result["score"],
            metric_details={"judge_model": result["judge_model"]},
            judge_raw=result["raw"],
            judge_tokens_in=result["tokens_in"],
            judge_tokens_out=result["tokens_out"],
        )
