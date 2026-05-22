"""LogicKor (maywell/LogicKor).

- 42 questions, 6 categories (추론·수학·글쓰기·코딩·이해·문법)
- multi-turn (2 turns)
- LLM-as-Judge with reference
"""
import random
from typing import Iterable

from datasets import load_dataset

from eval.benchmarks.base import Benchmark, Sample, TurnResult


class LogicKor(Benchmark):
    name = "logickor"
    metric_type = "judge"

    HF_ID = "maywell/LogicKor"

    def samples(self, limit: int | None = None, seed: int = 42) -> Iterable[Sample]:
        ds = load_dataset(self.HF_ID, split="train")
        items = list(ds)
        if limit is not None and limit < len(items):
            random.Random(seed).shuffle(items)
            items = items[:limit]

        for item in items:
            qid = item.get("id", item.get("idx", "unknown"))
            category = item.get("category", "unknown")
            questions = item.get("questions") or []
            references = item.get("references") or []

            if not questions:
                continue

            yield Sample(
                id=f"logickor-{qid}",
                prompt=questions[0],
                reference=references if references else None,
                follow_up_prompts=questions[1:],
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
            raise ValueError("LogicKor는 judge가 필요합니다.")

        ref = None
        if isinstance(sample.reference, list) and len(sample.reference) >= turn_idx:
            ref_for_turn = sample.reference[turn_idx - 1]
            ref = ref_for_turn if ref_for_turn else None

        context = None
        if turn_idx > 1:
            context = f"턴 1 질문: {sample.prompt}"

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
