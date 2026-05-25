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
        prev_turn: TurnResult | None = None,
    ) -> TurnResult:
        if judge is None:
            raise ValueError("Ko-MT-Bench는 judge가 필요합니다.")

        # turn별 reference 추출
        ref = None
        if isinstance(sample.reference, list) and len(sample.reference) >= turn_idx:
            ref_for_turn = sample.reference[turn_idx - 1]
            ref = ref_for_turn if ref_for_turn else None

        # 멀티턴: 이전 turn 정보 전달 (실제 이전 답변 포함)
        prev_question = None
        prev_answer = None
        if turn_idx > 1 and prev_turn is not None:
            prev_question = prev_turn.prompt
            prev_answer = prev_turn.model_output

        # 멀티턴이지만 prev_turn 없으면 (legacy 호출) — sample.prompt가 1턴 질문
        elif turn_idx > 1 and prev_turn is None:
            prev_question = sample.prompt
            prev_answer = "(이전 답변 생략)"

        result = await judge.score_dialogue(
            question=prompt,
            answer=model_output,
            reference=ref,
            prev_question=prev_question,
            prev_answer=prev_answer,
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
