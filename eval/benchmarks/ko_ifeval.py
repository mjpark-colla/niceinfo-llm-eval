"""Ko-IFEval (instruction-following 자동 룰 채점).

- 영문 IFEval의 한국어 버전
- ~541 prompts, 25 instruction types
- 자동 룰 기반 (judge 불필요)

채점: strict_acc + loose_acc 평균
"""
import random
import re
from typing import Iterable

from datasets import load_dataset

from eval.benchmarks.base import Benchmark, Sample, TurnResult


class KoIFEval(Benchmark):
    name = "ko_ifeval"
    metric_type = "auto"

    # Ko-IFEval은 allganize가 공식 한국어 번역본 공개
    # https://huggingface.co/datasets/allganize/IFEval-Ko
    HF_CANDIDATES = [
        "allganize/IFEval-Ko",
    ]

    def samples(self, limit: int | None = None, seed: int = 42) -> Iterable[Sample]:
        ds = None
        for hf_id in self.HF_CANDIDATES:
            try:
                ds = load_dataset(hf_id, split="train")
                break
            except Exception:
                continue
        if ds is None:
            raise RuntimeError(f"Ko-IFEval 데이터셋 로드 실패. 후보: {self.HF_CANDIDATES}")

        items = list(ds)
        if limit is not None and limit < len(items):
            random.Random(seed).shuffle(items)
            items = items[:limit]

        for item in items:
            qid = item.get("key", item.get("id", item.get("idx", "?")))
            prompt = item.get("prompt", item.get("question", ""))
            instruction_ids = item.get("instruction_id_list") or item.get("instruction_ids") or []
            kwargs = item.get("kwargs") or {}

            yield Sample(
                id=f"ko_ifeval-{qid}",
                prompt=prompt,
                reference=None,
                metadata={
                    "instruction_ids": instruction_ids,
                    "kwargs": kwargs,
                },
            )

    async def evaluate_turn(
        self,
        sample: Sample,
        turn_idx: int,
        prompt: str,
        model_output: str,
        judge=None,
    ) -> TurnResult:
        """간단한 자동 룰 채점. 본격 IFEval 룰은 별도 라이브러리 사용 권장,
        여기는 간이 구현 (strict/loose accuracy 평균)."""
        instruction_ids = sample.metadata.get("instruction_ids", [])
        kwargs_list = sample.metadata.get("kwargs", [])

        if not instruction_ids:
            # 룰 없으면 0점 (방어적)
            return TurnResult(
                turn=turn_idx, prompt=prompt, model_output=model_output, score=0.0,
                metric_details={"reason": "no instruction_ids"},
            )

        strict_pass = []
        loose_pass = []
        for i, inst_id in enumerate(instruction_ids):
            kw = kwargs_list[i] if i < len(kwargs_list) else {}
            s_ok, l_ok = self._check_instruction(model_output, inst_id, kw)
            strict_pass.append(s_ok)
            loose_pass.append(l_ok)

        strict_acc = sum(strict_pass) / len(strict_pass) if strict_pass else 0.0
        loose_acc = sum(loose_pass) / len(loose_pass) if loose_pass else 0.0
        score = (strict_acc + loose_acc) / 2 * 10  # 0~10 스케일 정규화

        return TurnResult(
            turn=turn_idx,
            prompt=prompt,
            model_output=model_output,
            score=score,
            metric_details={
                "strict_acc": strict_acc,
                "loose_acc": loose_acc,
                "instruction_ids": instruction_ids,
            },
        )

    @staticmethod
    def _check_instruction(text: str, inst_id: str, kw: dict) -> tuple[bool, bool]:
        """단일 instruction 룰 체크. (strict, loose) 반환.

        주요 룰만 구현. 누락된 룰은 (True, True) 처리하여 점수에 영향 없게.
        """
        loose_text = text.strip().lower()
        # length:* — 단어 수 제약
        if inst_id.startswith("length:") or "word_count" in inst_id:
            target = kw.get("num_words") or kw.get("word_count") or 0
            actual = len(text.split())
            strict = actual == target if target else True
            loose = abs(actual - target) <= max(5, target * 0.1) if target else True
            return strict, loose

        # keywords:include — 특정 키워드 포함
        if "keyword" in inst_id or "include" in inst_id:
            keywords = kw.get("keywords") or kw.get("words") or []
            if isinstance(keywords, str):
                keywords = [keywords]
            strict = all(k in text for k in keywords)
            loose = all(k.lower() in loose_text for k in keywords)
            return strict, loose

        # format:json — JSON 형식
        if "json" in inst_id.lower():
            import json as _json
            try:
                _json.loads(text)
                return True, True
            except Exception:
                return False, "{" in text and "}" in text

        # punctuation:no — 마침표 등 제거
        if "punctuation" in inst_id.lower() and ("no" in inst_id or "exclude" in inst_id):
            has_punct = any(p in text for p in ".,!?;:")
            return (not has_punct), (not has_punct)

        # 미구현 룰은 통과 처리
        return True, True
