"""AI Hub 요약 데이터셋 (582 + 90).

- 로컬 jsonl 파일 사용
- 자동 메트릭: ROUGE-L + BERTScore F1 평균
- 입력: passage / 정답: reference(s)
"""
import json
import random
from pathlib import Path
from typing import Iterable

from eval.benchmarks.base import Benchmark, Sample, TurnResult
from eval.metrics.rouge_korean import compute_rouge
from eval.metrics.bertscore_korean import compute_bertscore_single


SUMMARIZATION_PROMPT = """다음 글을 한국어로 간결하고 정확하게 요약해주세요. 사실 오류 없이 핵심만 담으세요.

[원문]
{passage}

[요약]"""


class AIHubBase(Benchmark):
    """AI Hub 요약 공통 로직."""
    metric_type = "auto"

    jsonl_path: Path  # subclass에서 설정

    def samples(self, limit: int | None = None, seed: int = 42) -> Iterable[Sample]:
        items = []
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    items.append(json.loads(line))
                except Exception:
                    continue

        if limit is not None and limit < len(items):
            random.Random(seed).shuffle(items)
            items = items[:limit]

        for item in items:
            sid = item.get("id", "?")
            passage = item.get("passage", "")
            # 582: "references" (list), 90: "reference" (str)
            refs = item.get("references") or item.get("reference")
            if isinstance(refs, str):
                refs = [refs]

            yield Sample(
                id=f"{self.name}-{sid}",
                prompt=SUMMARIZATION_PROMPT.format(passage=passage),
                reference=refs,
                metadata={
                    "domain": item.get("domain"),
                    "category": item.get("category"),
                    "summary_length": item.get("summary_length"),
                },
            )

    def evaluate_turn(
        self,
        sample: Sample,
        turn_idx: int,
        prompt: str,
        model_output: str,
        judge=None,
    ) -> TurnResult:
        references = sample.reference or []
        if isinstance(references, str):
            references = [references]

        if not references:
            return TurnResult(
                turn=turn_idx, prompt=prompt, model_output=model_output, score=0.0,
                metric_details={"reason": "no reference"},
            )

        rouge = compute_rouge(model_output, references)
        try:
            bert_f1 = compute_bertscore_single(model_output, references)
        except Exception as e:
            bert_f1 = 0.0

        # 종합 점수: ROUGE-L F1 + BERTScore F1 평균 → 0~10 스케일
        combined = (rouge["rougeL"] + bert_f1) / 2 * 10

        return TurnResult(
            turn=turn_idx,
            prompt=prompt,
            model_output=model_output,
            score=combined,
            metric_details={
                "rouge1": rouge["rouge1"],
                "rouge2": rouge["rouge2"],
                "rougeL": rouge["rougeL"],
                "bertscore_f1": bert_f1,
            },
        )


class AIHub582(AIHubBase):
    name = "aihub_582"
    jsonl_path = Path("/app/data/aihub_582_sample.jsonl")


class AIHub90(AIHubBase):
    name = "aihub_90"
    jsonl_path = Path("/app/data/aihub_90_sample.jsonl")
