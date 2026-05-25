"""AI Hub 요약 데이터셋 (582 + 90).

- 로컬 jsonl 파일 사용
- 메인 메트릭: LLM-as-Judge (G-Eval 4차원 → 종합 1-10)
- 보조 메트릭: ROUGE-1/2/L + BERTScore F1 (metric_details에 보존)
- 입력: passage / 정답: reference(s)

설계 변경 (2026-05-25):
- ROUGE/BERTScore만 사용하던 방식에서 G-Eval 기반 LLM-Judge로 메인 전환
- 이유: ROUGE는 패러프레이즈·사실 충실도 평가 불가
- ROUGE/BERTScore는 보조 metric으로 보존하여 표준 reference-based 비교 가능
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
    metric_type = "judge"

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
            raise ValueError("AIHub 요약은 judge(G-Eval)가 필요합니다.")

        references = sample.reference or []
        if isinstance(references, str):
            references = [references]

        # 원문(passage) 추출 — prompt 안에 SUMMARIZATION_PROMPT 형식으로 들어있음
        # "[원문]\n...\n\n[요약]" 패턴
        passage = _extract_passage(prompt)

        # G-Eval Judge 호출 (메인 점수)
        judge_result = await judge.score_summarization(
            passage=passage,
            summary=model_output,
            reference=references if references else None,
        )

        # 보조 메트릭 — reference가 있을 때만
        rouge = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
        bert_f1 = 0.0
        if references:
            rouge = compute_rouge(model_output, references)
            try:
                bert_f1 = compute_bertscore_single(model_output, references)
            except Exception:
                bert_f1 = 0.0

        return TurnResult(
            turn=turn_idx,
            prompt=prompt,
            model_output=model_output,
            score=judge_result["score"],  # G-Eval 종합 (0-10)
            metric_details={
                # G-Eval 4 차원
                "geval_faithfulness": judge_result["details"].get("faithfulness"),
                "geval_relevance": judge_result["details"].get("relevance"),
                "geval_coherence": judge_result["details"].get("coherence"),
                "geval_conciseness": judge_result["details"].get("conciseness"),
                "judge_model": judge_result["judge_model"],
                # 보조 metric (reference-based)
                "rouge1": rouge["rouge1"],
                "rouge2": rouge["rouge2"],
                "rougeL": rouge["rougeL"],
                "bertscore_f1": bert_f1,
            },
            judge_raw=judge_result["raw"],
            judge_tokens_in=judge_result["tokens_in"],
            judge_tokens_out=judge_result["tokens_out"],
        )


def _extract_passage(prompt: str) -> str:
    """SUMMARIZATION_PROMPT 형식의 prompt에서 [원문] 영역 추출."""
    import re
    m = re.search(r"\[원문\]\s*\n(.*?)\n\s*\[요약\]", prompt, re.DOTALL)
    if m:
        return m.group(1).strip()
    return prompt  # fallback


class AIHub582(AIHubBase):
    name = "aihub_582"
    jsonl_path = Path("/app/data/aihub_582_sample.jsonl")


class AIHub90(AIHubBase):
    name = "aihub_90"
    jsonl_path = Path("/app/data/aihub_90_sample.jsonl")
