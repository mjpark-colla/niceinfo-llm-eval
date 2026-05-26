"""results_v2/<model>/aihub_*.jsonl 의 judge_raw 를 재파싱하여 score 복구.

배경: G-Eval 응답이 "종합: [9.6]" 형식인데 기존 regex가 대괄호를 처리 못해
210건 (per bench) 가 score=0 으로 저장됨. 모든 judge_raw 는 보존돼 있어
재추론·재judge 없이 score 필드만 다시 채울 수 있음.
"""
import json
import shutil
from pathlib import Path

import re

# parser 인라인 (openai 모듈 의존 없이 동작) — eval/metrics/llm_judge.py 의
# _parse_summarization 와 동일 로직, regex 만 수정 후 적용
def _parse_summarization(raw: str) -> tuple[float, dict]:
    dim_patterns = {
        "faithfulness": [r"사실\s*충실도", r"faithfulness"],
        "relevance": [r"핵심\s*정보\s*포함도", r"핵심\s*정보", r"relevance"],
        "coherence": [r"일관성", r"coherence"],
        "conciseness": [r"간결성", r"conciseness"],
    }
    details = {}
    for key, patterns in dim_patterns.items():
        for p in patterns:
            m = re.search(rf"{p}\s*[:：]\s*\[?\s*(\d+(?:\.\d+)?)\s*\]?", raw, re.IGNORECASE)
            if m:
                try:
                    val = float(m.group(1))
                    details[key] = max(1.0, min(5.0, val))
                    break
                except ValueError:
                    continue

    overall = None
    m = re.search(r"종합\s*(?:점수)?\s*[:：]\s*\[?\s*(\d+(?:\.\d+)?)\s*\]?", raw)
    if m:
        try:
            overall = float(m.group(1))
            overall = max(1.0, min(10.0, overall))
        except ValueError:
            pass

    if overall is None and len(details) == 4:
        overall = sum(details.values()) / 4 * 2
        overall = max(1.0, min(10.0, overall))
    elif overall is None and len(details) > 0:
        overall = sum(details.values()) / len(details) * 2
        overall = max(1.0, min(10.0, overall))
    elif overall is None:
        overall = 0.0
    return overall, details

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results_v2"
MODELS = ["Qwen3-32B-AWQ", "Qwen3-30B-A3B-BF16",
          "Qwen3.6-35B-A3B-BF16", "Qwen3.6-35B-A3B-FP8"]
BENCHES = ["aihub_582", "aihub_90"]

print("=" * 80)
print("results_v2 AIHub score 재파싱 (regex 수정 적용)")
print("=" * 80)

for model in MODELS:
    for bench in BENCHES:
        path = RESULTS / model / f"{bench}.jsonl"
        if not path.exists():
            continue

        # 백업
        backup = path.with_suffix(".jsonl.bak")
        if not backup.exists():
            shutil.copy(path, backup)

        records = []
        changed = 0
        old_scores, new_scores = [], []
        for line in open(backup):
            r = json.loads(line)
            old_score = r.get("score")
            jr = r["turns"][0].get("judge_raw") or ""
            new_score, details = _parse_summarization(jr)
            r["score"] = new_score
            r["turns"][0]["score"] = new_score
            # metric_details 에 4차원 채우기 (기존엔 비어있음)
            md = r["turns"][0].get("metric_details") or {}
            md.update(details)
            r["turns"][0]["metric_details"] = md
            records.append(r)
            if old_score is not None:
                old_scores.append(old_score)
            new_scores.append(new_score)
            if old_score != new_score:
                changed += 1

        with open(path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        old_avg = sum(old_scores) / len(old_scores) if old_scores else 0
        new_avg = sum(new_scores) / len(new_scores) if new_scores else 0
        print(f"[{model:<25} × {bench:<10}] "
              f"평균 {old_avg:.2f} → {new_avg:.2f}  변경 {changed}/{len(records)}")

print()
print("✅ 재파싱 완료. 백업: results_v2/<model>/<bench>.jsonl.bak")
