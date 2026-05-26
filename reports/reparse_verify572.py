"""results_verify572 의 AIHub 점수 재파싱 (옛 buggy 코드 모델 2개).

배경:
- Qwen3-32B-AWQ, Qwen3-30B-A3B-BF16 평가는 patched 전 옛 코드로 진행 → score=0 다수
- Qwen3.6-35B-A3B-BF16/FP8 평가는 patched 후 진행 → score 정상
- 모든 모델 judge_raw 는 보존 → 동일 reparse 적용 가능

조치: 4 모델 모두 재파싱 (이미 정상인 모델은 변화 없음)
"""
import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results_verify572"
MODELS = ["Qwen3-32B-AWQ", "Qwen3-30B-A3B-BF16",
          "Qwen3.6-35B-A3B-BF16", "Qwen3.6-35B-A3B-FP8"]
BENCHES = ["aihub_582", "aihub_90"]


def parse_summarization(raw: str):
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
                    details[key] = max(1.0, min(5.0, float(m.group(1))))
                    break
                except ValueError:
                    continue
    overall = None
    m = re.search(r"종합\s*(?:점수)?\s*[:：]\s*\[?\s*(\d+(?:\.\d+)?)\s*\]?", raw)
    if m:
        try:
            overall = max(1.0, min(10.0, float(m.group(1))))
        except ValueError:
            pass
    if overall is None and len(details) == 4:
        overall = max(1.0, min(10.0, sum(details.values()) / 4 * 2))
    elif overall is None and details:
        overall = max(1.0, min(10.0, sum(details.values()) / len(details) * 2))
    elif overall is None:
        overall = 0.0
    return overall, details


print("=" * 80)
print("results_verify572 AIHub score 재파싱")
print("=" * 80)

for model in MODELS:
    for bench in BENCHES:
        path = RESULTS / model / f"{bench}.jsonl"
        if not path.exists():
            continue
        backup = path.with_suffix(".jsonl.bak")
        if not backup.exists():
            shutil.copy(path, backup)
        records, changed, old_avg, new_avg = [], 0, 0, 0
        for line in open(backup):
            r = json.loads(line)
            old_s = r.get("score", 0)
            jr = r["turns"][0].get("judge_raw") or ""
            new_s, details = parse_summarization(jr)
            r["score"] = new_s
            r["turns"][0]["score"] = new_s
            md = r["turns"][0].get("metric_details") or {}
            md.update(details)
            r["turns"][0]["metric_details"] = md
            records.append(r)
            old_avg += old_s
            new_avg += new_s
            if abs(old_s - new_s) > 0.01:
                changed += 1
        with open(path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        n = len(records)
        print(f"[{model:<25} × {bench}] "
              f"평균 {old_avg/n:.2f} → {new_avg/n:.2f}  변경 {changed}/{n}")

print()
print("✅ 재파싱 완료")
