"""results_v2/ → reports/samples_v2.json (sample-centric)
동일 sample_id 기준 4 모델 답변·점수·judge 묶음.
HTML 대시보드의 "답변 비교" 탭이 fetch 또는 embed로 사용.
"""
import json
from pathlib import Path
from collections import defaultdict

MODELS = ["Qwen3-32B-AWQ", "Qwen3-30B-A3B-BF16",
          "Qwen3.6-35B-A3B-BF16", "Qwen3.6-35B-A3B-FP8"]
BENCHES = ["ko_mt_bench", "logickor", "ko_ifeval", "aihub_582", "aihub_90"]

PROMPT_MAX, OUTPUT_MAX, JUDGE_MAX = 3000, 4000, 2000

samples = defaultdict(lambda: defaultdict(dict))

for model in MODELS:
    for bench in BENCHES:
        path = Path(f"results_v2/{model}/{bench}.jsonl")
        if not path.exists():
            continue
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                sid = r["sample_id"]
                turns = []
                for t in r.get("turns", []):
                    turns.append({
                        "turn": t.get("turn"),
                        "prompt": (t.get("prompt") or "")[:PROMPT_MAX],
                        "model_output": (t.get("model_output") or "")[:OUTPUT_MAX],
                        "score": t.get("score"),
                        "judge_raw": ((t.get("judge_raw") or "")[:JUDGE_MAX]) or None,
                    })
                samples[bench][sid][model] = {
                    "score": r.get("score"),
                    "turns": turns,
                }

out = {b: dict(samples[b]) for b in BENCHES}
out_path = Path("reports/samples_v2.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, separators=(",", ":"))

size_mb = out_path.stat().st_size / (1024 * 1024)
total = sum(len(v) for v in out.values())
print(f"✅ {out_path} 저장")
print(f"   크기: {size_mb:.2f} MB")
print(f"   총 sample (벤치 합계): {total}")
for b in BENCHES:
    print(f"   {b}: {len(out[b])} sample")
