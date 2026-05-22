"""Sanity 결과를 사람이 보기 좋게 출력."""
import json
import sys
from pathlib import Path


def main(base_path: str, model_name: str):
    base = Path(base_path) / model_name
    benchmarks = ["ko_mt_bench", "logickor", "ko_ifeval", "aihub_582", "aihub_90"]

    for bench in benchmarks:
        path = base / f"{bench}.jsonl"
        if not path.exists():
            print(f"[{bench}] 결과 파일 없음")
            continue

        with open(path) as f:
            records = [json.loads(l) for l in f if l.strip()]

        scores = [r["score"] for r in records if "score" in r and "error" not in r]
        avg = sum(scores) / len(scores) if scores else 0.0

        print("=" * 70)
        print(f"[{bench}] 평균: {avg:.2f}/10  ({len(records)} samples)")
        print("=" * 70)

        if not records:
            continue

        r = records[0]
        meta = r.get("metadata") or {}
        cost = r.get("cost_usd", 0.0)
        elapsed = r.get("elapsed_sec", 0.0)
        sid = r["sample_id"]
        cat = meta.get("category") or meta.get("domain") or "-"

        print(f"  [예시] sample_id: {sid}")
        print(f"         category/domain: {cat}")
        print(f"         종합 score: {r['score']:.2f}  |  cost: ${cost:.4f}  |  elapsed: {elapsed:.1f}s")

        for t in r.get("turns", []):
            print()
            print(f"  ── Turn {t['turn']} (score: {t['score']:.1f}/10) ──")
            details = t.get("metric_details") or {}
            for k, v in details.items():
                if isinstance(v, float):
                    print(f"    {k}: {v:.3f}")
                else:
                    print(f"    {k}: {v}")
            prompt = t["prompt"]
            output = t["model_output"]
            judge = t.get("judge_raw")
            print(f"    Q: {prompt[:200]}{'...' if len(prompt) > 200 else ''}")
            print(f"    A: {output[:250]}{'...' if len(output) > 250 else ''}")
            if judge:
                judge_clean = judge.replace("\n", " / ").strip()[:220]
                print(f"    Judge: {judge_clean}")
        print()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/app/results/sanity",
         sys.argv[2] if len(sys.argv) > 2 else "Qwen3-32B-AWQ")
