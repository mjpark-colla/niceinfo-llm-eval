"""JSONL 읽기·쓰기 + Resume 지원."""
import json
from pathlib import Path
from typing import Iterator, Any


def append_jsonl(record: dict, path: Path) -> None:
    """jsonl에 한 줄 추가 (crash-safe)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


def read_jsonl(path: Path) -> Iterator[dict]:
    """jsonl 한 줄씩 yield."""
    if not path.exists():
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                # 손상된 라인 skip
                continue


def load_completed_ids(path: Path) -> set[str]:
    """이미 평가한 sample id 집합. Resume 시 skip 용."""
    return {r["sample_id"] for r in read_jsonl(path) if "sample_id" in r}


def write_json(data: Any, path: Path) -> None:
    """단일 JSON 객체 저장 (집계 결과 등)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def strip_thinking(text: str) -> str:
    """Qwen3 등의 <think>...</think> 블록 제거.

    multiline thinking 안전 처리.
    """
    import re
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()
