"""HTML 대시보드 v2 생성기 — 가중치 슬라이더 + 프리셋 + v1/v2 비교.

생성: reports/index_v2.html (self-contained, 외부 의존성 0)

기능:
- 프리셋 5개 (균등, 현재, 요약, 보고서, 대화 QA) — 클릭하면 슬라이더 자동 설정
- 슬라이더 5개 — 각 벤치 가중치 0-100, 자유 조정
- 모델 종합 점수·순위 즉시 재계산
- 구 vs 신 비교 탭
- G-Eval 4차원 레이더 (요약 평가)
- 시나리오별 1위 모델 카드
- Raw 답변 탐색기 (모델/벤치/sample 필터)
"""
import json
from pathlib import Path

ROOT = Path("/Users/minji/Documents/PolarPulse/niceinfo")
OLD = ROOT / "results"
NEW = ROOT / "results_v2"
OUT = ROOT / "reports" / "index_v2.html"

MODELS_ORDER = [
    "Qwen3.6-35B-A3B-BF16",
    "Qwen3.6-35B-A3B-FP8",
    "Qwen3-32B-AWQ",
    "Qwen3-30B-A3B-BF16",
]
BENCH_ORDER = ["ko_mt_bench", "logickor", "ko_ifeval", "aihub_582", "aihub_90"]
BENCH_LABEL = {
    "ko_mt_bench": "Ko-MT-Bench",
    "logickor": "LogicKor",
    "ko_ifeval": "Ko-IFEval",
    "aihub_582": "AIHub 582",
    "aihub_90": "AIHub 90",
}
PRESETS = {
    "균등": {"ko_mt_bench": 20, "logickor": 20, "ko_ifeval": 20, "aihub_582": 20, "aihub_90": 20},
    "현재": {"ko_mt_bench": 25, "logickor": 15, "ko_ifeval": 20, "aihub_582": 20, "aihub_90": 20},
    "요약 중심": {"ko_mt_bench": 10, "logickor": 10, "ko_ifeval": 20, "aihub_582": 30, "aihub_90": 30},
    "추론·형식 중심": {"ko_mt_bench": 20, "logickor": 25, "ko_ifeval": 25, "aihub_582": 15, "aihub_90": 15},
    "대화 QA": {"ko_mt_bench": 35, "logickor": 25, "ko_ifeval": 25, "aihub_582": 7, "aihub_90": 8},
}


def load_data(root: Path) -> dict:
    """root / <model> / <bench>.jsonl → {model: {bench: [samples]}}"""
    data = {}
    for model in MODELS_ORDER:
        data[model] = {}
        for bench in BENCH_ORDER:
            f = root / model / f"{bench}.jsonl"
            if not f.exists():
                data[model][bench] = []
                continue
            samples = []
            for line in f.read_text().splitlines():
                if line.strip():
                    try:
                        samples.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
            data[model][bench] = samples
    return data


def avg_score(samples: list) -> float:
    scores = [s["score"] for s in samples if "score" in s and "error" not in s]
    return sum(scores) / len(scores) if scores else 0


# 데이터 로드 (per-bench 점수만 집계 — raw 전체 inline은 너무 무거움)
old_data = load_data(OLD)
new_data = load_data(NEW)


def aggregate_scores(data: dict) -> dict:
    """{model: {bench: avg_score}}"""
    return {
        model: {bench: avg_score(data[model][bench]) for bench in BENCH_ORDER}
        for model in MODELS_ORDER
    }


# G-Eval 4차원 통계 (요약 평가만, 신 결과만)
def geval_stats(data: dict) -> dict:
    """{model: {dim: avg}}"""
    out = {}
    for model in MODELS_ORDER:
        dims = {"faithfulness": [], "relevance": [], "coherence": [], "conciseness": []}
        for bench in ("aihub_582", "aihub_90"):
            for s in data[model][bench]:
                if "error" in s: continue
                turn = (s.get("turns") or [{}])[0]
                md = turn.get("metric_details") or {}
                for dim in dims:
                    key = f"geval_{dim}"
                    if key in md and md[key] is not None:
                        dims[dim].append(md[key])
        out[model] = {d: (sum(v)/len(v) if v else 0) for d, v in dims.items()}
    return out


old_scores = aggregate_scores(old_data)
new_scores = aggregate_scores(new_data)
new_geval = geval_stats(new_data)


# ============================================================
# 속도 통계 (v1 jsonl 의 elapsed_sec / tokens_out)
# ⚠️ A100 측정, concurrency 8, judge 일부 포함 — 절대값 운영 재현 X
# ============================================================
def collect_speed_stats(data: dict) -> dict:
    out = {}
    for model in MODELS_ORDER:
        out[model] = {}
        for bench in BENCH_ORDER:
            samples = [s for s in data[model][bench] if "error" not in s and "elapsed_sec" in s]
            if not samples:
                continue
            elapsed = [s["elapsed_sec"] for s in samples]
            tin = []
            tout = []
            for s in samples:
                turns = s.get("turns") or []
                tin.append(sum(t.get("tokens_in", 0) for t in turns))
                tout.append(sum(t.get("tokens_out", 0) for t in turns))
            out[model][bench] = {
                "elapsed_avg": sum(elapsed) / len(elapsed),
                "tokens_out_avg": sum(tout) / len(tout) if tout else 0,
                "tokens_in_avg": sum(tin) / len(tin) if tin else 0,
                "approx_tps": (sum(tout) / sum(elapsed)) if sum(elapsed) else 0,
            }
    return out


# 모델별 종합 (judge 없는 벤치 기준)
def speed_summary(speed_stats: dict) -> dict:
    out = {}
    for model in MODELS_ORDER:
        e_tot, t_tot, n = 0, 0, 0
        for bench in ("ko_ifeval", "aihub_582", "aihub_90"):
            s = speed_stats.get(model, {}).get(bench)
            if s:
                e_tot += s["elapsed_avg"]
                t_tot += s["tokens_out_avg"]
                n += 1
        if n > 0:
            out[model] = {
                "elapsed_avg": e_tot / n,
                "tokens_out_avg": t_tot / n,
                "tps": (t_tot / e_tot) if e_tot else 0,
            }
    return out


speed_stats = collect_speed_stats(old_data)
speed_sum = speed_summary(speed_stats)

# Raw 답변 (간단한 sample 메타만 embed — 풀 raw는 v1 HTML 참조)
# 점수와 sample_id만 inline — 클릭 시 외부 jsonl 안내
raw_min = {}
for model in MODELS_ORDER:
    raw_min[model] = {}
    for bench in BENCH_ORDER:
        items = []
        for s in new_data[model][bench]:
            if "error" in s: continue
            items.append({
                "id": s.get("sample_id"),
                "score": s.get("score"),
                "cat": (s.get("metadata") or {}).get("category") or (s.get("metadata") or {}).get("domain") or "",
            })
        raw_min[model][bench] = items

config_json = json.dumps({
    "models": MODELS_ORDER,
    "benches": BENCH_ORDER,
    "bench_label": BENCH_LABEL,
    "presets": PRESETS,
}, ensure_ascii=False)
old_scores_json = json.dumps(old_scores, ensure_ascii=False)
new_scores_json = json.dumps(new_scores, ensure_ascii=False)
geval_json = json.dumps(new_geval, ensure_ascii=False)
speed_json = json.dumps({"per_bench": speed_stats, "summary": speed_sum}, ensure_ascii=False)


def safe_embed(j: str) -> str:
    """XSS 방지: </script>, <!--, U+2028/29 이스케이프."""
    return (j
        .replace("</", "<\\/")
        .replace("<!--", "<\\!--")
        .replace("-->", "--\\>")
        .replace(" ", "\\u2028")
        .replace(" ", "\\u2029"))


HTML = """<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8">
<title>niceinfo Phase 1 v2 — 가중치 인터랙티브 대시보드</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root {
  --bg: #0f1419; --bg-2: #1a1f2e; --bg-3: #232a3d;
  --text: #e6e6e6; --text-2: #a8b2c1; --text-dim: #6b7280;
  --accent: #5eead4; --accent-2: #fbbf24;
  --good: #34d399; --warn: #fb923c; --bad: #f87171;
  --border: #2d3748;
  --gold: #fcd34d; --silver: #d1d5db; --bronze: #d97706;
}
@media (prefers-color-scheme: light) {
  :root {
    --bg: #fafaf9; --bg-2: #ffffff; --bg-3: #f3f4f6;
    --text: #1f2937; --text-2: #4b5563; --text-dim: #9ca3af;
    --accent: #0d9488; --accent-2: #d97706; --border: #e5e7eb;
  }
}
* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; background: var(--bg); color: var(--text);
  font-family: -apple-system, "Pretendard", "SF Pro Text", "Apple SD Gothic Neo", "Noto Sans KR", sans-serif;
  line-height: 1.55; font-size: 14px; }
.container { max-width: 1280px; margin: 0 auto; padding: 24px; }
header { border-bottom: 1px solid var(--border); padding-bottom: 16px; margin-bottom: 24px; }
header h1 { margin: 0 0 4px 0; font-size: 22px; font-weight: 700; }
header .sub { color: var(--text-2); font-size: 13px; }
nav { display: flex; gap: 4px; margin: 20px 0; border-bottom: 1px solid var(--border); flex-wrap: wrap; }
nav button { background: none; border: none; color: var(--text-2); padding: 10px 16px; cursor: pointer;
  font-size: 13px; font-weight: 500; border-bottom: 2px solid transparent; transition: all 0.15s; }
nav button:hover { color: var(--text); }
nav button.active { color: var(--accent); border-bottom-color: var(--accent); }
section { display: none; }
section.active { display: block; }
h2 { font-size: 18px; margin: 24px 0 12px 0; font-weight: 600; }
h2:first-child { margin-top: 0; }
h3 { font-size: 15px; margin: 18px 0 8px 0; font-weight: 600; color: var(--text-2); }
.card { background: var(--bg-2); border: 1px solid var(--border); border-radius: 8px; padding: 16px;
  margin-bottom: 16px; }
.grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
.grid-3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; }
@media (max-width: 800px) {
  .grid-2 { grid-template-columns: 1fr; }
  .grid-3 { grid-template-columns: 1fr; }
}
table { width: 100%; border-collapse: collapse; font-size: 13px; }
th, td { text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--border); }
th { font-weight: 600; color: var(--text-2); background: var(--bg-3); font-size: 12px; }
tr:last-child td { border-bottom: none; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
.badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 500; }
.b-gold { background: rgba(252, 211, 77, 0.15); color: var(--gold); }
.b-silver { background: rgba(209, 213, 219, 0.15); color: var(--silver); }
.b-bronze { background: rgba(217, 119, 6, 0.15); color: var(--bronze); }
.b-good { background: rgba(52, 211, 153, 0.15); color: var(--good); }
.b-warn { background: rgba(251, 146, 60, 0.15); color: var(--warn); }
.b-bad  { background: rgba(248, 113, 113, 0.15); color: var(--bad); }
.b-up { color: var(--good); }
.b-down { color: var(--bad); }
.b-zero { color: var(--text-dim); }
.bar-bg { background: var(--bg-3); border-radius: 3px; height: 18px; overflow: hidden; position: relative; }
.bar-fg { background: linear-gradient(90deg, var(--accent), var(--accent-2)); height: 100%;
  border-radius: 3px; transition: width 0.3s ease; }
.bar-row { display: grid; grid-template-columns: 200px 1fr 70px; gap: 12px; align-items: center;
  padding: 5px 0; font-size: 13px; }
.bar-row .val { text-align: right; color: var(--text-2); font-variant-numeric: tabular-nums; }

/* 슬라이더·프리셋 영역 */
.weight-panel { background: var(--bg-2); border: 1px solid var(--border); border-radius: 8px;
  padding: 16px; margin-bottom: 16px; }
.preset-bar { display: flex; gap: 8px; margin-bottom: 16px; flex-wrap: wrap; }
.preset-btn { background: var(--bg-3); color: var(--text); border: 1px solid var(--border);
  padding: 6px 12px; border-radius: 16px; cursor: pointer; font-size: 12px; transition: all 0.15s; }
.preset-btn:hover { border-color: var(--accent); }
.preset-btn.active { background: var(--accent); color: var(--bg); border-color: var(--accent);
  font-weight: 600; }
.slider-grid { display: grid; grid-template-columns: 130px 1fr 50px; gap: 10px;
  align-items: center; margin: 6px 0; font-size: 13px; }
.slider-grid input[type="range"] { width: 100%; accent-color: var(--accent); }
.slider-grid .val { text-align: right; font-variant-numeric: tabular-nums; color: var(--text-2);
  font-weight: 500; }
.weight-sum { font-size: 12px; color: var(--text-dim); text-align: right; padding-top: 8px;
  border-top: 1px solid var(--border); margin-top: 8px; }
.weight-sum.warn { color: var(--warn); }
.weight-sum.ok { color: var(--good); }

/* 모델 순위 카드 */
.rank-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; }
@media (max-width: 800px) { .rank-grid { grid-template-columns: repeat(2, 1fr); } }
.rank-card { background: var(--bg-3); border: 1px solid var(--border); border-radius: 8px;
  padding: 14px; text-align: center; transition: all 0.2s; }
.rank-card.gold { border-color: var(--gold); background: rgba(252, 211, 77, 0.05); }
.rank-card.silver { border-color: var(--silver); background: rgba(209, 213, 219, 0.05); }
.rank-card.bronze { border-color: var(--bronze); background: rgba(217, 119, 6, 0.05); }
.rank-card .medal { font-size: 24px; }
.rank-card .model { font-weight: 600; font-size: 13px; margin: 6px 0 4px 0; word-break: break-word; }
.rank-card .score { font-size: 18px; font-variant-numeric: tabular-nums; color: var(--accent); }
.rank-card .delta { font-size: 11px; color: var(--text-dim); margin-top: 4px; }

/* 레이더 차트 (G-Eval 4차원) */
.radar-wrap { display: flex; justify-content: center; padding: 12px; }
svg { display: block; }
.radar-legend { display: flex; gap: 12px; flex-wrap: wrap; justify-content: center; font-size: 11px;
  margin-top: 8px; color: var(--text-2); }
.radar-legend span::before { content: "●"; margin-right: 4px; font-size: 14px; }

/* 시나리오별 1위 카드 */
.scenario-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px; }
.scenario-card { background: var(--bg-3); border: 1px solid var(--border); border-radius: 8px; padding: 12px; }
.scenario-card .label { font-size: 11px; color: var(--text-dim); text-transform: uppercase;
  letter-spacing: 0.5px; }
.scenario-card .winner { font-weight: 700; font-size: 14px; margin-top: 6px; color: var(--accent); }
.scenario-card .score { font-size: 12px; color: var(--text-2); }

.muted { color: var(--text-dim); font-size: 12px; }
.callout { background: rgba(94, 234, 212, 0.05); border-left: 3px solid var(--accent);
  padding: 12px 14px; border-radius: 4px; margin: 12px 0; font-size: 13px; }
.callout-warn { background: rgba(251, 146, 60, 0.05); border-left-color: var(--warn); }
footer { margin-top: 40px; padding-top: 16px; border-top: 1px solid var(--border);
  color: var(--text-dim); font-size: 12px; text-align: center; }
</style>
</head>
<body>
<div class="container">
<header>
  <h1>Phase 1 v2 — 가중치 인터랙티브 대시보드</h1>
  <div class="sub">Judge prompt 재설계 (G-Eval / MT-Bench / LogicKor) · 2026-05-25 · 4 모델 × 5 벤치</div>
</header>

<nav>
  <button class="active" data-tab="dashboard">⚖️ 가중치 시뮬레이션</button>
  <button data-tab="compare">🔄 구 vs 신 비교</button>
  <button data-tab="geval">🎯 G-Eval 4차원</button>
  <button data-tab="speed">⚡ 속도 (대략 추정)</button>
  <button data-tab="bench">📋 벤치 상세</button>
</nav>

<!-- ============================================================ -->
<!-- TAB 1: 가중치 시뮬레이션 -->
<!-- ============================================================ -->
<section id="dashboard" class="active">
  <div class="weight-panel">
    <h2 style="margin-top:0">⚖️ 가중치 설정</h2>

    <div>
      <div class="muted" style="margin-bottom:6px">시나리오 프리셋 (클릭하면 슬라이더 자동 설정)</div>
      <div class="preset-bar" id="preset-bar"></div>
    </div>

    <h3>슬라이더로 미세 조정 (각 0-50)</h3>
    <div id="sliders"></div>
    <div class="weight-sum ok" id="weight-sum">합: 100%</div>
  </div>

  <h2>🏆 종합 점수 및 모델 순위</h2>
  <div class="rank-grid" id="rank-cards"></div>

  <div class="card" style="margin-top:16px">
    <h2 style="margin-top:0">점수 분포</h2>
    <div id="rank-bars"></div>
  </div>

  <div class="card">
    <h2 style="margin-top:0">시나리오별 1위 모델 한눈에 (참고)</h2>
    <div class="scenario-grid" id="scenario-cards"></div>
  </div>
</section>

<!-- ============================================================ -->
<!-- TAB 2: 구 vs 신 비교 -->
<!-- ============================================================ -->
<section id="compare">
  <div class="card">
    <h2 style="margin-top:0">Judge prompt 재설계 효과 — 모델 순위 변화</h2>
    <div class="muted">현재 가중치 (25/15/20/20/20) 기준</div>
    <div class="grid-2" style="margin-top:12px">
      <div>
        <h3>구 prompt (v1)</h3>
        <div id="old-ranks"></div>
      </div>
      <div>
        <h3>신 prompt (v2)</h3>
        <div id="new-ranks"></div>
      </div>
    </div>
  </div>

  <div class="card">
    <h2 style="margin-top:0">벤치별 점수 변화 (구→신)</h2>
    <table>
      <thead><tr><th>모델</th><th>Ko-MT</th><th>LogicKor</th><th>Ko-IFEval</th><th>AIHub 582</th><th>AIHub 90</th></tr></thead>
      <tbody id="compare-bench-table"></tbody>
    </table>
  </div>

  <div class="card">
    <h2 style="margin-top:0">변별력 변화 (모델 간 spread)</h2>
    <table>
      <thead><tr><th>벤치</th><th class="num">v1 spread</th><th class="num">v2 spread</th><th class="num">배수</th><th>평가</th></tr></thead>
      <tbody id="spread-table"></tbody>
    </table>
    <div class="callout" style="margin-top:12px">
      <strong>AIHub 요약 평가에서 변별력 3-5배 증가</strong> — ROUGE/BERTScore의 한계가 명확히 확인됨.
      G-Eval 4차원이 핵심 정보 포함도와 사실 충실도(faithfulness)를 잡아냈기 때문.
    </div>
  </div>
</section>

<!-- ============================================================ -->
<!-- TAB 3: G-Eval 4차원 -->
<!-- ============================================================ -->
<section id="geval">
  <div class="card">
    <h2 style="margin-top:0">G-Eval 4차원 통계 (요약 평가, AIHub 582+90 통합)</h2>
    <div class="muted">각 차원 1-5점. 모델의 요약 강점·약점 진단.</div>
    <div class="grid-2" style="margin-top:12px">
      <div class="radar-wrap" id="radar"></div>
      <div>
        <table>
          <thead><tr><th>모델</th><th class="num">Faithful</th><th class="num">Relevance</th><th class="num">Coherence</th><th class="num">Concise</th></tr></thead>
          <tbody id="geval-table"></tbody>
        </table>
        <div class="radar-legend" id="radar-legend" style="margin-top:16px"></div>
      </div>
    </div>
    <div class="callout" style="margin-top:12px" id="geval-insight"></div>
  </div>
</section>

<!-- ============================================================ -->
<!-- TAB 4: 속도 (대략 추정) -->
<!-- ============================================================ -->
<section id="speed">
  <div class="card callout callout-warn">
    <strong>⚠️ 본 속도 분석의 한계 (반드시 확인)</strong><br>
    아래 수치는 <strong>운영 의사결정에 그대로 적용할 수 없습니다</strong>:
    <ul style="margin:8px 0; padding-left:20px">
      <li><strong>A100 ≠ H100 NVL</strong> — 평가는 A100 80GB, 운영은 H100 NVL 94GB. 절대값 재현 X</li>
      <li><strong>단독 latency 아님</strong> — concurrency=8 환경, GPU 경합 포함</li>
      <li><strong>Judge 호출 시간 일부 포함</strong> — Ko-MT-Bench·LogicKor 의 elapsed_sec 에 OpenAI judge 호출 시간 섞임</li>
      <li><strong>TTFT 측정 불가</strong> — streaming 아닌 일괄 응답</li>
      <li><strong>AWQ/FP8 측정값 caveat</strong> — A100 dequant/emulation overhead 가능, H100 결과 다를 수 있음</li>
      <li><strong>임베딩·리랭커 동시 부하 미반영</strong> — 본 측정은 LLM 단독</li>
    </ul>
    <strong>운영 SLA·실제 응답 시간은 Phase 3 폐쇄망 H100 NVL 측정이 필수.</strong> 이 탭은 "4 모델 동등 조건 상대 비교"만 유효.
  </div>

  <div class="card">
    <h2 style="margin-top:0">속도 순위 — Judge 없는 벤치 (Ko-IFEval + AIHub 582/90)</h2>
    <div class="muted">가장 깨끗한 비교 — judge 호출 시간 제외</div>
    <div id="speed-summary" style="margin-top:12px"></div>
  </div>

  <div class="card">
    <h2 style="margin-top:0">품질 + 속도 통합 (A100 기준)</h2>
    <table>
      <thead><tr><th>모델</th><th class="num">품질 (v2)</th><th class="num">속도 (tok/s)</th><th class="num">메모리</th><th>평가</th></tr></thead>
      <tbody id="quality-speed-table"></tbody>
    </table>
    <div class="muted" style="margin-top:8px">
      ⚠️ 속도 순위는 A100 한정. H100 NVL 운영에서 AWQ·FP8 순위가 바뀔 수 있음.
    </div>
  </div>

  <div class="card">
    <h2 style="margin-top:0">벤치별 sample 시간 (concurrency 8 환경)</h2>
    <div class="legend">
      <span class="cb-judge">judge 호출 없음 (가장 깨끗)</span>
      <span class="cb-prompt">judge 호출 포함 (시간 부풀려짐)</span>
    </div>
    <div id="speed-per-bench"></div>
  </div>
</section>

<!-- ============================================================ -->
<!-- TAB 5: 벤치 상세 -->
<!-- ============================================================ -->
<section id="bench">
  <div class="card">
    <h2 style="margin-top:0">벤치별 모델 점수 (신 v2 기준)</h2>
    <div id="bench-bars"></div>
  </div>
</section>

<footer>
  생성: 2026-05-25 · 가중치 슬라이더로 자유 시뮬레이션 가능 · 데이터: results_v2 (rejudge 결과)
</footer>
</div>

<script>
const CFG = __CONFIG_JSON__;
const OLD = __OLD_SCORES_JSON__;
const NEW = __NEW_SCORES_JSON__;
const GEVAL = __GEVAL_JSON__;
const SPEED = __SPEED_JSON__;
const JUDGE_BENCHES = ["ko_mt_bench", "logickor"];  // judge 호출 포함 (시간 부풀려짐)

// 모델 색 매핑 (레이더)
const MODEL_COLORS = {
  "Qwen3.6-35B-A3B-BF16": "#f87171",
  "Qwen3.6-35B-A3B-FP8":  "#fbbf24",
  "Qwen3-32B-AWQ":        "#5eead4",
  "Qwen3-30B-A3B-BF16":   "#a78bfa",
};

// ============================================================
// Tabs
// ============================================================
document.querySelectorAll("nav button").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll("nav button").forEach(b => b.classList.remove("active"));
    document.querySelectorAll("section").forEach(s => s.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById(btn.dataset.tab).classList.add("active");
  });
});

// ============================================================
// 가중치 슬라이더·프리셋
// ============================================================
const sliderDiv = document.getElementById("sliders");
const presetBar = document.getElementById("preset-bar");
const sumDiv = document.getElementById("weight-sum");

const sliders = {};
CFG.benches.forEach(bench => {
  const row = document.createElement("div");
  row.className = "slider-grid";
  row.innerHTML = `
    <label>${CFG.bench_label[bench]}</label>
    <input type="range" id="s-${bench}" min="0" max="50" step="1" value="20">
    <span class="val" id="v-${bench}">20</span>
  `;
  sliderDiv.appendChild(row);
  sliders[bench] = row.querySelector("input");
  sliders[bench].addEventListener("input", () => {
    document.getElementById(`v-${bench}`).textContent = sliders[bench].value;
    updateAll();
    // 슬라이더 변경 시 프리셋 active 해제 (직접 조정 모드)
    document.querySelectorAll(".preset-btn").forEach(b => b.classList.remove("active"));
  });
});

// 프리셋 버튼
Object.entries(CFG.presets).forEach(([name, weights], i) => {
  const btn = document.createElement("button");
  btn.className = "preset-btn" + (i === 1 ? " active" : "");  // 기본: "현재"
  btn.textContent = name;
  btn.dataset.preset = name;
  btn.addEventListener("click", () => {
    Object.entries(weights).forEach(([bench, val]) => {
      sliders[bench].value = val;
      document.getElementById(`v-${bench}`).textContent = val;
    });
    document.querySelectorAll(".preset-btn").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    updateAll();
  });
  presetBar.appendChild(btn);
});

function getCurrentWeights() {
  const w = {};
  CFG.benches.forEach(b => { w[b] = parseFloat(sliders[b].value); });
  return w;
}

function getNormalizedWeights() {
  const w = getCurrentWeights();
  const sum = Object.values(w).reduce((a, b) => a + b, 0);
  if (sum === 0) return Object.fromEntries(CFG.benches.map(b => [b, 0]));
  const norm = {};
  for (const [k, v] of Object.entries(w)) norm[k] = v / sum;
  return norm;
}

function computeWeightedScores(scoresMap, weights) {
  const out = {};
  for (const model of CFG.models) {
    let total = 0;
    for (const bench of CFG.benches) {
      total += (scoresMap[model]?.[bench] || 0) * weights[bench];
    }
    out[model] = total;
  }
  return out;
}

function updateSum() {
  const w = getCurrentWeights();
  const sum = Object.values(w).reduce((a, b) => a + b, 0);
  sumDiv.textContent = `합: ${sum}% (normalize: ${(100).toFixed(0)}%)`;
  sumDiv.className = "weight-sum " + (Math.abs(sum - 100) <= 5 ? "ok" : (sum === 0 ? "warn" : "warn"));
}

function updateAll() {
  updateSum();
  renderRanks();
  renderRankBars();
}

// ============================================================
// 순위 카드·바차트
// ============================================================
function renderRanks() {
  const weights = getNormalizedWeights();
  const scores = computeWeightedScores(NEW, weights);
  const sorted = Object.entries(scores).sort((a, b) => b[1] - a[1]);

  // 이전 (구 가중치 25/15/20/20/20 기준) 점수로 변화 계산
  const v1Sorted = Object.entries(computeWeightedScores(NEW, normalizeWeights(CFG.presets["현재"])))
    .sort((a, b) => b[1] - a[1]);

  const cards = document.getElementById("rank-cards");
  cards.innerHTML = sorted.map(([m, s], i) => {
    const medal = ["🥇", "🥈", "🥉", "4️⃣"][i] || `${i+1}`;
    const cls = ["gold", "silver", "bronze", ""][i] || "";
    return `
      <div class="rank-card ${cls}">
        <div class="medal">${medal}</div>
        <div class="model">${m}</div>
        <div class="score">${s.toFixed(3)}</div>
      </div>
    `;
  }).join("");
}

function normalizeWeights(presetVals) {
  const sum = Object.values(presetVals).reduce((a, b) => a + b, 0);
  return Object.fromEntries(Object.entries(presetVals).map(([k, v]) => [k, v / sum]));
}

function renderRankBars() {
  const weights = getNormalizedWeights();
  const scores = computeWeightedScores(NEW, weights);
  const sorted = Object.entries(scores).sort((a, b) => b[1] - a[1]);
  const max = Math.max(...sorted.map(([_, v]) => v), 1);
  document.getElementById("rank-bars").innerHTML = sorted.map(([m, s]) => `
    <div class="bar-row">
      <div>${m}</div>
      <div class="bar-bg"><div class="bar-fg" style="width:${s/max*100}%"></div></div>
      <div class="val">${s.toFixed(3)}</div>
    </div>
  `).join("");
}

// ============================================================
// 시나리오별 1위 카드 (참고)
// ============================================================
function renderScenarios() {
  const div = document.getElementById("scenario-cards");
  const html = Object.entries(CFG.presets).map(([name, presetVals]) => {
    const weights = normalizeWeights(presetVals);
    const scores = computeWeightedScores(NEW, weights);
    const sorted = Object.entries(scores).sort((a, b) => b[1] - a[1]);
    const [winner, ws] = sorted[0];
    return `
      <div class="scenario-card">
        <div class="label">${name}</div>
        <div class="winner">🥇 ${winner}</div>
        <div class="score">${ws.toFixed(3)} · 2위 ${sorted[1][0].split('-').slice(-2).join('-')} (${sorted[1][1].toFixed(2)})</div>
      </div>
    `;
  }).join("");
  div.innerHTML = html;
}

// ============================================================
// 구 vs 신 비교 탭
// ============================================================
function renderCompareTab() {
  const w = normalizeWeights(CFG.presets["현재"]);
  const oldT = computeWeightedScores(OLD, w);
  const newT = computeWeightedScores(NEW, w);
  const oldSorted = Object.entries(oldT).sort((a, b) => b[1] - a[1]);
  const newSorted = Object.entries(newT).sort((a, b) => b[1] - a[1]);
  const oldPos = Object.fromEntries(oldSorted.map(([m], i) => [m, i]));

  document.getElementById("old-ranks").innerHTML = oldSorted.map(([m, s], i) => `
    <div class="bar-row" style="grid-template-columns:30px 1fr 60px">
      <div>${["🥇","🥈","🥉","4️⃣"][i]}</div>
      <div>${m}</div>
      <div class="val">${s.toFixed(3)}</div>
    </div>`).join("");

  document.getElementById("new-ranks").innerHTML = newSorted.map(([m, s], i) => {
    const diff = oldPos[m] - i;
    const change = diff > 0 ? `🚀 +${diff}` : diff < 0 ? `📉 ${diff}` : "→";
    return `
      <div class="bar-row" style="grid-template-columns:30px 1fr 60px 60px">
        <div>${["🥇","🥈","🥉","4️⃣"][i]}</div>
        <div>${m}</div>
        <div class="val">${s.toFixed(3)}</div>
        <div class="val" style="color:var(--text-dim)">${change}</div>
      </div>`;
  }).join("");

  // 벤치별 점수 변화 표
  const benchTbl = document.getElementById("compare-bench-table");
  benchTbl.innerHTML = CFG.models.map(m => {
    let row = `<tr><td>${m}</td>`;
    for (const b of CFG.benches) {
      const o = OLD[m]?.[b] || 0;
      const n = NEW[m]?.[b] || 0;
      const d = n - o;
      const arrow = d > 0.1 ? '<span class="b-up">↑</span>' :
                    d < -0.1 ? '<span class="b-down">↓</span>' : '<span class="b-zero">→</span>';
      row += `<td class="num">${o.toFixed(2)}→${n.toFixed(2)} ${arrow}</td>`;
    }
    return row + "</tr>";
  }).join("");

  // 변별력 표
  const spreadTbl = document.getElementById("spread-table");
  spreadTbl.innerHTML = CFG.benches.map(b => {
    const oldVals = CFG.models.map(m => OLD[m]?.[b] || 0);
    const newVals = CFG.models.map(m => NEW[m]?.[b] || 0);
    const oldSpread = Math.max(...oldVals) - Math.min(...oldVals);
    const newSpread = Math.max(...newVals) - Math.min(...newVals);
    const ratio = oldSpread > 0 ? (newSpread / oldSpread) : 0;
    let tag = '<span class="badge b-warn">→ 비슷</span>';
    if (ratio > 1.5) tag = '<span class="badge b-good">✅ 변별력↑</span>';
    else if (ratio < 0.8) tag = '<span class="badge b-bad">⚠️ 변별력↓</span>';
    return `<tr>
      <td>${CFG.bench_label[b]}</td>
      <td class="num">${oldSpread.toFixed(3)}</td>
      <td class="num">${newSpread.toFixed(3)}</td>
      <td class="num">${ratio.toFixed(2)}×</td>
      <td>${tag}</td>
    </tr>`;
  }).join("");
}

// ============================================================
// G-Eval 레이더 차트
// ============================================================
function renderRadar() {
  const dims = ["faithfulness", "relevance", "coherence", "conciseness"];
  const dimLabel = {faithfulness: "Faithfulness", relevance: "Relevance",
                    coherence: "Coherence", conciseness: "Conciseness"};
  const W = 360, H = 360, cx = W/2, cy = H/2, r = 130;
  let svg = `<svg width="${W}" height="${H}">`;

  // 동심원
  for (let v = 1; v <= 5; v++) {
    const rr = r * v / 5;
    const pts = dims.map((_, i) => {
      const a = -Math.PI/2 + i * Math.PI * 2 / 4;
      return `${cx + rr*Math.cos(a)},${cy + rr*Math.sin(a)}`;
    }).join(" ");
    svg += `<polygon points="${pts}" fill="none" stroke="${v === 5 ? '#666' : '#333'}" stroke-width="0.5" opacity="0.5"/>`;
  }
  // 축
  dims.forEach((d, i) => {
    const a = -Math.PI/2 + i * Math.PI * 2 / 4;
    const x = cx + r * Math.cos(a);
    const y = cy + r * Math.sin(a);
    svg += `<line x1="${cx}" y1="${cy}" x2="${x}" y2="${y}" stroke="#444" stroke-width="0.5" opacity="0.5"/>`;
    const lx = cx + (r+20)*Math.cos(a);
    const ly = cy + (r+20)*Math.sin(a);
    svg += `<text x="${lx}" y="${ly}" font-size="11" fill="currentColor" opacity="0.7" text-anchor="middle" dy="0.35em">${dimLabel[d]}</text>`;
  });

  // 각 모델 polygon
  CFG.models.forEach(model => {
    const color = MODEL_COLORS[model];
    const pts = dims.map((d, i) => {
      const v = GEVAL[model]?.[d] || 0;
      const rr = r * Math.min(5, v) / 5;
      const a = -Math.PI/2 + i * Math.PI * 2 / 4;
      return `${cx + rr*Math.cos(a)},${cy + rr*Math.sin(a)}`;
    }).join(" ");
    svg += `<polygon points="${pts}" fill="${color}" fill-opacity="0.10" stroke="${color}" stroke-width="2"/>`;
    // 점
    dims.forEach((d, i) => {
      const v = GEVAL[model]?.[d] || 0;
      const rr = r * Math.min(5, v) / 5;
      const a = -Math.PI/2 + i * Math.PI * 2 / 4;
      const px = cx + rr*Math.cos(a);
      const py = cy + rr*Math.sin(a);
      svg += `<circle cx="${px}" cy="${py}" r="3" fill="${color}"/>`;
    });
  });
  svg += `</svg>`;
  document.getElementById("radar").innerHTML = svg;

  // 표
  document.getElementById("geval-table").innerHTML = CFG.models.map(m => {
    const row = [m];
    for (const d of dims) row.push((GEVAL[m]?.[d] || 0).toFixed(2));
    return `<tr><td>${row[0]}</td><td class="num">${row[1]}</td><td class="num">${row[2]}</td><td class="num">${row[3]}</td><td class="num">${row[4]}</td></tr>`;
  }).join("");

  // 범례
  document.getElementById("radar-legend").innerHTML = CFG.models.map(m =>
    `<span style="color:${MODEL_COLORS[m]}">${m}</span>`
  ).join("");

  // 인사이트 — 차원별 1위·꼴찌
  const insights = dims.map(d => {
    const sorted = CFG.models.map(m => [m, GEVAL[m]?.[d] || 0]).sort((a, b) => b[1] - a[1]);
    return `<strong>${dimLabel[d]}</strong>: 🥇 ${sorted[0][0]} (${sorted[0][1].toFixed(2)}) / 🔻 ${sorted[3][0]} (${sorted[3][1].toFixed(2)})`;
  });
  document.getElementById("geval-insight").innerHTML = insights.join("<br>");
}

// ============================================================
// 벤치별 점수 바차트
// ============================================================
function renderBenchTab() {
  const html = CFG.benches.map(b => {
    const rows = CFG.models.map(m => ({ m, v: NEW[m]?.[b] || 0 })).sort((a, b) => b.v - a.v);
    const max = Math.max(...rows.map(r => r.v), 1);
    return `<h3>${CFG.bench_label[b]}</h3>` + rows.map(r => `
      <div class="bar-row">
        <div>${r.m}</div>
        <div class="bar-bg"><div class="bar-fg" style="width:${r.v/max*100}%"></div></div>
        <div class="val">${r.v.toFixed(3)}</div>
      </div>
    `).join("");
  }).join("");
  document.getElementById("bench-bars").innerHTML = html;
}

// ============================================================
// 초기 렌더
// ============================================================
// 기본 프리셋 "현재" 적용
const presetCurrent = CFG.presets["현재"];
Object.entries(presetCurrent).forEach(([b, v]) => {
  sliders[b].value = v;
  document.getElementById(`v-${b}`).textContent = v;
});

updateAll();
renderScenarios();
renderCompareTab();
renderRadar();
renderSpeedTab();
renderBenchTab();

// ============================================================
// 속도 탭 (한계 강조)
// ============================================================
function renderSpeedTab() {
  // 1. 속도 순위 요약 (judge 없는 벤치 기준)
  const summary = SPEED.summary;
  const ranked = Object.entries(summary).sort((a, b) => b[1].tps - a[1].tps);
  const maxTps = Math.max(...ranked.map(([_, v]) => v.tps), 1);
  document.getElementById("speed-summary").innerHTML = ranked.map(([m, v], i) => `
    <div class="bar-row" style="grid-template-columns:30px 200px 1fr 100px 80px">
      <div>${["🥇","🥈","🥉","4️⃣"][i] || ""}</div>
      <div>${m}</div>
      <div class="bar-bg"><div class="bar-fg" style="width:${v.tps/maxTps*100}%"></div></div>
      <div class="val">${v.tps.toFixed(1)} tok/s</div>
      <div class="val">${v.elapsed_avg.toFixed(2)}s/sample</div>
    </div>
  `).join("");

  // 2. 품질 + 속도 통합 표
  const memMap = {
    "Qwen3.6-35B-A3B-BF16": "70GB",
    "Qwen3.6-35B-A3B-FP8":  "35GB",
    "Qwen3-32B-AWQ":        "19GB",
    "Qwen3-30B-A3B-BF16":   "60GB",
  };
  const w = normalizeWeights(CFG.presets["현재"]);
  const qualityScores = computeWeightedScores(NEW, w);

  // 모델별 결합 평가
  function evalLabel(model) {
    const q = qualityScores[model];
    const s = summary[model]?.tps || 0;
    const qRank = Object.values(qualityScores).sort((a,b)=>b-a).indexOf(q);
    const sRank = ranked.findIndex(([m]) => m === model);
    const labels = [];
    if (qRank === 0) labels.push('<span class="badge b-gold">품질 1위</span>');
    if (sRank === 0) labels.push('<span class="badge b-gold">속도 1위</span>');
    if (qRank === 3) labels.push('<span class="badge b-bad">품질 4위</span>');
    if (sRank === 3) labels.push('<span class="badge b-bad">속도 4위 (A100)</span>');
    if (qRank <= 1 && sRank <= 1) labels.push('<span class="badge b-good">⭐ 종합 강력</span>');
    return labels.join(" ");
  }

  const tbl = document.getElementById("quality-speed-table");
  tbl.innerHTML = CFG.models.map(m => {
    const q = qualityScores[m];
    const s = summary[m]?.tps || 0;
    return `<tr>
      <td>${m}</td>
      <td class="num">${q.toFixed(3)}</td>
      <td class="num">${s.toFixed(1)}</td>
      <td class="num">${memMap[m] || "-"}</td>
      <td>${evalLabel(m)}</td>
    </tr>`;
  }).join("");

  // 3. 벤치별 sample 시간 표시
  const perBenchDiv = document.getElementById("speed-per-bench");
  let html = "";
  for (const bench of CFG.benches) {
    const judgeMark = JUDGE_BENCHES.includes(bench) ?
      ' <span class="muted" style="color:var(--accent-2)">(⚠️ judge 호출 시간 포함)</span>' :
      ' <span class="muted">(judge 없음)</span>';
    html += `<h3>${CFG.bench_label[bench]}${judgeMark}</h3>`;
    const rows = CFG.models.map(m => ({
      m, ...((SPEED.per_bench[m] || {})[bench] || {elapsed_avg: 0, tokens_out_avg: 0, approx_tps: 0})
    })).sort((a, b) => a.elapsed_avg - b.elapsed_avg);
    const maxE = Math.max(...rows.map(r => r.elapsed_avg), 1);
    html += rows.map(r => `
      <div class="bar-row" style="grid-template-columns:200px 1fr 80px 80px">
        <div>${r.m}</div>
        <div class="bar-bg"><div class="bar-fg" style="width:${r.elapsed_avg/maxE*100}%"></div></div>
        <div class="val">${r.elapsed_avg.toFixed(2)}s</div>
        <div class="val">${r.approx_tps.toFixed(1)} tok/s</div>
      </div>
    `).join("");
  }
  perBenchDiv.innerHTML = html;
}
</script>
</body>
</html>
"""

html_out = (HTML
    .replace("__CONFIG_JSON__", safe_embed(config_json))
    .replace("__OLD_SCORES_JSON__", safe_embed(old_scores_json))
    .replace("__NEW_SCORES_JSON__", safe_embed(new_scores_json))
    .replace("__GEVAL_JSON__", safe_embed(geval_json))
    .replace("__SPEED_JSON__", safe_embed(speed_json)))

OUT.write_text(html_out, encoding="utf-8")
print(f"✅ {OUT}")
print(f"   크기: {OUT.stat().st_size / 1024:.1f} KB")
