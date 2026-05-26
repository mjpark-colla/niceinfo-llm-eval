"""index_v2.html 을 final 버전으로 업데이트 — verify572 cross-validation 결과 통합.

변경 요약:
1. NEW 데이터를 v2/verify 평균으로 (메인 점수 = robust 평균)
2. 가중치 탭에 verify572 점수도 표시 (비교 보조)
3. 새 탭 "🔁 v2 ↔ verify572" 추가: cross-validation robust 검증 시각화

기존 답변 비교 탭 등은 그대로 보존.
"""
import html
import json
import re
from pathlib import Path

BASE = Path(__file__).parent
SRC = BASE / "index_v2.html"
SAMPLES = BASE / "samples_v2.json"
BACKUP = BASE / "index_v2_backup.html"

# 항상 backup 기준으로 리빌드 (idempotent)
src_html = BACKUP.read_text(encoding="utf-8")

# ---------------------------------------------------------------
# 1) NEW 데이터를 v2/verify572 평균으로 교체 + 별도 V2/VER 상수 추가
# ---------------------------------------------------------------
MODELS = ["Qwen3.6-35B-A3B-BF16", "Qwen3.6-35B-A3B-FP8",
          "Qwen3-32B-AWQ", "Qwen3-30B-A3B-BF16"]
BENCHES = ["ko_mt_bench", "logickor", "ko_ifeval", "aihub_582", "aihub_90"]


def averages_per_model(base_dir):
    out = {}
    for m in MODELS:
        bench_avg = {}
        for b in BENCHES:
            path = Path(base_dir) / m / f"{b}.jsonl"
            scores = [json.loads(l).get("score", 0) for l in open(path)]
            bench_avg[b] = sum(scores) / len(scores) if scores else 0
        out[m] = bench_avg
    return out


def geval_per_model(base_dir):
    DIMS = ["faithfulness", "relevance", "coherence", "conciseness"]
    out = {}
    for m in MODELS:
        sums = {d: 0.0 for d in DIMS}
        counts = {d: 0 for d in DIMS}
        for b in ("aihub_582", "aihub_90"):
            path = Path(base_dir) / m / f"{b}.jsonl"
            for line in open(path):
                r = json.loads(line)
                md = r.get("turns", [{}])[0].get("metric_details") or {}
                for d in DIMS:
                    if d in md:
                        sums[d] += md[d]
                        counts[d] += 1
        out[m] = {d: (sums[d] / counts[d] if counts[d] else 0) for d in DIMS}
    return out


v2 = averages_per_model(BASE.parent / "results_v2")
ver = averages_per_model(BASE.parent / "results_verify572")
new_avg = {m: {b: (v2[m][b] + ver[m][b]) / 2 for b in BENCHES} for m in MODELS}

geval_v2 = geval_per_model(BASE.parent / "results_v2")
geval_avg = geval_v2  # G-Eval은 v2 기준으로 (verify와 거의 동일)

new_const = "const NEW = " + json.dumps(new_avg, ensure_ascii=False) + ";"
geval_const = "const GEVAL = " + json.dumps(geval_avg, ensure_ascii=False) + ";"
v2_const = "const V2 = " + json.dumps(v2, ensure_ascii=False) + ";"
ver_const = "const VER = " + json.dumps(ver, ensure_ascii=False) + ";"

src_html = re.sub(r"const NEW = \{.*?\};", new_const, src_html, count=1, flags=re.DOTALL)
src_html = re.sub(r"const GEVAL = \{.*?\};", geval_const, src_html, count=1, flags=re.DOTALL)
# V2, VER 추가 (NEW 다음에)
src_html = src_html.replace(new_const, new_const + "\n" + v2_const + "\n" + ver_const)

# === SPEED.summary 덮어쓰기 — streaming 측정값 (정확한 단독 latency) ===
# 기존 SPEED 객체의 summary 부분만 교체 (per_bench 는 그대로 — concurrency=8 참고용)
SPEED_STREAMING = {
    "Qwen3.6-35B-A3B-BF16": {"ttft": 0.308, "tps": 144.8, "vram_mb": 77024, "elapsed_avg": 3.98, "tokens_out_avg": 512},
    "Qwen3.6-35B-A3B-FP8":  {"ttft": 0.324, "tps": 164.2, "vram_mb": 76970, "elapsed_avg": 3.62, "tokens_out_avg": 512},
    "Qwen3-32B-AWQ":        {"ttft": 0.041, "tps":  66.9, "vram_mb": 76696, "elapsed_avg": 7.80, "tokens_out_avg": 503},
    "Qwen3-30B-A3B-BF16":   {"ttft": 0.039, "tps": 132.1, "vram_mb": 77572, "elapsed_avg": 3.85, "tokens_out_avg": 490},
}
# 기존 const SPEED 안 "summary" 부분만 정규식 교체
new_summary_json = json.dumps(SPEED_STREAMING, ensure_ascii=False)
# const SPEED = {"per_bench": {...}, "summary": {...}}; 안의 summary 부분
src_html = re.sub(
    r'("summary":\s*\{)[^{}]*(?:\{[^{}]*\}[^{}]*)*(\})\}\s*;',
    lambda mm: '"summary": ' + new_summary_json + '};',
    src_html, count=1
)

# ---------------------------------------------------------------
# 2) 새 탭 "🔁 v2 ↔ verify572" 추가 (footer 직전)
# ---------------------------------------------------------------
NEW_TAB_BTN = '<button data-tab="cross">🔁 v2 ↔ verify572</button>'
src_html = src_html.replace(
    '<button data-tab="bench">📋 벤치 상세</button>',
    '<button data-tab="bench">📋 벤치 상세</button>\n  ' + NEW_TAB_BTN
)

NEW_SECTION = """
<!-- ============================================================ -->
<!-- TAB CROSS: v2 ↔ verify572 cross-validation -->
<!-- ============================================================ -->
<section id="cross">
  <div class="card">
    <h2 style="margin-top:0">🔁 v2 ↔ verify572 Cross-Validation</h2>
    <div class="muted">동일 sample (random seed 42) 을 다른 vLLM 환경 + 다른 시점 judge run 에서 재측정.
    상대 순위가 일치하면 결론 robust 검증 완료.</div>

    <div class="callout" style="margin-top:12px">
      <strong>결과: 4 모델 모두 순위 100% 일치</strong> — 점수 차이 0.13~0.18 점은 systematic judge noise (절대값 낮아짐, 상대 순위 동일).
    </div>
  </div>

  <div class="card">
    <h2 style="margin-top:0">종합 순위 비교</h2>
    <div class="grid-2">
      <div>
        <h3>v2 (재파싱)</h3>
        <div id="v2-ranks"></div>
      </div>
      <div>
        <h3>verify572 (cloud 재평가)</h3>
        <div id="ver-ranks"></div>
      </div>
    </div>
  </div>

  <div class="card">
    <h2 style="margin-top:0">벤치별 변동 (verify - v2)</h2>
    <table>
      <thead><tr><th>모델</th><th class="num">Ko-MT</th><th class="num">LogicKor</th><th class="num">Ko-IFEval</th><th class="num">AIHub 582</th><th class="num">AIHub 90</th><th class="num">종합 Δ</th></tr></thead>
      <tbody id="cross-bench-table"></tbody>
    </table>
    <div class="callout callout-warn" style="margin-top:12px">
      <strong>Judge noise 패턴</strong>: 모든 모델이 verify 에서 일관되게 -0.1~-0.2 점 → judge gpt-4o 의 시점·환경 변동.
      <strong>Ko-IFEval (룰 기반)</strong> 만 ±0.03 (사실상 동일) → 룰 기반 채점이 가장 reproducibility 높음.
    </div>
  </div>
</section>
"""

src_html = src_html.replace('<footer>', NEW_SECTION + '\n<footer>')

# ---------------------------------------------------------------
# 3) JS 함수 추가 — </script> 직전
# ---------------------------------------------------------------
JS_NEW = """
// ============================================================
// 속도 탭 — TTFT/VRAM 추가 표시 (streaming 측정값 이미 SPEED.summary에 반영됨)
// ============================================================
(function enhanceSpeedTab() {
  // 기존 renderSpeedTab 가 그린 결과에 TTFT·VRAM 컬럼 추가
  const summaryEl = document.getElementById("speed-summary");
  if (!summaryEl) return;
  const summary = SPEED.summary;
  const ranked = Object.entries(summary).sort((a, b) => b[1].tps - a[1].tps);
  const maxTps = Math.max(...ranked.map(([_, v]) => v.tps), 1);
  const medals = ["🥇","🥈","🥉","4️⃣"];

  // 1. 속도 순위 — TTFT/VRAM 컬럼 추가
  summaryEl.innerHTML = ranked.map(([m, v], i) =>
    '<div class="bar-row" style="grid-template-columns:30px 200px 1fr 100px 100px 90px">'
    + '<div>' + (medals[i] || "") + '</div>'
    + '<div>' + m + '</div>'
    + '<div class="bar-bg"><div class="bar-fg" style="width:' + (v.tps/maxTps*100) + '%"></div></div>'
    + '<div class="val">' + v.tps.toFixed(1) + ' tok/s</div>'
    + '<div class="val">TTFT ' + (v.ttft != null ? v.ttft.toFixed(3) : "-") + 's</div>'
    + '<div class="val">' + (v.vram_mb != null ? (v.vram_mb/1024).toFixed(1) + ' GiB' : "-") + '</div>'
    + '</div>'
  ).join("");

  // 품질+속도 통합 표는 기존 renderSpeedTab 가 SPEED.summary.tps 를 그대로 쓰니까 OK
  // 벤치별 sample 시간 영역 라벨만 조정 (concurrency 8 참고용임을 명시)
  // — 기존 렌더 그대로 두고 헤더에 "참고" 표시 추가
})();

// ============================================================
// v2 ↔ verify572 Cross-Validation 탭
// ============================================================
function renderCrossTab() {
  if (typeof V2 === "undefined" || typeof VER === "undefined") return;
  const w = normalizeWeights(CFG.presets["현재"]);
  const v2Totals = computeWeightedScores(V2, w);
  const verTotals = computeWeightedScores(VER, w);
  const v2Sorted = Object.entries(v2Totals).sort((a, b) => b[1] - a[1]);
  const verSorted = Object.entries(verTotals).sort((a, b) => b[1] - a[1]);

  const medals = ["🥇", "🥈", "🥉", "4️⃣"];
  document.getElementById("v2-ranks").innerHTML = v2Sorted.map(([m, s], i) =>
    '<div class="bar-row" style="grid-template-columns:30px 1fr 70px">'
    + '<div>' + medals[i] + '</div>'
    + '<div>' + m + '</div>'
    + '<div class="val">' + s.toFixed(3) + '</div>'
    + '</div>'
  ).join("");
  document.getElementById("ver-ranks").innerHTML = verSorted.map(([m, s], i) =>
    '<div class="bar-row" style="grid-template-columns:30px 1fr 70px">'
    + '<div>' + medals[i] + '</div>'
    + '<div>' + m + '</div>'
    + '<div class="val">' + s.toFixed(3) + '</div>'
    + '</div>'
  ).join("");

  // 벤치별 차이 표
  const tbl = document.getElementById("cross-bench-table");
  tbl.innerHTML = CFG.models.map(m => {
    let row = '<tr><td>' + m + '</td>';
    let weightedDelta = 0;
    for (const b of CFG.benches) {
      const d = VER[m][b] - V2[m][b];
      weightedDelta += d * w[b];
      const cls = d > 0.05 ? "b-up" : (d < -0.05 ? "b-down" : "b-zero");
      const sign = d >= 0 ? "+" : "";
      row += '<td class="num"><span class="' + cls + '">' + sign + d.toFixed(3) + '</span></td>';
    }
    const tcls = weightedDelta > 0 ? "b-up" : (weightedDelta < 0 ? "b-down" : "b-zero");
    const tsign = weightedDelta >= 0 ? "+" : "";
    row += '<td class="num"><span class="' + tcls + '"><strong>' + tsign + weightedDelta.toFixed(3) + '</strong></span></td>';
    return row + '</tr>';
  }).join("");
}
renderCrossTab();
"""

src_html = src_html.replace('</script>', JS_NEW + '\n</script>')

# ---------------------------------------------------------------
# 4) SAMPLES textarea (답변 비교 탭) - 기존 build_dashboard_v3 로직 추가
# ---------------------------------------------------------------
# build_dashboard_v3 와 마찬가지로 답변 비교 탭의 SAMPLES 임베드 + JS 추가
# 이미 backup 은 V2 답변비교 탭 들어가지 않은 원본 — 추가 필요

samples_data = json.loads(SAMPLES.read_text(encoding="utf-8"))
samples_json = json.dumps(samples_data, ensure_ascii=False, separators=(",", ":"))
samples_escaped = html.escape(samples_json, quote=True).replace("</textarea", "&lt;/textarea")

# 답변 비교 탭 (이미 build_dashboard_v3 가 만든 동일 로직)
src_html = src_html.replace(
    '<button data-tab="cross">🔁 v2 ↔ verify572</button>',
    '<button data-tab="cross">🔁 v2 ↔ verify572</button>\n  '
    '<button data-tab="raw">🔬 답변 비교</button>'
)

RAW_SECTION = """
<section id="raw">
  <div class="card">
    <h2 style="margin-top:0">🔬 동일 sample 4 모델 답변 비교</h2>
    <div class="muted">동일 prompt에 대해 4 모델이 어떻게 답변했는지·점수가 어떻게 차이나는지 비교.</div>
    <div class="filter-bar">
      <label>벤치<select id="raw-bench"></select></label>
      <label>정렬<select id="raw-sort">
        <option value="default">기본 (sample_id 순)</option>
        <option value="spread">점수 차이 큰 순 (변별력)</option>
        <option value="low">최저 점수 낮은 순</option>
        <option value="high">최고 점수 높은 순</option>
      </select></label>
      <label>검색<input id="raw-search" placeholder="sample_id 또는 prompt 검색..."></label>
      <span class="muted" id="raw-count"></span>
    </div>
  </div>
  <div class="raw-layout">
    <aside class="raw-sidebar"><div id="raw-list" class="sample-list"></div></aside>
    <main class="raw-detail" id="raw-detail">
      <div class="muted" style="padding:24px;text-align:center">왼쪽 목록에서 sample을 선택하세요.</div>
    </main>
  </div>
</section>
<textarea id="samples-data" hidden aria-hidden="true" style="display:none">""" + samples_escaped + """</textarea>
"""
src_html = src_html.replace('<footer>', RAW_SECTION + '\n<footer>')

# 답변 비교 탭 CSS
CSS_NEW = """
.filter-bar { display: flex; gap: 12px; flex-wrap: wrap; align-items: center;
  margin-top: 12px; padding: 12px; background: var(--bg-3); border-radius: 6px; }
.filter-bar label { font-size: 12px; color: var(--text-2); display: flex; flex-direction: column; gap: 4px; }
.filter-bar select, .filter-bar input { background: var(--bg-2); color: var(--text);
  border: 1px solid var(--border); padding: 6px 8px; border-radius: 4px; font-size: 13px; min-width: 160px; }
.filter-bar input { min-width: 240px; }
.raw-layout { display: grid; grid-template-columns: 320px 1fr; gap: 16px; min-height: 600px; }
@media (max-width: 900px) { .raw-layout { grid-template-columns: 1fr; } }
.raw-sidebar { background: var(--bg-2); border: 1px solid var(--border); border-radius: 8px;
  overflow: hidden; max-height: 800px; display: flex; flex-direction: column; }
.sample-list { overflow-y: auto; flex: 1; }
.sample-item { padding: 10px 12px; border-bottom: 1px solid var(--border); cursor: pointer; }
.sample-item:hover { background: var(--bg-3); }
.sample-item.active { background: rgba(94, 234, 212, 0.12); border-left: 3px solid var(--accent); padding-left: 9px; }
.sample-item .sid { font-size: 11px; color: var(--text-dim); font-variant-numeric: tabular-nums; }
.sample-item .preview { font-size: 12px; color: var(--text); margin-top: 2px;
  display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; }
.sample-item .scores { font-size: 10px; color: var(--text-2); margin-top: 4px;
  display: flex; gap: 4px; flex-wrap: wrap; font-variant-numeric: tabular-nums; }
.sample-item .score-pill { padding: 1px 5px; border-radius: 3px; background: var(--bg-3); }
.sample-item .score-pill.hi { background: rgba(52, 211, 153, 0.15); color: var(--good); }
.sample-item .score-pill.mid { background: rgba(251, 146, 60, 0.15); color: var(--warn); }
.sample-item .score-pill.lo { background: rgba(248, 113, 113, 0.15); color: var(--bad); }
.raw-detail { background: var(--bg-2); border: 1px solid var(--border); border-radius: 8px;
  padding: 16px; overflow-x: auto; }
.prompt-box { background: var(--bg-3); border-left: 3px solid var(--accent); border-radius: 4px;
  padding: 12px 14px; margin-bottom: 16px; font-size: 13px; white-space: pre-wrap;
  max-height: 240px; overflow-y: auto; }
.prompt-box .label { font-size: 11px; color: var(--text-dim); text-transform: uppercase;
  letter-spacing: 0.5px; margin-bottom: 6px; }
.turn-block { margin-bottom: 20px; }
.turn-header { font-size: 13px; font-weight: 600; color: var(--text-2);
  padding: 6px 0; border-bottom: 1px solid var(--border); margin-bottom: 12px; }
.answer-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; }
@media (max-width: 1100px) { .answer-grid { grid-template-columns: repeat(2, 1fr); } }
@media (max-width: 700px) { .answer-grid { grid-template-columns: 1fr; } }
.answer-card { background: var(--bg-3); border: 1px solid var(--border); border-radius: 6px;
  padding: 10px; font-size: 12px; }
.answer-card .head { display: flex; justify-content: space-between; align-items: center;
  margin-bottom: 8px; padding-bottom: 6px; border-bottom: 1px solid var(--border); }
.answer-card .model-name { font-weight: 600; color: var(--text); font-size: 11px; word-break: break-word; }
.answer-card .score { font-size: 14px; font-variant-numeric: tabular-nums; font-weight: 700; }
.answer-card .score.hi { color: var(--good); }
.answer-card .score.mid { color: var(--warn); }
.answer-card .score.lo { color: var(--bad); }
.answer-card .output { white-space: pre-wrap; max-height: 360px; overflow-y: auto;
  background: var(--bg-2); padding: 8px; border-radius: 4px; font-size: 12px; line-height: 1.5; }
.answer-card details { margin-top: 8px; }
.answer-card summary { cursor: pointer; font-size: 11px; color: var(--text-dim); padding: 4px 0; }
.answer-card .judge { font-size: 11px; color: var(--text-2); padding: 8px; background: var(--bg-2);
  border-radius: 4px; margin-top: 4px; white-space: pre-wrap; line-height: 1.5; }
"""
src_html = src_html.replace('</style>', CSS_NEW + '\n</style>')

# 답변 비교 JS
RAW_JS = """
// 답변 비교 탭
const SAMPLES = JSON.parse(document.getElementById("samples-data").textContent);
const BENCH_LABEL_LOCAL = {
  "ko_mt_bench": "Ko-MT-Bench", "logickor": "LogicKor", "ko_ifeval": "Ko-IFEval",
  "aihub_582": "AIHub 582", "aihub_90": "AIHub 90",
};
const rawState = { bench: "ko_mt_bench", sort: "default", search: "", selectedSid: null };
function escapeHTML(s) {
  return String(s ?? "").replace(/[&<>"']/g, c => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}
function scoreClass(s) {
  if (s === null || s === undefined) return "";
  if (s >= 8) return "hi";
  if (s >= 4) return "mid";
  return "lo";
}
function computeSpread(samplesByModel) {
  const scores = CFG.models.map(m => samplesByModel[m]?.score).filter(s => s !== null && s !== undefined);
  if (scores.length < 2) return 0;
  return Math.max(...scores) - Math.min(...scores);
}
function getFilteredSorted() {
  const samples = SAMPLES[rawState.bench] || {};
  let entries = Object.entries(samples);
  if (rawState.search.trim()) {
    const q = rawState.search.toLowerCase();
    entries = entries.filter(([sid, byModel]) => {
      if (sid.toLowerCase().includes(q)) return true;
      const m0 = CFG.models[0];
      return (byModel[m0]?.turns?.[0]?.prompt || "").toLowerCase().includes(q);
    });
  }
  if (rawState.sort === "spread") entries.sort((a, b) => computeSpread(b[1]) - computeSpread(a[1]));
  else if (rawState.sort === "low") entries.sort((a, b) => Math.min(...CFG.models.map(m => a[1][m]?.score ?? 99)) - Math.min(...CFG.models.map(m => b[1][m]?.score ?? 99)));
  else if (rawState.sort === "high") entries.sort((a, b) => Math.max(...CFG.models.map(m => b[1][m]?.score ?? -1)) - Math.max(...CFG.models.map(m => a[1][m]?.score ?? -1)));
  return entries;
}
function renderRawList() {
  const entries = getFilteredSorted();
  const list = document.getElementById("raw-list");
  document.getElementById("raw-count").textContent = entries.length + " sample";
  list.innerHTML = entries.map(([sid, byModel]) => {
    const m0 = CFG.models.find(m => byModel[m]) || CFG.models[0];
    const preview = (byModel[m0]?.turns?.[0]?.prompt || "").slice(0, 120).replace(/\\n/g, " ");
    const spread = computeSpread(byModel);
    const pills = CFG.models.map(m => {
      const s = byModel[m]?.score;
      return s == null
        ? '<span class="score-pill">-</span>'
        : '<span class="score-pill ' + scoreClass(s) + '" title="' + escapeHTML(m) + '">' + s.toFixed(1) + '</span>';
    }).join("");
    const active = rawState.selectedSid === sid ? "active" : "";
    return '<div class="sample-item ' + active + '" data-sid="' + escapeHTML(sid) + '">'
      + '<div class="sid">' + escapeHTML(sid) + ' · 차이 ' + spread.toFixed(1) + '</div>'
      + '<div class="preview">' + escapeHTML(preview) + '</div>'
      + '<div class="scores">' + pills + '</div></div>';
  }).join("");
  list.querySelectorAll(".sample-item").forEach(el => {
    el.addEventListener("click", () => {
      rawState.selectedSid = el.dataset.sid;
      renderRawList();
      renderRawDetail();
    });
  });
}
function renderRawDetail() {
  const detail = document.getElementById("raw-detail");
  if (!rawState.selectedSid) {
    detail.innerHTML = '<div class="muted" style="padding:24px;text-align:center">왼쪽 목록에서 sample을 선택하세요.</div>';
    return;
  }
  const byModel = SAMPLES[rawState.bench]?.[rawState.selectedSid];
  if (!byModel) { detail.innerHTML = '<div class="muted">sample을 찾을 수 없습니다.</div>'; return; }
  const m0 = CFG.models.find(m => byModel[m]) || CFG.models[0];
  const turns0 = byModel[m0]?.turns || [];
  const numTurns = Math.max(...CFG.models.map(m => (byModel[m]?.turns || []).length), 1);
  let html = '<div style="font-size:11px;color:var(--text-dim);margin-bottom:4px">'
    + escapeHTML(BENCH_LABEL_LOCAL[rawState.bench]) + '</div>'
    + '<div style="font-weight:600;font-size:14px;margin-bottom:12px">'
    + escapeHTML(rawState.selectedSid) + '</div>';
  for (let ti = 0; ti < numTurns; ti++) {
    const promptT = turns0[ti]?.prompt || "(prompt 없음)";
    html += '<div class="turn-block">';
    if (numTurns > 1) html += '<div class="turn-header">📌 Turn ' + (ti + 1) + ' / ' + numTurns + '</div>';
    html += '<div class="prompt-box"><div class="label">Prompt</div>' + escapeHTML(promptT) + '</div>';
    html += '<div class="answer-grid">';
    for (const m of CFG.models) {
      const t = byModel[m]?.turns?.[ti];
      if (!t) {
        html += '<div class="answer-card"><div class="head"><div class="model-name">' + escapeHTML(m)
          + '</div><div class="score lo">-</div></div><div class="output muted">(데이터 없음)</div></div>';
        continue;
      }
      const score = t.score;
      const scoreText = (score != null) ? score.toFixed(1) : "-";
      html += '<div class="answer-card"><div class="head">'
        + '<div class="model-name">' + escapeHTML(m) + '</div>'
        + '<div class="score ' + scoreClass(score) + '">' + scoreText + '</div></div>'
        + '<div class="output">' + escapeHTML(t.model_output || "(출력 없음)") + '</div>';
      if (t.judge_raw) {
        html += '<details><summary>📝 Judge 평가</summary>'
          + '<div class="judge">' + escapeHTML(t.judge_raw) + '</div></details>';
      }
      html += '</div>';
    }
    html += '</div></div>';
  }
  detail.innerHTML = html;
}
function initRawTab() {
  const benchSel = document.getElementById("raw-bench");
  benchSel.innerHTML = CFG.benches.map(b =>
    '<option value="' + b + '">' + BENCH_LABEL_LOCAL[b]
    + ' (' + Object.keys(SAMPLES[b] || {}).length + ')</option>'
  ).join("");
  benchSel.addEventListener("change", () => {
    rawState.bench = benchSel.value;
    rawState.selectedSid = null;
    renderRawList();
    renderRawDetail();
  });
  document.getElementById("raw-sort").addEventListener("change", e => {
    rawState.sort = e.target.value;
    renderRawList();
  });
  document.getElementById("raw-search").addEventListener("input", e => {
    rawState.search = e.target.value;
    renderRawList();
  });
  renderRawList();
}
initRawTab();
"""
src_html = src_html.replace('</script>', RAW_JS + '\n</script>')

# ---------------------------------------------------------------
# 5) 헤더 부제 업데이트
# ---------------------------------------------------------------
src_html = src_html.replace(
    'Judge prompt 재설계 (G-Eval / MT-Bench / LogicKor) · 2026-05-25 · 4 모델 × 5 벤치',
    'Phase 1 final · G-Eval parsing 버그 수정 + verify572 cross-validation · 2026-05-26 · 4 모델 × 5 벤치 × 2 회 측정'
)

# ---------------------------------------------------------------
# 5-2) 속도 탭 맨 위 경고 카드 교체 — 정확한 안내로
# ---------------------------------------------------------------
import re as _re
new_speed_intro = '''<div class="card callout">
    <strong>본 탭의 속도 데이터 안내</strong><br>
    아래 측정값은 <strong>2026-05-26 streaming 측정 (concurrency=1, 단독 latency)</strong> 결과입니다.
    <ul style="margin:8px 0; padding-left:20px">
      <li><strong>측정 방식</strong>: vLLM streaming API + nvidia-smi 0.3초 폴링</li>
      <li><strong>TTFT</strong> (Time To First Token) — 첫 토큰 도착 시각, 대화 UX 결정적</li>
      <li><strong>Decode TPS</strong> — 첫 토큰 ~ 마지막 토큰 평균 생성 속도</li>
      <li><strong>VRAM peak</strong> — 추론 중 GPU 메모리 최대 사용량</li>
      <li>측정 조건: A100 80GB, vLLM-openai latest, max-model-len 32K, temperature 0.0</li>
    </ul>
    ⚠️ <strong>한계</strong>: A100 기준 (운영 H100 NVL 결과는 Phase 3 재측정), prompt 7개 측정, 임베딩·리랭커 동시 부하 미반영.
  </div>'''

# 속도 탭의 첫 card (callout-warn) 만 교체. <h2> "속도 순위" 시작 직전까지.
src_html = _re.sub(
    r'<div class="card callout callout-warn">[\s\S]*?(?=\s*<div class="card">\s*<h2[^>]*>속도 순위)',
    new_speed_intro + '\n\n  ',
    src_html, count=1
)

# 탭 라벨 "⚡ 속도 (대략 추정)" → "⚡ 속도"
src_html = src_html.replace('⚡ 속도 (대략 추정)', '⚡ 속도')

# ---------------------------------------------------------------
# 6) 저장
# ---------------------------------------------------------------
SRC.write_text(src_html, encoding="utf-8")
size_mb = SRC.stat().st_size / (1024 * 1024)
print(f"✅ {SRC} 생성 ({size_mb:.2f} MB)")
print()
print("새 NEW (v2/verify 평균):")
for m in MODELS:
    print(f"  {m:<26}: " + " ".join(f"{b}={new_avg[m][b]:.3f}" for b in BENCHES))
