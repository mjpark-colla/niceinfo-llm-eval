"""평가 결과 → self-contained HTML 대시보드 생성.

생성물: reports/index.html
- 종합 점수 표 + BF16/FP8 비교 + 벤치별 차트
- Raw 답변 탐색기 (모델/벤치/sample/점수 필터)
- 외부 의존성 0 (브라우저에서 더블클릭으로 열기)
"""
import json
from pathlib import Path

ROOT = Path("/Users/minji/Documents/PolarPulse/niceinfo")
RESULTS = ROOT / "results"
OUT = ROOT / "reports" / "index.html"

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
    "aihub_582": "AIHub 582 (한국어 요약)",
    "aihub_90": "AIHub 90 (논문/특허 요약)",
}
BENCH_WEIGHT = {"ko_mt_bench": 0.25, "logickor": 0.15, "ko_ifeval": 0.20, "aihub_582": 0.20, "aihub_90": 0.20}

summary = json.loads((RESULTS / "summary.json").read_text())

data = {}
for model in MODELS_ORDER:
    model_dir = RESULTS / model
    if not model_dir.exists():
        continue
    data[model] = {}
    for bench in BENCH_ORDER:
        f = model_dir / f"{bench}.jsonl"
        if not f.exists():
            data[model][bench] = []
            continue
        samples = []
        for line in f.read_text().splitlines():
            if line.strip():
                samples.append(json.loads(line))
        data[model][bench] = samples

data_json = json.dumps(data, ensure_ascii=False)
summary_json = json.dumps(summary, ensure_ascii=False)
config_json = json.dumps({
    "models_order": MODELS_ORDER,
    "bench_order": BENCH_ORDER,
    "bench_label": BENCH_LABEL,
    "bench_weight": BENCH_WEIGHT,
}, ensure_ascii=False)

HTML = """<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8">
<title>niceinfo Phase 1 — 한국어 텍스트 생성 능력 평가</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root {
  --bg: #0f1419;
  --bg-2: #1a1f2e;
  --bg-3: #232a3d;
  --text: #e6e6e6;
  --text-2: #a8b2c1;
  --text-dim: #6b7280;
  --accent: #5eead4;
  --accent-2: #fbbf24;
  --good: #34d399;
  --warn: #fb923c;
  --bad: #f87171;
  --border: #2d3748;
  --gold: #fcd34d;
  --silver: #d1d5db;
  --bronze: #d97706;
}
@media (prefers-color-scheme: light) {
  :root {
    --bg: #fafaf9;
    --bg-2: #ffffff;
    --bg-3: #f3f4f6;
    --text: #1f2937;
    --text-2: #4b5563;
    --text-dim: #9ca3af;
    --accent: #0d9488;
    --accent-2: #d97706;
    --border: #e5e7eb;
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
nav { display: flex; gap: 4px; margin: 20px 0; border-bottom: 1px solid var(--border); }
nav button { background: none; border: none; color: var(--text-2); padding: 10px 16px; cursor: pointer;
  font-size: 13px; font-weight: 500; border-bottom: 2px solid transparent; transition: all 0.15s; }
nav button:hover { color: var(--text); }
nav button.active { color: var(--accent); border-bottom-color: var(--accent); }
section { display: none; }
section.active { display: block; }
h2 { font-size: 18px; margin: 28px 0 12px 0; font-weight: 600; }
h2:first-child { margin-top: 0; }
h3 { font-size: 15px; margin: 20px 0 8px 0; font-weight: 600; color: var(--text-2); }
.card { background: var(--bg-2); border: 1px solid var(--border); border-radius: 8px; padding: 16px;
  margin-bottom: 16px; }
.grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
@media (max-width: 800px) { .grid-2 { grid-template-columns: 1fr; } }
table { width: 100%; border-collapse: collapse; font-size: 13px; }
th, td { text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--border); }
th { font-weight: 600; color: var(--text-2); background: var(--bg-3); }
tr:last-child td { border-bottom: none; }
tr:hover td { background: var(--bg-3); }
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
.metric-row { display: flex; gap: 16px; margin: 12px 0; flex-wrap: wrap; }
.metric { background: var(--bg-3); padding: 12px 16px; border-radius: 6px; min-width: 140px; }
.metric .label { font-size: 11px; color: var(--text-2); text-transform: uppercase; letter-spacing: 0.5px; }
.metric .value { font-size: 22px; font-weight: 700; margin-top: 4px; }
.bar-chart { padding: 4px 0; }
.bar-row { display: grid; grid-template-columns: 200px 1fr 60px; gap: 12px; align-items: center;
  padding: 5px 0; font-size: 13px; }
.bar-bg { background: var(--bg-3); border-radius: 3px; height: 18px; overflow: hidden; position: relative; }
.bar-fg { background: linear-gradient(90deg, var(--accent), var(--accent-2)); height: 100%;
  border-radius: 3px; transition: width 0.4s ease; }
.bar-row .val { text-align: right; color: var(--text-2); font-variant-numeric: tabular-nums; }
.controls { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 12px; align-items: center;
  padding: 12px; background: var(--bg-2); border-radius: 8px; border: 1px solid var(--border); }
.controls label { font-size: 12px; color: var(--text-2); display: flex; flex-direction: column; gap: 4px; }
.controls select, .controls input { background: var(--bg-3); color: var(--text); border: 1px solid var(--border);
  padding: 6px 8px; border-radius: 4px; font-size: 13px; min-width: 140px; }
.controls .count { margin-left: auto; color: var(--text-dim); font-size: 12px; }
.sample-list { display: flex; flex-direction: column; gap: 8px; }
.sample { background: var(--bg-2); border: 1px solid var(--border); border-radius: 6px; overflow: hidden; }
.sample-head { padding: 10px 14px; cursor: pointer; display: flex; align-items: center;
  gap: 12px; user-select: none; }
.sample-head:hover { background: var(--bg-3); }
.sample-head .id { color: var(--text-dim); font-size: 12px; min-width: 60px; }
.sample-head .meta { color: var(--text-2); font-size: 12px; flex: 1;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.sample-head .score-pill { padding: 3px 10px; border-radius: 12px; font-size: 12px; font-weight: 600;
  font-variant-numeric: tabular-nums; }
.sample-body { display: none; padding: 0 14px 14px 14px; border-top: 1px solid var(--border); }
.sample.open .sample-body { display: block; }
.turn-block { margin-top: 14px; }
.turn-block .turn-label { font-size: 11px; color: var(--text-dim); text-transform: uppercase;
  letter-spacing: 0.5px; margin-bottom: 6px; }
.text-box { background: var(--bg-3); padding: 10px 12px; border-radius: 4px; white-space: pre-wrap;
  word-break: break-word; font-size: 13px; line-height: 1.6; max-height: 400px; overflow-y: auto;
  border-left: 3px solid var(--border); }
.text-box.prompt { border-left-color: var(--accent-2); }
.text-box.output { border-left-color: var(--accent); }
.text-box.judge { border-left-color: var(--text-dim); font-size: 12px; color: var(--text-2); }
.sample-body .row-label { font-size: 12px; color: var(--text-2); margin: 12px 0 4px 0; font-weight: 500; }
.muted { color: var(--text-dim); font-size: 12px; }
.legend { font-size: 11px; color: var(--text-dim); margin: 8px 0; display: flex; gap: 16px; flex-wrap: wrap; }
.legend span::before { content: "■"; margin-right: 4px; }
.cb-prompt::before { color: var(--accent-2); }
.cb-output::before { color: var(--accent); }
.cb-judge::before { color: var(--text-dim); }
.pagination { display: flex; gap: 4px; justify-content: center; margin: 16px 0; }
.pagination button { background: var(--bg-3); color: var(--text-2); border: 1px solid var(--border);
  padding: 5px 10px; border-radius: 4px; cursor: pointer; font-size: 12px; min-width: 32px; }
.pagination button:hover:not(:disabled) { color: var(--text); border-color: var(--accent); }
.pagination button.active { background: var(--accent); color: var(--bg); border-color: var(--accent); }
.pagination button:disabled { opacity: 0.4; cursor: not-allowed; }
footer { margin-top: 40px; padding-top: 16px; border-top: 1px solid var(--border);
  color: var(--text-dim); font-size: 12px; text-align: center; }
hr { border: none; border-top: 1px solid var(--border); margin: 20px 0; }
.callout { background: rgba(94, 234, 212, 0.05); border-left: 3px solid var(--accent);
  padding: 12px 14px; border-radius: 4px; margin: 12px 0; font-size: 13px; }
.callout-warn { background: rgba(251, 146, 60, 0.05); border-left-color: var(--warn); }
</style>
</head>
<body>
<div class="container">
<header>
  <h1>Phase 1 한국어 텍스트 생성 능력 평가</h1>
  <div class="sub">PolarPulse · niceinfo · 2026-05-22 · 4 모델 × 5 벤치 × 572 sample</div>
</header>

<nav>
  <button class="active" data-tab="summary">📊 종합</button>
  <button data-tab="benches">📋 벤치별</button>
  <button data-tab="bf16fp8">⚡ BF16 vs FP8</button>
  <button data-tab="raw">🔍 Raw 탐색</button>
</nav>

<section id="summary" class="active">
  <div class="metric-row" id="kpi"></div>
  <div class="card">
    <h2>종합 점수 (weighted total)</h2>
    <div id="overall-bars" class="bar-chart"></div>
  </div>
  <div class="card">
    <h2>운영 권장</h2>
    <div class="callout">
      <strong>🥇 Qwen3.6-35B-A3B-FP8</strong> 운영 채택 권장.<br>
      BF16 대비 양자화 손실 <strong>0.14%</strong> — 운영 결정 기준 (1-2% 이내) 충족.<br>
      메모리 ~35GB (BF16의 절반), H100 NVL native FP8 지원.
    </div>
    <div class="callout callout-warn">
      ⚠️ <strong>벤치별 강점이 다름</strong>: 추론·대화는 35B 모델군이, 요약·instruction은 30B-A3B가 우세.
      Target task 형태 확정 시 재평가 필요.
    </div>
  </div>
</section>

<section id="benches">
  <div class="card">
    <h2>벤치별 모델 비교</h2>
    <div class="legend">
      <span class="cb-judge">막대: 모델 점수</span>
      <span class="muted">스케일: Ko-MT/LogicKor/Ko-IFEval = 1~10, AIHub = 0~10 환산</span>
    </div>
    <div id="bench-bars"></div>
  </div>
  <div class="card">
    <h2>벤치별 1위</h2>
    <div id="bench-winners"></div>
  </div>
</section>

<section id="bf16fp8">
  <div class="card">
    <h2>BF16 vs FP8 양자화 손실 비교</h2>
    <div class="muted">동일 모델 (Qwen3.6-35B-A3B)의 BF16 원본과 FP8 양자화 점수 차이</div>
    <div id="bf16fp8-table"></div>
  </div>
  <div class="card">
    <h2>해석</h2>
    <div class="callout">
      종합 점수 차이는 <strong>0.14%</strong> 수준 — 통계적·실용적으로 무의미한 손실.
      Ko-MT-Bench / LogicKor에선 FP8가 오히려 소폭 우세 (judge noise 가능).
      Ko-IFEval만 −1.7% — 형식 엄격 준수 항목에서 약간 영향, 운영상 수용 가능.
    </div>
  </div>
</section>

<section id="raw">
  <div class="controls">
    <label>모델
      <select id="f-model"><option value="">(전체)</option></select>
    </label>
    <label>벤치
      <select id="f-bench"><option value="">(전체)</option></select>
    </label>
    <label>카테고리
      <select id="f-cat"><option value="">(전체)</option></select>
    </label>
    <label>점수
      <select id="f-score">
        <option value="">(전체)</option>
        <option value="hi">≥ 8 (높음)</option>
        <option value="mid">4 ~ 8 (중간)</option>
        <option value="lo">&lt; 4 (낮음)</option>
      </select>
    </label>
    <label>검색
      <input id="f-q" placeholder="prompt/output 검색">
    </label>
    <span class="count" id="raw-count"></span>
  </div>
  <div id="raw-list" class="sample-list"></div>
  <div class="pagination" id="raw-pagination"></div>
</section>

<footer>
  생성: 2026-05-22 · GitHub: github.com/mjpark-colla/niceinfo-llm-eval · Judge: gpt-4o
</footer>

</div>

<script>
const CFG = __CONFIG_JSON__;
const SUMMARY = __SUMMARY_JSON__;
const DATA = __DATA_JSON__;

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
// Summary tab
// ============================================================
function renderKPI() {
  const totals = CFG.models_order.map(m => ({m, total: SUMMARY[m]?.weighted_total || 0}));
  totals.sort((a, b) => b.total - a.total);
  const top = totals[0];
  const fp8 = SUMMARY["Qwen3.6-35B-A3B-FP8"]?.weighted_total || 0;
  const bf16 = SUMMARY["Qwen3.6-35B-A3B-BF16"]?.weighted_total || 0;
  const diff = ((fp8 - bf16) / bf16 * 100).toFixed(2);
  document.getElementById("kpi").innerHTML = `
    <div class="metric"><div class="label">총 sample</div><div class="value">572</div></div>
    <div class="metric"><div class="label">평가 모델 수</div><div class="value">4</div></div>
    <div class="metric"><div class="label">1위 모델 점수</div><div class="value">${top.total.toFixed(2)}</div></div>
    <div class="metric"><div class="label">FP8 손실 (vs BF16)</div><div class="value" style="color:var(--good)">${diff}%</div></div>
  `;
}

function renderOverall() {
  const rows = CFG.models_order.map(m => ({m, v: SUMMARY[m]?.weighted_total || 0}));
  rows.sort((a, b) => b.v - a.v);
  const max = Math.max(...rows.map(r => r.v));
  const medals = ["🥇", "🥈", "🥉", " "];
  const html = rows.map((r, i) => `
    <div class="bar-row">
      <div>${medals[i] || ""} ${r.m}</div>
      <div class="bar-bg"><div class="bar-fg" style="width:${r.v/max*100}%"></div></div>
      <div class="val">${r.v.toFixed(3)}</div>
    </div>
  `).join("");
  document.getElementById("overall-bars").innerHTML = html;
}

// ============================================================
// Benches tab
// ============================================================
function renderBenches() {
  const container = document.getElementById("bench-bars");
  const winnersDiv = document.getElementById("bench-winners");
  let html = "";
  const winners = [];
  for (const bench of CFG.bench_order) {
    const rows = CFG.models_order.map(m => ({
      m, v: SUMMARY[m]?.per_benchmark?.[bench] ?? 0
    })).sort((a, b) => b.v - a.v);
    const max = Math.max(...rows.map(r => r.v));
    winners.push({bench, top: rows[0]});
    html += `<h3>${CFG.bench_label[bench]} <span class="muted">(가중치 ${(CFG.bench_weight[bench]*100).toFixed(0)}%)</span></h3>`;
    html += rows.map(r => `
      <div class="bar-row">
        <div>${r.m}</div>
        <div class="bar-bg"><div class="bar-fg" style="width:${r.v/max*100}%"></div></div>
        <div class="val">${r.v.toFixed(3)}</div>
      </div>
    `).join("");
  }
  container.innerHTML = html;

  winnersDiv.innerHTML = `<table>
    <tr><th>벤치</th><th>1위 모델</th><th>점수</th></tr>
    ${winners.map(w => `<tr>
      <td>${CFG.bench_label[w.bench]}</td>
      <td><span class="badge b-gold">🥇</span> ${w.top.m}</td>
      <td class="num">${w.top.v.toFixed(3)}</td>
    </tr>`).join("")}
  </table>`;
}

// ============================================================
// BF16 vs FP8 tab
// ============================================================
function renderBF16FP8() {
  const bf16 = SUMMARY["Qwen3.6-35B-A3B-BF16"];
  const fp8 = SUMMARY["Qwen3.6-35B-A3B-FP8"];
  if (!bf16 || !fp8) {
    document.getElementById("bf16fp8-table").innerHTML = "<p>데이터 없음</p>";
    return;
  }
  const rows = CFG.bench_order.map(b => {
    const a = bf16.per_benchmark[b] ?? 0;
    const c = fp8.per_benchmark[b] ?? 0;
    const diff = c - a;
    const pct = a !== 0 ? (diff / a * 100) : 0;
    return {b, a, c, diff, pct};
  });
  const totalA = bf16.weighted_total;
  const totalC = fp8.weighted_total;
  const totalDiff = totalC - totalA;
  const totalPct = totalA !== 0 ? (totalDiff / totalA * 100) : 0;

  const fmt = (v) => {
    const cls = v > 0 ? "b-up" : v < 0 ? "b-down" : "b-zero";
    const sign = v > 0 ? "+" : "";
    return `<span class="${cls}">${sign}${v.toFixed(3)}</span>`;
  };
  const fmtPct = (v) => {
    const cls = v > 0 ? "b-up" : v < 0 ? "b-down" : "b-zero";
    const sign = v > 0 ? "+" : "";
    return `<span class="${cls}">${sign}${v.toFixed(2)}%</span>`;
  };
  const html = `<table>
    <tr><th>벤치</th><th class="num">BF16</th><th class="num">FP8</th><th class="num">절대차</th><th class="num">상대차</th></tr>
    ${rows.map(r => `<tr>
      <td>${CFG.bench_label[r.b]}</td>
      <td class="num">${r.a.toFixed(3)}</td>
      <td class="num">${r.c.toFixed(3)}</td>
      <td class="num">${fmt(r.diff)}</td>
      <td class="num">${fmtPct(r.pct)}</td>
    </tr>`).join("")}
    <tr style="background:var(--bg-3); font-weight:600;">
      <td>Weighted Total</td>
      <td class="num">${totalA.toFixed(3)}</td>
      <td class="num">${totalC.toFixed(3)}</td>
      <td class="num">${fmt(totalDiff)}</td>
      <td class="num">${fmtPct(totalPct)}</td>
    </tr>
  </table>`;
  document.getElementById("bf16fp8-table").innerHTML = html;
}

// ============================================================
// Raw explorer tab
// ============================================================
const fModel = document.getElementById("f-model");
const fBench = document.getElementById("f-bench");
const fCat = document.getElementById("f-cat");
const fScore = document.getElementById("f-score");
const fQ = document.getElementById("f-q");
const rawList = document.getElementById("raw-list");
const rawCount = document.getElementById("raw-count");
const rawPagination = document.getElementById("raw-pagination");

CFG.models_order.forEach(m => {
  if (DATA[m]) fModel.innerHTML += `<option value="${m}">${m}</option>`;
});
CFG.bench_order.forEach(b => {
  fBench.innerHTML += `<option value="${b}">${CFG.bench_label[b]}</option>`;
});

let allSamples = [];
let filtered = [];
let currentPage = 1;
const PER_PAGE = 20;

function buildAllSamples() {
  allSamples = [];
  for (const model of Object.keys(DATA)) {
    for (const bench of Object.keys(DATA[model])) {
      for (const s of DATA[model][bench]) {
        allSamples.push({...s, _model: model, _bench: bench,
          _cat: s.metadata?.category || s.metadata?.domain || ""});
      }
    }
  }
}

function updateCategories() {
  const bench = fBench.value;
  const cats = new Set();
  for (const s of allSamples) {
    if (bench && s._bench !== bench) continue;
    if (s._cat) cats.add(s._cat);
  }
  fCat.innerHTML = `<option value="">(전체)</option>` +
    [...cats].sort().map(c => `<option value="${c}">${c}</option>`).join("");
}

function applyFilters() {
  const model = fModel.value;
  const bench = fBench.value;
  const cat = fCat.value;
  const sc = fScore.value;
  const q = fQ.value.toLowerCase().trim();
  filtered = allSamples.filter(s => {
    if (model && s._model !== model) return false;
    if (bench && s._bench !== bench) return false;
    if (cat && s._cat !== cat) return false;
    if (sc === "hi" && s.score < 8) return false;
    if (sc === "mid" && (s.score < 4 || s.score >= 8)) return false;
    if (sc === "lo" && s.score >= 4) return false;
    if (q) {
      const hay = JSON.stringify(s.turns).toLowerCase();
      if (!hay.includes(q)) return false;
    }
    return true;
  });
  currentPage = 1;
  render();
}

function scorePillClass(score) {
  if (score >= 8) return "b-good";
  if (score >= 4) return "b-warn";
  return "b-bad";
}

function escapeHTML(s) {
  if (typeof s !== "string") s = String(s ?? "");
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function renderSample(s, idx) {
  const turns = (s.turns || []).map((t, i) => {
    const turnLabel = (s.turns.length > 1) ? `[Turn ${i+1}]` : "";
    return `
      <div class="turn-block">
        <div class="turn-label">${turnLabel} Prompt</div>
        <div class="text-box prompt">${escapeHTML(t.prompt)}</div>
        <div class="turn-label" style="margin-top:8px">Model Output</div>
        <div class="text-box output">${escapeHTML(t.model_output)}</div>
        ${t.judge_raw ? `
          <div class="turn-label" style="margin-top:8px">Judge (score: ${t.score ?? "-"})</div>
          <div class="text-box judge">${escapeHTML(t.judge_raw)}</div>
        ` : ""}
        ${t.metric_details ? `
          <div class="turn-label" style="margin-top:8px">Metric details</div>
          <div class="text-box judge">${escapeHTML(JSON.stringify(t.metric_details, null, 2))}</div>
        ` : ""}
      </div>
    `;
  }).join("");

  return `
    <div class="sample" data-i="${idx}">
      <div class="sample-head" onclick="this.parentElement.classList.toggle('open')">
        <span class="id">${s.sample_id ?? "-"}</span>
        <span class="meta">${s._model} · ${CFG.bench_label[s._bench]} · ${s._cat || ""}</span>
        <span class="score-pill badge ${scorePillClass(s.score)}">${
          (typeof s.score === "number") ? s.score.toFixed(2) : "-"
        }</span>
      </div>
      <div class="sample-body">${turns}</div>
    </div>
  `;
}

function render() {
  const total = filtered.length;
  const totalPages = Math.max(1, Math.ceil(total / PER_PAGE));
  if (currentPage > totalPages) currentPage = totalPages;
  const start = (currentPage - 1) * PER_PAGE;
  const slice = filtered.slice(start, start + PER_PAGE);
  rawList.innerHTML = slice.map((s, i) => renderSample(s, start + i)).join("");
  rawCount.textContent = `총 ${total}건 (페이지 ${currentPage}/${totalPages})`;
  renderPagination(totalPages);
}

function renderPagination(totalPages) {
  if (totalPages <= 1) { rawPagination.innerHTML = ""; return; }
  const buttons = [];
  buttons.push(`<button ${currentPage === 1 ? "disabled" : ""} onclick="changePage(${currentPage-1})">‹</button>`);
  const maxBtns = 7;
  let start = Math.max(1, currentPage - 3);
  let end = Math.min(totalPages, start + maxBtns - 1);
  if (end - start < maxBtns - 1) start = Math.max(1, end - maxBtns + 1);
  if (start > 1) buttons.push(`<button onclick="changePage(1)">1</button>`);
  if (start > 2) buttons.push(`<span class="muted" style="padding:5px">…</span>`);
  for (let p = start; p <= end; p++) {
    buttons.push(`<button class="${p === currentPage ? 'active' : ''}" onclick="changePage(${p})">${p}</button>`);
  }
  if (end < totalPages - 1) buttons.push(`<span class="muted" style="padding:5px">…</span>`);
  if (end < totalPages) buttons.push(`<button onclick="changePage(${totalPages})">${totalPages}</button>`);
  buttons.push(`<button ${currentPage === totalPages ? "disabled" : ""} onclick="changePage(${currentPage+1})">›</button>`);
  rawPagination.innerHTML = buttons.join("");
}

window.changePage = function(p) { currentPage = p; render(); window.scrollTo(0, 0); };

fModel.addEventListener("change", applyFilters);
fBench.addEventListener("change", () => { updateCategories(); applyFilters(); });
fCat.addEventListener("change", applyFilters);
fScore.addEventListener("change", applyFilters);
fQ.addEventListener("input", () => { clearTimeout(window._qt); window._qt = setTimeout(applyFilters, 250); });

// Init
buildAllSamples();
updateCategories();
applyFilters();
renderKPI();
renderOverall();
renderBenches();
renderBF16FP8();
</script>
</body>
</html>
"""

def safe_for_inline_script(j: str) -> str:
    """HTML <script> 블록 안에 JSON을 안전하게 임베드.

    데이터에 '</script>', '<!--' 같은 게 있으면 HTML 파서가 블록을 조기 종료시켜
    뒤따르는 데이터가 본문으로 렌더링되고 XSS가 발생할 수 있다.
    JSON에서 '<\\/'는 '/'로 디코드되므로 의미 보존된다.
    """
    return (j
        .replace("</", "<\\/")
        .replace("<!--", "<\\!--")
        .replace("-->", "--\\>")
        .replace(" ", "\\u2028")
        .replace(" ", "\\u2029"))

html_out = (HTML
    .replace("__CONFIG_JSON__", safe_for_inline_script(config_json))
    .replace("__SUMMARY_JSON__", safe_for_inline_script(summary_json))
    .replace("__DATA_JSON__", safe_for_inline_script(data_json)))

OUT.write_text(html_out, encoding="utf-8")
print(f"✅ {OUT}")
print(f"   크기: {OUT.stat().st_size / 1024 / 1024:.1f} MB")

total_samples = sum(len(data[m][b]) for m in data for b in data[m])
print(f"   embed sample: {total_samples}건")
