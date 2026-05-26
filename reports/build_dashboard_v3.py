"""index_v2.html 에 sample-centric "답변 비교" 탭 추가 (in-place 업데이트).

핵심: SAMPLES 데이터를 inline <script> 안에 넣으면 모델 답변 안의 </script>·
<!--·<button>·alert() 등이 HTML로 새어나옴. 그래서 hidden <textarea> + HTML escape
+ textContent + JSON.parse 패턴으로 데이터 임베드. HTML 파서는 textarea 내용을
절대 코드로 해석하지 않음.
"""
import html
import json
from pathlib import Path

BASE = Path(__file__).parent
SRC = BASE / "index_v2.html"
SAMPLES = BASE / "samples_v2.json"
OUT = BASE / "index_v2.html"
BACKUP = BASE / "index_v2_backup.html"

# 백업 (없을 때만)
if SRC.exists() and not BACKUP.exists():
    BACKUP.write_bytes(SRC.read_bytes())
    print(f"📦 백업: {BACKUP}")

# 이전 빌드 결과에 답변 비교 탭이 이미 들어가 있을 수 있음 → 백업이 있다면 원본 v2로 리셋
if BACKUP.exists():
    src_html = BACKUP.read_text(encoding="utf-8")
    print("↩️  원본 v2 (백업) 기준으로 리셋 후 재빌드")
else:
    src_html = SRC.read_text(encoding="utf-8")

samples_data = json.loads(SAMPLES.read_text(encoding="utf-8"))

# 1) nav 버튼 추가
src_html = src_html.replace(
    '<button data-tab="bench">📋 벤치 상세</button>',
    '<button data-tab="bench">📋 벤치 상세</button>\n  '
    '<button data-tab="raw">🔬 답변 비교</button>',
)

# 2) 새 section 추가
SECTION_NEW = """
<!-- ============================================================ -->
<!-- TAB 6: 답변 비교 (sample-centric) -->
<!-- ============================================================ -->
<section id="raw">
  <div class="card">
    <h2 style="margin-top:0">🔬 동일 sample 4 모델 답변 비교</h2>
    <div class="muted">동일 prompt에 대해 4 모델이 어떻게 답변했는지, 점수가 어떻게 차이나는지 비교.</div>

    <div class="filter-bar">
      <label>벤치
        <select id="raw-bench"></select>
      </label>
      <label>정렬
        <select id="raw-sort">
          <option value="default">기본 (sample_id 순)</option>
          <option value="spread">점수 차이 큰 순 (변별력)</option>
          <option value="low">최저 점수 낮은 순</option>
          <option value="high">최고 점수 높은 순</option>
        </select>
      </label>
      <label>검색
        <input id="raw-search" placeholder="sample_id 또는 prompt 검색...">
      </label>
      <span class="muted" id="raw-count"></span>
    </div>
  </div>

  <div class="raw-layout">
    <aside class="raw-sidebar">
      <div id="raw-list" class="sample-list"></div>
    </aside>
    <main class="raw-detail" id="raw-detail">
      <div class="muted" style="padding:24px;text-align:center">왼쪽 목록에서 sample을 선택하세요.</div>
    </main>
  </div>
</section>
"""
src_html = src_html.replace('<footer>', SECTION_NEW + '\n<footer>')

# 3) CSS 추가
CSS_NEW = """
/* === 답변 비교 탭 === */
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

# 4) SAMPLES 데이터를 hidden <textarea> 안에 HTML escape 해서 넣기 (XSS 안전)
samples_json = json.dumps(samples_data, ensure_ascii=False, separators=(",", ":"))
samples_escaped = html.escape(samples_json, quote=True)
samples_escaped = samples_escaped.replace("</textarea", "&lt;/textarea")
TEXTAREA_BLOCK = (
    '\n<textarea id="samples-data" hidden aria-hidden="true" style="display:none">'
    + samples_escaped
    + '</textarea>\n'
)
src_html = src_html.replace('<footer>', TEXTAREA_BLOCK + '<footer>')

# 5) JS 함수 추가 — </script> 직전
JS_NEW = """
// ============================================================
// 답변 비교 탭 (sample-centric) — SAMPLES 데이터는 textarea 에서 로드
// ============================================================
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
  const bench = rawState.bench;
  const samples = SAMPLES[bench] || {};
  let entries = Object.entries(samples);

  if (rawState.search.trim()) {
    const q = rawState.search.toLowerCase();
    entries = entries.filter(([sid, byModel]) => {
      if (sid.toLowerCase().includes(q)) return true;
      const m0 = CFG.models[0];
      const prompt = (byModel[m0]?.turns?.[0]?.prompt || "").toLowerCase();
      return prompt.includes(q);
    });
  }

  if (rawState.sort === "spread") {
    entries.sort((a, b) => computeSpread(b[1]) - computeSpread(a[1]));
  } else if (rawState.sort === "low") {
    entries.sort((a, b) => {
      const am = Math.min(...CFG.models.map(m => a[1][m]?.score ?? 99));
      const bm = Math.min(...CFG.models.map(m => b[1][m]?.score ?? 99));
      return am - bm;
    });
  } else if (rawState.sort === "high") {
    entries.sort((a, b) => {
      const am = Math.max(...CFG.models.map(m => a[1][m]?.score ?? -1));
      const bm = Math.max(...CFG.models.map(m => b[1][m]?.score ?? -1));
      return bm - am;
    });
  }
  return entries;
}

function renderRawList() {
  const entries = getFilteredSorted();
  const list = document.getElementById("raw-list");
  document.getElementById("raw-count").textContent = entries.length + " sample";

  list.innerHTML = entries.map(([sid, byModel]) => {
    const m0 = CFG.models.find(m => byModel[m]) || CFG.models[0];
    const prompt = byModel[m0]?.turns?.[0]?.prompt || "";
    const preview = prompt.slice(0, 120).replace(/\\n/g, " ");
    const spread = computeSpread(byModel);
    const pills = CFG.models.map(m => {
      const s = byModel[m]?.score;
      if (s === null || s === undefined) return '<span class="score-pill">-</span>';
      return '<span class="score-pill ' + scoreClass(s) + '" title="' + escapeHTML(m) + '">' + s.toFixed(1) + '</span>';
    }).join("");
    const active = rawState.selectedSid === sid ? "active" : "";
    return '<div class="sample-item ' + active + '" data-sid="' + escapeHTML(sid) + '">'
      + '<div class="sid">' + escapeHTML(sid) + ' · 차이 ' + spread.toFixed(1) + '</div>'
      + '<div class="preview">' + escapeHTML(preview) + '</div>'
      + '<div class="scores">' + pills + '</div>'
      + '</div>';
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
  if (!byModel) {
    detail.innerHTML = '<div class="muted">sample을 찾을 수 없습니다.</div>';
    return;
  }

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
    if (numTurns > 1) {
      html += '<div class="turn-header">📌 Turn ' + (ti + 1) + ' / ' + numTurns + '</div>';
    }
    html += '<div class="prompt-box"><div class="label">Prompt</div>'
      + escapeHTML(promptT) + '</div>';
    html += '<div class="answer-grid">';
    for (const m of CFG.models) {
      const t = byModel[m]?.turns?.[ti];
      if (!t) {
        html += '<div class="answer-card"><div class="head"><div class="model-name">'
          + escapeHTML(m) + '</div><div class="score lo">-</div></div>'
          + '<div class="output muted">(데이터 없음)</div></div>';
        continue;
      }
      const score = t.score;
      const cls = scoreClass(score);
      const scoreText = (score !== null && score !== undefined) ? score.toFixed(1) : "-";
      html += '<div class="answer-card">'
        + '<div class="head"><div class="model-name">' + escapeHTML(m) + '</div>'
        + '<div class="score ' + cls + '">' + scoreText + '</div></div>'
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
src_html = src_html.replace('</script>', JS_NEW + '\n</script>')

OUT.write_text(src_html, encoding="utf-8")
size_mb = OUT.stat().st_size / (1024 * 1024)
print(f"✅ {OUT} 생성 ({size_mb:.2f} MB)")
