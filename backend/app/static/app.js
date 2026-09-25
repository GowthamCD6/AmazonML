/**
 * JavaScript logic for Amazon ML Challenge 2026 Interactive Dashboard
 */

const API_BASE = "/api/v1";

document.addEventListener("DOMContentLoaded", () => {
  checkHealth();
  loadStats();
  loadExperiments();
  loadLeaderboard();
  loadReport("validation_report");
});

// Tab Switching
function switchTab(tabName) {
  document.querySelectorAll(".nav-tabs .tab-btn").forEach(btn => btn.classList.remove("active"));
  document.querySelectorAll(".view-panel").forEach(panel => panel.classList.remove("active"));

  const targetBtn = Array.from(document.querySelectorAll(".nav-tabs .tab-btn")).find(b => b.innerText.toLowerCase().includes(tabName.slice(0, 4)));
  if (targetBtn) targetBtn.classList.add("active");

  const targetPanel = document.getElementById(`tab-${tabName}`);
  if (targetPanel) targetPanel.classList.add("active");

  if (tabName === "experiments") {
    loadExperiments();
    loadLeaderboard();
  }
}

// Health Check
async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE}/health`);
    if (res.ok) {
      const data = await res.json();
      document.getElementById("healthText").innerText = `Model Ready (${data.target_records_count.toLocaleString()} Targets)`;
    }
  } catch (err) {
    document.getElementById("healthText").innerText = "Connecting...";
  }
}

// Load Dataset Stats
async function loadStats() {
  try {
    const res = await fetch(`${API_BASE}/stats/dataset`);
    if (res.ok) {
      const d = await res.json();
      document.getElementById("statTrainS1").innerText = d.num_train_s1.toLocaleString();
      document.getElementById("statTestS1").innerText = d.num_test_s1.toLocaleString();
    }
  } catch (e) {
    console.log("Stats fetch fallback");
  }
}

// Sample Query
function loadSampleQuery() {
  document.getElementById("queryName").value = "Apex Logistics 128 Pty Ltd";
  document.getElementById("queryAddr").value = "4820 George Street, Sydney, 2000, AU";
  document.getElementById("queryCountry").value = "AU";
}

// Handle Single Entity Resolution
async function handleResolve(event) {
  event.preventDefault();
  const btn = document.getElementById("btnResolve");
  btn.disabled = true;
  btn.innerHTML = "<span>⏳ Resolving...</span>";

  const name = document.getElementById("queryName").value.trim();
  const addr = document.getElementById("queryAddr").value.trim();
  const country = document.getElementById("queryCountry").value.trim();
  const threshold = parseFloat(document.getElementById("queryThreshold").value);

  try {
    const res = await fetch(`${API_BASE}/resolve?threshold=${threshold}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        business_name: name,
        business_address: addr,
        country: country
      })
    });

    const data = await res.json();
    renderResults(data);
  } catch (err) {
    alert("Resolution request failed: " + err.message);
  } finally {
    btn.disabled = false;
    btn.innerHTML = "<span>⚡ Resolve Entity</span>";
  }
}

function renderResults(data) {
  const card = document.getElementById("resultsCard");
  const list = document.getElementById("matchList");
  const badge = document.getElementById("resultSummaryBadge");
  card.style.display = "block";

  badge.innerText = `${data.matches_count} Matches Found (${data.candidates_count} candidates evaluated in ${data.latency_ms}ms)`;

  if (data.matches.length === 0) {
    list.innerHTML = `
      <div style="text-align: center; padding: 32px; color: var(--text-muted);">
        <div style="font-size: 2rem; margin-bottom: 8px;">🔍</div>
        <h4>No Matches Found Above Threshold (${data.threshold_applied})</h4>
        <p>This entity is treated as a <strong>Singleton</strong>. Empty match list will be emitted.</p>
      </div>
    `;
    return;
  }

  let html = "";
  data.matches.forEach(m => {
    const matchClass = m.is_match ? "is-match" : "is-non-match";
    const statusText = m.is_match ? "✅ MATCH" : "❌ CANDIDATE (Below Threshold)";
    const statusColor = m.is_match ? "var(--accent-green)" : "var(--text-muted)";

    const sigs = Object.entries(m.blocking_signals)
      .filter(([_, v]) => v === 1)
      .map(([k, _]) => `<span style="font-size: 0.75rem; background: rgba(255,255,255,0.06); padding: 2px 6px; border-radius: 4px; margin-right: 4px;">${k}</span>`)
      .join("");

    html += `
      <div class="match-card ${matchClass}">
        <div class="entity-info" style="flex: 1; padding-right: 20px;">
          <h4>
            ${m.business_name}
            <span class="entity-id-tag">${m.candidate_entity_id}</span>
            <span style="font-size: 0.75rem; color: var(--text-muted);">(${m.source})</span>
          </h4>
          <div class="entity-addr">📍 ${m.business_address} ${m.country ? `• 🌍 ${m.country}` : ""}</div>
          <div style="margin-top: 8px; display: flex; align-items: center; gap: 8px;">
            <span style="font-size: 0.75rem; color: var(--text-muted);">Blocking Keys Hit:</span>
            ${sigs}
          </div>
        </div>

        <div class="match-score-group">
          <div style="font-size: 0.75rem; color: ${statusColor}; font-weight: 700; margin-bottom: 2px;">${statusText}</div>
          <div class="score-value">${(m.match_probability * 100).toFixed(1)}%</div>
          <div class="score-bar-bg">
            <div class="score-bar-fill" style="width: ${m.match_probability * 100}%;"></div>
          </div>
        </div>
      </div>
    `;
  });

  list.innerHTML = html;
}

// Handle Pairwise Comparison
async function handleCompare(event) {
  event.preventDefault();
  const e1 = {
    business_name: document.getElementById("p1Name").value.trim(),
    business_address: document.getElementById("p1Addr").value.trim(),
    country: document.getElementById("p1Country").value.trim()
  };
  const e2 = {
    business_name: document.getElementById("p2Name").value.trim(),
    business_address: document.getElementById("p2Addr").value.trim(),
    country: document.getElementById("p2Country").value.trim()
  };

  try {
    const res = await fetch(`${API_BASE}/match/pair`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ entity_1: e1, entity_2: e2, threshold: 0.30 })
    });
    const d = await res.json();
    renderPairResults(d);
  } catch (err) {
    alert("Pair comparison failed: " + err.message);
  }
}

function renderPairResults(d) {
  const card = document.getElementById("pairResultsCard");
  const verdict = document.getElementById("pairVerdict");
  const tbody = document.querySelector("#featureTable tbody");
  card.style.display = "block";

  const isMatch = d.predicted_same_business;
  verdict.innerHTML = `
    <div style="padding: 16px; border-radius: var(--radius-md); background: ${isMatch ? 'rgba(16, 185, 129, 0.1)' : 'rgba(244, 63, 94, 0.1)'}; border: 1px solid ${isMatch ? 'rgba(16, 185, 129, 0.3)' : 'rgba(244, 63, 94, 0.3)'};">
      <h3 style="color: ${isMatch ? 'var(--accent-green)' : 'var(--accent-rose)'}; margin-bottom: 4px;">
        ${isMatch ? 'MATCH CONFIRMED (Same Real-World Business)' : 'NON-MATCH (Different Businesses)'}
      </h3>
      <p style="font-size: 0.9rem; color: var(--text-muted);">
        Model Predicted Match Probability: <strong>${(d.match_probability * 100).toFixed(2)}%</strong> (Threshold: ${d.decision_threshold})
      </p>
    </div>
  `;

  let rows = "";
  for (const [k, v] of Object.entries(d.features)) {
    let cat = "Cross-Field / Meta";
    if (k.startsWith("f_name")) cat = "Name Similarity";
    else if (k.startsWith("f_addr")) cat = "Address Similarity";
    else if (k.startsWith("f_country")) cat = "Country Alignment";
    else if (k.startsWith("f_block")) cat = "Blocking Signal";

    rows += `
      <tr>
        <td><code>${k}</code></td>
        <td><span style="font-size: 0.8rem; background: rgba(255,255,255,0.06); padding: 2px 8px; border-radius: 4px;">${cat}</span></td>
        <td style="font-weight: 700; color: var(--accent-cyan);">${v}</td>
        <td style="font-size: 0.85rem; color: var(--text-muted);">${getInterpretation(k, v)}</td>
      </tr>
    `;
  }
  tbody.innerHTML = rows;
}

function getInterpretation(k, v) {
  if (k.includes("exact")) return v === 1 ? "Exact String Equality" : "Not Exact Match";
  if (k.includes("ratio") || k.includes("jaccard")) return `${(v * 100).toFixed(1)}% Token / Char Agreement`;
  if (k.includes("harmonic")) return `Combined Harmonic Mean: ${v}`;
  if (k.includes("num_overlap")) return `${v} Shared Numerical Digits`;
  return `Signal value: ${v}`;
}

// Load Experiments
async function loadExperiments() {
  try {
    const res = await fetch(`${API_BASE}/experiments`);
    if (res.ok) {
      const data = await res.json();
      const tbody = document.querySelector("#experimentsTable tbody");
      tbody.innerHTML = data.map(row => `
        <tr>
          <td><strong style="color: var(--accent-cyan);">${row.version}</strong></td>
          <td>${row.day}</td>
          <td>${row.change}</td>
          <td><code>${row.model}</code></td>
          <td>${row.threshold}</td>
          <td>${row.candidate_recall}</td>
          <td>${row.validation_precision}</td>
          <td>${row.validation_recall}</td>
          <td><strong style="color: var(--accent-green);">${row.validation_f05}</strong></td>
          <td><span style="font-size: 0.75rem; background: rgba(16,185,129,0.15); color: var(--accent-green); padding: 2px 8px; border-radius: 4px;">${row.decision}</span></td>
        </tr>
      `).join("");
    }
  } catch (e) {
    console.log("Experiments fetch error", e);
  }
}

// Load Leaderboard
async function loadLeaderboard() {
  try {
    const res = await fetch(`${API_BASE}/leaderboard`);
    if (res.ok) {
      const data = await res.json();
      const tbody = document.querySelector("#leaderboardTable tbody");
      tbody.innerHTML = data.map(row => `
        <tr>
          <td><strong>${row.submission_version}</strong></td>
          <td>Day ${row.day}</td>
          <td>${row.submission_time}</td>
          <td><strong style="color: var(--accent-green);">${row.validation_f05}</strong></td>
          <td>${row.public_score}</td>
          <td>${row.private_score}</td>
          <td style="color: var(--text-muted);">${row.notes}</td>
        </tr>
      `).join("");
    }
  } catch (e) {
    console.log("Leaderboard fetch error", e);
  }
}

// Load Markdown Report
async function loadReport(reportName) {
  const container = document.getElementById("reportContainer");
  container.innerHTML = "<p>Loading report...</p>";

  try {
    const res = await fetch(`${API_BASE}/reports/${reportName}`);
    if (res.ok) {
      const data = await res.json();
      if (window.marked) {
        container.innerHTML = marked.parse(data.content);
      } else {
        container.innerText = data.content;
      }
    } else {
      container.innerHTML = `<p style="color: var(--accent-rose);">Report ${reportName} not found.</p>`;
    }
  } catch (err) {
    container.innerHTML = `<p style="color: var(--accent-rose);">Failed to load report: ${err.message}</p>`;
  }
}
