/**
 * API service helper connecting to FastAPI Backend
 */

const BASE_URL = "/api/v1";

export async function fetchHealth() {
  const res = await fetch(`${BASE_URL}/health`);
  if (!res.ok) throw new Error("Health check failed");
  return res.json();
}

export async function fetchStats() {
  const res = await fetch(`${BASE_URL}/stats/dataset`);
  if (!res.ok) throw new Error("Stats fetch failed");
  return res.json();
}

export async function resolveEntity(query, threshold) {
  const url = threshold !== undefined ? `${BASE_URL}/resolve?threshold=${threshold}` : `${BASE_URL}/resolve`;
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(query)
  });
  if (!res.ok) throw new Error(`Resolution failed: ${res.statusText}`);
  return res.json();
}

export async function comparePair(entity1, entity2, threshold = 0.30) {
  const res = await fetch(`${BASE_URL}/match/pair`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ entity_1: entity1, entity_2: entity2, threshold })
  });
  if (!res.ok) throw new Error(`Pair comparison failed: ${res.statusText}`);
  return res.json();
}

export async function fetchExperiments() {
  const res = await fetch(`${BASE_URL}/experiments`);
  if (!res.ok) throw new Error("Experiments fetch failed");
  return res.json();
}

export async function fetchLeaderboard() {
  const res = await fetch(`${BASE_URL}/leaderboard`);
  if (!res.ok) throw new Error("Leaderboard fetch failed");
  return res.json();
}

export async function fetchReport(reportName) {
  const res = await fetch(`${BASE_URL}/reports/${reportName}`);
  if (!res.ok) throw new Error(`Report ${reportName} fetch failed`);
  return res.json();
}
