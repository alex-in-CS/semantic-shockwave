// Bridge Builder: the daily challenge. Pure functions, so they're testable in Node.
//
// You get two far-apart concepts and pick your own stepping stones. Your chain is
// scored with the app's bridge metric, the sum of (1 - similarity)^2 over each step,
// against the optimal bridge. Your steps don't have to be graph edges, so a
// clever chain can beat the algorithm (score shown as 100, flagged).

export const EPOCH = '2026-10-01'; // challenge #1
export const NEAR_MATCH = 0.7; // similarity that counts as "close to a stone on the optimal bridge"

const DAY_MS = 86400000;

export function localISODate(date = new Date()) {
  const y = date.getFullYear();
  const m = String(date.getMonth() + 1).padStart(2, '0');
  const d = String(date.getDate()).padStart(2, '0');
  return `${y}-${m}-${d}`;
}

export function dayNumber(iso, epoch = EPOCH) {
  return Math.round((Date.parse(`${iso}T00:00:00Z`) - Date.parse(`${epoch}T00:00:00Z`)) / DAY_MS) + 1;
}

/** The scheduled challenge for a date, falling back (past the schedule) to a stable pick. */
export function challengeFor(challenges, iso) {
  if (!challenges?.length) return null;
  const exact = challenges.find((c) => c.date === iso);
  if (exact) return exact;
  const n = Math.abs(dayNumber(iso)) % challenges.length;
  return { ...challenges[n], date: iso };
}

function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i += 1) s += a[i] * b[i];
  return s;
}

/** Bridge cost of any chain of concepts, edges or not. `vectorOf(id)` gives unit vectors. */
export function chainCost(chain, vectorOf) {
  let total = 0;
  for (let i = 0; i < chain.length - 1; i += 1) {
    const sim = Math.min(1, Math.max(-1, dot(vectorOf(chain[i]), vectorOf(chain[i + 1]))));
    total += (1 - sim) ** 2;
  }
  return total;
}

/**
 * Score a player's chain (including both endpoints) against the optimal path.
 * Returns { score 1-100, beatOptimal, userCost, optimalCost, marks: one emoji per stone }.
 */
export function scoreChain(optimalPath, userChain, vectorOf) {
  const optimalCost = chainCost(optimalPath, vectorOf);
  const userCost = chainCost(userChain, vectorOf);
  const beatOptimal = userCost < optimalCost - 1e-9;
  const tied = Math.abs(userCost - optimalCost) <= 1e-9;
  // Round down, so only a chain that really ties (or beats) the optimum shows 100.
  const score = beatOptimal || tied ? 100 : Math.max(1, Math.floor((100 * optimalCost) / userCost));
  const optimalStones = optimalPath.slice(1, -1);
  const marks = userChain.slice(1, -1).map((stone) => {
    if (optimalStones.includes(stone)) return '🟩';
    const closest = Math.max(...optimalStones.map((s) => dot(vectorOf(stone), vectorOf(s))), -1);
    return closest >= NEAR_MATCH ? '🟨' : '⬛';
  });
  return { score, beatOptimal, userCost, optimalCost, marks };
}

export function shareText({ number, source, target, result, url }) {
  const verdict = result.beatOptimal ? '100% (beat the algorithm!)' : `${result.score}%`;
  return [
    `Semantic Shockwave #${number} 💥`,
    `${source} → ${target}`,
    `${result.marks.join('')} ${verdict}`,
    url,
  ].join('\n');
}

/** Per-viewer history in localStorage; every access is guarded (private mode, blocked storage). */
const KEY = 'shockwave.challenge.v1';

export function loadHistory(storage = globalThis.localStorage) {
  try {
    return JSON.parse(storage.getItem(KEY)) ?? {};
  } catch {
    return {};
  }
}

export function saveResult(iso, record, storage = globalThis.localStorage) {
  const history = loadHistory(storage);
  history[iso] = record;
  try { storage.setItem(KEY, JSON.stringify(history)); } catch { /* storage unavailable */ }
  return history;
}

/** Consecutive days played, counting back from `iso` (today, or yesterday if today isn't played). */
export function streak(history, iso) {
  let day = Date.parse(`${iso}T00:00:00Z`);
  if (!history[iso]) day -= DAY_MS;
  let n = 0;
  while (history[new Date(day).toISOString().slice(0, 10)]) { n += 1; day -= DAY_MS; }
  return n;
}
