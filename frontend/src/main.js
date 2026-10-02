import * as THREE from 'three';
import { createApi } from './api.js';
import { boom, chime, isSoundOn, setSound, tick, whoosh } from './audio.js';
import {
  challengeFor, dayNumber, loadHistory, localISODate, saveResult, scoreChain, shareText, streak,
} from './game.js';
import { animate, createScene } from './scene.js';
import { ALGORITHMS, ALGORITHM_BY_ID, buildGraph, runSearch } from './search.js';
import { decodeVectors } from './vectors.js';
import './style.css';

const COLORS = {
  source: '#f472b6',
  target: '#facc15',
  bridge: '#a3e635',
  forward: '#38bdf8',
  backward: '#fb923c',
  pheromone: '#c084fc',
  failed: '#f87171',
  yours: '#fb923c',
  alternatives: ['#22d3ee', '#e879f9'],
};
const BLAST_DISTANCE = 1500;
const DEFAULT_ALGORITHM = 'dijkstra';
const TOUR_STEP_MS = 3600;
const PLACED_COLOR = '#f8fafc';

const $ = (id) => document.getElementById(id);
const ui = {
  form: $('controls'), source: $('source'), target: $('target'), list: $('concepts'),
  swap: $('swap'), examples: $('examples'), algorithm: $('algorithm'), blurb: $('algo-blurb'),
  speed: $('speed'), execute: $('execute'), compare: $('compare'), surprise: $('surprise'),
  reset: $('reset'), status: $('status'), stats: $('stats'), chain: $('chain'),
  summary: $('summary'), comparison: $('comparison'), hint: $('hint'), tooltip: $('tooltip'),
  legend: $('legend'), tour: $('tour'), intro: $('intro'), help: $('help'), sound: $('sound'),
  tabExplore: $('tab-explore'), tabChallenge: $('tab-challenge'),
  explore: $('explore'), challenge: $('challenge'),
};

const state = {
  api: null,
  features: { free_text: false, narration: false },
  space: null, // the payload: nodes, links, groups, vectors, curated
  nodes: [], // base nodes with .color (THREE.Color)
  byLower: new Map(),
  vectors: new Map(), // id -> Float32Array
  groupColor: new Map(),
  graph: null, // search graph for the current pair (base + any free-text extras)
  baseGraph: null,
  extras: [], // placed free-text concepts in the scene right now
  busy: false,
  run: 0,
  shown: null, // what's on screen: { pair, result } or { challenge }
  narrations: new Map(), // path key -> { steps, summary }
  legendGroup: null,
  tour: null,
  displaced: false, // nodes blown away from home by a blast
};

// ---------------------------------------------------------------------------
// Scene

const scene = createScene($('graph'), { onPick: pickNode, onHover: hoverNode });
scene.setInsets(() => {
  const panel = document.querySelector('.panel').getBoundingClientRect();
  return window.innerWidth > 800
    ? { left: panel.right + 16, bottom: 0 }
    : { left: 0, bottom: window.innerHeight - panel.top + 16 };
});

const idOf = (i) => state.graph.ids[i];
const indexOf = (id) => state.graph.index.get(id);
const pathKey = (ids) => ids.join(' → ');

function groupPalette(groups) {
  // Evenly spaced hues, nudged so neighbouring sections don't look alike.
  groups.forEach((g, i) => {
    const hue = ((i * 7) % groups.length) / groups.length;
    state.groupColor.set(g, new THREE.Color().setHSL(hue, 0.72, 0.62));
  });
  state.groupColor.set('Your phrases', new THREE.Color(PLACED_COLOR));
}

/** Region label per section, at its medoid in 3D (the member nearest the section's centroid). */
function regions(nodes, groups) {
  return groups.map((group) => {
    const members = nodes.filter((n) => n.group === group);
    const c = members.reduce((acc, n) => ({ x: acc.x + n.x / members.length, y: acc.y + n.y / members.length, z: acc.z + n.z / members.length }), { x: 0, y: 0, z: 0 });
    const medoid = members.reduce((best, n) => {
      const d = Math.hypot(n.x - c.x, n.y - c.y, n.z - c.z);
      return d < best.d ? { n, d } : best;
    }, { n: members[0], d: Infinity }).n;
    return { label: group, x: medoid.x, y: medoid.y + 22, z: medoid.z, color: `#${state.groupColor.get(group).getHexString()}` };
  });
}

function setStatus(text, isError = false) {
  ui.status.textContent = text;
  ui.status.classList.toggle('error', isError);
}

// ---------------------------------------------------------------------------
// Idle styling, hover, picking

function idleStyle() {
  scene.resetStyle();
  if (state.legendGroup) {
    state.nodes.forEach((n, i) => {
      if (n.group !== state.legendGroup) scene.setNode(i, { brightness: 0.18, labelAlpha: 0.25 });
      else scene.setNode(i, { labelScale: 1.25, labelAlpha: 1 });
    });
  }
  const picks = [[ui.source.value, COLORS.source], [ui.target.value, COLORS.target]];
  picks.forEach(([name, color]) => {
    const id = state.byLower.get(name.trim().toLowerCase());
    if (id !== undefined) scene.setNode(state.baseGraph.index.get(id), { color, scale: 2, labelScale: 1.8, labelAlpha: 1 });
  });
}

function hoverNode(i, x, y) {
  if (i === null || i === undefined || !state.graph || i >= state.graph.ids.length) {
    ui.tooltip.hidden = true;
    return;
  }
  const id = idOf(i);
  const group = state.nodes[i]?.group ?? 'Your phrases';
  ui.tooltip.innerHTML = '';
  ui.tooltip.append(id, Object.assign(document.createElement('small'), { textContent: group }));
  ui.tooltip.style.left = `${x + 14}px`;
  ui.tooltip.style.top = `${y + 12}px`;
  ui.tooltip.hidden = false;
}

function pickNode(i) {
  if (state.busy || i >= state.baseGraph.ids.length) return;
  const id = state.baseGraph.ids[i];
  if (!ui.challenge.hidden) return addStone(id);
  if (state.shown) return; // finish or reset the current bridge first
  if (!ui.source.value || ui.target.value) {
    ui.source.value = id;
    ui.target.value = '';
  } else {
    ui.target.value = id;
  }
  idleStyle();
}

// ---------------------------------------------------------------------------
// Resolving a pair (vocabulary or free text) into a search graph

async function resolvePair(source, target) {
  const known = [source, target].map((t) => state.byLower.get(t.trim().toLowerCase()));
  if (known.every(Boolean)) {
    state.graph = state.baseGraph;
    return known;
  }
  if (!state.features.free_text) {
    const missing = [source, target][known.findIndex((k) => !k)];
    throw new Error(`"${missing}" isn't in the concept space`);
  }
  const body = await state.api.bridge(source, target);
  const placed = body.placed.map((p) => ({ ...p, color: new THREE.Color(PLACED_COLOR) }));
  placed.forEach((p) => state.vectors.set(p.id, Float32Array.from(p.vector)));
  state.extras = placed;
  state.graph = buildGraph([...state.nodes, ...placed], [...state.space.links, ...body.links], state.vectors);
  scene.addExtras(
    placed.map((p) => ({ ...p, label: p.id })),
    body.links.map((l) => ({ a: state.graph.index.get(l.source), b: state.graph.index.get(l.target) })),
  );
  return [source, target].map((t, k) => known[k] ?? t.trim());
}

// ---------------------------------------------------------------------------
// Search replay

const speedMs = () => Number(ui.speed.value);

/** Replay an algorithm's trace: explored nodes light up, frontier edges glow. */
async function replaySearch(result, s, t, run) {
  scene.setAllNodes({ brightness: 0.3, labelAlpha: 0.22 });
  scene.dimAllEdges(0.5);
  const sideColor = [COLORS.forward, COLORS.backward, COLORS.pheromone];
  const markEnds = () => {
    scene.setNode(s, { color: COLORS.source, brightness: 1, scale: 2.2, labelScale: 2, labelAlpha: 1 });
    scene.setNode(t, { color: COLORS.target, brightness: 1, scale: 2.2, labelScale: 2, labelAlpha: 1 });
  };
  markEnds();
  const trace = result.trace;
  const seen = new Set();
  const apply = (event) => {
    if (event[0] === 'v') {
      const [, n, side] = event;
      seen.add(n);
      scene.setNode(n, { color: sideColor[side], brightness: 1, scale: 1.25, labelAlpha: 0.9 });
    } else if (event[0] === 'e') {
      const [, a, b, side] = event;
      scene.setEdge(scene.edgeIndex(a, b), sideColor[side], side === 2 ? 1 : 0.85);
    } else if (event[0] === 'p') {
      scene.showPaths([{ nodes: event[1], color: COLORS.pheromone, radius: 0.55, flow: false, opacity: 0.55 }]);
    }
  };
  const total = speedMs();
  if (total === 0 || trace.length === 0) {
    trace.forEach(apply);
  } else {
    // Long traces play faster, short ones slower, within the chosen speed.
    const duration = Math.min(total, Math.max(total * 0.35, trace.length * 6));
    let done = 0;
    await animate(duration, (p) => {
      if (run !== state.run) return;
      const upto = Math.floor(trace.length * p);
      for (; done < upto; done += 1) apply(trace[done]);
      if (upto > 0 && trace[upto - 1]?.[0] === 'v') tick(seen.size / Math.max(1, state.graph.ids.length));
      ui.stats.textContent = `${result.algorithm.name}: explored ${seen.size.toLocaleString()} nodes…`;
    });
    for (; done < trace.length; done += 1) apply(trace[done]);
  }
  markEnds();
}

/** The finale: the path glows, comets collide, the blast clears everything else. */
async function finale(paths, s, t) {
  const onPath = new Set(paths.flat());
  const main = paths[0];
  scene.clearPaths();
  scene.setAllNodes({ brightness: 0.3 });
  main.forEach((i) => scene.setNode(i, { color: COLORS.bridge, brightness: 1, scale: 1.8, labelScale: 2.1, labelAlpha: 1 }));
  paths.slice(1).forEach((p, k) => p.forEach((i) => {
    if (!main.includes(i)) scene.setNode(i, { color: COLORS.alternatives[k], brightness: 1, scale: 1.5, labelScale: 1.7, labelAlpha: 1 });
  }));
  scene.setNode(s, { color: COLORS.source, scale: 2.3, labelScale: 2.3 });
  scene.setNode(t, { color: COLORS.target, scale: 2.3, labelScale: 2.3 });

  const a = new THREE.Vector3(...Object.values(scene.home(s)));
  const b = new THREE.Vector3(...Object.values(scene.home(t)));
  const mid = a.clone().add(b).multiplyScalar(0.5);
  whoosh();
  await scene.comets(a, b, mid, [COLORS.source, COLORS.target]);
  boom();
  scene.dimAllEdges(0.04);
  scene.setRegionsVisible(false);
  for (let i = 0; i < scene.count; i += 1) if (!onPath.has(i)) scene.setNode(i, { brightness: 0.06, labelAlpha: 0 });
  scene.showPaths([
    { nodes: main, color: COLORS.bridge },
    ...paths.slice(1).map((p, k) => ({ nodes: p, color: COLORS.alternatives[k], radius: 0.7, opacity: 0.8 })),
  ]);
  state.displaced = true;
  await Promise.all([
    scene.flash(mid),
    scene.tweenPositions((i) => {
      if (onPath.has(i)) return null;
      const p = scene.home(i);
      const d = { x: p.x - mid.x, y: p.y - mid.y, z: p.z - mid.z };
      const len = Math.hypot(d.x, d.y, d.z) || 1;
      // Closer to ground zero = thrown harder, which reads as a blast wave.
      const force = BLAST_DISTANCE * (0.6 + 0.4 * Math.exp(-len / 200));
      return { x: p.x + (d.x / len) * force, y: p.y + (d.y / len) * force, z: p.z + (d.z / len) * force };
    }, 1400),
  ]);
  const { center, radius } = scene.boundsOf([...onPath]);
  await scene.fit(center, Math.max(radius * 1.2, 40), 1600);
}

// ---------------------------------------------------------------------------
// Narration and the chain list

async function narrationFor(ids) {
  const key = pathKey(ids);
  const curated = state.space.curated?.narrations?.[key];
  if (curated) return curated;
  if (state.narrations.has(key)) return state.narrations.get(key);
  if (!state.features.narration) return null;
  const result = await state.api.narrate(ids);
  state.narrations.set(key, result);
  return result;
}

function renderChain(ids, sims, { placedIds = new Set(), failed = false } = {}) {
  ui.summary.textContent = '';
  ui.chain.replaceChildren(...ids.map((id, i) => {
    const li = document.createElement('li');
    if (failed) li.className = 'failed';
    li.append(Object.assign(document.createElement('span'), { className: 'name', textContent: id }));
    if (placedIds.has(id)) li.append(Object.assign(document.createElement('span'), { className: 'tag', textContent: 'new' }));
    if (i > 0 && sims[i - 1] !== undefined) {
      li.append(Object.assign(document.createElement('span'), { className: 'sim', textContent: `${Math.round(sims[i - 1] * 100)}% similar` }));
    }
    return li;
  }));
}

function showNarration(narration) {
  if (!narration) return;
  [...ui.chain.children].slice(1).forEach((li, i) => {
    if (narration.steps[i] && !li.querySelector('.why')) {
      li.append(Object.assign(document.createElement('p'), { className: 'why', textContent: narration.steps[i] }));
    }
  });
  ui.summary.textContent = narration.summary ?? '';
}

function formatStats(r) {
  const parts = [`${r.algorithm.name}: explored ${r.explored.toLocaleString()} nodes`];
  if (r.path) parts.push(`${r.hops} hops`, `cost ${r.cost.toFixed(3)}`, `weakest link ${Math.round(r.weakest * 100)}%`);
  parts.push(`${r.ms < 1 ? '<1' : Math.round(r.ms)} ms`);
  return parts.join(' · ');
}

// ---------------------------------------------------------------------------
// The hop-by-hop tour

function stopTour() {
  if (state.tour) clearTimeout(state.tour.timer);
  state.tour = null;
  ui.tour.hidden = true;
}

function startTour(ids, sims, narration) {
  stopTour();
  if (ids.length < 2) return;
  state.tour = { ids, sims, narration, step: 0, playing: true, timer: null };
  ui.tour.hidden = false;
  showTourStep();
}

function showTourStep() {
  const tour = state.tour;
  if (!tour) return;
  const { ids, sims, narration, step } = tour;
  const [a, b] = [ids[step], ids[step + 1]];
  ui.tour.querySelector('.tour-step').textContent = `Hop ${step + 1} of ${ids.length - 1}`;
  const hop = ui.tour.querySelector('.tour-hop');
  hop.replaceChildren(`${a} → ${b} `, Object.assign(document.createElement('span'), { className: 'sim', textContent: `${Math.round(sims[step] * 100)}% similar` }));
  ui.tour.querySelector('.tour-why').textContent = narration?.steps?.[step] ?? '';
  ui.tour.querySelector('[data-tour="play"]').textContent = tour.playing ? '❚❚' : '▶';
  chime(step);
  const ia = indexOf(a);
  const ib = indexOf(b);
  scene.setNode(ia, { scale: 2.6, labelScale: 2.6 });
  scene.setNode(ib, { scale: 2.6, labelScale: 2.6 });
  if (step > 0) scene.setNode(indexOf(ids[step - 1]), { scale: 1.8, labelScale: 2.1 });
  const { center, radius } = scene.boundsOf([ia, ib]);
  scene.fit(center, Math.max(radius * 2.4, 60), 1100);
  clearTimeout(tour.timer);
  if (tour.playing) {
    tour.timer = setTimeout(() => {
      if (state.tour !== tour) return;
      if (tour.step < ids.length - 2) { tour.step += 1; showTourStep(); } else endTour();
    }, TOUR_STEP_MS);
  }
}

function endTour() {
  const tour = state.tour;
  if (!tour) return;
  stopTour();
  tour.ids.forEach((id, i) => scene.setNode(indexOf(id), { scale: i === 0 || i === tour.ids.length - 1 ? 2.3 : 1.8 }));
  const { center, radius } = scene.boundsOf(tour.ids.map(indexOf));
  scene.fit(center, Math.max(radius * 1.2, 40), 1300);
}

ui.tour.addEventListener('click', (e) => {
  const action = e.target.closest('[data-tour]')?.dataset.tour;
  const tour = state.tour;
  if (!action || !tour) return;
  if (action === 'close') return endTour();
  if (action === 'play') { tour.playing = !tour.playing; return showTourStep(); }
  tour.playing = false;
  if (action === 'next' && tour.step < tour.ids.length - 2) tour.step += 1;
  if (action === 'prev' && tour.step > 0) tour.step -= 1;
  showTourStep();
});

// ---------------------------------------------------------------------------
// Running a bridge

function setBusy(busy) {
  state.busy = busy;
  [ui.execute, ui.compare, ui.surprise].forEach((b) => { b.disabled = busy; });
}

async function runBridge(sourceText, targetText, { algorithm = ui.algorithm.value } = {}) {
  if (state.busy) return;
  const [source, target] = [sourceText.trim(), targetText.trim()];
  if (!source || !target) return;
  if (source.toLowerCase() === target.toLowerCase()) {
    return setStatus("That's the same concept twice. Pick two different ones.", true);
  }
  setBusy(true);
  const run = ++state.run;
  try {
    await resetView({ keepInputs: true });
    setStatus(`Resolving ${source} → ${target}…`);
    const [s, t] = await resolvePair(source, target);
    if (run !== state.run) return;
    ui.source.value = s;
    ui.target.value = t;
    const si = indexOf(s);
    const ti = indexOf(t);
    setStatus(`Searching with ${ALGORITHM_BY_ID.get(algorithm).name}…`);
    const result = runSearch(state.graph, algorithm, s, t);
    // Pull back to the whole cloud so the search's spread is visible.
    scene.overview(speedMs() ? 900 : 0);
    state.shown = { pair: [s, t], result };
    history.replaceState(null, '', `?${new URLSearchParams({ from: s, to: t, algo: algorithm })}`);
    await replaySearch(result, si, ti, run);
    if (run !== state.run) return;
    ui.stats.textContent = formatStats(result);

    const placedIds = new Set(state.extras.map((p) => p.id));
    if (!result.path) {
      const partial = result.partial?.map(idOf) ?? [s];
      renderChain(partial, [], { placedIds, failed: true });
      if (result.partial) scene.showPaths([{ nodes: result.partial, color: COLORS.failed, flow: false }]);
      setStatus(`${result.algorithm.name} failed: ${result.note}`, true);
      return;
    }
    const ids = result.path.map(idOf);
    renderChain(ids, result.sims, { placedIds });
    const alt = result.paths?.length > 1 ? ` (+${result.paths.length - 1} alternatives)` : '';
    setStatus(`${result.hops} hops from ${s} to ${t}${alt}.${result.note ? ` ${result.note}.` : ''}`);
    const narrationPromise = narrationFor(ids).catch(() => null);
    if (state.features.narration && !state.space.curated?.narrations?.[pathKey(ids)]) {
      ui.summary.textContent = 'Narrating the bridge…';
      ui.summary.classList.add('pending');
    }
    await finale(result.paths ?? [result.path], si, ti);
    if (run !== state.run) return;
    // The animation is done: free the controls even if narration is still on its way.
    setBusy(false);
    const narration = await narrationPromise;
    ui.summary.classList.remove('pending');
    if (run !== state.run) return;
    ui.summary.textContent = '';
    showNarration(narration);
    startTour(ids, result.sims, narration);
  } catch (err) {
    setStatus(`Bridge failed: ${err.message}`, true);
  } finally {
    if (run === state.run) setBusy(false);
  }
}

async function resetView({ keepInputs = false } = {}) {
  stopTour();
  state.shown = null;
  ui.chain.replaceChildren();
  ui.summary.textContent = '';
  ui.stats.textContent = '';
  if (!keepInputs) ui.comparison.replaceChildren();
  scene.clearPaths();
  if (state.extras.length) {
    scene.clearExtras();
    state.extras.forEach((p) => state.vectors.delete(p.id));
    state.extras = [];
  }
  state.graph = state.baseGraph;
  scene.setRegionsVisible(true);
  idleStyle();
  if (state.displaced) {
    state.displaced = false;
    await scene.tweenPositions(() => null, 700);
  }
}

// ---------------------------------------------------------------------------
// Compare all algorithms on the current pair

async function compareAll() {
  if (state.busy) return;
  const [source, target] = [ui.source.value.trim(), ui.target.value.trim()];
  if (!source || !target) return setStatus('Pick two concepts to compare on.', true);
  setBusy(true);
  try {
    await resetView({ keepInputs: true });
    const [s, t] = await resolvePair(source, target);
    setStatus(`Running all ${ALGORITHMS.length} algorithms…`);
    await new Promise((r) => setTimeout(r, 20));
    const results = ALGORITHMS.map((algo) => runSearch(state.graph, algo.id, s, t));
    const ok = results.filter((r) => r.path);
    const best = {
      explored: Math.min(...ok.map((r) => r.explored)),
      cost: Math.min(...ok.map((r) => r.cost)),
      hops: Math.min(...ok.map((r) => r.hops)),
    };
    const cell = (value, isBest, text = value) => {
      const td = document.createElement('td');
      td.textContent = text;
      if (isBest) td.className = 'best';
      return td;
    };
    const table = document.createElement('table');
    table.innerHTML = '<thead><tr><th>Algorithm</th><th title="Distinct nodes expanded">Explored</th><th>Hops</th><th title="Sum of (1 − similarity)² over the hops; lower is a smoother bridge">Cost</th><th title="Lowest similarity on the path">Weakest</th></tr></thead>';
    const body = document.createElement('tbody');
    results.forEach((r) => {
      const tr = document.createElement('tr');
      tr.title = `${r.algorithm.blurb} Click to watch it.`;
      tr.append(Object.assign(document.createElement('td'), { textContent: r.algorithm.name }));
      tr.append(cell(r.explored, r.path && r.explored === best.explored, r.explored.toLocaleString()));
      if (r.path) {
        tr.append(cell(r.hops, r.hops === best.hops));
        tr.append(cell(r.cost, Math.abs(r.cost - best.cost) < 1e-9, r.cost.toFixed(3)));
        tr.append(cell(r.weakest, false, `${Math.round(r.weakest * 100)}%`));
      } else {
        const td = Object.assign(document.createElement('td'), { colSpan: 3, className: 'fail', textContent: 'failed' });
        td.title = r.note ?? '';
        tr.append(td);
      }
      tr.addEventListener('click', () => {
        ui.algorithm.value = r.algorithm.id;
        updateBlurb();
        runBridge(s, t, { algorithm: r.algorithm.id });
      });
      body.append(tr);
    });
    table.append(body);
    table.append(Object.assign(document.createElement('caption'), { textContent: 'Green = best in column. Click a row to watch that algorithm search.' }));
    ui.comparison.replaceChildren(table);
    setStatus(`${ok.length} of ${results.length} algorithms found a bridge from ${s} to ${t}.`);
  } catch (err) {
    setStatus(`Compare failed: ${err.message}`, true);
  } finally {
    setBusy(false);
  }
}

// ---------------------------------------------------------------------------
// Daily challenge

const challenge = { today: localISODate(), current: null, practice: false, stones: [] };

function vectorOf(id) {
  return state.vectors.get(id) ?? state.vectors.get(state.byLower.get(id.toLowerCase()));
}

function optimalPathFor(c) {
  if (c.path) return c.path;
  const r = runSearch(state.baseGraph, 'dijkstra', c.source, c.target);
  return r.path.map((i) => state.baseGraph.ids[i]);
}

function randomPractice() {
  const ids = state.baseGraph.ids;
  for (let tries = 0; tries < 200; tries += 1) {
    const a = ids[Math.floor(Math.random() * ids.length)];
    const b = ids[Math.floor(Math.random() * ids.length)];
    const ga = state.nodes[state.baseGraph.index.get(a)].group;
    const gb = state.nodes[state.baseGraph.index.get(b)].group;
    if (a === b || ga === gb) continue;
    const r = runSearch(state.baseGraph, 'dijkstra', a, b);
    if (r.hops >= 4 && r.hops <= 7) return { source: a, target: b, path: r.path.map((i) => ids[i]), date: null };
  }
  return null;
}

function renderChallenge() {
  const c = challenge.current;
  const root = ui.challenge;
  root.replaceChildren();
  if (!c) {
    root.append(Object.assign(document.createElement('p'), { className: 'challenge-meta', textContent: 'No challenge available.' }));
    return;
  }
  const history = loadHistory();
  const number = c.date ? dayNumber(c.date) : null;
  const done = c.date ? history[c.date] : null;
  const optimalPath = optimalPathFor(c);
  root.insertAdjacentHTML('beforeend', `<h2>${c.date ? `Daily challenge #${number}` : 'Practice round'}</h2>`);
  const pair = document.createElement('div');
  pair.className = 'challenge-pair';
  pair.append(Object.assign(document.createElement('span'), { className: 'src', textContent: c.source }), ' → ',
    Object.assign(document.createElement('span'), { className: 'dst', textContent: c.target }));
  root.append(pair);
  root.append(Object.assign(document.createElement('p'), {
    className: 'challenge-meta',
    textContent: `The optimal bridge takes ${optimalPath.length - 1} hops. Pick your own stepping stones (1–6): type them, or click nodes in the cloud. Your chain is scored on how smooth each step is, so you can even beat the algorithm.`,
  }));

  if (done) {
    root.append(resultCard(c, done.result, done.stones, optimalPath));
  } else {
    const list = document.createElement('ol');
    list.id = 'stones';
    const fixed = (text) => Object.assign(document.createElement('li'), { className: 'fixed', textContent: text });
    list.append(fixed(c.source));
    if (!challenge.stones.length) challenge.stones = Array(Math.max(1, optimalPath.length - 2)).fill('');
    challenge.stones.forEach((value, k) => {
      const li = document.createElement('li');
      const input = Object.assign(document.createElement('input'), { value, placeholder: `stepping stone ${k + 1}`, maxLength: 60 });
      input.setAttribute('list', 'concepts');
      input.addEventListener('input', () => { challenge.stones[k] = input.value; });
      const remove = Object.assign(document.createElement('button'), { type: 'button', className: 'icon', textContent: '✕', title: 'Remove' });
      remove.setAttribute('aria-label', `Remove stepping stone ${k + 1}`);
      remove.addEventListener('click', () => { challenge.stones.splice(k, 1); renderChallenge(); });
      li.append(input, remove);
      list.append(li);
    });
    list.append(fixed(c.target));
    root.append(list);
    const buttons = document.createElement('div');
    buttons.className = 'buttons';
    const add = Object.assign(document.createElement('button'), { type: 'button', className: 'ghost', textContent: '+ Add stone' });
    add.disabled = challenge.stones.length >= 6;
    add.addEventListener('click', () => { challenge.stones.push(''); renderChallenge(); });
    const submit = Object.assign(document.createElement('button'), { type: 'button', textContent: 'Submit chain' });
    submit.addEventListener('click', submitChallenge);
    buttons.append(add, submit);
    root.append(buttons);
    root.append(Object.assign(document.createElement('p'), { id: 'challenge-error', className: 'challenge-meta' }));
  }

  const footer = document.createElement('div');
  footer.className = 'buttons';
  const practice = Object.assign(document.createElement('button'), { type: 'button', className: 'ghost', textContent: 'Practice round' });
  practice.addEventListener('click', () => {
    challenge.current = randomPractice();
    challenge.practice = true;
    challenge.stones = [];
    renderChallenge();
    showChallengePair();
  });
  footer.append(practice);
  if (challenge.practice) {
    const back = Object.assign(document.createElement('button'), { type: 'button', className: 'ghost', textContent: "Today's challenge" });
    back.addEventListener('click', () => { loadTodaysChallenge(); renderChallenge(); showChallengePair(); });
    footer.append(back);
  }
  root.append(footer);
  const days = streak(history, challenge.today);
  if (days) root.append(Object.assign(document.createElement('p'), { className: 'challenge-meta', textContent: `🔥 ${days}-day streak` }));
}

function resultCard(c, result, stones, optimalPath) {
  const card = document.createElement('div');
  card.className = 'result';
  card.append(Object.assign(document.createElement('div'), { className: 'score', textContent: result.beatOptimal ? '100% · beat the algorithm!' : `${result.score}%` }));
  card.append(Object.assign(document.createElement('div'), { className: 'marks', textContent: result.marks.join('') }));
  card.append(Object.assign(document.createElement('p'), { textContent: `Yours: ${[c.source, ...stones, c.target].join(' → ')}` }));
  card.append(Object.assign(document.createElement('p'), { textContent: `Optimal: ${optimalPath.join(' → ')}` }));
  card.append(Object.assign(document.createElement('p'), { textContent: '🟩 on the optimal bridge · 🟨 close to one · ⬛ your own route' }));
  const buttons = document.createElement('div');
  buttons.className = 'buttons';
  const show = Object.assign(document.createElement('button'), { type: 'button', className: 'ghost', textContent: 'Show both bridges' });
  show.addEventListener('click', () => revealChallenge(c, stones, optimalPath));
  buttons.append(show);
  if (c.date) {
    const share = Object.assign(document.createElement('button'), { type: 'button', textContent: 'Copy result' });
    share.addEventListener('click', async () => {
      const text = shareText({ number: dayNumber(c.date), source: c.source, target: c.target, result, url: location.origin + location.pathname });
      try { await navigator.clipboard.writeText(text); share.textContent = 'Copied!'; } catch { share.textContent = 'Copy failed'; }
    });
    buttons.append(share);
  }
  card.append(buttons);
  return card;
}

function addStone(id) {
  if (!challenge.current || loadHistory()[challenge.current.date]) return;
  const empty = challenge.stones.findIndex((s) => !s.trim());
  if (empty >= 0) challenge.stones[empty] = id;
  else if (challenge.stones.length < 6) challenge.stones.push(id);
  renderChallenge();
  const ii = state.baseGraph.index.get(id);
  scene.setNode(ii, { color: COLORS.yours, scale: 1.9, labelScale: 1.8, labelAlpha: 1 });
}

function submitChallenge() {
  const c = challenge.current;
  const error = $('challenge-error');
  const stones = challenge.stones.map((s) => s.trim()).filter(Boolean);
  if (!stones.length) { error.textContent = 'Add at least one stepping stone.'; return; }
  const resolved = [];
  for (const s of stones) {
    const id = state.byLower.get(s.toLowerCase());
    if (!id) { error.textContent = `"${s}" isn't in the concept space (pick from the suggestions).`; return; }
    if (id === c.source || id === c.target || resolved.includes(id)) { error.textContent = `"${id}" is already in your chain.`; return; }
    resolved.push(id);
  }
  const optimalPath = optimalPathFor(c);
  const result = scoreChain(optimalPath, [c.source, ...resolved, c.target], vectorOf);
  if (c.date) saveResult(c.date, { result, stones: resolved });
  else challenge.lastPractice = { result, stones: resolved };
  challenge.stones = resolved;
  renderChallenge();
  if (!c.date) ui.challenge.insertBefore(resultCard(c, result, resolved, optimalPath), ui.challenge.querySelector('#stones'));
  revealChallenge(c, resolved, optimalPath);
}

async function revealChallenge(c, stones, optimalPath) {
  if (state.busy) return;
  setBusy(true);
  try {
    await resetView({ keepInputs: true });
    const index = (id) => state.baseGraph.index.get(id);
    const optimal = optimalPath.map(index);
    const yours = [c.source, ...stones, c.target].map(index);
    scene.setAllNodes({ brightness: 0.12, labelAlpha: 0.1 });
    scene.dimAllEdges(0.15);
    scene.setRegionsVisible(false);
    optimal.forEach((i) => scene.setNode(i, { color: COLORS.bridge, brightness: 1, scale: 1.8, labelScale: 2, labelAlpha: 1 }));
    yours.forEach((i) => { if (!optimal.includes(i)) scene.setNode(i, { color: COLORS.yours, brightness: 1, scale: 1.8, labelScale: 2, labelAlpha: 1 }); });
    scene.setNode(optimal[0], { color: COLORS.source, scale: 2.3, labelScale: 2.3 });
    scene.setNode(optimal.at(-1), { color: COLORS.target, scale: 2.3, labelScale: 2.3 });
    scene.showPaths([{ nodes: optimal, color: COLORS.bridge }, { nodes: yours, color: COLORS.yours, radius: 0.8 }]);
    state.shown = { challenge: c };
    const { center, radius } = scene.boundsOf([...new Set([...optimal, ...yours])]);
    await scene.fit(center, Math.max(radius * 1.2, 40), 1500);
    const narration = await narrationFor(optimalPath).catch(() => null);
    if (narration && state.shown?.challenge === c) {
      ui.tabExplore.click();
      renderChain(optimalPath, optimalPath.slice(1).map((id, i) => runSearchSim(optimalPath[i], id)));
      showNarration(narration);
      setStatus(`The optimal bridge (green) vs yours (orange).`);
    }
  } finally {
    setBusy(false);
  }
}

function runSearchSim(a, b) {
  const va = vectorOf(a);
  const vb = vectorOf(b);
  let s = 0;
  for (let k = 0; k < va.length; k += 1) s += va[k] * vb[k];
  return s;
}

function loadTodaysChallenge() {
  challenge.practice = false;
  challenge.stones = [];
  challenge.current = challengeFor(state.space.curated?.challenges, challenge.today) ?? randomPractice();
  const done = challenge.current?.date && loadHistory()[challenge.current.date];
  if (done) challenge.stones = done.stones;
}

function showChallengePair() {
  if (state.busy || !challenge.current) return;
  stopTour();
  resetView({ keepInputs: true }).then(() => {
    scene.resetStyle(); // drop the Explore tab's picks: only the challenge pair is highlighted
    const ids = [challenge.current.source, challenge.current.target].map((id) => state.baseGraph.index.get(id));
    scene.setNode(ids[0], { color: COLORS.source, scale: 2.4, labelScale: 2.4, labelAlpha: 1 });
    scene.setNode(ids[1], { color: COLORS.target, scale: 2.4, labelScale: 2.4, labelAlpha: 1 });
    const { center, radius } = scene.boundsOf(ids);
    scene.fit(center, Math.max(radius * 1.3, 60), 1400);
  });
}

function selectTab(name) {
  const isChallenge = name === 'challenge';
  ui.tabExplore.setAttribute('aria-selected', String(!isChallenge));
  ui.tabChallenge.setAttribute('aria-selected', String(isChallenge));
  ui.explore.hidden = isChallenge;
  ui.challenge.hidden = !isChallenge;
  if (isChallenge) { renderChallenge(); showChallengePair(); }
}
ui.tabExplore.addEventListener('click', () => selectTab('explore'));
ui.tabChallenge.addEventListener('click', () => selectTab('challenge'));

// ---------------------------------------------------------------------------
// Panel wiring

function updateBlurb() {
  const algo = ALGORITHM_BY_ID.get(ui.algorithm.value);
  const metric = { squared: 'cost: (1 − similarity)² per hop', angular: 'cost: angle between meanings', hops: 'cost: number of hops' }[algo.metric];
  ui.blurb.textContent = `${algo.blurb} (${metric})`;
}

function buildAlgorithmSelect() {
  const families = [...new Set(ALGORITHMS.map((a) => a.family))];
  ui.algorithm.replaceChildren(...families.map((family) => {
    const group = document.createElement('optgroup');
    group.label = family;
    ALGORITHMS.filter((a) => a.family === family).forEach((a) => group.append(new Option(a.name, a.id)));
    return group;
  }));
  ui.algorithm.value = DEFAULT_ALGORITHM;
  ui.algorithm.addEventListener('change', updateBlurb);
  updateBlurb();
}

function buildLegend(groups) {
  ui.legend.replaceChildren(...groups.map((g) => {
    const b = document.createElement('button');
    b.type = 'button';
    b.setAttribute('aria-pressed', 'false');
    b.title = `Highlight ${g}`;
    const dot = document.createElement('i');
    dot.style.background = `#${state.groupColor.get(g).getHexString()}`;
    b.append(dot, g);
    b.addEventListener('click', () => {
      if (state.shown || state.busy) return;
      state.legendGroup = state.legendGroup === g ? null : g;
      [...ui.legend.children].forEach((x) => x.setAttribute('aria-pressed', String(x === b && state.legendGroup === g)));
      idleStyle();
    });
    return b;
  }));
}

function buildExamples(examples) {
  ui.examples.replaceChildren(...examples.map(({ source, target }) => {
    const b = Object.assign(document.createElement('button'), { type: 'button', textContent: `${source} → ${target}` });
    b.addEventListener('click', () => { ui.source.value = source; ui.target.value = target; runBridge(source, target); });
    return b;
  }));
}

function surprise() {
  const ids = state.baseGraph.ids;
  const pick = () => ids[Math.floor(Math.random() * ids.length)];
  let [a, b] = [pick(), pick()];
  while (a === b) b = pick();
  ui.source.value = a;
  ui.target.value = b;
  runBridge(a, b);
}

ui.form.addEventListener('submit', (e) => { e.preventDefault(); runBridge(ui.source.value, ui.target.value); });
ui.compare.addEventListener('click', compareAll);
ui.surprise.addEventListener('click', surprise);
ui.swap.addEventListener('click', () => {
  [ui.source.value, ui.target.value] = [ui.target.value, ui.source.value];
  if (!state.shown) idleStyle();
});
[ui.source, ui.target].forEach((input) => input.addEventListener('change', () => { if (!state.shown) idleStyle(); }));
ui.reset.addEventListener('click', async () => {
  if (state.busy) return;
  state.run += 1;
  ui.source.value = ui.target.value = '';
  history.replaceState(null, '', location.pathname);
  setStatus('Pick two concepts.');
  await resetView();
  scene.overview(900);
});

function updateSoundButton() {
  ui.sound.textContent = isSoundOn() ? '🔊' : '🔇';
  ui.sound.setAttribute('aria-label', isSoundOn() ? 'Turn sound off' : 'Turn sound on');
}
ui.sound.addEventListener('click', () => { setSound(!isSoundOn()); updateSoundButton(); });
updateSoundButton();

const INTRO_KEY = 'shockwave.intro.seen';
ui.help.addEventListener('click', () => ui.intro.showModal());
ui.intro.addEventListener('click', (e) => {
  const action = e.target.closest('[data-intro]')?.dataset.intro;
  if (!action) return;
  ui.intro.close();
  if (action === 'demo') { selectTab('explore'); ui.source.value = 'black hole'; ui.target.value = 'jazz'; runBridge('black hole', 'jazz'); }
  if (action === 'challenge') selectTab('challenge');
});
ui.intro.addEventListener('close', () => { try { localStorage.setItem(INTRO_KEY, '1'); } catch { /* storage unavailable */ } });

// ---------------------------------------------------------------------------
// Load

async function load() {
  try {
    state.api = await createApi({ onStatus: (text) => setStatus(text) });
    const [health, space] = await Promise.all([state.api.health(), state.api.space()]);
    state.features = health;
    state.space = space;
    const groups = space.groups ?? [...new Set(space.nodes.map((n) => n.group))];
    groupPalette(groups);
    state.nodes = space.nodes.map((n) => ({ ...n, color: state.groupColor.get(n.group) ?? new THREE.Color(PLACED_COLOR) }));
    const vectors = decodeVectors(space.vectors, space.nodes.length);
    state.nodes.forEach((n, i) => {
      state.vectors.set(n.id, vectors[i]);
      state.byLower.set(n.id.toLowerCase(), n.id);
    });
    state.baseGraph = state.graph = buildGraph(state.nodes, space.links, state.vectors);

    scene.setData({
      nodes: state.nodes.map((n) => ({ ...n, label: n.id })),
      links: space.links.map((l) => ({ a: state.graph.index.get(l.source), b: state.graph.index.get(l.target) })),
      regions: regions(state.nodes, groups),
    });
    ui.list.replaceChildren(...state.nodes.map((n) => new Option('', n.id)));
    buildAlgorithmSelect();
    buildLegend(groups);
    buildExamples(space.curated?.examples ?? []);
    $('intro-count').textContent = state.nodes.length.toLocaleString();
    loadTodaysChallenge();

    let hint = `Tip: click two nodes to pick them${health.free_text ? ', or type <em>anything</em>: new phrases get embedded and placed on the fly' : ''}. Drag to rotate, scroll to zoom.`;
    if (health.static) hint += ' This demo runs entirely in your browser; <a href="https://github.com/alex-in-CS/semantic-shockwave" target="_blank" rel="noopener">run the full app</a> for live narration of any bridge.';
    ui.hint.innerHTML = hint;

    scene.setAutoRotate(true);
    await scene.overview(0);
    setStatus(`${state.nodes.length.toLocaleString()} concepts in ${groups.length} regions. Pick two.`);

    const params = new URLSearchParams(location.search);
    if (params.get('algo') && ALGORITHM_BY_ID.has(params.get('algo'))) { ui.algorithm.value = params.get('algo'); updateBlurb(); }
    if (params.get('from') && params.get('to')) {
      ui.source.value = params.get('from');
      ui.target.value = params.get('to');
      setTimeout(() => runBridge(ui.source.value, ui.target.value), 900);
    } else if (params.get('tab') === 'challenge') {
      selectTab('challenge');
    } else {
      let seen = false;
      try { seen = localStorage.getItem(INTRO_KEY) === '1'; } catch { /* storage unavailable */ }
      if (!seen) ui.intro.showModal();
    }
  } catch (err) {
    setStatus(`Can't load the concept space (${err.message}).`, true);
  }
}

load();
