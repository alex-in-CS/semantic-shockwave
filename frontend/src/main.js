import ForceGraph3D from '3d-force-graph';
import * as THREE from 'three';
import SpriteText from 'three-spritetext';
import { createApi } from './api.js';
import './style.css';

// The FastAPI backend, or (static build) the same logic running in the browser.
let api;

const COLLIDE_MS = 1100; // particle streams race from both picks to the midpoint
const FLASH_MS = 900;
const BLAST_DISTANCE = 1400; // how far non-bridge nodes are flung
const BLAST_MS = 1400;
const RESET_MS = 900;

const COLORS = {
  idle: '#7dd3fc',
  source: '#f472b6',
  target: '#facc15',
  bridge: '#a3e635',
  faded: 'rgba(125, 211, 252, 0.07)',
  link: 'rgba(148, 163, 184, 0.16)',
  linkFaded: 'rgba(148, 163, 184, 0.015)',
};

const $ = (id) => document.getElementById(id);
const ui = {
  form: $('controls'), source: $('source'), target: $('target'), list: $('concepts'),
  execute: $('execute'), reset: $('reset'), surprise: $('surprise'), status: $('status'),
  chain: $('chain'), summary: $('summary'), hint: $('hint'),
};

// `bridge` is null in the idle view; during a shockwave it holds the path's node
// ids and its edge keys so styling accessors can tell bridge from background.
// `placed` are free-text concepts the backend embedded for the current bridge.
const state = {
  nodes: [], links: [], byId: new Map(), source: null, target: null, bridge: null,
  placed: [], features: { free_text: false, narration: false }, run: 0, busy: false,
};

const edgeKey = (a, b) => (a < b ? `${a}\u0000${b}` : `${b}\u0000${a}`);
const endId = (end) => (typeof end === 'object' ? end.id : end);
// The bridge is only drawn once revealed (at impact), so the collision plays over the full cloud.
const shown = () => (state.bridge?.revealed ? state.bridge : null);
const isBridgeLink = (l) => shown()?.edges.has(edgeKey(endId(l.source), endId(l.target)));
const isPick = (n) => n.id === state.source || n.id === state.target;
const isHighlighted = (n) => isPick(n) || shown()?.nodes.has(n.id);

function nodeColor(n) {
  if (n.id === state.source) return COLORS.source;
  if (n.id === state.target) return COLORS.target;
  if (!shown()) return COLORS.idle;
  return shown().nodes.has(n.id) ? COLORS.bridge : COLORS.faded;
}

function linkColor(l) {
  if (!shown()) return COLORS.link;
  return isBridgeLink(l) ? COLORS.bridge : COLORS.linkFaded;
}

const labels = new Map();
function label(n) {
  if (!labels.has(n.id)) {
    const sprite = new SpriteText(n.id, 7, '#f8fafc');
    sprite.fontFace = 'Inter, system-ui, sans-serif';
    sprite.backgroundColor = 'rgba(5, 6, 10, 0.55)';
    sprite.padding = 2;
    sprite.borderRadius = 3;
    sprite.position.y = 12;
    labels.set(n.id, sprite);
  }
  return labels.get(n.id);
}

const graph = new ForceGraph3D($('graph'))
  .backgroundColor('#05060a')
  .nodeLabel('id')
  .nodeRelSize(4)
  .nodeOpacity(1)
  .linkOpacity(1)
  .linkDirectionalParticleWidth(2.5)
  .linkDirectionalParticleSpeed(0.012)
  .linkDirectionalParticleColor(() => COLORS.bridge)
  .nodeThreeObjectExtend(true)
  .onNodeClick(pickNode)
  // Positions come from the backend (UMAP), so physics is off and nodes stay pinned.
  // The engine is kept ticking forever so pinned positions we animate get redrawn.
  .warmupTicks(0)
  .cooldownTime(Infinity)
  .d3AlphaMin(0);
graph.d3Force('charge', null);
graph.d3Force('center', null);
// The link force stays: it can't move pinned nodes, but it is what resolves each link's
// source/target ids into node objects, and links aren't drawn without that.
graph.d3Force('link').strength(0);

// Accessors read `state`; re-setting them (with fresh closures) forces a restyle.
function restyle() {
  graph
    .nodeColor((n) => nodeColor(n))
    .nodeVal((n) => (isHighlighted(n) ? 3 : 1))
    // Returning nothing keeps the default sphere; highlighted nodes also get a text label.
    .nodeThreeObject((n) => (isHighlighted(n) ? label(n) : null))
    .linkColor((l) => linkColor(l))
    .linkWidth((l) => (isBridgeLink(l) ? 2 : 0))
    .linkDirectionalParticles((l) => (isBridgeLink(l) ? 4 : 0));
}

const easeOutCubic = (t) => 1 - (1 - t) ** 3;
const easeInCubic = (t) => t ** 3;

/** Run `onFrame(progress)` every animation frame for `ms`, resolving when done. */
function animate(ms, onFrame) {
  const start = performance.now();
  return new Promise((resolve) => {
    function frame(now) {
      const t = Math.min(1, (now - start) / ms);
      onFrame(t);
      if (t < 1) requestAnimationFrame(frame);
      else resolve();
    }
    requestAnimationFrame(frame);
  });
}

/** Tween every node's pinned position from where it is now to `destination(node)`. */
function tweenNodes(destination, ms) {
  const nodes = graph.graphData().nodes;
  const from = nodes.map((n) => ({ x: n.fx, y: n.fy, z: n.fz }));
  const to = nodes.map(destination);
  return animate(ms, (t) => {
    const e = easeOutCubic(t);
    nodes.forEach((n, i) => {
      n.fx = from[i].x + (to[i].x - from[i].x) * e;
      n.fy = from[i].y + (to[i].y - from[i].y) * e;
      n.fz = from[i].z + (to[i].z - from[i].z) * e;
    });
  });
}

const vec = (p) => new THREE.Vector3(p.x, p.y, p.z);

function centroid(points) {
  const c = { x: 0, y: 0, z: 0 };
  points.forEach((p) => { c.x += p.x; c.y += p.y; c.z += p.z; });
  return { x: c.x / points.length, y: c.y / points.length, z: c.z / points.length };
}

/** A comet: a bright head with a fading tail, drawn as additive points. */
function makeComet(color, count = 120) {
  const positions = new Float32Array(count * 3);
  const colors = new Float32Array(count * 3);
  const base = new THREE.Color(color);
  for (let i = 0; i < count; i += 1) {
    // Additive blending: darker vertices are more transparent, so the tail fades out.
    base.clone().multiplyScalar((1 - i / count) ** 1.6).toArray(colors, i * 3);
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  const material = new THREE.PointsMaterial({
    size: 11, vertexColors: true, transparent: true, depthWrite: false,
    blending: THREE.AdditiveBlending,
  });
  return new THREE.Points(geometry, material);
}

function disposeAll(scene, objects) {
  objects.forEach((o) => { scene.remove(o); o.geometry.dispose(); o.material.dispose(); });
}

/** Phase 1: particle streams fire from both picks and accelerate into each other. */
async function collide(a, b, mid) {
  const scene = graph.scene();
  const streams = [[a, makeComet(COLORS.source)], [b, makeComet(COLORS.target)]];
  streams.forEach(([, comet]) => scene.add(comet));
  const point = new THREE.Vector3();
  await animate(COLLIDE_MS, (t) => {
    streams.forEach(([start, comet]) => {
      const attr = comet.geometry.attributes.position;
      for (let i = 0; i < attr.count; i += 1) {
        // Each tail particle lags the head a little, with jitter that shrinks near the head.
        const lag = Math.max(0, t - i * 0.004);
        const jitter = (i / attr.count) * 10;
        point.lerpVectors(start, mid, easeInCubic(lag));
        attr.setXYZ(i, point.x + (Math.random() - 0.5) * jitter,
          point.y + (Math.random() - 0.5) * jitter, point.z + (Math.random() - 0.5) * jitter);
      }
      attr.needsUpdate = true;
    });
  });
  disposeAll(scene, streams.map(([, comet]) => comet));
}

/** Phase 2a: the impact flash and an expanding shock ring at ground zero. */
function flash(mid) {
  const scene = graph.scene();
  const additive = { transparent: true, depthWrite: false, blending: THREE.AdditiveBlending };
  const core = new THREE.Mesh(new THREE.SphereGeometry(1, 24, 24),
    new THREE.MeshBasicMaterial({ color: '#ffffff', ...additive }));
  const ring = new THREE.Mesh(new THREE.RingGeometry(0.97, 1, 96),
    new THREE.MeshBasicMaterial({ color: '#e0f2fe', side: THREE.DoubleSide, ...additive }));
  core.position.copy(mid);
  ring.position.copy(mid);
  ring.lookAt(graph.camera().position);
  scene.add(core, ring);
  return animate(FLASH_MS, (t) => {
    const e = easeOutCubic(t);
    core.scale.setScalar(4 + 40 * e);
    core.material.opacity = (1 - e) ** 2;
    ring.scale.setScalar(10 + 700 * e);
    ring.material.opacity = 0.6 * (1 - t) ** 1.5;
  }).then(() => disposeAll(scene, [core, ring]));
}

/** Phase 2b: fling every non-bridge node radially away from ground zero. */
function blast(center) {
  return tweenNodes((n) => {
    if (state.bridge.nodes.has(n.id)) return n.home;
    const d = { x: n.home.x - center.x, y: n.home.y - center.y, z: n.home.z - center.z };
    const len = Math.hypot(d.x, d.y, d.z) || 1;
    // Closer to ground zero = thrown harder, which reads as a blast wave.
    const force = BLAST_DISTANCE * (0.6 + 0.4 * Math.exp(-len / 200));
    return { x: n.home.x + (d.x / len) * force, y: n.home.y + (d.y / len) * force, z: n.home.z + (d.z / len) * force };
  }, BLAST_MS);
}

async function shockwave(pathIds) {
  const a = vec(state.byId.get(state.source).home);
  const b = vec(state.byId.get(state.target).home);
  const mid = a.clone().add(b).multiplyScalar(0.5);
  await collide(a, b, mid);
  state.bridge.revealed = true;
  restyle();
  await Promise.all([flash(mid), blast(mid)]);
  focusCamera(pathIds);
}

/** Frame the whole cloud. Computed from the layout itself: zoomToFit measures rendered
 *  objects, which can still be sitting at the origin right after the data loads. */
function overview(ms) {
  const radius = Math.max(...state.nodes.map((n) => Math.hypot(n.home.x, n.home.y, n.home.z)), 1);
  lookAt({ x: 0, y: 0, z: 0 }, radius, ms);
}

function focusCamera(pathIds) {
  const points = pathIds.map((id) => state.byId.get(id).home);
  const c = centroid(points);
  const spread = Math.max(...points.map((p) => Math.hypot(p.x - c.x, p.y - c.y, p.z - c.z)), 40);
  lookAt(c, spread * 1.25, 1600);
}

/** Point the camera at `center` from far enough away to fit a sphere of `radius`,
 *  shifted so the subject sits in the space beside the panel rather than under it. */
function lookAt(center, radius, ms) {
  const camera = graph.camera();
  const halfV = (camera.fov * Math.PI) / 360;
  const halfH = Math.atan(Math.tan(halfV) * camera.aspect);
  const panel = document.querySelector('.panel').getBoundingClientRect();
  // Wide screens: the panel covers a strip on the left. Phones: a band along the bottom.
  const wide = window.innerWidth > 800;
  const coveredX = wide ? (panel.right + 16) / window.innerWidth : 0;
  const coveredY = wide ? 0 : (window.innerHeight - panel.top + 16) / window.innerHeight;
  const usableH = Math.atan(Math.tan(halfH) * (1 - coveredX));
  const usableV = Math.atan(Math.tan(halfV) * (1 - coveredY));
  const distance = (radius / Math.sin(Math.min(usableV, usableH))) * 1.05;
  // Shift by the covered share of the visible extent, so the subject centres in what's left.
  const shiftX = distance * Math.tan(halfH) * coveredX;
  const shiftY = distance * Math.tan(halfV) * coveredY;
  const at = { x: center.x - shiftX, y: center.y - shiftY, z: center.z };
  graph.cameraPosition({ ...at, z: center.z + distance }, at, ms);
}

function setStatus(text, isError = false) {
  ui.status.textContent = text;
  ui.status.classList.toggle('error', isError);
}

function renderChain(path, hops) {
  ui.summary.textContent = '';
  ui.chain.replaceChildren(
    ...path.map((id, i) => {
      const li = document.createElement('li');
      const name = document.createElement('span');
      name.className = 'name';
      name.textContent = id;
      li.append(name);
      if (state.placed.some((p) => p.id === id)) {
        li.append(Object.assign(document.createElement('span'), { className: 'tag', textContent: 'new' }));
      }
      if (i > 0) {
        const sim = document.createElement('span');
        sim.className = 'sim';
        sim.textContent = `${Math.round(hops[i - 1].similarity * 100)}% similar`;
        li.append(sim);
      }
      return li;
    }),
  );
}

/** Ask the backend's LLM why each hop connects, and slot the answers under each step. */
async function narrateBridge(path, run) {
  if (!state.features.narration || path.length < 2) return;
  ui.summary.textContent = 'Narrating the bridge…';
  ui.summary.classList.add('pending');
  try {
    const body = await api.narrate(path);
    if (run !== state.run) return; // a newer bridge replaced this one
    [...ui.chain.children].slice(1).forEach((li, i) => {
      li.append(Object.assign(document.createElement('p'), { className: 'why', textContent: body.steps[i] }));
    });
    ui.summary.textContent = body.summary;
  } catch (err) {
    if (run === state.run) ui.summary.textContent = `No narration: ${err.message}`;
  } finally {
    if (run === state.run) ui.summary.classList.remove('pending');
  }
}

function pickNode(node) {
  if (state.bridge || state.busy) return; // finish or reset the current shockwave first
  if (!state.source || state.target) {
    state.source = node.id;
    state.target = null;
  } else {
    state.target = node.id;
  }
  ui.source.value = state.source ?? '';
  ui.target.value = state.target ?? '';
  restyle();
}

/** Add the backend's free-text concepts (and their links) to the scene for this bridge. */
function showPlaced(placed, hops) {
  state.placed = placed.map((p) => ({ ...p, home: { x: p.x, y: p.y, z: p.z }, fx: p.x, fy: p.y, fz: p.z }));
  state.placed.forEach((p) => state.byId.set(p.id, p));
  const placedIds = new Set(placed.map((p) => p.id));
  const extraLinks = hops
    .filter((h) => placedIds.has(h.source) || placedIds.has(h.target))
    .map((h) => ({ ...h }));
  graph.graphData({ nodes: [...state.nodes, ...state.placed], links: [...state.links, ...extraLinks] });
}

function clearPlaced() {
  if (!state.placed.length) return;
  state.placed.forEach((p) => state.byId.delete(p.id));
  state.placed = [];
  graph.graphData({ nodes: state.nodes, links: state.links });
}

async function runBridge(source, target) {
  if (state.busy) return;
  if (source.trim().toLowerCase() === target.trim().toLowerCase()) {
    return setStatus('That\'s the same concept twice. Pick two different ones.', true);
  }
  if (!state.features.free_text) {
    for (const name of [source, target]) {
      if (!findConcept(name)) return setStatus(`"${name}" isn't in the concept space.`, true);
    }
  }
  state.busy = true;
  ui.execute.disabled = ui.surprise.disabled = true;
  const run = ++state.run;
  try {
    if (state.bridge) await reset();
    setStatus(`Routing ${source} → ${target}…`);
    const body = await api.bridge(source, target);

    showPlaced(body.placed, body.hops);
    // The backend resolves case and whitespace, so trust its names over what was typed.
    [state.source, state.target] = [body.path[0], body.path.at(-1)];
    ui.source.value = state.source;
    ui.target.value = state.target;
    state.bridge = {
      nodes: new Set(body.path),
      edges: new Set(body.hops.map((h) => edgeKey(h.source, h.target))),
    };
    restyle();
    renderChain(body.path, body.hops);
    const hops = body.hops.length;
    setStatus(`${hops} hop${hops === 1 ? '' : 's'} from ${state.source} to ${state.target}.`);
    history.replaceState(null, '', `?${new URLSearchParams({ from: state.source, to: state.target })}`);
    await shockwave(body.path);
    narrateBridge(body.path, run);
  } catch (err) {
    setStatus(`Bridge failed: ${err.message}`, true);
  } finally {
    state.busy = false;
    ui.execute.disabled = ui.surprise.disabled = false;
  }
}

function execute(event) {
  event.preventDefault();
  const source = ui.source.value.trim();
  const target = ui.target.value.trim();
  if (source && target) runBridge(source, target);
}

async function reset() {
  state.bridge = null;
  clearPlaced();
  restyle();
  ui.chain.replaceChildren();
  ui.summary.textContent = '';
  await tweenNodes((n) => n.home, RESET_MS);
  overview(800);
}

function findConcept(name) {
  const lower = name.toLowerCase();
  return state.nodes.find((n) => n.id.toLowerCase() === lower);
}

function surprise() {
  const pick = () => state.nodes[Math.floor(Math.random() * state.nodes.length)].id;
  let [a, b] = [pick(), pick()];
  while (a === b) b = pick();
  ui.source.value = a;
  ui.target.value = b;
  runBridge(a, b);
}

async function load() {
  try {
    api = await createApi({ onStatus: (text) => setStatus(text) });
    const [health, space] = await Promise.all([api.health(), api.space()]);
    state.features = health;
    const { nodes, links } = space;
    nodes.forEach((n) => {
      n.home = { x: n.x, y: n.y, z: n.z };
      n.fx = n.x; n.fy = n.y; n.fz = n.z;
      state.byId.set(n.id, n);
    });
    state.nodes = nodes;
    state.links = links;
    ui.list.replaceChildren(...nodes.map((n) => Object.assign(document.createElement('option'), { value: n.id })));
    if (health.free_text) ui.hint.innerHTML = 'Tip: click nodes to pick, or type <em>anything</em>: new phrases get embedded and placed on the fly.';
    if (health.static) {
      ui.hint.innerHTML += ' This demo runs entirely in your browser; <a href="https://github.com/alex-in-CS/semantic-shockwave" target="_blank" rel="noopener">run the full app</a> for LLM narration of each hop.';
    }
    restyle();
    graph.graphData({ nodes, links });
    overview(0);
    setStatus(`${nodes.length} concepts loaded. Pick two.`);

    const params = new URLSearchParams(location.search);
    if (params.get('from') && params.get('to')) {
      ui.source.value = params.get('from');
      ui.target.value = params.get('to');
      setTimeout(() => runBridge(ui.source.value, ui.target.value), 1200);
    }
  } catch (err) {
    setStatus(`Can't reach the backend (${err.message}). Is it running?`, true);
  }
}

ui.form.addEventListener('submit', execute);
ui.surprise.addEventListener('click', surprise);
ui.reset.addEventListener('click', () => {
  if (state.busy) return;
  state.run += 1;
  state.source = state.target = null;
  ui.source.value = ui.target.value = '';
  history.replaceState(null, '', location.pathname);
  setStatus('Pick two concepts.');
  reset();
});
window.addEventListener('resize', () => graph.width(window.innerWidth).height(window.innerHeight));

load();
