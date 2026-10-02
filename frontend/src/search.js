// Search algorithms over the concept graph, written to be watched.
//
// Every algorithm returns { path, trace, explored, note?, paths? }:
//   path      node indices from source to target, or null if it gave up
//   trace     what it did, in order, for the 3D replay:
//               ['v', node, side]        expanded / visited a node
//               ['e', from, to, side]    looked along an edge (frontier, relaxation)
//               ['p', path, side]        a candidate path (ants, Yen, hill climbing)
//             side: 0 = forward, 1 = backward (bidirectional), 2 = pheromone / candidate
//   explored  how many distinct nodes it expanded
//
// Costs: "squared" is the app's bridge metric, (1 - similarity)^2 per hop, which
// prefers many small steps. "angular" is the angle between unit vectors, a true
// metric, so the straight-line angle to the target is an admissible, consistent
// A* heuristic. "hops" counts edges.

const TRACE_LIMIT = 150000; // keeps pathological runs (IDDFS, Bellman-Ford) bounded

// ---------------------------------------------------------------------------
// Graph

/** Build an index-based graph. `vectors` maps id -> Float32Array unit vector (optional). */
export function buildGraph(nodes, links, vectors = new Map()) {
  const ids = nodes.map((n) => n.id);
  const index = new Map(ids.map((id, i) => [id, i]));
  const adj = ids.map(() => []);
  const seen = new Set();
  for (const l of links) {
    const a = index.get(typeof l.source === 'object' ? l.source.id : l.source);
    const b = index.get(typeof l.target === 'object' ? l.target.id : l.target);
    if (a === undefined || b === undefined || a === b) continue;
    const key = a < b ? `${a},${b}` : `${b},${a}`;
    if (seen.has(key)) continue;
    seen.add(key);
    adj[a].push({ to: b, sim: l.similarity });
    adj[b].push({ to: a, sim: l.similarity });
  }
  // Most similar first: gives DFS, beam and hill climbing a sensible neighbour order.
  adj.forEach((edges) => edges.sort((x, y) => y.sim - x.sim));
  return { ids, index, adj, vec: ids.map((id) => vectors.get(id) ?? null) };
}

const clampSim = (s) => Math.min(1, Math.max(-1, s));
export const squaredCost = (sim) => (1 - clampSim(sim)) ** 2;
export const angularCost = (sim) => Math.acos(clampSim(sim));

function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i += 1) s += a[i] * b[i];
  return s;
}

/** Straight-line angle between two nodes' vectors; 0 when either vector is unknown. */
function angleTo(graph, a, b) {
  const va = graph.vec[a];
  const vb = graph.vec[b];
  return va && vb ? Math.acos(clampSim(dot(va, vb))) : 0;
}

function similarity(graph, a, b) {
  const edge = graph.adj[a].find((e) => e.to === b);
  if (edge) return edge.sim;
  const va = graph.vec[a];
  const vb = graph.vec[b];
  return va && vb ? dot(va, vb) : 0;
}

// ---------------------------------------------------------------------------
// Helpers

class MinHeap {
  constructor() { this.items = []; }
  get size() { return this.items.length; }
  peek() { return this.items[0]; }
  push(priority, value) {
    const a = this.items;
    a.push([priority, value]);
    let i = a.length - 1;
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (a[p][0] <= a[i][0]) break;
      [a[p], a[i]] = [a[i], a[p]];
      i = p;
    }
  }
  pop() {
    const a = this.items;
    const top = a[0];
    const last = a.pop();
    if (a.length) {
      a[0] = last;
      let i = 0;
      for (;;) {
        const l = 2 * i + 1;
        const r = l + 1;
        let m = i;
        if (l < a.length && a[l][0] < a[m][0]) m = l;
        if (r < a.length && a[r][0] < a[m][0]) m = r;
        if (m === i) break;
        [a[m], a[i]] = [a[i], a[m]];
        i = m;
      }
    }
    return top;
  }
}

/** Deterministic PRNG, so a replay of the same pair behaves the same. */
export function mulberry32(seed) {
  let t = seed >>> 0;
  return () => {
    t = (t + 0x6d2b79f5) >>> 0;
    let r = Math.imul(t ^ (t >>> 15), 1 | t);
    r = (r + Math.imul(r ^ (r >>> 7), 61 | r)) ^ r;
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}

function walkBack(prev, target) {
  const path = [target];
  while (prev.has(path[0])) path.unshift(prev.get(path[0]));
  return path;
}

function makeTrace() {
  const trace = [];
  const push = (event) => { if (trace.length < TRACE_LIMIT) trace.push(event); };
  return { trace, push };
}

// ---------------------------------------------------------------------------
// Uninformed

function bfs(graph, s, t) {
  const { trace, push } = makeTrace();
  const prev = new Map();
  const seen = new Set([s]);
  const queue = [s];
  let explored = 0;
  for (let head = 0; head < queue.length; head += 1) {
    const u = queue[head];
    push(['v', u, 0]);
    explored += 1;
    if (u === t) return { path: walkBack(prev, t), trace, explored };
    for (const { to } of graph.adj[u]) {
      if (seen.has(to)) continue;
      seen.add(to);
      prev.set(to, u);
      push(['e', u, to, 0]);
      queue.push(to);
    }
  }
  return { path: null, trace, explored };
}

function dfs(graph, s, t) {
  const { trace, push } = makeTrace();
  const prev = new Map();
  const seen = new Set();
  const stack = [s];
  let explored = 0;
  while (stack.length) {
    const u = stack.pop();
    if (seen.has(u)) continue;
    seen.add(u);
    push(['v', u, 0]);
    explored += 1;
    if (u === t) return { path: walkBack(prev, t), trace, explored };
    // Push least similar first, so the most similar neighbour is explored next.
    for (let i = graph.adj[u].length - 1; i >= 0; i -= 1) {
      const { to } = graph.adj[u][i];
      if (seen.has(to)) continue;
      prev.set(to, u);
      push(['e', u, to, 0]);
      stack.push(to);
    }
  }
  return { path: null, trace, explored };
}

function iddfs(graph, s, t, { maxDepth = 40, budget = 400000 } = {}) {
  const { trace, push } = makeTrace();
  const expandedEver = new Set();
  let work = 0;
  for (let limit = 0; limit <= maxDepth; limit += 1) {
    // Best remaining depth each node was reached with in this round: prunes repeats.
    const bestRemaining = new Map();
    const path = [s];
    const onPath = new Set([s]);
    const search = (u, remaining) => {
      work += 1;
      if (work > budget) return 'budget';
      push(['v', u, 0]);
      expandedEver.add(u);
      if (u === t) return 'found';
      if (remaining === 0) return 'cut';
      bestRemaining.set(u, remaining);
      for (const { to } of graph.adj[u]) {
        if (onPath.has(to) || (bestRemaining.get(to) ?? -1) >= remaining - 1) continue;
        push(['e', u, to, 0]);
        path.push(to);
        onPath.add(to);
        const result = search(to, remaining - 1);
        if (result === 'found' || result === 'budget') return result;
        path.pop();
        onPath.delete(to);
      }
      return 'cut';
    };
    const result = search(s, limit);
    if (result === 'found') {
      return { path: [...path], trace, explored: expandedEver.size, note: `found at depth limit ${limit}` };
    }
    if (result === 'budget') break;
  }
  return { path: null, trace, explored: expandedEver.size, note: 'gave up: depth limit or work budget reached' };
}

function bidirectionalBfs(graph, s, t) {
  const { trace, push } = makeTrace();
  if (s === t) return { path: [s], trace, explored: 1 };
  const prev = [new Map(), new Map()];
  const seen = [new Set([s]), new Set([t])];
  let frontier = [[s], [t]];
  let explored = 0;
  while (frontier[0].length && frontier[1].length) {
    // Expand the smaller frontier by one full level.
    const side = frontier[0].length <= frontier[1].length ? 0 : 1;
    const next = [];
    for (const u of frontier[side]) {
      push(['v', u, side]);
      explored += 1;
      for (const { to } of graph.adj[u]) {
        if (seen[side].has(to)) continue;
        seen[side].add(to);
        prev[side].set(to, u);
        push(['e', u, to, side]);
        if (seen[1 - side].has(to)) {
          const forward = walkBack(prev[0], to);
          const backward = walkBack(prev[1], to).reverse();
          return { path: [...forward, ...backward.slice(1)], trace, explored };
        }
        next.push(to);
      }
    }
    frontier[side] = next;
  }
  return { path: null, trace, explored };
}

// ---------------------------------------------------------------------------
// Optimal, weighted

function dijkstraCore(graph, s, t, cost, { banned = null, bannedEdges = null, push = () => {} } = {}) {
  const dist = new Map([[s, 0]]);
  const prev = new Map();
  const done = new Set();
  const heap = new MinHeap();
  heap.push(0, s);
  let explored = 0;
  while (heap.size) {
    const [d, u] = heap.pop();
    if (done.has(u)) continue;
    done.add(u);
    push(['v', u, 0]);
    explored += 1;
    if (u === t) return { path: walkBack(prev, t), cost: d, explored };
    for (const { to, sim } of graph.adj[u]) {
      if (done.has(to) || banned?.has(to) || bannedEdges?.has(`${u},${to}`)) continue;
      const nd = d + cost(sim);
      if (nd < (dist.get(to) ?? Infinity)) {
        dist.set(to, nd);
        prev.set(to, u);
        push(['e', u, to, 0]);
        heap.push(nd, to);
      }
    }
  }
  return { path: null, cost: Infinity, explored };
}

function dijkstra(graph, s, t) {
  const { trace, push } = makeTrace();
  const { path, explored } = dijkstraCore(graph, s, t, squaredCost, { push });
  return { path, trace, explored };
}

function bidirectionalDijkstra(graph, s, t) {
  const { trace, push } = makeTrace();
  if (s === t) return { path: [s], trace, explored: 1 };
  const dist = [new Map([[s, 0]]), new Map([[t, 0]])];
  const prev = [new Map(), new Map()];
  const done = [new Set(), new Set()];
  const heaps = [new MinHeap(), new MinHeap()];
  heaps[0].push(0, s);
  heaps[1].push(0, t);
  let best = Infinity;
  let meet = null;
  let explored = 0;
  while (heaps[0].size && heaps[1].size) {
    // Stop once no path through unsettled nodes can beat the best meeting found.
    if (heaps[0].peek()[0] + heaps[1].peek()[0] >= best) break;
    const side = heaps[0].peek()[0] <= heaps[1].peek()[0] ? 0 : 1;
    const [d, u] = heaps[side].pop();
    if (done[side].has(u)) continue;
    done[side].add(u);
    push(['v', u, side]);
    explored += 1;
    for (const { to, sim } of graph.adj[u]) {
      const nd = d + squaredCost(sim);
      if (nd < (dist[side].get(to) ?? Infinity)) {
        dist[side].set(to, nd);
        prev[side].set(to, u);
        push(['e', u, to, side]);
        heaps[side].push(nd, to);
      }
      const other = dist[1 - side].get(to);
      if (other !== undefined && nd + other < best) {
        best = nd + other;
        meet = to;
      }
    }
  }
  if (meet === null) return { path: null, trace, explored };
  const forward = walkBack(prev[0], meet);
  const backward = walkBack(prev[1], meet).reverse();
  return { path: [...forward, ...backward.slice(1)], trace, explored };
}

function bellmanFord(graph, s, t) {
  const { trace, push } = makeTrace();
  const n = graph.ids.length;
  const dist = new Float64Array(n).fill(Infinity);
  const prev = new Map();
  dist[s] = 0;
  const touched = new Set([s]);
  let rounds = 0;
  for (; rounds < n - 1; rounds += 1) {
    let changed = false;
    for (let u = 0; u < n; u += 1) {
      if (dist[u] === Infinity) continue;
      for (const { to, sim } of graph.adj[u]) {
        const nd = dist[u] + squaredCost(sim);
        if (nd < dist[to] - 1e-12) {
          dist[to] = nd;
          prev.set(to, u);
          push(['e', u, to, 0]);
          if (!touched.has(to)) { touched.add(to); push(['v', to, 0]); }
          changed = true;
        }
      }
    }
    if (!changed) break;
  }
  const path = dist[t] === Infinity ? null : walkBack(prev, t);
  return { path, trace, explored: touched.size, note: `converged after ${rounds + 1} relaxation rounds over every edge` };
}

function aStar(graph, s, t, { weight = 1 } = {}) {
  const { trace, push } = makeTrace();
  const g = new Map([[s, 0]]);
  const prev = new Map();
  const done = new Set();
  const heap = new MinHeap();
  heap.push(weight * angleTo(graph, s, t), s);
  let explored = 0;
  while (heap.size) {
    const [, u] = heap.pop();
    if (done.has(u)) continue;
    done.add(u);
    push(['v', u, 0]);
    explored += 1;
    if (u === t) return { path: walkBack(prev, t), trace, explored };
    for (const { to, sim } of graph.adj[u]) {
      if (done.has(to)) continue;
      const ng = g.get(u) + angularCost(sim);
      if (ng < (g.get(to) ?? Infinity)) {
        g.set(to, ng);
        prev.set(to, u);
        push(['e', u, to, 0]);
        heap.push(ng + weight * angleTo(graph, to, t), to);
      }
    }
  }
  return { path: null, trace, explored };
}

function idaStar(graph, s, t, { budget = 300000 } = {}) {
  const { trace, push } = makeTrace();
  const expanded = new Set();
  let work = 0;
  let bound = angleTo(graph, s, t);
  const path = [s];
  const onPath = new Set([s]);
  const search = (u, g) => {
    const f = g + angleTo(graph, u, t);
    if (f > bound + 1e-12) return f;
    work += 1;
    if (work > budget) return -1;
    push(['v', u, 0]);
    expanded.add(u);
    if (u === t) return 0;
    let min = Infinity;
    for (const { to, sim } of graph.adj[u]) {
      if (onPath.has(to)) continue;
      push(['e', u, to, 0]);
      path.push(to);
      onPath.add(to);
      const r = search(to, g + angularCost(sim));
      if (r === 0 || r === -1) return r;
      if (r < min) min = r;
      path.pop();
      onPath.delete(to);
    }
    return min;
  };
  for (let iteration = 1; iteration <= 200; iteration += 1) {
    const r = search(s, 0);
    if (r === 0) {
      return { path: [...path], trace, explored: expanded.size, note: `found on iteration ${iteration}` };
    }
    if (r === -1 || r === Infinity) break;
    bound = r;
  }
  return { path: null, trace, explored: expanded.size, note: 'gave up: work budget reached. IDA* keeps no memory between passes, and with real-valued costs each pass only raises its limit a tiny bit, so it re-explores the same nodes thousands of times' };
}

// ---------------------------------------------------------------------------
// Heuristic and approximate

function greedy(graph, s, t) {
  const { trace, push } = makeTrace();
  const prev = new Map();
  const seen = new Set([s]);
  const heap = new MinHeap();
  heap.push(angleTo(graph, s, t), s);
  let explored = 0;
  while (heap.size) {
    const [, u] = heap.pop();
    push(['v', u, 0]);
    explored += 1;
    if (u === t) return { path: walkBack(prev, t), trace, explored };
    for (const { to } of graph.adj[u]) {
      if (seen.has(to)) continue;
      seen.add(to);
      prev.set(to, u);
      push(['e', u, to, 0]);
      heap.push(angleTo(graph, to, t), to);
    }
  }
  return { path: null, trace, explored };
}

function beam(graph, s, t, { width = 3, maxDepth = 60 } = {}) {
  const { trace, push } = makeTrace();
  const prev = new Map();
  const seen = new Set([s]);
  let level = [s];
  let explored = 0;
  for (let depth = 0; depth < maxDepth && level.length; depth += 1) {
    const candidates = [];
    for (const u of level) {
      push(['v', u, 0]);
      explored += 1;
      if (u === t) return { path: walkBack(prev, t), trace, explored, note: `beam width ${width}` };
      for (const { to } of graph.adj[u]) {
        if (seen.has(to)) continue;
        seen.add(to);
        prev.set(to, u);
        push(['e', u, to, 0]);
        candidates.push(to);
      }
    }
    // Keep only the `width` candidates that look closest to the target.
    candidates.sort((a, b) => angleTo(graph, a, t) - angleTo(graph, b, t));
    level = candidates.slice(0, width);
  }
  return { path: null, trace, explored, note: `beam width ${width}: every kept candidate led to a dead end` };
}

function hillClimbing(graph, s, t) {
  const { trace, push } = makeTrace();
  const path = [s];
  const onPath = new Set([s]);
  let u = s;
  while (u !== t) {
    push(['v', u, 0]);
    let best = null;
    let bestH = angleTo(graph, u, t);
    for (const { to } of graph.adj[u]) {
      if (onPath.has(to)) continue;
      push(['e', u, to, 0]);
      const h = to === t ? -1 : angleTo(graph, to, t);
      if (h < bestH) { bestH = h; best = to; }
    }
    if (best === null) {
      push(['p', [...path], 2]);
      return {
        path: null,
        partial: path,
        trace,
        explored: path.length,
        note: `stuck at "${graph.ids[u]}": no neighbour is closer to the target (a local optimum)`,
      };
    }
    u = best;
    path.push(u);
    onPath.add(u);
  }
  push(['v', t, 0]);
  return { path, trace, explored: path.length };
}

function randomWalk(graph, s, t, { maxSteps = 5000, bias = 6 } = {}) {
  const { trace, push } = makeTrace();
  const rand = mulberry32(s * 7919 + t);
  const walk = [s];
  const visited = new Set([s]);
  let u = s;
  for (let step = 0; step < maxSteps && u !== t; step += 1) {
    // Biased: neighbours that point toward the target are likelier, but anything can happen.
    const edges = graph.adj[u];
    const weights = edges.map(({ to }) => Math.exp(-bias * angleTo(graph, to, t)));
    let r = rand() * weights.reduce((a, b) => a + b, 0);
    let next = edges[edges.length - 1].to;
    for (let i = 0; i < edges.length; i += 1) {
      r -= weights[i];
      if (r <= 0) { next = edges[i].to; break; }
    }
    push(['e', u, next, 0]);
    push(['v', next, 0]);
    visited.add(next);
    walk.push(next);
    u = next;
  }
  if (u !== t) {
    return { path: null, trace, explored: visited.size, note: `wandered ${maxSteps} steps without arriving` };
  }
  // Loop-erase the walk into a simple path.
  const path = [];
  const position = new Map();
  for (const v of walk) {
    if (position.has(v)) {
      path.length = position.get(v) + 1;
      for (const [k, i] of position) if (i >= path.length) position.delete(k);
    } else {
      position.set(v, path.length);
      path.push(v);
    }
  }
  return { path, trace, explored: visited.size, note: `walked ${walk.length - 1} steps, loop-erased to ${path.length - 1} hops` };
}

function antColony(graph, s, t, { ants = 24, iterations = 14, maxSteps = 80, alpha = 1, beta = 4, evaporation = 0.35 } = {}) {
  const { trace, push } = makeTrace();
  const rand = mulberry32(s * 104729 + t);
  const pheromone = new Map();
  const tau = (a, b) => pheromone.get(a < b ? `${a},${b}` : `${b},${a}`) ?? 1;
  const visited = new Set();
  let best = null;
  let bestCost = Infinity;
  for (let it = 0; it < iterations; it += 1) {
    const found = [];
    for (let ant = 0; ant < ants; ant += 1) {
      const path = [s];
      const onPath = new Set([s]);
      let u = s;
      for (let step = 0; step < maxSteps && u !== t; step += 1) {
        const options = graph.adj[u].filter(({ to }) => !onPath.has(to));
        if (!options.length) break;
        // Pheromone (what earlier ants liked) times desirability (pointing at the target).
        const weights = options.map(({ to }) => tau(u, to) ** alpha * (1 / (0.05 + angleTo(graph, to, t))) ** beta);
        let r = rand() * weights.reduce((a, b) => a + b, 0);
        let next = options[options.length - 1].to;
        for (let i = 0; i < options.length; i += 1) {
          r -= weights[i];
          if (r <= 0) { next = options[i].to; break; }
        }
        path.push(next);
        onPath.add(next);
        visited.add(next);
        u = next;
      }
      if (u === t) found.push(path);
    }
    for (const [key, value] of pheromone) pheromone.set(key, value * (1 - evaporation));
    for (const path of found) {
      const cost = pathCost(graph, path, squaredCost);
      for (let i = 0; i < path.length - 1; i += 1) {
        const [a, b] = [path[i], path[i + 1]];
        const key = a < b ? `${a},${b}` : `${b},${a}`;
        pheromone.set(key, (pheromone.get(key) ?? 1) + 1 / (cost + 0.05));
        push(['e', a, b, 2]);
      }
      if (cost < bestCost) { bestCost = cost; best = path; }
    }
    if (best) push(['p', best, 2]);
  }
  return {
    path: best,
    trace,
    explored: visited.size,
    note: best ? `${ants} ants x ${iterations} rounds` : 'no ant reached the target',
  };
}

function yen(graph, s, t, { k = 3 } = {}) {
  const { trace, push } = makeTrace();
  const first = dijkstraCore(graph, s, t, squaredCost, { push });
  if (!first.path) return { path: null, trace, explored: first.explored };
  const paths = [first.path];
  push(['p', first.path, 2]);
  const candidates = [];
  const seenPaths = new Set([first.path.join(',')]);
  let explored = first.explored;
  for (let n = 1; n < k; n += 1) {
    const last = paths[n - 1];
    for (let i = 0; i < last.length - 1; i += 1) {
      const spur = last[i];
      const root = last.slice(0, i + 1);
      const bannedEdges = new Set();
      for (const p of paths) {
        if (p.length > i && root.every((v, j) => p[j] === v)) {
          bannedEdges.add(`${p[i]},${p[i + 1]}`);
        }
      }
      const banned = new Set(root.slice(0, -1));
      const spurPath = dijkstraCore(graph, spur, t, squaredCost, { banned, bannedEdges });
      explored += spurPath.explored;
      if (!spurPath.path) continue;
      const total = [...root.slice(0, -1), ...spurPath.path];
      const key = total.join(',');
      if (seenPaths.has(key)) continue;
      seenPaths.add(key);
      candidates.push([pathCost(graph, total, squaredCost), total]);
    }
    if (!candidates.length) break;
    candidates.sort((a, b) => a[0] - b[0]);
    const [, next] = candidates.shift();
    paths.push(next);
    push(['p', next, 2]);
  }
  return { path: paths[0], paths, trace, explored, note: `${paths.length} best distinct bridges` };
}

// ---------------------------------------------------------------------------
// Catalogue

export const ALGORITHMS = [
  { id: 'dijkstra', name: 'Dijkstra', family: 'Optimal', metric: 'squared', run: dijkstra,
    blurb: 'Expands outward in order of total cost. Always finds the cheapest bridge (the app\'s default).' },
  { id: 'bidijkstra', name: 'Bidirectional Dijkstra', family: 'Optimal', metric: 'squared', run: bidirectionalDijkstra,
    blurb: 'Two Dijkstra searches, one from each end, meeting in the middle. Same answer, far fewer nodes.' },
  { id: 'astar', name: 'A*', family: 'Optimal', metric: 'angular', run: aStar,
    blurb: 'Dijkstra plus a compass: the straight-line angle to the target. Optimal for angular distance.' },
  { id: 'idastar', name: 'IDA*', family: 'Optimal', metric: 'angular', run: idaStar,
    blurb: 'A* with almost no memory: repeated depth-first passes with a rising cost limit.' },
  { id: 'bellmanford', name: 'Bellman-Ford', family: 'Optimal', metric: 'squared', run: bellmanFord,
    blurb: 'Relaxes every edge, round after round, until nothing improves. Slow but simple.' },
  { id: 'bfs', name: 'Breadth-first search', family: 'Uninformed', metric: 'hops', run: bfs,
    blurb: 'Ring by ring outward. Fewest hops, but ignores how similar each step is.' },
  { id: 'bibfs', name: 'Bidirectional BFS', family: 'Uninformed', metric: 'hops', run: bidirectionalBfs,
    blurb: 'BFS from both ends at once until the two waves touch.' },
  { id: 'dfs', name: 'Depth-first search', family: 'Uninformed', metric: 'hops', run: dfs,
    blurb: 'Dives down one trail as far as it goes before backtracking. Finds a path, rarely a good one.' },
  { id: 'iddfs', name: 'Iterative deepening DFS', family: 'Uninformed', metric: 'hops', run: iddfs,
    blurb: 'Depth-first, but with a depth limit that grows by one each round: BFS-like answers, DFS-like memory.' },
  { id: 'weightedastar', name: 'Weighted A* (w = 2.5)', family: 'Heuristic', metric: 'angular', run: (g, s, t) => aStar(g, s, t, { weight: 2.5 }),
    blurb: 'Trusts the compass 2.5x more: much faster, no longer guaranteed optimal.' },
  { id: 'greedy', name: 'Greedy best-first', family: 'Heuristic', metric: 'angular', run: greedy,
    blurb: 'Always expands whatever looks closest to the target. Quick, often wanders.' },
  { id: 'beam', name: 'Beam search (width 3)', family: 'Heuristic', metric: 'angular', run: beam,
    blurb: 'Keeps only the 3 most promising nodes per level. Cheap, can prune the only way through.' },
  { id: 'hillclimb', name: 'Hill climbing', family: 'Heuristic', metric: 'angular', run: hillClimbing,
    blurb: 'Always steps to a neighbour closer to the target, never back. Gets stuck on local optima.' },
  { id: 'randomwalk', name: 'Random walk (biased)', family: 'Stochastic', metric: 'hops', run: randomWalk,
    blurb: 'Stumbles from neighbour to neighbour, nudged toward the target. Loops are erased at the end.' },
  { id: 'aco', name: 'Ant colony optimization', family: 'Stochastic', metric: 'squared', run: antColony,
    blurb: 'Swarms of ants lay pheromone on good bridges; later ants follow the strongest trails.' },
  { id: 'yen', name: "Yen's k-shortest paths (k = 3)", family: 'Multi-path', metric: 'squared', run: yen,
    blurb: 'The best bridge, then the 2nd and 3rd best that differ from it. Shows the alternatives.' },
];

export const ALGORITHM_BY_ID = new Map(ALGORITHMS.map((a) => [a.id, a]));

export function pathCost(graph, path, cost = squaredCost) {
  let total = 0;
  for (let i = 0; i < path.length - 1; i += 1) total += cost(similarity(graph, path[i], path[i + 1]));
  return total;
}

/** Run one algorithm by id and attach comparable metrics. */
export function runSearch(graph, algorithmId, source, target) {
  const algo = ALGORITHM_BY_ID.get(algorithmId);
  if (!algo) throw new Error(`unknown algorithm: ${algorithmId}`);
  const s = graph.index.get(source);
  const t = graph.index.get(target);
  if (s === undefined || t === undefined) throw new Error('both concepts must be in the graph');
  const started = performance.now();
  const result = algo.run(graph, s, t);
  const ms = performance.now() - started;
  const path = result.path;
  const sims = path ? path.slice(1).map((v, i) => similarity(graph, path[i], v)) : [];
  return {
    ...result,
    algorithm: algo,
    ms,
    hops: path ? path.length - 1 : null,
    cost: path ? pathCost(graph, path, squaredCost) : null,
    weakest: sims.length ? Math.min(...sims) : null,
    sims,
  };
}
