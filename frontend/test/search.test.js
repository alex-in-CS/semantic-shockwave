import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import { test } from 'node:test';

import {
  ALGORITHMS, angularCost, buildGraph, mulberry32, pathCost, runSearch, squaredCost,
} from '../src/search.js';

// A random "concept space": unit vectors in 8D, each linked to its 4 nearest
// neighbours, plus a chain through all nodes so the graph is always connected.
function randomSpace(n = 60, seed = 1) {
  const rand = mulberry32(seed);
  const vectors = new Map();
  const nodes = [];
  for (let i = 0; i < n; i += 1) {
    const v = Float32Array.from({ length: 8 }, () => rand() * 2 - 1);
    const norm = Math.hypot(...v);
    vectors.set(`n${i}`, v.map((x) => x / norm));
    nodes.push({ id: `n${i}` });
  }
  const sim = (a, b) => vectors.get(a).reduce((s, x, i) => s + x * vectors.get(b)[i], 0);
  const links = [];
  for (const { id: a } of nodes) {
    nodes.filter(({ id }) => id !== a)
      .map(({ id }) => [sim(a, id), id])
      .sort((x, y) => y[0] - x[0])
      .slice(0, 4)
      .forEach(([s, b]) => links.push({ source: a, target: b, similarity: s }));
  }
  for (let i = 0; i < n - 1; i += 1) {
    links.push({ source: `n${i}`, target: `n${i + 1}`, similarity: sim(`n${i}`, `n${i + 1}`) });
  }
  return buildGraph(nodes, links, vectors);
}

/** Brute-force reference: Dijkstra with a plain array scan, for any edge cost. */
function reference(graph, s, t, cost) {
  const n = graph.ids.length;
  const dist = new Array(n).fill(Infinity);
  const done = new Array(n).fill(false);
  dist[s] = 0;
  for (;;) {
    let u = -1;
    for (let i = 0; i < n; i += 1) if (!done[i] && (u < 0 || dist[i] < dist[u])) u = i;
    if (u < 0 || dist[u] === Infinity) break;
    done[u] = true;
    for (const { to, sim } of graph.adj[u]) dist[to] = Math.min(dist[to], dist[u] + cost(sim));
  }
  return dist[t];
}

function fewestHops(graph, s, t) {
  const depth = new Map([[s, 0]]);
  const queue = [s];
  for (let i = 0; i < queue.length; i += 1) {
    for (const { to } of graph.adj[queue[i]]) {
      if (!depth.has(to)) { depth.set(to, depth.get(queue[i]) + 1); queue.push(to); }
    }
  }
  return depth.get(t);
}

function assertValidPath(graph, path, s, t) {
  assert.equal(path[0], s);
  assert.equal(path.at(-1), t);
  assert.equal(new Set(path).size, path.length, 'path must not repeat nodes');
  for (let i = 0; i < path.length - 1; i += 1) {
    assert.ok(graph.adj[path[i]].some((e) => e.to === path[i + 1]), `no edge ${path[i]}-${path[i + 1]}`);
  }
}

const PAIRS = [[0, 59], [3, 41], [17, 22], [50, 8]];

test('every algorithm returns a valid simple path, or an honest failure', () => {
  const graph = randomSpace();
  for (const algo of ALGORITHMS) {
    for (const [s, t] of PAIRS) {
      const r = runSearch(graph, algo.id, `n${s}`, `n${t}`);
      assert.ok(Array.isArray(r.trace) && r.trace.length > 0, `${algo.id} should leave a trace`);
      if (r.path) assertValidPath(graph, r.path, s, t);
      else assert.ok(r.note, `${algo.id} failed without explaining why`);
    }
  }
});

test('squared-cost optimal algorithms all find the cheapest bridge', () => {
  const graph = randomSpace();
  for (const [s, t] of PAIRS) {
    const best = reference(graph, s, t, squaredCost);
    for (const id of ['dijkstra', 'bidijkstra', 'bellmanford', 'yen']) {
      const r = runSearch(graph, id, `n${s}`, `n${t}`);
      assert.ok(Math.abs(r.cost - best) < 1e-9, `${id}: ${r.cost} vs optimal ${best}`);
    }
  }
});

test('A* and IDA* are optimal for angular distance', () => {
  const graph = randomSpace();
  for (const [s, t] of PAIRS) {
    const best = reference(graph, s, t, angularCost);
    for (const id of ['astar', 'idastar']) {
      const r = runSearch(graph, id, `n${s}`, `n${t}`);
      if (!r.path) { assert.equal(id, 'idastar', 'only IDA* may give up (work budget)'); continue; }
      assert.ok(Math.abs(pathCost(graph, r.path, angularCost) - best) < 1e-9, `${id} not optimal`);
    }
  }
});

test('the A* heuristic prunes part of the graph', () => {
  const graph = randomSpace(200, 7);
  const astar = runSearch(graph, 'astar', 'n0', 'n150');
  const bfsAll = graph.ids.length;
  assert.ok(astar.explored < bfsAll, 'the heuristic should prune something');
});

test('BFS and bidirectional BFS find the fewest hops; so does IDDFS', () => {
  const graph = randomSpace();
  for (const [s, t] of PAIRS) {
    const hops = fewestHops(graph, s, t);
    for (const id of ['bfs', 'bibfs', 'iddfs']) {
      assert.equal(runSearch(graph, id, `n${s}`, `n${t}`).hops, hops, id);
    }
  }
});

test("Yen's paths are distinct and in cost order", () => {
  const graph = randomSpace();
  const r = runSearch(graph, 'yen', 'n0', 'n59');
  assert.equal(r.paths.length, 3);
  assert.equal(new Set(r.paths.map((p) => p.join())).size, 3);
  const costs = r.paths.map((p) => pathCost(graph, p));
  assert.ok(costs[0] <= costs[1] + 1e-12 && costs[1] <= costs[2] + 1e-12, costs.join());
  r.paths.forEach((p) => assertValidPath(graph, p, 0, 59));
});

test('stochastic algorithms are deterministic for the same pair', () => {
  const graph = randomSpace();
  for (const id of ['randomwalk', 'aco']) {
    const a = runSearch(graph, id, 'n3', 'n41');
    const b = runSearch(graph, id, 'n3', 'n41');
    assert.deepEqual(a.path, b.path, id);
  }
});

test('same source and target is a zero-hop path', () => {
  const graph = randomSpace();
  for (const id of ['dijkstra', 'bidijkstra', 'bfs', 'bibfs', 'astar']) {
    assert.deepEqual(runSearch(graph, id, 'n5', 'n5').path, [5], id);
  }
});

// Cross-check against the Python backend (networkx) when an export is present.
const SPACE = new URL('../public/space.json', import.meta.url);
test('JS Dijkstra finds the same bridges as the Python backend', { skip: !existsSync(SPACE) && 'no public/space.json (run shockwave.export)' }, () => {
  const space = JSON.parse(readFileSync(SPACE, 'utf8'));
  const graph = buildGraph(space.nodes, space.links);
  const curated = [...(space.curated.examples ?? []), ...(space.curated.challenges ?? []).slice(0, 60)];
  assert.ok(curated.length > 0, 'export has no curated bridges to compare');
  let same = 0;
  for (const { source, target, path } of curated) {
    const r = runSearch(graph, 'dijkstra', source, target);
    const ids = r.path.map((i) => graph.ids[i]);
    if (ids.join('|') === path.join('|')) same += 1;
    else {
      // Different only if tied: costs must agree to rounding.
      const pyCost = pathCost(graph, path.map((id) => graph.index.get(id)));
      assert.ok(Math.abs(pyCost - r.cost) < 1e-3, `${source} -> ${target}: ${ids.join(' > ')} vs ${path.join(' > ')}`);
    }
  }
  assert.ok(same / curated.length > 0.95, `only ${same}/${curated.length} identical`);
});
