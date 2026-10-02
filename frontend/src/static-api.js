// The backend, reimplemented in the browser for the static build.
//
// space.json (from `python -m shockwave.export`) carries the UMAP layout, the
// kNN + MST graph and int8-quantized vocabulary vectors. Routing is the same
// squared-cosine-distance shortest path. Free text is embedded with the same
// model via transformers.js; since UMAP can't run here, a new phrase is placed at
// the similarity-weighted centroid of its nearest concepts instead of UMAP.transform.

const MAX_CONCEPT_LENGTH = 60;
// Same all-MiniLM-L6-v2 as the backend; this repo ships the 8-bit quantized ONNX file.
const MODEL = 'Xenova/all-MiniLM-L6-v2';

class ApiError extends Error {}

function decodeVectors({ data, dims, scale }, count) {
  const bytes = Uint8Array.from(atob(data), (c) => c.charCodeAt(0));
  const raw = new Int8Array(bytes.buffer);
  const vectors = [];
  for (let i = 0; i < count; i += 1) {
    const v = Float32Array.from(raw.subarray(i * dims, (i + 1) * dims), (x) => x / scale);
    vectors.push(normalize(v));
  }
  return vectors;
}

function normalize(v) {
  const norm = Math.hypot(...v) || 1;
  return v.map((x) => x / norm);
}

const dot = (a, b) => a.reduce((sum, x, i) => sum + x * b[i], 0);
const clampDistance = (similarity) => Math.min(2, Math.max(0, 1 - similarity));

/** Plain O(V^2) Dijkstra: a few hundred nodes don't need a heap. */
function shortestPath(adjacency, source, target) {
  const dist = new Map([[source, 0]]);
  const prev = new Map();
  const done = new Set();
  while (true) {
    let current = null;
    for (const [node, d] of dist) {
      if (!done.has(node) && (current === null || d < dist.get(current))) current = node;
    }
    if (current === null) throw new ApiError('no bridge between those concepts');
    if (current === target) break;
    done.add(current);
    for (const { to, cost } of adjacency.get(current) ?? []) {
      const candidate = dist.get(current) + cost;
      if (!done.has(to) && candidate < (dist.get(to) ?? Infinity)) {
        dist.set(to, candidate);
        prev.set(to, current);
      }
    }
  }
  const path = [target];
  while (path[0] !== source) path.unshift(prev.get(path[0]));
  return path;
}

export async function staticApi({ onStatus = () => {} } = {}) {
  const space = await fetch(`${import.meta.env.BASE_URL}space.json`).then((res) => {
    if (!res.ok) throw new Error(`space.json: HTTP ${res.status}`);
    return res.json();
  });
  const labels = space.nodes.map((n) => n.id);
  const byLower = new Map(labels.map((id) => [id.toLowerCase(), id]));
  const coords = new Map(space.nodes.map((n) => [n.id, n]));
  const vectors = decodeVectors(space.vectors, labels.length);

  const link = ({ adjacency, similarity }, a, b, sim) => {
    const cost = clampDistance(sim) ** 2;
    adjacency.get(a).push({ to: b, cost });
    adjacency.get(b).push({ to: a, cost });
    similarity.set(`${a}\u0000${b}`, sim).set(`${b}\u0000${a}`, sim);
  };
  const base = { adjacency: new Map(labels.map((id) => [id, []])), similarity: new Map() };
  space.links.forEach((l) => link(base, l.source, l.target, l.similarity));

  let extractor = null;
  async function embed(text) {
    extractor ??= (async () => {
      onStatus('Loading the embedding model (first time only, about 23 MB)…');
      const { pipeline } = await import('@huggingface/transformers');
      return pipeline('feature-extraction', MODEL, { dtype: 'q8' });
    })().catch((err) => { extractor = null; throw err; });
    const output = await (await extractor)(text, { pooling: 'mean', normalize: true });
    return Float32Array.from(output.data);
  }

  return {
    health: async () => ({ status: 'ok', free_text: true, narration: false, static: true }),

    // Fresh copies: the 3D graph rewrites link endpoints into node objects.
    space: async () => ({
      nodes: space.nodes.map((n) => ({ ...n })),
      links: space.links.map((l) => ({ ...l })),
    }),

    async bridge(source, target) {
      const extras = new Map(); // name -> unit vector
      const names = [];
      for (const term of [source.trim(), target.trim()]) {
        if (!term || term.length > MAX_CONCEPT_LENGTH) {
          throw new ApiError(`concepts must be 1-${MAX_CONCEPT_LENGTH} characters`);
        }
        const known = byLower.get(term.toLowerCase());
        if (known) { names.push(known); continue; }
        names.push(term);
        if (!extras.has(term)) extras.set(term, await embed(term));
      }

      // Per-request copy, as on the server: extras never leak into the shared graph.
      const graph = {
        adjacency: new Map([...base.adjacency].map(([id, edges]) => [id, [...edges]])),
        similarity: new Map(base.similarity),
      };
      const poolLabels = [...labels];
      const poolVectors = [...vectors];
      const placed = [];
      for (const [name, vector] of extras) {
        const sims = poolVectors.map((v) => dot(v, vector));
        const nearest = sims.map((s, i) => [s, i]).sort((a, b) => b[0] - a[0]).slice(0, space.k);
        graph.adjacency.set(name, []);
        nearest.forEach(([sim, i]) => link(graph, name, poolLabels[i], sim));

        // Similarity-weighted centroid of the nearest concepts that already have positions.
        let [wx, wy, wz, total] = [0, 0, 0, 0];
        for (const [sim, i] of nearest) {
          const p = coords.get(poolLabels[i]);
          if (!p) continue;
          const w = Math.max(sim, 0) ** 4 + 1e-6; // sharpen toward the closest neighbours
          wx += p.x * w; wy += p.y * w; wz += p.z * w; total += w;
        }
        const point = { id: name, x: wx / total, y: wy / total, z: wz / total };
        coords.set(name, point);
        placed.push(point);
        poolLabels.push(name);
        poolVectors.push(vector);
      }

      const path = shortestPath(graph.adjacency, names[0], names[1]);
      extras.forEach((_, name) => coords.delete(name));
      const hops = path.slice(1).map((b, i) => ({
        source: path[i],
        target: b,
        similarity: Math.round(graph.similarity.get(`${path[i]}\u0000${b}`) * 1e4) / 1e4,
      }));
      return { path, hops, placed };
    },

    narrate: async () => { throw new ApiError('narration needs the full app with an LLM; see the README'); },
  };
}
