// The backend, reimplemented in the browser for the static build.
//
// space.json (from `python -m shockwave.export`) is exactly what /api/v1/space
// serves. Free text is embedded with the same model via transformers.js; since
// UMAP can't run here, a new phrase is placed at the similarity-weighted centroid
// of its nearest concepts instead of UMAP.transform. Searching happens in the
// app itself (search.js) for both builds, so this only has to resolve and place.

import { decodeVectors } from './vectors.js';

const MAX_CONCEPT_LENGTH = 60;
const PLACED_GROUP = 'Your phrases';

class ApiError extends Error {}

const dot = (a, b) => a.reduce((sum, x, i) => sum + x * b[i], 0);

export async function staticApi({ onStatus = () => {} } = {}) {
  const space = await fetch(`${import.meta.env.BASE_URL}space.json`).then((res) => {
    if (!res.ok) throw new Error(`space.json: HTTP ${res.status}`);
    return res.json();
  });
  const labels = space.nodes.map((n) => n.id);
  const byLower = new Map(labels.map((id) => [id.toLowerCase(), id]));
  const coords = new Map(space.nodes.map((n) => [n.id, n]));
  const vectors = decodeVectors(space.vectors, labels.length);
  const model = space.browser_model ?? { id: 'Xenova/bge-small-en-v1.5', pooling: 'cls', dtype: 'q8' };

  let extractor = null;
  async function embed(text) {
    extractor ??= (async () => {
      onStatus('Loading the embedding model (first time only, about 34 MB)…');
      const { pipeline } = await import('@huggingface/transformers');
      return pipeline('feature-extraction', model.id, { dtype: model.dtype });
    })().catch((err) => { extractor = null; throw err; });
    const output = await (await extractor)(text, { pooling: model.pooling, normalize: true });
    return Float32Array.from(output.data);
  }

  return {
    health: async () => ({ status: 'ok', free_text: true, narration: false, static: true }),

    // Fresh copies, so callers can annotate nodes without touching the cached payload.
    space: async () => ({
      ...space,
      nodes: space.nodes.map((n) => ({ ...n })),
      links: space.links.map((l) => ({ ...l })),
    }),

    async bridge(source, target) {
      const extras = new Map(); // name -> unit vector
      for (const term of [source.trim(), target.trim()]) {
        if (!term || term.length > MAX_CONCEPT_LENGTH) {
          throw new ApiError(`concepts must be 1-${MAX_CONCEPT_LENGTH} characters`);
        }
        if (!byLower.has(term.toLowerCase()) && !extras.has(term)) extras.set(term, await embed(term));
      }

      const poolLabels = [...labels];
      const poolVectors = [...vectors];
      const placed = [];
      const links = [];
      for (const [name, vector] of extras) {
        const nearest = poolVectors
          .map((v, i) => [dot(v, vector), i])
          .sort((a, b) => b[0] - a[0])
          .slice(0, space.k);
        nearest.forEach(([sim, i]) => links.push({ source: name, target: poolLabels[i], similarity: Math.round(sim * 1e4) / 1e4 }));

        // Similarity-weighted centroid of the nearest concepts that already have positions.
        let [wx, wy, wz, total] = [0, 0, 0, 0];
        for (const [sim, i] of nearest) {
          const p = coords.get(poolLabels[i]) ?? placed.find((q) => q.id === poolLabels[i]);
          if (!p) continue;
          const w = Math.max(sim, 0) ** 4 + 1e-6; // sharpen toward the closest neighbours
          wx += p.x * w; wy += p.y * w; wz += p.z * w; total += w;
        }
        placed.push({
          id: name, group: PLACED_GROUP, x: wx / total, y: wy / total, z: wz / total,
          vector: Array.from(vector, (x) => Math.round(x * 1e4) / 1e4),
        });
        poolLabels.push(name);
        poolVectors.push(vector);
      }
      // The full app returns the server's Dijkstra path too; here the app searches itself.
      return { path: null, hops: [], placed, links };
    },

    narrate: async () => { throw new ApiError('narration needs the full app with an LLM; see the README'); },
  };
}
