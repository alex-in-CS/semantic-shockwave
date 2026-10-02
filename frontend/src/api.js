// Two ways to answer the same four questions. The full app asks the FastAPI backend;
// the static build (VITE_STATIC=1, e.g. GitHub Pages) answers in the browser from a
// precomputed space.json. Both resolve to the backend's JSON shapes and throw on errors.

// Empty = same origin: the dev server proxies /api, and in production FastAPI serves this page.
const API = import.meta.env.VITE_API_URL ?? '';

async function call(path, body) {
  const res = await fetch(`${API}/api/v1/${path}`, body && {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(typeof data.detail === 'string' ? data.detail : `HTTP ${res.status}`);
  return data;
}

function httpApi() {
  return {
    health: () => call('health'),
    space: () => call('space'),
    bridge: (source, target) => call('bridge', { source, target }),
    narrate: (path) => call('narrate', { path }),
  };
}

export async function createApi(options = {}) {
  if (!import.meta.env.VITE_STATIC) return httpApi();
  const { staticApi } = await import('./static-api.js');
  return staticApi(options);
}
