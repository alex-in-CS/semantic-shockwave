import { defineConfig } from 'vite';

export default defineConfig({
  // Relative asset paths: the same build works at a domain root or under /semantic-shockwave/.
  base: './',
  // In dev, /api goes to the FastAPI backend, so the app always talks to its own origin.
  server: { proxy: { '/api': 'http://127.0.0.1:8000' } },
  // three.js alone is ~1.2 MB minified; one chunk is fine for a single-page demo.
  build: { chunkSizeWarningLimit: 1600 },
});
