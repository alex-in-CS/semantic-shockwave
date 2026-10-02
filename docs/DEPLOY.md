# Deploying Semantic Shockwave

There are two builds, and they host differently.

| Build | Needs | Narration | Cost |
|---|---|---|---|
| **Static demo** (`VITE_STATIC=1`) | any static file host | no | free |
| **Full app** (Docker image) | a container host with ~1 GB RAM | yes | depends on the host |

## Static demo on GitHub Pages (what the live demo uses)

[`.github/workflows/pages.yml`](../.github/workflows/pages.yml) does everything
on each push to `main`:

1. Installs the backend and runs `python -m shockwave.export`, which writes
   `space.json` (the UMAP layout, the bridge graph and int8 vocabulary vectors).
2. Builds the frontend with `VITE_STATIC=1`, which swaps the HTTP API for an
   in-browser one (Dijkstra in JavaScript, free text via transformers.js).
3. Publishes `frontend/dist` to GitHub Pages.

One-time setup: in the repo's *Settings → Pages*, set **Source** to
**GitHub Actions** (or run
`gh api -X POST repos/<owner>/<repo>/pages -f build_type=workflow`).

Any other static host works the same way: run the two commands from the
README's "Build the static demo yourself" section and upload `frontend/dist`.
The build uses relative paths, so it can live at a domain root or in a subfolder.

## Full app as a container

```bash
docker build -t semantic-shockwave .
docker run -p 7860:7860 -e GROQ_API_KEY=gsk_... semantic-shockwave
```

The image bakes in the model and vocabulary embeddings and boots without network
access (`HF_HUB_OFFLINE=1`). The platform must route traffic to the container's
`PORT` (default 7860). Measured locally: about 700 MB of RAM, and about 90
seconds from start until `/api/v1/health` answers (UMAP plus JIT compilation).
So:

- **Hosts with 512 MB free tiers are too small.** Give it 1 GB at least.
- **Scale-to-zero hosts** (Cloud Run and similar) make the first visit after an
  idle period wait for that 90-second boot, unless one instance stays warm.
- **Hugging Face Spaces:** Docker Spaces need a PRO subscription (since 2026;
  only static Spaces are free). With PRO, create a Docker Space with
  `app_port: 7860` in its README front matter, upload `backend/`, `frontend/`,
  `Dockerfile`, `.dockerignore` and `LICENSE`, and add `GROQ_API_KEY` as a
  Space secret.

Set `GROQ_API_KEY` (or `SHOCKWAVE_LLM_BASE_URL`) as a secret in whatever host
you use; never commit it.

To host the frontend separately from the API, build it with
`VITE_API_URL=https://your-api` and set `SHOCKWAVE_CORS_ORIGINS` on the backend
to the frontend's origin.
