# One container serves the whole app: FastAPI on $PORT, with the built frontend at /.
# Defaults suit Hugging Face Spaces (Docker SDK: port 7860, non-root uid 1000).

# --- 1. Frontend ---------------------------------------------------------------
FROM node:22-slim AS frontend
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# --- 2. Backend ----------------------------------------------------------------
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.10 /uv /bin/uv

RUN useradd --create-home --uid 1000 user
# UV_NO_CACHE: otherwise uv's wheel cache lands in the layer and stores torch twice.
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1 \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH=/app/.venv/bin:$PATH \
    HF_HOME=/app/.hf \
    NUMBA_CACHE_DIR=/tmp/numba \
    HOST=0.0.0.0 \
    PORT=7860

WORKDIR /app/backend
# Dependencies first (CPU-only torch, per pyproject), so code edits don't reinstall them.
COPY backend/pyproject.toml backend/uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY backend/ ./
RUN uv sync --frozen --no-dev

# Bake the model and the vocabulary's embeddings into the image: startup needs no network.
RUN python -c "from pathlib import Path; \
from shockwave.concepts import load_concepts; \
from shockwave.embeddings import cached_embed; \
cached_embed(load_concepts(Path('data/concepts.txt')), Path('.cache'))"

# The model is baked in above, so never contact the Hugging Face Hub at runtime.
ENV HF_HUB_OFFLINE=1

COPY --from=frontend /app/frontend/dist /app/frontend/dist
RUN chown -R user /app
USER user

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s \
  CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/api/v1/health')"
CMD ["python", "-m", "shockwave"]
