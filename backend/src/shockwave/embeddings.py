"""Turn text into unit-length vectors, cached on disk so restarts skip the model."""

from __future__ import annotations

import hashlib
import threading
from collections.abc import Callable
from pathlib import Path

import numpy as np

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

Embedder = Callable[[list[str]], np.ndarray]


def sentence_transformer_embedder(model_name: str = DEFAULT_MODEL) -> Embedder:
    # Heavy import (torch); only paid on a cache miss.
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)

    def embed(texts: list[str]) -> np.ndarray:
        return model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

    return embed


def lazy_embedder(
    model_name: str = DEFAULT_MODEL,
    make_embedder: Callable[[str], Embedder] = sentence_transformer_embedder,
) -> Embedder:
    """Embed free-text queries as unit vectors. The model loads on first use, once,
    so a server that never sees free text never pays for it."""
    lock = threading.Lock()
    loaded: list[Embedder] = []

    def embed(texts: list[str]) -> np.ndarray:
        with lock:
            if not loaded:
                loaded.append(make_embedder(model_name))
        return normalize(loaded[0](texts))

    return embed


def normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return (vectors / np.clip(norms, 1e-12, None)).astype(np.float32)


def cached_embed(
    texts: list[str],
    cache_dir: Path,
    model_name: str = DEFAULT_MODEL,
    make_embedder: Callable[[str], Embedder] = sentence_transformer_embedder,
) -> np.ndarray:
    """Embed `texts` as an (N, D) array of unit vectors, reusing a cached copy when present.

    The cache key covers the model name and the exact text list, so editing the
    vocabulary or switching models invalidates it automatically.
    """
    key = hashlib.sha256("\n".join([model_name, *texts]).encode()).hexdigest()[:16]
    path = cache_dir / f"embeddings-{key}.npy"
    if path.exists():
        return np.load(path)

    vectors = normalize(make_embedder(model_name)(texts))
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(path, vectors)
    return vectors
