"""Squash 384-dimensional embeddings into 3D coordinates the browser can render."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

DEFAULT_RADIUS = 300.0


@dataclass(frozen=True)
class Layout:
    """3D coordinates for the vocabulary, plus a way to place new vectors in the same space."""

    coords: np.ndarray
    place: Callable[[np.ndarray], np.ndarray]


def reduce_to_3d(
    vectors: np.ndarray, *, n_neighbors: int = 15, seed: int = 42, radius: float = DEFAULT_RADIUS
) -> Layout:
    """UMAP (cosine metric) down to 3D, then fit the cloud inside a sphere of `radius`.

    A fixed seed keeps the layout stable between restarts, so the same concept
    always appears in the same place. The fitted reducer is kept so free-text
    concepts can be dropped into the existing cloud with `UMAP.transform`.
    """
    import umap  # slow import (numba JIT); keep it out of module load

    if len(vectors) < 5:
        raise ValueError("need at least 5 concepts to build a 3D layout")
    reducer = umap.UMAP(
        n_components=3,
        n_neighbors=min(n_neighbors, len(vectors) - 1),
        min_dist=0.1,
        metric="cosine",
        random_state=seed,
    )
    raw = reducer.fit_transform(vectors)
    center, scale = sphere_fit(raw, radius)
    return Layout(
        coords=(raw - center) * scale,
        place=lambda new: (reducer.transform(new) - center) * scale,
    )


def sphere_fit(coords: np.ndarray, radius: float = DEFAULT_RADIUS) -> tuple[np.ndarray, float]:
    """The shift and scale that center the cloud and put its farthest point on `radius`."""
    center = coords.mean(axis=0)
    farthest = float(np.linalg.norm(coords - center, axis=1).max())
    return center, (radius / farthest if farthest else 1.0)


def fit_to_sphere(coords: np.ndarray, radius: float = DEFAULT_RADIUS) -> np.ndarray:
    """Center the cloud on the origin and scale it so the farthest point sits on `radius`."""
    center, scale = sphere_fit(coords, radius)
    return (coords - center) * scale
