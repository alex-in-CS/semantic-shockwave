"""The bridge: the chain of stepping-stone concepts between two ideas.

Every concept is a node. Each node links to its k nearest neighbours by cosine
distance, and the minimum spanning tree's edges are added too, so the graph is
always connected and a bridge exists between any two concepts.

The bridge is the weighted shortest path. Edge cost is distance **squared**:
two short hops beat one long jump, so the path walks through close, intuitive
neighbours instead of leaping straight across the space.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np


class UnknownConceptError(KeyError):
    def __init__(self, concept: str) -> None:
        super().__init__(concept)
        self.concept = concept


@dataclass(frozen=True)
class Hop:
    source: str
    target: str
    similarity: float


@dataclass(frozen=True)
class Bridge:
    path: list[str]
    hops: list[Hop]


def cosine_distances(vectors: np.ndarray) -> np.ndarray:
    """Pairwise cosine distance for unit vectors: 0 = same meaning, 2 = opposite."""
    return np.clip(1.0 - vectors @ vectors.T, 0.0, 2.0)


def _mst_edges(dist: np.ndarray) -> list[tuple[int, int]]:
    """Prim's algorithm on a dense distance matrix. O(n^2), fine for a few thousand nodes."""
    n = len(dist)
    in_tree = np.zeros(n, dtype=bool)
    in_tree[0] = True
    best = dist[0].copy()
    parent = np.zeros(n, dtype=int)
    edges: list[tuple[int, int]] = []
    for _ in range(n - 1):
        j = int(np.argmin(np.where(in_tree, np.inf, best)))
        edges.append((int(parent[j]), j))
        in_tree[j] = True
        closer = dist[j] < best
        best = np.where(closer, dist[j], best)
        parent = np.where(closer, j, parent)
    return edges


def build_graph(labels: list[str], vectors: np.ndarray, k: int = 6) -> nx.Graph:
    """k-nearest-neighbour graph plus MST edges. Nodes are labels; edges carry
    `distance` (cosine) and `cost` (distance squared, used for routing)."""
    if len(labels) != len(vectors):
        raise ValueError("labels and vectors must have the same length")
    dist = cosine_distances(vectors)
    graph = nx.Graph()
    graph.add_nodes_from(labels)

    def link(i: int, j: int) -> None:
        d = float(dist[i, j])
        graph.add_edge(labels[i], labels[j], distance=d, cost=d * d)

    k = min(k, len(labels) - 1)
    for i in range(len(labels)):
        neighbours = [j for j in np.argsort(dist[i]) if j != i][:k]
        for j in neighbours:
            link(i, int(j))
    for i, j in _mst_edges(dist):
        link(i, j)
    return graph


def with_extra_concepts(
    graph: nx.Graph,
    labels: list[str],
    vectors: np.ndarray,
    extras: dict[str, np.ndarray],
    k: int = 6,
) -> nx.Graph:
    """A copy of `graph` with each extra concept linked to its k nearest neighbours.

    The shared graph is never mutated, so concurrent requests can't see each
    other's free-text concepts. Extras can link to one another too.
    """
    graph = graph.copy()
    pool_labels = list(labels)
    pool = vectors
    for name, vector in extras.items():
        dist = np.clip(1.0 - pool @ vector, 0.0, 2.0)
        for j in np.argsort(dist)[: min(k, len(pool_labels))]:
            d = float(dist[j])
            graph.add_edge(name, pool_labels[j], distance=d, cost=d * d)
        pool_labels.append(name)
        pool = np.vstack([pool, vector[None, :]])
    return graph


def find_bridge(graph: nx.Graph, source: str, target: str) -> Bridge:
    for concept in (source, target):
        if concept not in graph:
            raise UnknownConceptError(concept)
    path = nx.shortest_path(graph, source, target, weight="cost")
    hops = [
        Hop(a, b, round(1.0 - graph.edges[a, b]["distance"], 4))
        for a, b in zip(path, path[1:], strict=False)
    ]
    return Bridge(path=path, hops=hops)
