import networkx as nx
import numpy as np
import pytest

from shockwave.bridge import (
    UnknownConceptError,
    build_graph,
    cosine_distances,
    find_bridge,
    with_extra_concepts,
)


def test_bridge_walks_the_arc_in_order(arc):
    labels, vectors = arc
    bridge = find_bridge(build_graph(labels, vectors, k=3), "c0", "c9")
    # Squared cost makes small steps win, so every intermediate concept is visited.
    assert bridge.path == labels
    assert [h.source for h in bridge.hops] == labels[:-1]
    assert all(0.9 < h.similarity <= 1.0 for h in bridge.hops)


def test_same_concept_is_a_zero_hop_bridge(arc):
    labels, vectors = arc
    bridge = find_bridge(build_graph(labels, vectors), "c4", "c4")
    assert bridge.path == ["c4"]
    assert bridge.hops == []


def test_mst_connects_clusters_knn_alone_would_split():
    # Two tight, far-apart clusters: with k=1 the kNN graph has two components.
    rng = np.random.default_rng(0)
    a = np.array([1.0, 0.0, 0.0]) + rng.normal(0, 0.01, (5, 3))
    b = np.array([0.0, 1.0, 0.0]) + rng.normal(0, 0.01, (5, 3))
    vectors = np.vstack([a, b])
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    labels = [f"a{i}" for i in range(5)] + [f"b{i}" for i in range(5)]

    graph = build_graph(labels, vectors, k=1)

    assert nx.is_connected(graph)
    path = find_bridge(graph, "a0", "b0").path
    assert path[0] == "a0" and path[-1] == "b0"


def test_unknown_concept_raises(arc):
    labels, vectors = arc
    with pytest.raises(UnknownConceptError) as exc:
        find_bridge(build_graph(labels, vectors), "c0", "nope")
    assert exc.value.concept == "nope"


def test_identical_concepts_still_get_an_edge():
    vectors = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    graph = build_graph(["x", "x-twin", "y"], vectors, k=1)
    assert graph.has_edge("x", "x-twin")
    assert graph.edges["x", "x-twin"]["distance"] == pytest.approx(0.0)


def test_cosine_distance_range():
    vectors = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]])
    dist = cosine_distances(vectors)
    assert dist[0, 1] == pytest.approx(2.0)
    assert dist[0, 2] == pytest.approx(1.0)
    assert dist[0, 0] == pytest.approx(0.0)


def test_extra_concepts_link_to_nearest_without_touching_the_original(arc):
    labels, vectors = arc
    graph = build_graph(labels, vectors, k=2)
    edges_before = graph.number_of_edges()
    # Just past the end of the arc (c9 sits at 170 degrees), so c9 is the way in.
    past_c9 = np.zeros(vectors.shape[1], dtype=np.float32)
    past_c9[:2] = np.cos(np.radians(175.0)), np.sin(np.radians(175.0))

    extended = with_extra_concepts(graph, labels, vectors, {"new": past_c9}, k=2)

    assert set(extended["new"]) == {"c9", "c8"}
    assert "new" not in graph and graph.number_of_edges() == edges_before
    assert find_bridge(extended, "c0", "new").path == [*labels, "new"]


def test_two_extra_concepts_can_link_to_each_other(arc):
    labels, vectors = arc
    graph = build_graph(labels, vectors, k=2)
    extended = with_extra_concepts(graph, labels, vectors, {"x": vectors[3], "y": vectors[3]}, k=1)
    assert extended.has_edge("x", "y") or extended.has_edge("x", "c3")
    assert find_bridge(extended, "x", "y").path[0] == "x"


def test_length_mismatch_rejected():
    with pytest.raises(ValueError):
        build_graph(["a"], np.zeros((2, 3)))
