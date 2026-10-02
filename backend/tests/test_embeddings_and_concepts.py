import numpy as np
import pytest

from shockwave.concepts import load_concepts
from shockwave.embeddings import cached_embed, lazy_embedder
from shockwave.layout import fit_to_sphere


def test_load_concepts_strips_comments_blanks_and_duplicates(tmp_path):
    path = tmp_path / "concepts.txt"
    path.write_text("# header\njazz\n\nBlack Hole  # inline note\njazz\nblack hole\n")
    assert load_concepts(path) == ["jazz", "Black Hole"]


def test_cached_embed_normalizes_and_hits_cache(tmp_path):
    calls = []

    def fake_factory(model_name):
        calls.append(model_name)
        return lambda texts: np.array([[3.0, 4.0]] * len(texts))

    first = cached_embed(["a", "b"], tmp_path, "m", make_embedder=fake_factory)
    second = cached_embed(["a", "b"], tmp_path, "m", make_embedder=fake_factory)

    assert calls == ["m"], "model should load once; second call reads the cache"
    np.testing.assert_allclose(first, [[0.6, 0.8], [0.6, 0.8]], rtol=1e-6)
    np.testing.assert_array_equal(first, second)


def test_cache_key_changes_with_vocabulary(tmp_path):
    calls = []

    def fake_factory(model_name):
        calls.append(model_name)
        return lambda texts: np.ones((len(texts), 2))

    cached_embed(["a"], tmp_path, "m", make_embedder=fake_factory)
    cached_embed(["a", "b"], tmp_path, "m", make_embedder=fake_factory)
    assert len(calls) == 2


def test_lazy_embedder_loads_model_once_on_first_use():
    calls = []

    def fake_factory(model_name):
        calls.append(model_name)
        return lambda texts: np.array([[0.0, 2.0]] * len(texts))

    embed = lazy_embedder("m", make_embedder=fake_factory)
    assert calls == [], "nothing loads until the first query"
    np.testing.assert_allclose(embed(["a"]), [[0.0, 1.0]])
    embed(["b"])
    assert calls == ["m"]


def test_fit_to_sphere_centers_and_scales():
    coords = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [5.0, 5.0, 0.0]])
    fitted = fit_to_sphere(coords, radius=100.0)
    np.testing.assert_allclose(fitted.mean(axis=0), 0.0, atol=1e-9)
    assert np.linalg.norm(fitted, axis=1).max() == pytest.approx(100.0)
