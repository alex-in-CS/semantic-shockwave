import base64
import json

import numpy as np

from shockwave.app import build_space, space_payload
from shockwave.export import INT8_SCALE, export_payload, quantize
from tests.conftest import arc_vectors
from tests.test_api import LABELS, flat_layout


def test_export_matches_space_endpoint_and_is_json():
    space = build_space(LABELS, arc_vectors(10), reducer=flat_layout)
    payload = export_payload(space, model="m")

    assert payload["nodes"] == space_payload(space)["nodes"]
    assert payload["links"] == space_payload(space)["links"]
    assert (payload["k"], payload["model"]) == (space.k, "m")
    json.dumps(payload)  # must serialize as-is


def test_vectors_roundtrip_through_int8():
    vectors = arc_vectors(10)
    payload = export_payload(build_space(LABELS, vectors, reducer=flat_layout))
    meta = payload["vectors"]

    raw = np.frombuffer(base64.b64decode(meta["data"]), dtype=np.int8)
    decoded = raw.reshape(len(LABELS), meta["dims"]).astype(np.float32) / meta["scale"]

    np.testing.assert_allclose(decoded, vectors, atol=0.5 / INT8_SCALE + 1e-6)


def test_quantize_keeps_nearest_neighbours():
    vectors = arc_vectors(10)
    q = quantize(vectors).astype(np.float32)
    # On the arc, c3's neighbours are tied pairs (c2/c4, c1/c5), so compare as sets.
    exact = np.argsort(-(vectors @ vectors[3]))[:5]
    approx = np.argsort(-(q @ q[3]))[:5]
    assert set(exact) == set(approx) == {1, 2, 3, 4, 5}
