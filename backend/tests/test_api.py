import json

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from shockwave.app import build_space, create_app
from shockwave.layout import Layout
from shockwave.narrate import LLMConfig
from tests.conftest import arc_vectors

LABELS = [f"c{i}" for i in range(10)]


def flat_layout(vectors: np.ndarray) -> Layout:
    # Fake reducer: no UMAP, no model. The first 3 dims are a valid layout.
    return Layout(coords=vectors[:, :3], place=lambda new: new[:, :3])


def fake_embed(texts: list[str]) -> np.ndarray:
    """'mid' lands halfway between c4 and c5 on the arc; anything else sits on c0."""
    angle = np.radians(170.0 * 4.5 / 9 if texts[0] == "mid" else 0.0)
    vector = np.zeros(8, dtype=np.float32)
    vector[:2] = np.cos(angle), np.sin(angle)
    return vector[None, :]


def make_client(*, embed=fake_embed, llm=None, handler=None, static_dir=None) -> TestClient:
    def http_client():
        transport = httpx.MockTransport(handler or (lambda req: httpx.Response(500)))
        return httpx.AsyncClient(transport=transport)

    app = create_app(
        lambda: build_space(LABELS, arc_vectors(10), reducer=flat_layout, embed=embed),
        llm_config_factory=lambda: llm,
        http_client_factory=http_client,
        static_dir=static_dir,
    )
    return TestClient(app)


@pytest.fixture
def client():
    with make_client() as c:
        yield c


def test_health_reports_features(client):
    assert client.get("/api/v1/health").json() == {
        "status": "ok", "free_text": True, "narration": False,
    }


def test_space_shape(client):
    body = client.get("/api/v1/space").json()
    assert [n["id"] for n in body["nodes"]] == LABELS
    assert set(body["nodes"][0]) == {"id", "group", "x", "y", "z"}
    assert body["links"], "graph should have edges"
    assert body["k"] == 6 and body["vectors"]["dims"] == 8
    assert body["groups"] == ["Other"]
    ids = set(LABELS)
    assert all(link["source"] in ids and link["target"] in ids for link in body["links"])


def test_bridge(client):
    res = client.post("/api/v1/bridge", json={"source": "c0", "target": "c9"})
    assert res.status_code == 200
    body = res.json()
    assert body["path"][0] == "c0" and body["path"][-1] == "c9"
    assert len(body["hops"]) == len(body["path"]) - 1
    assert body["placed"] == [] and body["links"] == []


def test_bridge_matches_vocabulary_case_insensitively(client):
    body = client.post("/api/v1/bridge", json={"source": " C0 ", "target": "C2"}).json()
    assert body["path"] == ["c0", "c1", "c2"]
    assert body["placed"] == []


def test_free_text_concept_is_placed_and_routed(client):
    body = client.post("/api/v1/bridge", json={"source": "c0", "target": "mid"}).json()
    assert body["path"][0] == "c0" and body["path"][-1] == "mid"
    # It sits between c4 and c5, so the walk along the arc reaches c4 first.
    assert body["path"][-2] in {"c4", "c5"}
    assert [p["id"] for p in body["placed"]] == ["mid"]
    assert set(body["placed"][0]) == {"id", "group", "x", "y", "z", "vector"}
    assert len(body["placed"][0]["vector"]) == 8
    # Every edge of the placed concept is returned, so the browser can search through it.
    links = body["links"]
    assert len(links) == 6
    assert all("mid" in (link["source"], link["target"]) for link in links)


def test_free_text_does_not_leak_into_shared_graph(client):
    client.post("/api/v1/bridge", json={"source": "c0", "target": "mid"})
    nodes = [n["id"] for n in client.get("/api/v1/space").json()["nodes"]]
    assert "mid" not in nodes
    assert client.app.state.space.graph.number_of_nodes() == len(LABELS)


def test_unknown_concept_is_404_when_free_text_is_off():
    with make_client(embed=None) as c:
        res = c.post("/api/v1/bridge", json={"source": "c0", "target": "unicorn"})
    assert res.status_code == 404
    assert "unicorn" in res.json()["detail"]


@pytest.mark.parametrize("body", [
    {"source": "", "target": "c1"},
    {"source": "   ", "target": "c1"},
    {"source": "x" * 61, "target": "c1"},
])
def test_bridge_rejects_bad_input(client, body):
    assert client.post("/api/v1/bridge", json=body).status_code == 422


def test_narrate_is_503_when_not_configured(client):
    res = client.post("/api/v1/narrate", json={"path": ["c0", "c1"]})
    assert res.status_code == 503


LLM = LLMConfig(base_url="http://llm.test/v1", model="test-model", api_key="secret")


def llm_reply(content: str):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content))
        assert request.url == "http://llm.test/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer secret"
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    return handler, calls


def test_narrate_returns_one_step_per_hop_and_caches():
    handler, calls = llm_reply('{"steps": ["a to b", "b to c"], "summary": "done"}')
    with make_client(llm=LLM, handler=handler) as c:
        assert c.get("/api/v1/health").json()["narration"] is True
        first = c.post("/api/v1/narrate", json={"path": ["c0", "c1", "c2"]})
        second = c.post("/api/v1/narrate", json={"path": ["c0", "c1", "c2"]})
    assert first.status_code == 200
    assert first.json() == {"steps": ["a to b", "b to c"], "summary": "done"}
    assert second.json() == first.json()
    assert len(calls) == 1, "second identical request should be served from cache"
    assert calls[0]["model"] == "test-model"
    assert "c0 -> c1 -> c2" in calls[0]["messages"][1]["content"]


def test_narrate_bad_model_output_is_502():
    handler, _ = llm_reply('{"steps": ["only one"]}')
    with make_client(llm=LLM, handler=handler) as c:
        res = c.post("/api/v1/narrate", json={"path": ["c0", "c1", "c2"]})
    assert res.status_code == 502


def test_narrate_llm_http_error_is_502():
    with make_client(llm=LLM, handler=lambda req: httpx.Response(429)) as c:
        res = c.post("/api/v1/narrate", json={"path": ["c0", "c1"]})
    assert res.status_code == 502
    assert "429" in res.json()["detail"]


def test_serves_built_frontend(tmp_path):
    (tmp_path / "index.html").write_text("<h1>shockwave</h1>")
    with make_client(static_dir=tmp_path) as c:
        assert "shockwave" in c.get("/").text
        assert c.get("/api/v1/health").status_code == 200
