"""HTTP API: serves the 3D concept space and computes bridges between concepts."""

from __future__ import annotations

import logging
import os
from collections import OrderedDict
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated

import httpx
import networkx as nx
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from shockwave.bridge import UnknownConceptError, build_graph, find_bridge, with_extra_concepts
from shockwave.concepts import load_concepts
from shockwave.embeddings import Embedder, cached_embed, lazy_embedder
from shockwave.layout import Layout, reduce_to_3d
from shockwave.narrate import LLMConfig, NarrationError, llm_config_from_env, narrate

log = logging.getLogger("shockwave")

BACKEND_DIR = Path(__file__).resolve().parents[2]
MAX_CONCEPT_LENGTH = 60
NARRATION_CACHE_SIZE = 256


@dataclass(frozen=True)
class Space:
    labels: list[str]
    vectors: np.ndarray
    layout: Layout
    graph: nx.Graph
    k: int = 6
    # Embeds free text on the fly; None means only vocabulary concepts are accepted.
    embed: Embedder | None = None
    by_lower: dict[str, str] = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "by_lower", {label.lower(): label for label in self.labels})


def build_space(
    labels: list[str],
    vectors: np.ndarray,
    reducer: Callable[[np.ndarray], Layout] = reduce_to_3d,
    k: int = 6,
    embed: Embedder | None = None,
) -> Space:
    return Space(labels=labels, vectors=vectors, layout=reducer(vectors),
                 graph=build_graph(labels, vectors, k=k), k=k, embed=embed)


def default_space() -> Space:
    concepts_path = Path(os.getenv("SHOCKWAVE_CONCEPTS", BACKEND_DIR / "data" / "concepts.txt"))
    cache_dir = Path(os.getenv("SHOCKWAVE_CACHE_DIR", BACKEND_DIR / ".cache"))
    labels = load_concepts(concepts_path)
    log.info("embedding %d concepts from %s", len(labels), concepts_path)
    free_text = os.getenv("SHOCKWAVE_FREE_TEXT", "1") != "0"
    space = build_space(labels, cached_embed(labels, cache_dir),
                        embed=lazy_embedder() if free_text else None)
    if space.embed:
        # Load the model and JIT-compile UMAP.transform now, not on the first visitor's query.
        space.layout.place(space.embed(["warm up"]))
    return space


def default_static_dir() -> Path | None:
    """The built frontend, if there is one, so a single process can serve the whole app."""
    configured = os.getenv("SHOCKWAVE_STATIC_DIR")
    path = Path(configured) if configured else BACKEND_DIR.parent / "frontend" / "dist"
    return path if (path / "index.html").exists() else None


Concept = Annotated[str, Field(min_length=1, max_length=MAX_CONCEPT_LENGTH)]


class BridgeRequest(BaseModel):
    source: Concept
    target: Concept


class NarrateRequest(BaseModel):
    path: list[str] = Field(min_length=2, max_length=24)


def resolve(space: Space, terms: list[str]) -> tuple[list[str], dict[str, np.ndarray]]:
    """Map each term to a vocabulary label (case-insensitive), or embed it as a new concept."""
    names: list[str] = []
    extras: dict[str, np.ndarray] = {}
    for term in (t.strip() for t in terms):
        if not term:
            raise HTTPException(422, "concepts can't be blank")
        if term.lower() in space.by_lower:
            names.append(space.by_lower[term.lower()])
            continue
        if space.embed is None:
            raise UnknownConceptError(term)
        names.append(term)
        extras.setdefault(term, space.embed([term])[0])
    return names, extras


def create_app(
    space_factory: Callable[[], Space] = default_space,
    llm_config_factory: Callable[[], LLMConfig | None] = llm_config_from_env,
    http_client_factory: Callable[[], httpx.AsyncClient] = httpx.AsyncClient,
    static_dir: Path | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Built once at startup and read-only afterwards: every visitor shares it safely.
        app.state.space = space_factory()
        app.state.llm = llm_config_factory()
        app.state.narrations = OrderedDict()
        async with http_client_factory() as client:
            app.state.http = client
            log.info("narration %s", f"via {app.state.llm.model}" if app.state.llm else "off")
            yield

    app = FastAPI(title="Semantic Shockwave", version="0.2.0", lifespan=lifespan)
    origins = os.getenv("SHOCKWAVE_CORS_ORIGINS", "http://localhost:5173").split(",")
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["*"],
                       allow_headers=["*"])

    @app.get("/api/v1/health")
    def health(request: Request) -> dict:
        return {
            "status": "ok",
            "free_text": request.app.state.space.embed is not None,
            "narration": request.app.state.llm is not None,
        }

    @app.get("/api/v1/space")
    def space(request: Request) -> dict:
        s: Space = request.app.state.space
        nodes = [
            {"id": label, "x": float(x), "y": float(y), "z": float(z)}
            for label, (x, y, z) in zip(s.labels, s.layout.coords, strict=True)
        ]
        links = [
            {"source": a, "target": b, "similarity": round(1.0 - d["distance"], 4)}
            for a, b, d in s.graph.edges(data=True)
        ]
        return {"nodes": nodes, "links": links}

    @app.post("/api/v1/bridge")
    def bridge(body: BridgeRequest, request: Request) -> dict:
        s: Space = request.app.state.space
        try:
            (source, target), extras = resolve(s, [body.source, body.target])
            graph = s.graph
            placed: list[dict] = []
            if extras:
                graph = with_extra_concepts(s.graph, s.labels, s.vectors, extras, k=s.k)
                coords = s.layout.place(np.stack(list(extras.values())))
                placed = [
                    {"id": name, "x": float(x), "y": float(y), "z": float(z)}
                    for name, (x, y, z) in zip(extras, coords, strict=True)
                ]
            result = find_bridge(graph, source, target)
        except UnknownConceptError as exc:
            raise HTTPException(404, f"unknown concept: {exc.concept!r}") from exc
        return {
            "path": result.path,
            "hops": [
                {"source": h.source, "target": h.target, "similarity": h.similarity}
                for h in result.hops
            ],
            "placed": placed,
        }

    @app.post("/api/v1/narrate")
    async def narrate_bridge(body: NarrateRequest, request: Request) -> dict:
        config: LLMConfig | None = request.app.state.llm
        if config is None:
            raise HTTPException(503, "narration is off: set GROQ_API_KEY or SHOCKWAVE_LLM_BASE_URL")
        if any(not 0 < len(p.strip()) <= MAX_CONCEPT_LENGTH for p in body.path):
            raise HTTPException(422, f"each concept must be 1-{MAX_CONCEPT_LENGTH} characters")

        # Same bridge, same story: cache so replays and shared links don't re-bill the LLM.
        cache: OrderedDict = request.app.state.narrations
        key = tuple(body.path)
        if key not in cache:
            try:
                cache[key] = await narrate(body.path, config, request.app.state.http)
            except NarrationError as exc:
                raise HTTPException(502, f"narration failed: {exc}") from exc
            if len(cache) > NARRATION_CACHE_SIZE:
                cache.popitem(last=False)
        result = cache[key]
        return {"steps": result.steps, "summary": result.summary}

    static_dir = static_dir or default_static_dir()
    if static_dir:
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="frontend")

    return app


app = create_app()
