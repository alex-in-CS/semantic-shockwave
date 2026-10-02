"""Export the concept space as one static JSON file, so the frontend can run with no backend.

    uv run python -m shockwave.export ../frontend/public/space.json

The file holds what `/api/v1/space` serves, plus what the browser needs to route
and to place free text itself: the graph's k, the embedding model's name, and the
vocabulary vectors quantized to int8 (unit-vector components fit in [-1, 1], and
nearest-neighbour ranking survives the rounding).
"""

from __future__ import annotations

import base64
import json
import logging
import sys
from pathlib import Path

import numpy as np

from shockwave.app import Space, default_space, space_payload
from shockwave.embeddings import DEFAULT_MODEL

INT8_SCALE = 127


def quantize(vectors: np.ndarray) -> np.ndarray:
    return np.clip(np.round(vectors * INT8_SCALE), -INT8_SCALE, INT8_SCALE).astype(np.int8)


def export_payload(space: Space, model: str = DEFAULT_MODEL) -> dict:
    vectors = quantize(space.vectors)
    return {
        **space_payload(space),
        "k": space.k,
        "model": model,
        "vectors": {
            "dtype": "int8",
            "scale": INT8_SCALE,
            "dims": int(vectors.shape[1]),
            "data": base64.b64encode(vectors.tobytes()).decode("ascii"),
        },
    }


def main(argv: list[str]) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if len(argv) != 1:
        sys.exit("usage: python -m shockwave.export OUTPUT.json")
    out = Path(argv[0])
    out.parent.mkdir(parents=True, exist_ok=True)
    # Free text is the browser's job in the static build, so the server-side model stays unloaded.
    payload = export_payload(default_space(free_text=False))
    out.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    logging.info("wrote %d concepts to %s (%.0f KB)", len(payload["nodes"]), out,
                 out.stat().st_size / 1024)


if __name__ == "__main__":
    main(sys.argv[1:])
