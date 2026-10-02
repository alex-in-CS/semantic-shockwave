"""Export the concept space as one static JSON file, so the frontend can run with no backend.

    uv run python -m shockwave.export ../frontend/public/space.json

The file is exactly what `/api/v1/space` serves (nodes, links, groups, k, int8
vocabulary vectors, curated extras) plus the embedding model's name and its
transformers.js twin, so the browser embeds free text the same way.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from shockwave.app import Space, default_space, space_payload
from shockwave.embeddings import BROWSER_MODEL, DEFAULT_MODEL


def export_payload(space: Space, model: str = DEFAULT_MODEL) -> dict:
    return {**space_payload(space), "model": model, "browser_model": BROWSER_MODEL}


def main(argv: list[str]) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if len(argv) != 1:
        sys.exit("usage: python -m shockwave.export OUTPUT.json")
    out = Path(argv[0])
    out.parent.mkdir(parents=True, exist_ok=True)
    # Free text is the browser's job in the static build, so the server-side model stays unloaded.
    payload = export_payload(default_space(free_text=False))
    out.write_text(json.dumps(payload, separators=(",", ":"), ensure_ascii=False),
                   encoding="utf-8")
    logging.info("wrote %d concepts to %s (%.0f KB)", len(payload["nodes"]), out,
                 out.stat().st_size / 1024)


if __name__ == "__main__":
    main(sys.argv[1:])
