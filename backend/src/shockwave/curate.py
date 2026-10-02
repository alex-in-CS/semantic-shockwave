"""Generate data/curated.json: example pairs, a year of daily challenges, and narrations.

    uv run python -m shockwave.curate [--start 2026-10-01] [--days 365] [--no-llm]

Narrations are pre-generated so the static demo (which has no LLM) can still
explain the example and challenge bridges. Configure the LLM the same way as the
server (GROQ_API_KEY or SHOCKWAVE_LLM_BASE_URL). Re-running is incremental:
narrations for paths that haven't changed are kept, and only new ones are requested.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import logging
import random

import httpx

from shockwave.app import CURATED_PATH, Space, default_space, load_curated
from shockwave.bridge import find_bridge
from shockwave.narrate import LLMConfig, NarrationError, llm_config_from_env, narrate

log = logging.getLogger("shockwave.curate")

EXAMPLES = [
    ("black hole", "jazz"),
    ("coffee", "revolution"),
    ("heartbreak", "volcano"),
    ("dinosaur", "smartphone"),
    ("pizza", "philosophy"),
    ("Viking", "K-pop"),
    ("chess", "octopus"),
    ("lullaby", "nuclear reactor"),
]
CHALLENGE_SEED = 20261001
MIN_HOPS, MAX_HOPS = 4, 7


def path_key(path: list[str]) -> str:
    """How narrations are keyed in curated.json (and looked up by the frontend)."""
    return " → ".join(path)


def pick_challenges(space: Space, start: dt.date, days: int, seed: int = CHALLENGE_SEED,
                    ) -> list[dict]:
    """Deterministic daily pairs from different sections whose bridge is 4-7 hops long:
    far enough to be a puzzle, short enough to be fair."""
    rng = random.Random(seed)
    group_of = dict(zip(space.labels, space.groups, strict=True))
    used: set[str] = set()
    challenges = []
    for day in range(days):
        while True:
            a, b = rng.sample(space.labels, 2)
            if group_of[a] == group_of[b] or a in used or b in used:
                continue
            path = find_bridge(space.graph, a, b).path
            if MIN_HOPS <= len(path) - 1 <= MAX_HOPS:
                break
        # Don't reuse an endpoint for a while, so consecutive days feel different.
        used.update((a, b))
        if len(used) > 120:
            used.clear()
        challenges.append({
            "date": (start + dt.timedelta(days=day)).isoformat(),
            "source": a,
            "target": b,
            "path": path,
        })
    return challenges


async def narrate_all(paths: list[list[str]], existing: dict, config: LLMConfig | None,
                      ) -> dict:
    narrations = {}
    todo = []
    for path in paths:
        key = path_key(path)
        if key in existing:
            narrations[key] = existing[key]
        elif key not in narrations:
            todo.append(path)
    if config is None:
        log.warning("no LLM configured: %d paths left without narration", len(todo))
        return narrations
    log.info("narrating %d new paths via %s (%d kept)", len(todo), config.model,
             len(narrations))
    async with httpx.AsyncClient() as client:
        for i, path in enumerate(todo, 1):
            try:
                result = await narrate(path, config, client)
            except NarrationError as exc:
                log.warning("skipped %s: %s", path_key(path), exc)
                continue
            narrations[path_key(path)] = {"steps": result.steps, "summary": result.summary}
            if i % 25 == 0:
                log.info("  %d/%d", i, len(todo))
    return narrations


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--start", type=dt.date.fromisoformat, default=dt.date(2026, 10, 1))
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--no-llm", action="store_true", help="skip narration")
    args = parser.parse_args()

    space = default_space(free_text=False)
    missing = [c for pair in EXAMPLES for c in pair if c.lower() not in space.by_lower]
    if missing:
        raise SystemExit(f"examples not in the vocabulary: {missing}")
    examples = [
        {"source": a, "target": b, "path": find_bridge(space.graph, a, b).path}
        for a, b in EXAMPLES
    ]
    challenges = pick_challenges(space, args.start, args.days)

    existing = load_curated().get("narrations", {})
    paths = [e["path"] for e in examples] + [c["path"] for c in challenges]
    config = None if args.no_llm else llm_config_from_env()
    narrations = asyncio.run(narrate_all(paths, existing, config))

    curated = {
        "examples": examples,
        "challenges": challenges,
        "narrations": narrations,
        "narrated_by": config.model if config else load_curated().get("narrated_by"),
    }
    CURATED_PATH.write_text(json.dumps(curated, ensure_ascii=False, indent=1) + "\n",
                            encoding="utf-8")
    log.info("wrote %s: %d examples, %d challenges, %d narrations", CURATED_PATH,
             len(examples), len(challenges), len(narrations))


if __name__ == "__main__":
    main()
