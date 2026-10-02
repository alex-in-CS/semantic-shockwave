import asyncio
import datetime as dt

import numpy as np

from shockwave.app import build_space
from shockwave.curate import MAX_HOPS, MIN_HOPS, narrate_all, path_key, pick_challenges
from tests.test_api import flat_layout


def circle_space(n: int = 60):
    """Concepts spread around a circle, alternating between two groups."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    vectors = np.zeros((n, 8), dtype=np.float32)
    vectors[:, 0], vectors[:, 1] = np.cos(angles), np.sin(angles)
    labels = [f"c{i}" for i in range(n)]
    groups = ["even" if i % 2 == 0 else "odd" for i in range(n)]
    return build_space(labels, vectors, reducer=flat_layout, k=2, groups=groups)


def test_challenges_are_dated_cross_group_and_mid_length():
    space = circle_space()
    challenges = pick_challenges(space, dt.date(2026, 10, 1), days=20)
    group_of = dict(zip(space.labels, space.groups, strict=True))

    assert [c["date"] for c in challenges[:2]] == ["2026-10-01", "2026-10-02"]
    for c in challenges:
        assert group_of[c["source"]] != group_of[c["target"]]
        assert MIN_HOPS <= len(c["path"]) - 1 <= MAX_HOPS
        assert (c["path"][0], c["path"][-1]) == (c["source"], c["target"])


def test_challenges_are_deterministic():
    space = circle_space()
    first = pick_challenges(space, dt.date(2026, 10, 1), days=10)
    again = pick_challenges(space, dt.date(2026, 10, 1), days=10)
    assert first == again


def test_existing_narrations_are_kept_and_nothing_is_requested_without_an_llm():
    paths = [["a", "b"], ["b", "c"]]
    existing = {path_key(["a", "b"]): {"steps": ["kept"], "summary": ""}}
    narrations = asyncio.run(narrate_all(paths, existing, config=None))
    assert narrations == existing
    assert path_key(["a", "b", "c"]) == "a → b → c"
