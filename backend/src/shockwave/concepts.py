"""Load the concept vocabulary: one concept per line, `#` starts a comment."""

from __future__ import annotations

from pathlib import Path


def load_concepts(path: Path) -> list[str]:
    """Read concepts in file order, dropping blanks, comments and case-insensitive duplicates."""
    seen: set[str] = set()
    concepts: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        concept = raw.split("#", 1)[0].strip()
        if concept and concept.lower() not in seen:
            seen.add(concept.lower())
            concepts.append(concept)
    return concepts
