"""Load the concept vocabulary: one concept per line, `#` starts a comment.

A line starting with `## ` names a section; the concepts below it belong to that
group until the next one. Groups color the nodes and name the cloud's regions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

UNGROUPED = "Other"


@dataclass(frozen=True)
class Vocabulary:
    labels: list[str]
    groups: list[str]  # one per label


def load_vocabulary(path: Path) -> Vocabulary:
    """Read concepts in file order, dropping blanks, comments and case-insensitive
    duplicates (a repeated concept stays in the first section it appeared in)."""
    seen: set[str] = set()
    labels: list[str] = []
    groups: list[str] = []
    group = UNGROUPED
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("## "):
            group = line[3:].strip() or UNGROUPED
            continue
        concept = line.split("#", 1)[0].strip()
        if concept and concept.lower() not in seen:
            seen.add(concept.lower())
            labels.append(concept)
            groups.append(group)
    return Vocabulary(labels=labels, groups=groups)


def load_concepts(path: Path) -> list[str]:
    return load_vocabulary(path).labels
