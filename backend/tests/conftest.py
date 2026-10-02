import numpy as np
import pytest


def arc_vectors(n: int, dims: int = 8, span_degrees: float = 170.0) -> np.ndarray:
    """Unit vectors spread evenly along an arc: neighbours on the arc are nearest in cosine."""
    angles = np.radians(np.linspace(0.0, span_degrees, n))
    vectors = np.zeros((n, dims), dtype=np.float32)
    vectors[:, 0] = np.cos(angles)
    vectors[:, 1] = np.sin(angles)
    return vectors


@pytest.fixture
def arc() -> tuple[list[str], np.ndarray]:
    labels = [f"c{i}" for i in range(10)]
    return labels, arc_vectors(10)
