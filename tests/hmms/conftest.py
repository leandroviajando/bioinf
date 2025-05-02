import numpy as np
import pytest

EXPECTED_SEQUENCE_LENGTH = 300
r = (EXPECTED_SEQUENCE_LENGTH - 1) / EXPECTED_SEQUENCE_LENGTH
one_minus_r = 1 / EXPECTED_SEQUENCE_LENGTH


@pytest.fixture
def A() -> np.ndarray:
    return np.array(
        [
            [0.0, 0.5, 0.5, 0.0],
            [0.0, r * 0.95, r * 0.05, one_minus_r],
            [0.0, r * 0.10, r * 0.90, one_minus_r],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


@pytest.fixture
def E() -> np.ndarray:
    return np.array(
        [
            [1.0] + [0.0] * 7,
            [0.0] + [1 / 6] * 6 + [0.0],
            [0.0] + [1 / 10] * 5 + [1 / 2] + [0.0],  # Biased die: 6 is more likely
            [0.0] * 7 + [1.0],
        ]
    )
