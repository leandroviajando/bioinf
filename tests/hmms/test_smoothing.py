import numpy as np

from src.hmms.filtering import forward
from src.hmms.smoothing import backward


def test_backward(A: np.ndarray, E: np.ndarray) -> None:
    x = [0, 3, 6, 6, 7]
    _, __, s = forward(x, A, E)

    B = backward(x, A, E, s)

    np.testing.assert_array_almost_equal(
        B,
        np.array(
            [
                [1.0, 9.4265, 3.1859, 0.0, 0.0],
                [0.5329, 2.8976, 1.7464, 2.7585, 0.0],
                [1.4089, 15.1706, 4.4453, 2.7585, 0.0],
                [0.0, 0.0, 0.0, 827.5483, 300.0],
            ]
        ),
        decimal=4,
    )
