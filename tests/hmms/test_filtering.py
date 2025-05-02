import numpy as np

from src.hmms.filtering import forward


def test_forward(A: np.ndarray, E: np.ndarray) -> None:
    x = [0, 3, 6, 6, 7]

    log_prob, F, s = forward(x, A, E)

    np.testing.assert_almost_equal(log_prob, -9.9760, decimal=4)
    np.testing.assert_array_almost_equal(
        F,
        np.array(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.6250, 0.3633, 0.1873, 0.0],
                [0.0, 0.3750, 0.6367, 0.8127, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
            ]
        ),
        decimal=4,
    )
    np.testing.assert_array_almost_equal(s, np.array([1.0, 0.1333, 0.2886, 0.3625, 0.0033]), decimal=4)
