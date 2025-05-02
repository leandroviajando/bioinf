import numpy as np


def backward(x: list[int], A: np.ndarray, E: np.ndarray, s: np.ndarray) -> np.ndarray:
    """
    :param x: Observation sequence (L)
    :param A: Transition matrix (K, K)
    :param E: Emission matrix (K, U)
    :param s: Scaling factors from the forward algorithm (L)

    :return B: (K, L)
    """

    K, L = A.shape[1], len(x)

    B = np.zeros((K, L), dtype=np.float64)
    B[K - 1, L - 1] = 1.0 / s[L - 1]

    for i in range(L - 2, -1, -1):
        B[:, i] = (1 / s[i]) * (A * E[:, x[i + 1]] * B[:, i + 1]).sum(axis=1)

    return B
