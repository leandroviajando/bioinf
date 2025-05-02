import numpy as np


def viterbi(x: list[int], A: np.ndarray, E: np.ndarray) -> list[int]:
    """
    :param x: Observation sequence (L)
    :param A: Transition matrix (K, K)
    :param E: Emission matrix (K, U)

    :return path: Most probable state path
    """

    K, L = A.shape[1], len(x)

    V = np.full((K, L), fill_value=float("-inf"), dtype=np.float64)
    V[0, 0] = 0.0
    ptr = np.zeros((K, L), dtype=int)

    for i in range(L - 1):
        for l in range(K):  # noqa: E741
            with np.errstate(divide="ignore"):
                V[l, i + 1] = np.log(E[l, x[i + 1]]) + np.max((probs := V[:, i] + np.log(A[:, l])))
                ptr[l, i] = np.argmax(probs)

    path = np.zeros(L, dtype=int)
    path[L - 1] = np.argmax(V[:, L - 1])
    for i in range(L - 2, -1, -1):
        path[i] = ptr[path[i + 1], i]
    return path.tolist()


def posterior(x: list[int], F: np.ndarray, B: np.ndarray, s: np.ndarray) -> list[int]:
    """
    :param x: Observation sequence (L)
    :param F: (K, L)
    :param B: (K, L)
    :param s: Scaling factors from the forward algorithm (L)

    :return posterior_path: (L)
    """

    L = len(x)

    posterior_prob = F * B  # * (1 / s)
    posterior_path = np.zeros(L, dtype=int)

    for i in range(L):
        posterior_path[i] = np.argmax(posterior_prob[:, i])

    return posterior_path.tolist()
