import numpy as np


def forward_naive(x: list[int], A: np.ndarray, E: np.ndarray) -> tuple[float, np.ndarray]:
    """
    :param x: Observation sequence (L)
    :param A: Transition matrix (K, K)
    :param E: Emission matrix (K, U)

    :return prob: Probability of sequence x
    :return F: (K, L)
    """

    K, L = A.shape[1], len(x)

    F = np.zeros((K, L), dtype=np.float64)
    F[0, 0] = 1.0

    for i in range(L - 1):
        F[:, i + 1] = E[:, x[i + 1]] * np.dot(F[:, i], A)

    prob = np.dot(F[:, i], A).sum()

    return prob, F


def forward_log(x: list[int], A: np.ndarray, E: np.ndarray) -> tuple[float, np.ndarray]:
    """
    :param x: Observation sequence (L)
    :param A: Transition matrix (K, K)
    :param E: Emission matrix (K, U)

    :return prob: Probability of sequence x
    :return F: (K, L)
    """

    K, L = A.shape[1], len(x)

    F = np.full((K, L), fill_value=float("-inf"), dtype=np.float64)
    F[0, 0] = 0.0

    for i in range(L - 1):
        with np.errstate(divide="ignore", invalid="ignore"):
            F[:, i + 1] = np.log(E[:, x[i + 1]]) + np.log(np.dot(F[:, i], A))

    prob = np.dot(F[:, i], A).sum()

    return prob, F


def forward(x: list[int], A: np.ndarray, E: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """
    :param x: Observation sequence (L)
    :param A: Transition matrix (K, K)
    :param E: Emission matrix (K, U)

    :return log_prob: Log probability of sequence x
    :return F: (K, L)
    :return s: Scaling factors (L)
    """

    K, L = A.shape[1], len(x)

    F = np.zeros((K, L), dtype=np.float64)
    s = np.zeros(L, dtype=np.float64)
    F[0, 0] = s[0] = 1.0

    for i in range(L - 1):
        unnormalised_probs = E[:, x[i + 1]] * np.dot(F[:, i], A)
        partition_function = unnormalised_probs.sum()

        F[:, i + 1] = unnormalised_probs / partition_function
        s[i + 1] = partition_function

    log_prob = np.log(s).sum()

    return float(log_prob), F, s
