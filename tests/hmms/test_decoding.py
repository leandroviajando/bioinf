import numpy as np

from src.hmms.decoding import posterior, viterbi
from src.hmms.filtering import forward
from src.hmms.smoothing import backward


def test_viterbi(A: np.ndarray, E: np.ndarray) -> None:
    x = [0, 3, 6, 6, 7]

    viterbi_path = viterbi(x, A, E)

    assert viterbi_path == [0, 2, 2, 2, 3]


def test_posterior(A: np.ndarray, E: np.ndarray) -> None:
    x = [0, 3, 6, 6, 7]
    _, F, s = forward(x, A, E)
    B = backward(x, A, E, s)

    posterior_path = posterior(x, F, B, s)

    assert posterior_path == [0, 2, 2, 2, 3]
