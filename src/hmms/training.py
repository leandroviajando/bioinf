import numpy as np
from tqdm import tqdm

from .filtering import forward
from .smoothing import backward


def baum_welch(X, K, U, max_iter=300, eps=0.001):
    """
    Implementation of Baum-Welch algorithm for HMM parameter estimation

    Args:
        X: List of observation sequencesRunning
        K: Number of states
        U: Number of possible emissions
        max_iter: Maximum number of iterations
        eps: Convergence threshold

    Returns:
        A_est: Estimated transition matrix
        E_est: Estimated emission matrix
    """

    # Initialize A_est randomly
    A_est = np.random.randint(1, 102, size=(K, K)).astype(float)

    # Start state (index 0) cannot transition to end state (index K-1)
    A_est[0, K - 1] = 0

    # Cannot transition to start state
    A_est[:, 0] = 0

    # End state only transitions to itself
    A_est[K - 1, :] = 0
    A_est[K - 1, K - 1] = 1

    # Normalize rows to get probabilities
    A_est = A_est / A_est.sum(axis=1, keepdims=True)

    # Initialize E_est randomly
    E_est = np.random.randint(1, 102, size=(K, U)).astype(float)

    # Special emissions for start and end states
    E_est[0, :] = 0
    E_est[K - 1, :] = 0
    E_est[0, 0] = 1  # Start state only emits START symbol
    E_est[K - 1, U - 1] = 1  # End state only emits END symbol

    # Normalize rows to get probabilities
    E_est = E_est / E_est.sum(axis=1, keepdims=True)

    N = len(X)
    prev_logp = 0
    total_logp = 0

    for iter in tqdm(range(max_iter)):
        # These will hold the updates for A and E
        A_c = np.zeros((K, K))
        E_c = np.zeros((K, U))

        prev_logp = total_logp
        total_logp = 0

        for n in range(N):
            x = X[n]
            L = len(x)

            logp, F_t, s = forward(x, A_est, E_est)
            B_t = backward(x, A_est, E_est, s)

            total_logp += logp
            update_E = np.zeros((K, U))

            # Process all states except start and end
            for k in range(1, K - 1):
                # Start to k transitions
                A_c[0, k] += A_est[0, k] * E_est[k, x[1]] * B_t[k, 1]

                # k to end transitions
                A_c[k, K - 1] += A_est[k, K - 1] * F_t[k, L - 2] / s[L - 1]

                # k to other states transitions
                for l in range(1, K - 1):  # noqa: E741
                    A_c[k, l] += np.sum(F_t[k, 1 : L - 2] * A_est[k, l] * E_est[l, x[2 : L - 1]] * B_t[l, 2 : L - 1])

                # Update emission counts
                for i in range(L - 2):
                    update_E[k, x[i + 1]] += s[i + 1] * F_t[k, i + 1] * B_t[k, i + 1]

            E_c += update_E

        # Ensure special states have correct emissions
        E_c[0, 0] = 1
        E_c[K - 1, U - 1] = 1

        # Update transition and emission probabilities
        for k in range(1, K - 1):
            A_est[0, k] = A_c[0, k] / np.sum(A_c[0, :])

            for l in range(1, K):  # noqa: E741
                A_est[k, l] = A_c[k, l] / np.sum(A_c[k, :])

            for u in range(U):
                E_est[k, u] = E_c[k, u] / np.sum(E_c[k, :])

        if abs(total_logp - prev_logp) < eps:
            break

    return A_est, E_est
