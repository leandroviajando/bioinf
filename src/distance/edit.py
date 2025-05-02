from pprint import pprint

import Bio.Seq
import numpy as np

from .plots import plot_scoring_matrix


def levenshtein(
    seq1: Bio.Seq.Seq | str,
    seq2: Bio.Seq.Seq | str,
    *,
    gap_penalty: int = +1,
    verbose: bool = False,
    plot: bool = False,
) -> int:
    seq1_len, seq2_len = len(seq1), len(seq2)

    F: np.ndarray = np.zeros((seq1_len + 1, seq2_len + 1), dtype=int)
    """The edit distance scoring matrix (for Dynamic Programming, DP)."""

    def s(x: str, y: str) -> int:
        """The substitution function."""
        return 0 if x == y else gap_penalty

    for r in range(1, seq1_len + 1):
        F[r][0] = r * gap_penalty
    for c in range(1, seq2_len + 1):
        F[0][c] = c * gap_penalty

    for r in range(1, seq1_len + 1):
        for c in range(1, seq2_len + 1):
            substitution_val = F[r - 1][c - 1] + s(seq1[r - 1], seq2[c - 1])
            deletion_val = F[r - 1][c] + gap_penalty
            insertion_val = F[r][c - 1] + gap_penalty

            F[r][c] = min(substitution_val, deletion_val, insertion_val)

    score = F[seq1_len][seq2_len]

    if plot:
        plot_scoring_matrix(seq1, seq2, F)
    elif verbose:
        print()
        pprint(F.tolist())
        print()

    return int(score)


def hamming(
    seq1: Bio.Seq.Seq | str,
    seq2: Bio.Seq.Seq | str,
    *,
    gap_penalty: int = +1,
    verbose: bool = False,
    plot: bool = False,
) -> int:
    seq1_len, seq2_len = len(seq1), len(seq2)
    if seq1_len != seq2_len:
        raise ValueError("Sequences must be of the same length")

    count = 0
    for char1, char2 in zip(seq1, seq2):
        if char1 != char2:
            if verbose or plot:
                print(f"Mismatch at position {count}: {char1} != {char2}")

            count += gap_penalty

    return count
