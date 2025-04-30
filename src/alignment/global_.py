from pprint import pprint
from typing import Optional

import Bio.Align
import Bio.Seq
import numpy as np

from .alignment import path_to_alignment
from .plots import plot_alignment_path


def nwalign(
    seq1: Bio.Seq.Seq | str,
    seq2: Bio.Seq.Seq | str,
    *,
    gap_penalty: int = -8,
    substitution_matrix: str = "BLOSUM50",
    verbose: bool = False,
    plot: bool = False,
) -> tuple[int, str]:
    """Needleman-Wunsch algorithm. Equivalent to Matlab's nwalign function.

    :param substitution_matrix: Available options: BLOSUM45, BLOSUM50, BLOSUM62, BLOSUM80, BLOSUM90, PAM30, PAM70, PAM250.

    :return global_score, global_alignment:"""

    seq1_len, seq2_len = len(seq1), len(seq2)

    F: np.ndarray = np.zeros((seq1_len + 1, seq2_len + 1), dtype=int)
    """The alignment scoring matrix (for Dynamic Programming, DP)."""

    b = np.empty((seq1_len + 1, seq2_len + 1), dtype=object)
    """The backtracking matrix."""

    Fpath: list[tuple[int, int]] = []
    """The backtracking path through the alignment scoring matrix."""

    s: Bio.Align.substitution_matrices.Array = Bio.Align.substitution_matrices.load(substitution_matrix)
    """The substitution matrix."""

    """!!!!!!!!!!!!! Global alignment !!!!!!!!!!!!!"""
    for r in range(1, seq1_len + 1):
        F[r][0], b[r][0] = r * gap_penalty, (r - 1, 0)
    for c in range(1, seq2_len + 1):
        F[0][c], b[0][c] = c * gap_penalty, (0, c - 1)
    """!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"""

    for r in range(1, seq1_len + 1):
        for c in range(1, seq2_len + 1):
            """!!!!!!!!!! Global alignment !!!!!!!!!!"""
            max_val, max_val_src = float("-inf"), (0, 0)
            """!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"""

            if (substitution_val := F[r - 1][c - 1] + s[seq1[r - 1]][seq2[c - 1]]) > max_val:
                max_val, max_val_src = substitution_val, (r - 1, c - 1)

            if (deletion_val := F[r - 1][c] + gap_penalty) > max_val:
                max_val, max_val_src = deletion_val, (r - 1, c)

            if (insertion_val := F[r][c - 1] + gap_penalty) > max_val:
                max_val, max_val_src = insertion_val, (r, c - 1)

            F[r][c] = max_val
            b[r][c] = max_val_src

    """!!!!!! Global alignment !!!!!!"""
    score = F[seq1_len][seq2_len]
    score_pos = ([seq1_len], [seq2_len])
    """!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"""

    src: Optional[tuple[int, int]] = (score_pos[0][-1], score_pos[1][-1])
    while src is not None:
        Fpath.append(src)
        src = b[src[0], src[1]]
    Fpath = list(reversed(Fpath))

    alignment = path_to_alignment(Fpath, seq1, seq2, s)

    if plot:
        plot_alignment_path(seq1, seq2, F, b, Fpath)
    elif verbose:
        print()
        pprint(F.tolist())
        print()
        pprint(b.tolist())
        print()
        pprint(Fpath)

    return int(score), alignment
