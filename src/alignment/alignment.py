import Bio.Align
import Bio.Seq

EXACT_MATCH = "|"
"""Mathworks: The symbol | indicates amino acids or nucleotides that match exactly."""
RELATION = ":"
"""Mathworks: The symbol : indicates amino acids or nucleotides that are related as defined by the scoring matrix (nonmatches with a zero or positive scoring matrix value)."""
GAP = " "


def path_to_alignment(
    Fpath: list[tuple[int, int]],
    seq1: Bio.Seq.Seq | str,
    seq2: Bio.Seq.Seq | str,
    substitution_matrix: Bio.Align.substitution_matrices.Array,
) -> str:
    SEQ_GAP = "-"
    aligned_seq1, aligned_seq2 = "", ""
    i, j = 0, 0

    for idx in range(len(Fpath)):
        current = Fpath[idx]
        if idx > 0:
            prev = Fpath[idx - 1]
            # Diagonal move (match/mismatch)
            if current[0] == prev[0] + 1 and current[1] == prev[1] + 1:
                aligned_seq1 += seq1[prev[0]]
                aligned_seq2 += seq2[prev[1]]
                i, j = i + 1, j + 1
            # Vertical move (gap in seq2)
            elif current[0] == prev[0] + 1 and current[1] == prev[1]:
                aligned_seq1 += seq1[prev[0]]
                aligned_seq2 += SEQ_GAP
                i += 1
            # Horizontal move (gap in seq1)
            elif current[0] == prev[0] and current[1] == prev[1] + 1:
                aligned_seq1 += SEQ_GAP
                aligned_seq2 += seq2[prev[1]]
                j += 1

    match_line = ""
    for s1, s2 in zip(aligned_seq1, aligned_seq2):
        if s1 == s2 and s1 != SEQ_GAP:
            match_line += EXACT_MATCH
        elif s1 != SEQ_GAP and s2 != SEQ_GAP:
            if substitution_matrix[s1, s2] > 0:
                match_line += RELATION
            else:
                match_line += GAP
        else:
            match_line += GAP

    global_alignment = f"\n{aligned_seq1}\n{match_line}\n{aligned_seq2}"

    return global_alignment
