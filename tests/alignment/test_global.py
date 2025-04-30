from src.alignment.global_ import nwalign


def test_nwalign() -> None:
    global_score, global_alignment = nwalign(
        seq1="HEAGAWGHEE",
        seq2="PAWHEAE",
        gap_penalty=-8,
        substitution_matrix="BLOSUM50",
        verbose=False,
    )

    assert global_score == +1
    assert isinstance(global_score, int)

    assert (
        global_alignment
        == """
HEAGAWGHE-E
    || || |
--P-AW-HEAE"""
    )
