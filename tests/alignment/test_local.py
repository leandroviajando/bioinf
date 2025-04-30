from src.alignment.local import swalign


def test_swalign() -> None:
    local_score, local_alignment = swalign(
        seq1="HEAGAWGHEE",
        seq2="PAWHEAE",
        gap_penalty=-8,
        substitution_matrix="BLOSUM50",
        verbose=False,
    )

    assert local_score == +28
    assert isinstance(local_score, int)

    assert (
        local_alignment
        == """
AWGHE
|| ||
AW-HE"""
    )
