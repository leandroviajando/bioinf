from src.distance.edit import hamming, levenshtein


def test_levenshtein() -> None:
    assert levenshtein(seq1="horse", seq2="ros", gap_penalty=+1, verbose=False) == 3
    assert levenshtein(seq1="intention", seq2="execution", gap_penalty=+1, verbose=False) == 5
    assert levenshtein(seq1="HEAGAWGHEE", seq2="PAWHEAE", gap_penalty=+1, verbose=False) == 6


def test_hamming() -> None:
    assert hamming(seq1="intention", seq2="execution", gap_penalty=+1, verbose=False) == 5
    assert hamming(seq1="1011101", seq2="1001001", gap_penalty=+1, verbose=False) == 2
