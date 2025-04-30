import Bio.Seq
import matplotlib.pyplot as plt
import numpy as np


def plot_alignment_path(
    seq1: Bio.Seq.Seq | str,
    seq2: Bio.Seq.Seq | str,
    F: np.ndarray,
    b: np.ndarray,
    Fpath: list[tuple[int, int]],
) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    ax1.imshow(np.ones_like(F), cmap="Greys", alpha=0.1)
    for r, c in Fpath:
        ax1.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=True, color="mistyrose", alpha=0.7))
    for i in range(F.shape[0]):
        for j in range(F.shape[1]):
            is_in_path = (i, j) in Fpath
            color = "red" if is_in_path else "black"
            fontweight = "bold" if is_in_path else "normal"
            ax1.text(j, i, F[i, j], ha="center", va="center", color=color, fontweight=fontweight)
    ax1.set_xticks(np.arange(-0.5, F.shape[1], 1), minor=True)
    ax1.set_yticks(np.arange(-0.5, F.shape[0], 1), minor=True)
    ax1.grid(which="minor", color="gray", linestyle="-", linewidth=0.5)
    if seq1 is not None and seq2 is not None:
        ax1.set_xticks(range(F.shape[1]))
        x_labels = [""] + list(seq2)  # empty string for position 0
        ax1.set_xticklabels(x_labels)
        ax1.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)

        ax1.set_yticks(range(F.shape[0]))
        y_labels = [""] + list(seq1)  # empty string for position 0
        ax1.set_yticklabels(y_labels)
    else:
        ax1.set_xticks(range(F.shape[1]))
        ax1.set_yticks(range(F.shape[0]))
    ax1.set_title("Alignment Score Matrix with Path")

    ax2.imshow(np.ones_like(F), cmap="Greys", alpha=0.1)
    for r, c in Fpath:
        ax2.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=True, color="mistyrose", alpha=0.7))
    for i in range(b.shape[0]):
        for j in range(b.shape[1]):
            is_in_path = (i, j) in Fpath
            color = "red" if is_in_path else "black"
            fontweight = "bold" if is_in_path else "normal"
            if b[i, j] is not None:
                text = f"({b[i, j][0]},{b[i, j][1]})"
            else:
                text = "None"
            ax2.text(j, i, text, ha="center", va="center", color=color, fontweight=fontweight, fontsize=8)
    ax2.set_xticks(np.arange(-0.5, b.shape[1], 1), minor=True)
    ax2.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    ax2.set_yticks(np.arange(-0.5, b.shape[0], 1), minor=True)
    ax2.grid(which="minor", color="gray", linestyle="-", linewidth=0.5)
    ax2.set_xticks(range(b.shape[1]))
    ax2.set_yticks(range(b.shape[0]))
    ax2.set_title("Backtracking Matrix with Path")

    plt.tight_layout()
    plt.show()
