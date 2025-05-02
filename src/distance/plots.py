import Bio.Seq
import matplotlib.pyplot as plt
import numpy as np


def plot_scoring_matrix(
    seq1: Bio.Seq.Seq | str,
    seq2: Bio.Seq.Seq | str,
    F: np.ndarray,
) -> None:
    fig, ax = plt.subplots(figsize=(18, 8))

    ax.imshow(np.ones_like(F), cmap="Greys", alpha=0.1)
    for i in range(F.shape[0]):
        for j in range(F.shape[1]):
            color = "black"
            fontweight = "normal"
            ax.text(j, i, F[i, j], ha="center", va="center", color=color, fontweight=fontweight)
    ax.set_xticks(np.arange(-0.5, F.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, F.shape[0], 1), minor=True)
    ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.5)
    if seq1 is not None and seq2 is not None:
        ax.set_xticks(range(F.shape[1]))
        x_labels = [""] + list(seq2)  # empty string for position 0
        ax.set_xticklabels(x_labels)
        ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)

        ax.set_yticks(range(F.shape[0]))
        y_labels = [""] + list(seq1)  # empty string for position 0
        ax.set_yticklabels(y_labels)
    else:
        ax.set_xticks(range(F.shape[1]))
        ax.set_yticks(range(F.shape[0]))
    ax.set_title("Edit Distance Score Matrix")

    plt.tight_layout()
    plt.show()
