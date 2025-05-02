from typing import Optional, Sequence

import matplotlib.pyplot as plt


def plot_paths(
    emissions: Sequence[int],
    hidden_path: Sequence[int],
    reconstructed_path: Sequence[int],
    *,
    biased_observations: Sequence[int],
    reconstruction_algorithm: Optional[str] = None,
) -> None:
    state_map = {0: "START", 1: "FAIR", 2: "BIASED", 3: "END"}
    if reconstruction_algorithm is None:
        reconstruction_algorithm = "Reconstructed"

    fig, ax = plt.subplots(figsize=(15, 6))
    positions = list(range(len(hidden_path)))

    ax.step(positions, hidden_path, "g-", where="mid", linewidth=2, alpha=0.7, label="True Hidden Path")
    ax.step(positions, reconstructed_path, "r--", where="mid", linewidth=2, label=f"{reconstruction_algorithm} Path")

    differences = [i for i in range(len(hidden_path)) if hidden_path[i] != reconstructed_path[i]]
    if differences:
        ax.scatter(
            [positions[i] for i in differences],
            [hidden_path[i] for i in differences],
            color="green",
            marker="o",
            s=50,
            alpha=0.5,
            zorder=3,
        )
        ax.scatter(
            [positions[i] for i in differences],
            [reconstructed_path[i] for i in differences],
            color="red",
            marker="o",
            s=50,
            alpha=0.5,
            zorder=3,
        )

    if biased_observations:
        for i, e in enumerate(emissions):
            if e in biased_observations:
                ax.axvline(x=i, color="gray", linestyle=":", alpha=0.4)

    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels([state_map[i] for i in range(4)])
    ax.set_title(f"Comparison of True Hidden Path vs. {reconstruction_algorithm} Path", fontsize=14)
    ax.set_xlabel("Sequence Position", fontsize=12)
    ax.set_ylabel("State", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)

    correct = sum(h == v for h, v in zip(hidden_path, reconstructed_path))
    accuracy = correct / len(hidden_path) * 100
    plt.suptitle(f"Accuracy: {accuracy:.2f}%", fontsize=16)

    plt.tight_layout()
    plt.show()
