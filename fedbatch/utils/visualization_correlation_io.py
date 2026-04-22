
import numpy as np
import matplotlib.pyplot as plt


def plot_parameter_correlation(
    cov,
    param_names,
    threshold=0.8,
    savepath=None
):
    """
    Plot parameter correlation matrix from covariance.

    Parameters
    ----------
    cov : ndarray
        Covariance matrix (n_params x n_params)
    param_names : list of str
        Parameter names in correct order
    threshold : float
        Absolute correlation threshold to highlight
    """

    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)

    fig, ax = plt.subplots(figsize=(8, 7))

    im = ax.imshow(
        corr,
        cmap="coolwarm",
        vmin=-1,
        vmax=1
    )

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Correlation coefficient")

    # Ticks
    ax.set_xticks(range(len(param_names)))
    ax.set_yticks(range(len(param_names)))
    ax.set_xticklabels(param_names, rotation=45, ha="right")
    ax.set_yticklabels(param_names)

    # Highlight strong correlations
    for i in range(len(param_names)):
        for j in range(len(param_names)):
            if i != j and abs(corr[i, j]) > threshold:
                ax.text(
                    j, i,
                    f"{corr[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=9,
                    fontweight="bold"
                )

    ax.set_title("Parameter correlation matrix")

    plt.tight_layout()

    if savepath:
        plt.savefig(savepath, dpi=300, bbox_inches="tight")

    # plt.show()

    return corr
