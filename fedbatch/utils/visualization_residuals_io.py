import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


def qq_plot_residuals(
    residuals,
    title="Q-Q plot of residuals",
    savepath=None
):
    residuals = np.asarray(residuals)
    residuals = residuals[~np.isnan(residuals)]

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)

    stats.probplot(residuals, dist="norm", plot=ax)

    ax.set_title(title)
    ax.grid(True)

    if savepath:
        plt.savefig(savepath, dpi=300, bbox_inches="tight")

    # plt.show()
