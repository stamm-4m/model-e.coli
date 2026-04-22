
import numpy as np

def compute_confidence_intervals(result, alpha=0.05):
    J = result.jac
    n_res, n_par = J.shape

    sigma2 = 2 * result.cost / (n_res - n_par)
    cov = sigma2 * np.linalg.inv(J.T @ J)
    std = np.sqrt(np.diag(cov))

    z = 1.96  # 95% CI
    ci = np.vstack([
        result.x - z * std,
        result.x + z * std
    ]).T

    return cov, std, ci
