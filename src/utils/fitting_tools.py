"""

Note: it can be implemented LMFIT for verosimilitud, correlation of paramteres, automtic reports and non-symetric intervals
"""


import numpy as np
from scipy.stats import t
# from lmfit import Parameters

def compute_confidence_intervals(result, alpha=0.05):
    J = result.jac
    n_res, n_par = J.shape

    sigma2 = 2 * result.cost / (n_res - n_par)
    # cov = sigma2 * np.linalg.inv(J.T @ J)
    cov = sigma2 * np.linalg.pinv(J.T @ J)
    std = np.sqrt(np.diag(cov))

    # z = 1.96  # 95% CI
    # ci = np.vstack([
    #     result.x - z * std,
    #     result.x + z * std
    # ]).T

    alpha = 0.05
    tval = t.ppf( 1 - alpha/2, n_res - n_par )

    ci = np.vstack([
        result.x - tval*std,
        result.x + tval*std
    ]).T

    return cov, std, ci


class MultiExperimentObjective:
    def __init__(self, datasets, simulators, kin, y0s, param_names, full_params):
        self.datasets = datasets
        self.simulators = simulators
        self.kin = kin
        self.y0s = y0s
        self.param_names = param_names
        self.full_params = full_params

    def __call__(self, theta):
        
        params = self.full_params.copy()

        for name, value in zip(self.param_names, theta):
            params[name] = value

        self.kin.set_params(params)

        residuals = []

        for dataset, sim, y0 in zip(self.datasets, self.simulators, self.y0s):

            sol = sim.run(
                t_span=(dataset.t[0], dataset.t[-1]),
                y0=y0,
                t_eval=dataset.t
            )

            if (not sol.success) or (sol.y.shape[1] != len(dataset.t)):
                print("FAILED")
                print(params)
                print(sol.message)
                residuals.extend(np.full(2 * len(dataset.t), 1e6))
                continue

            residuals.extend( (sol.y[0,:] - dataset.data["X"]) / dataset.data["X"].std() )
            residuals.extend( (sol.y[1,:] - dataset.data["S"]) / dataset.data["S"].std() )
            # residuals.extend( (sol.y[2,:] - dataset.data["P"]) / dataset.data["P"].std() )

        return np.array(residuals)
    
