
import numpy as np

class MultiExperimentObjective:
    def __init__(self, datasets, simulators, model, y0s, param_names, full_params):
        self.datasets = datasets
        self.simulators = simulators
        self.model = model
        self.y0s = y0s
        self.param_names = param_names
        self.full_params = full_params


    def __call__(self, theta):
        
        params = self.full_params.copy()

        for name, value in zip(self.param_names, theta):
            params[name] = value

        self.model.set_params(params)

        residuals = []

        for dataset, sim, y0 in zip(self.datasets, self.simulators, self.y0s):

            sol = sim.run(
                t_span=(dataset.t[0], dataset.t[-1]),
                y0=y0,
                t_eval=dataset.t
            )

            residuals.extend(sol.y[0,:] - dataset.data["X"])
            residuals.extend(sol.y[1,:] - dataset.data["S"])
            residuals.extend(sol.y[2,:] - dataset.data["P"])

        return np.array(residuals)
