
import numpy as np
from scipy.integrate import solve_ivp

class Simulator:
    def __init__(self, model, method, rtol, atol, max_step):
        self.model = model
        self.method = method
        self.rtol = rtol
        self.atol = atol
        self.max_step = max_step

    def run(self, y0, t_span, n_points=None, t_eval=None):   
        
        if t_eval is None:
            if n_points is None:
                raise ValueError("Provide either n_points or t_eval")
            t_eval = np.linspace(t_span[0], t_span[1], n_points)

        sol = solve_ivp(
            fun=self.model.dfdt,
            t_span=t_span,
            y0=y0,
            t_eval=t_eval,
            method=self.method,
            max_step=self.max_step,
            rtol=self.rtol,
            atol=self.atol
        )

        return sol
