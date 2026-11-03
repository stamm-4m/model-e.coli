"""
Nanobody-based Antivenom Production with E. coli Reactor Simulation 

simulator.py: This module contains the Simulator class, which is responsible for running simulations of 
the bioreactor model using numerical integration methods defined in the `scipy.integrate` library. 
The Simulator class takes a model, integration method, and solver parameters as input and provides a 
method to run the simulation over a specified time span.

Author: Juan Camilo Castaño Sanchez
Email: jcastano-san@insa-toulose.fr
Date: 01/09/2026
"""

import numpy as np
from scipy.integrate import solve_ivp

class Simulator:
    def __init__(self, model, method, rtol, atol, max_step):
        self.model = model
        self.method = method
        self.rtol = rtol
        self.atol = atol
        self.max_step = max_step

    def run(self, y0, t_span, n_points=None, t_eval=None, dense_output=False):   

        if t_eval is None:
            if n_points is None:
                raise ValueError("Provide either n_points or t_eval")
            t_eval = np.linspace(t_span[0], t_span[1], n_points)

        sol = solve_ivp(
            fun=self.model.ODEs,
            t_span=t_span,
            y0=y0,
            t_eval=t_eval,
            dense_output=dense_output,
            method=self.method,
            max_step=self.max_step,
            rtol=self.rtol,
            atol=self.atol
        )

        return sol  # sol.y (valores en t_exp) / sol.sol(t) (valores interpolados en cualquier t)