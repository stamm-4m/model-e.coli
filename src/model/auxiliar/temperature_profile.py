"""
Nanobody-based Antivenom Production with E. coli Reactor Simulation 

Creates a temperature profile based on the experimental data provided in the `t_exp` and `T_exp` arrays.

Author: Juan Camilo Castaño Sanchez
Email: jcastano-san@insa-toulose.fr
Date: 01/09/2026
"""

from scipy.interpolate import interp1d

class TemperatureProfile:
    def __init__(self, t_exp, T_exp):
        self.Temperature = interp1d(
            t_exp, T_exp,
            fill_value="extrapolate",
            bounds_error=False
        )

    def __call__(self, t):
        return round(float(self.Temperature(t)), 3)
