
from scipy.interpolate import interp1d

class TemperatureProfile:
    def __init__(self, t_exp, T_exp):
        self.T = interp1d(
            t_exp, T_exp,
            fill_value="extrapolate",
            bounds_error=False
        )

    def F(self, t):
        return float(self.T(t))
