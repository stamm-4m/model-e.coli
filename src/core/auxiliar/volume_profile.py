
from scipy.interpolate import UnivariateSpline
from scipy.optimize import minimize_scalar
from src.data_analysis.data_treatment.derivative import fun_obj_uni

class VolumeProfile:
    def __init__(self, t_exp, V_exp):

        bounds = (0.0001,0.01)
        res = minimize_scalar(fun_obj_uni, 
                            bounds=bounds,
                            method="bounded",
                            args=(t_exp, V_exp))
        
        s_calc_uni = res.x # type: ignore
        self.V = UnivariateSpline(t_exp, V_exp, s=s_calc_uni)
        self.dV = self.V.derivative()

    def F(self, t):
        return float(self.V(t)), float(self.dV(t))
