
from scipy.interpolate import UnivariateSpline
from scipy.optimize import minimize_scalar
from src.data_analysis.data_treatment.derivative import fun_obj_uni

class BiomassProfile:
    def __init__(self, t_exp, X_exp, V_profile):

        bounds = (3,5)
        res = minimize_scalar(fun_obj_uni, 
                            bounds=bounds,
                            method="bounded",
                            args=(t_exp, X_exp))
        
        s_calc_uni = res.x # type: ignore
        self.X = UnivariateSpline(t_exp, X_exp, s=s_calc_uni)
        self.dX = self.X.derivative()
        self.V_profile = V_profile
        
        self.t_max = t_exp.max()

    def F(self, t):
        V, dV = self.V_profile.F(t)
        mu    = (1/self.X(t)) * ( self.dX(t) ) + (1/V) * ( dV )
        return float(max(self.X(t),0)), float(max(self.dX(t),0)), float(max(mu,0))
