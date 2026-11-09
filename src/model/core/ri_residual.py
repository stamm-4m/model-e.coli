import numpy as np
from src.model.auxiliar.functions import I_mu_profile, I_profile, feed_concentration_profile, feed_profile, var_profile
from src.model.core.kinetics import mu_fun, mu_max_fun, qp_fun, rP_fun, rS_fun, rX_fun


def ri_residual(ri, t, state, derivatives, br_id):

    X, S, P, V = state

    rX, rS, rP = ri

    dXdt, dSdt, dPdt = derivatives

    dVdt = feed_profile(t, br_id, 'S')
    Sf = feed_concentration_profile(t, br_id, 'S')


    dXdt_calc = rX - (X/V)*dVdt
    dSdt_calc = rS - (S/V)*dVdt + (Sf/V)*dVdt
    dPdt_calc = rP - (P/V)*dVdt

    return np.array([dXdt_calc - dXdt, dSdt_calc - dSdt, dPdt_calc - dPdt])