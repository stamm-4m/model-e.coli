
import numpy as np
from src.utils.io import load_yaml
from src.model.auxiliar.functions import I_mu_profile, I_profile, feed_concentration_profile, feed_profile, var_profile
from src.model.core.kinetics import mu_fun, mu_max_fun, qp_fun, rP_fun, rS_fun, rX_fun


def kinetics_residual(kinetic_values, state, ri, br_id):

    X, S, P, V = state

    rX, rS, rP = ri

    params = load_yaml('src/config/params.yaml')

    mu, mu_max, qp = kinetic_values

    kinetics = {
        'mu': mu,
        'mu_max': mu_max,
        'qp': qp,
    }

    features = {}

    features['states'] = {
        'X': X,
        'S': S,
        'P': P,
    }

    features['kinetics'] = kinetics

    rX_calc = rX_fun(features, params)
    rS_calc = rS_fun(features, params)
    rP_calc = rP_fun(features, params)

    return np.array([rX_calc - rX, rS_calc - rS, rP_calc - rP])