
import numpy as np
from src.utils.io import load_yaml
from src.model.auxiliar.functions import I_mu_profile, I_profile, feed_concentration_profile, feed_profile, var_profile
from src.model.core.kinetics import mu_fun, mu_max_fun, qp_fun, rP_fun, rS_fun, rX_fun


def ODEs(self, t, state):

    # --------- States --------
    X, S, P, V = state
    states = {
        'X': X,
        'S': S,
        'P': P,
        'V': V
    }

    # --------- Parameters --------
    parameters = load_yaml('src/config/params.yaml')

    # --------- Inputs --------
    inputs = {}

    T = var_profile(t, self.br_id, 'T')
    inputs['T'] = T

    I = I_profile(t, self.br_id, 'I')
    inputs['I'] = I

    C = var_profile(t, self.br_id, 'C')
    inputs['C'] = C

    S0 = var_profile(t, self.br_id, 'S0')
    inputs['S0'] = S0

    SF = var_profile(t, self.br_id, 'SF')
    inputs['SF'] = SF

    I_mu = I_mu_profile(t, self.br_id, 'I_mu')
    inputs['I_mu'] = I_mu

    # --------- Features --------
    features = {}

    features.update(states)
    features.update(inputs)

    features['states'] = states
    features['inputs'] = inputs

    # --------- Volume --------
    dVdt = feed_profile(t, self.br_id, 'S')
    Sf = feed_concentration_profile(t, self.br_id, 'S')

    # --------- Kinetics --------
    kinetics = {}

    mu = mu_fun(features, parameters)
    kinetics['mu'] = mu

    mu_max = mu_max_fun(features, parameters)
    kinetics['mu_max'] = mu_max

    qp = qp_fun(features, parameters)
    kinetics['qp'] = qp

    features['kinetics'] = kinetics

    # --------- Balances --------
    rX = rX_fun(features, parameters)
    dXdt = rX - (X/V)*dVdt

    rS = rS_fun(features, parameters)
    dSdt = rS - (S/V)*dVdt + (Sf/V)*dVdt

    rP = rP_fun(features, parameters)
    dPdt = rP - (P/V)*dVdt

    return np.array([dXdt, dSdt, dPdt, dVdt])