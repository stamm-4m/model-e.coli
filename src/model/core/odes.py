import numpy as np

from src.utils.io import load_yaml
from src.utils import import_function

def ODEs(self, t, state):

    X, S, P, V = state

    states = {
        'X': X,
        'S': S,
        'P': P,
        'V': V
    }

    parameters = load_yaml('src/config/params.yaml')

    inputs = {}

    T_func = import_function('src.model.auxiliar.funtions.var_profile')
    T = T_func(t, self.br_id, 'T')
    inputs['T'] = T

    I_func = import_function('src.model.auxiliar.funtions.I_profile')
    I = I_func(t, self.br_id, 'I')
    inputs['I'] = I

    C_func = import_function('src.model.auxiliar.funtions.var_profile')
    C = C_func(t, self.br_id, 'C')
    inputs['C'] = C

    S0_func = import_function('src.model.auxiliar.funtions.var_profile')
    S0 = S0_func(t, self.br_id, 'S0')
    inputs['S0'] = S0

    SF_func = import_function('src.model.auxiliar.funtions.var_profile')
    SF = SF_func(t, self.br_id, 'SF')
    inputs['SF'] = SF

    features = {}

    features.update(states)
    features.update(inputs)

    features['states'] = states
    features['inputs'] = inputs

    feed_profile = import_function('src.model.auxiliar.feed_profile')
    dVdt, I_F = feed_profile(t, self.br_id, 'S')

    kinetics = {}

    mu_func = import_function('src.model.core.kinetics.mu')
    mu = mu_func(features, parameters)
    kinetics['mu'] = mu

    mu_max_func = import_function('src.model.core.kinetics.mu_max')
    mu_max = mu_max_func(features, parameters)
    kinetics['mu_max'] = mu_max

    qp_func = import_function('src.model.core.kinetics.qp')
    qp = qp_func(features, parameters)
    kinetics['qp'] = qp

    features['kinetics'] = kinetics

    rX_func = import_function('src.model.core.kinetics.rX')
    rX = rX_func(features)
    dXdt = rX - (X/V)*dVdt

    rS_func = import_function('src.model.core.kinetics.rS')
    rS = rS_func(features)
    dSdt = rS - (S/V)*dVdt + (Sf/V)*dVdt

    rP_func = import_function('src.model.core.kinetics.rP')
    rP = rP_func(features)
    dPdt = rP - (P/V)*dVdt


    return np.array([dXdt, dSdt, dPdt, dVdt])