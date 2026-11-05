
import numpy as np

def mu_max(features, params):
    states = features["states"]
    inputs = features["inputs"]
    X, S, P, V = states
    T, I, I_F, C, S0, SF = inputs
    mu_max_0 = params["mu_max_0"]
    mu_max_p = params["mu_max_p"]
    b = params["b"]
    if I_F == 0:
        mu_max = mu_max_0
    else:
        mu_max = mu_max_p * T + b
    return mu_max

def mu(features, params):
    states = features["states"]
    inputs = features["inputs"]
    X, S, P, V = states
    Ks = params["Ks"]
    mu_max_val = mu_max(states, inputs, params)
    mu = mu_max_val * S / (Ks + S)
    return mu

def qp(features, params):
    states = features["states"]
    inputs = features["inputs"]
    X, S, P, V = states
    T, I, I_F, C, S0, SF = inputs
    gamma_1 = params["gamma_1"]
    gamma_2 = params["gamma_2"]
    Ap_1 = params["Ap_1"]
    Ap_2 = params["Ap_2"]
    sigma = params["sigma"]
    alpha = params["alpha"]
    if I == 0:
        qp = 0
    else:
        mu_val = mu(states, inputs, params)
        beta = ( 
            ( gamma_1 * np.exp(- Ap_1 / T)) -
            ( gamma_2 * np.exp(- Ap_2 / T)) +
                sigma
            )
        qp = 0 # - (- alpha * mu_val + beta )
        # qp = abs(- alpha * mu_val + beta )
    return qp   # [mg Nb * L / (g X * h)]

def rX(features, params):
    states = features["states"]
    specific_rates = features["kinetic"]
    mu_val = specific_rates["mu"]
    X, S, P, V = states
    rate = mu_val * X
    return rate

def rS(features, params):
    Y_XS = params["Y_XS"]
    rX_val = rX(features, params)
    rate = - rX_val / Y_XS 
    return rate

def rP(features, params):
    states = features["states"]
    specific_rates = features["kinetic"]
    mu_val = specific_rates["mu"]
    qp_val = specific_rates["qp"]
    X, S, P, V = states
    rate = qp_val * X
    # rate = qp_val * X + mu_val * P 
    return rate
