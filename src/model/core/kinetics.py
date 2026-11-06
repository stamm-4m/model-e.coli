
import numpy as np

def mu_max_fun(features, params):
    T = features["inputs"]["T"]
    I_mu = features["inputs"]["I_mu"]
    mu_max_0 = params["mu_max_0"]
    mu_max_p = params["mu_max_p"]
    b = params["b"]
    if I_mu == 0:
        mu_max = mu_max_0
    else:
        mu_max = mu_max_p * T + b
    return mu_max

def mu_fun(features, params):
    S = features["states"]["S"]
    Ks = params["Ks"]
    mu_max_val = mu_max_fun(features, params)
    mu = mu_max_val * S / (Ks + S)
    return mu

def qp_fun(features, params):
    T = features["inputs"]["T"]
    I = features["inputs"]["I"]
    gamma_1 = params["gamma_1"]
    gamma_2 = params["gamma_2"]
    Ap_1 = params["Ap_1"]
    Ap_2 = params["Ap_2"]
    sigma = params["sigma"]
    alpha = params["alpha"]
    if I == 0:
        qp = 0
    else:
        mu_val = mu_fun(features, params)
        beta = ( 
            ( gamma_1 * np.exp(- Ap_1 / T)) -
            ( gamma_2 * np.exp(- Ap_2 / T)) +
                sigma
            )
        qp = 0 # - (- alpha * mu_val + beta )
        # qp = abs(- alpha * mu_val + beta )
    return qp   # [mg Nb * L / (g X * h)]

def rX_fun(features, params):
    X = features["states"]["X"]
    mu_val = features["kinetic"]["mu"]
    rate = mu_val * X
    return rate

def rS_fun(features, params):
    Y_XS = params["Y_XS"]
    rX_val = rX_fun(features, params)
    rate = - rX_val / Y_XS 
    return rate

def rP_fun(features, params):
    mu_val = features["kinetic"]["mu"]
    qp_val = features["kinetic"]["qp"]
    X = features["states"]["X"]
    rate = qp_val * X
    # rate = qp_val * X + mu_val * P 
    return rate

