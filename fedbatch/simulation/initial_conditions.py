import numpy as np 

def build_initial_state(cfg, br_id, dataset):
    """
    Build consistent initial conditions vector y0.
    Only substrate S is taken from data.
    Other states come from configuration.
    """
    # --- measured from dataset ---
    
    X0 = dataset.y0[0]
    S0 = dataset.y0[1]
    P0 = dataset.y0[2]
    V0 = dataset.y0[3] # V0 = dataset.V[0]

    # --- model-defined defaults ---

    # S0 = cfg["bioreactor"][br_id]["S0"]["value"]    

    # --- return in correct order ---
    return np.array([X0, S0, P0, V0], dtype=float)
