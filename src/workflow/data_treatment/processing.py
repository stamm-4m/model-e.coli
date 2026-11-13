
"""
Nanobody-based Antivenom Production with E. coli Reactor Simulation 

Author: Juan Camilo Castaño Sanchez
Email: jcastano-san@insa-toulose.fr
Date: 09/11/2026

"""

from src.utils.io import load_yaml
import numpy as np
import pandas as pd
from src.model.core.kinetics_residual import kinetics_residual
from src.model.core.ri_residual import ri_residual
from scipy.optimize import least_squares, fsolve

# ------------------- Computes qP and mu and unifies dataframes --------------
# @timer

def processing_data(datasets, model_path):

    df_global = []

    model_ins = load_yaml(model_path)["model_instructions"]

    generation_names = [cfg["generation"]for cfg in model_ins["states"].values()]

    # kinetic_names = list(model_ins["kinetic_functions"].keys())
    kinetic_names = [name for name, cfg in model_ins["kinetic_functions"].items() if cfg.get("estimation", False)]

    for br_id, data in datasets.items():
        df = pd.DataFrame(data)
        df["Run_ID"] = br_id
        df.insert(0, "Run_ID", df.pop("Run_ID"))
        results = []
        for _, row in df.iterrows():
            t, state, derivatives = build_row_inputs(row, model_ins)

            # Estimate ri
            # ri_solution = least_squares(ri_residual, x0=np.zeros(len(generation_names)), args=(t, state, derivatives, br_id))
            ri_solution = fsolve(ri_residual, x0=np.zeros(len(generation_names)), args=(t, state, derivatives, br_id))
            ri = ri_solution.x

            # Estimate kinetic parameters
            # kinetics_solution = least_squares(kinetics_residual, x0=np.zeros(len(kinetic_names)), args=(state, ri, br_id))
            kinetics_solution = fsolve(kinetics_residual, x0=np.zeros(len(kinetic_names)), args=(state, ri, br_id))
            kinetics = kinetics_solution.x
            
            row_result = {}
            row_result.update(dict(zip(generation_names, ri)))
            row_result.update(dict(zip(kinetic_names, kinetics)))

            results.append(row_result)

        results_df = pd.DataFrame(results)

        df = pd.concat([df.reset_index(drop=True),results_df.reset_index(drop=True)],axis=1)

        df_global.append(df)

    return pd.concat(df_global,ignore_index=True)

def build_row_inputs(row, model_ins):

    time_var = model_ins["independent"][0]
    state_names = list(model_ins["states"].keys())
    volume_name = model_ins["volume"]["name"]

    t = row[time_var]

    state = np.array([ row[var] for var in state_names + [volume_name] ])
    derivatives = np.array([ row[f"d{var}dt"] for var in state_names ])

    return t, state, derivatives