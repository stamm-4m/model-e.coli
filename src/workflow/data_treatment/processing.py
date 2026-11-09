
"""
Nanobody-based Antivenom Production with E. coli Reactor Simulation 

Author: Juan Camilo Castaño Sanchez
Email: jcastano-san@insa-toulose.fr
Date: 09/11/2026

"""

from src.utils.io import load_yaml
import numpy as np
import pandas as pd
from src.model.core import kinetics_residual, ri_residual
from scipy.optimize import least_squares

# ------------------- Computes qP and mu and unifies dataframes --------------
# @timer

def processing_data(self, datasets, model_path):

    df_global = []

    self.model_ins = load_yaml(model_path)["model_instructions"]

    generation_names = [cfg["generation"]for cfg in self.model_ins["states"].values()]

    kinetic_names = list(self.model_ins["kinetic_functions"].keys())

    for br_id, data in datasets.items():
        self.br_id = br_id
        df = pd.DataFrame(data)
        df["Run_ID"] = br_id
        df.insert(0, "Run_ID", df.pop("Run_ID"))
        results = []
        for _, row in df.iterrows():
            t, state, derivatives = self.build_row_inputs(row)

            # Estimate ri
            ri_solution = least_squares(ri_residual,x0=np.ones(len(generation_names)),args=(self, t, state, derivatives))
            ri = ri_solution.x

            # Estimate kinetic parameters
            kinetics_solution = least_squares(kinetics_residual, x0=np.ones(len(kinetic_names)),args=(self, state, ri))

            kinetics = kinetics_solution.x
            row_result = {}
            row_result.update(dict(zip(generation_names, ri)))
            row_result.update(dict(zip(kinetic_names, kinetics)))

            results.append(row_result)

        results_df = pd.DataFrame(results)

        df = pd.concat([df.reset_index(drop=True),results_df.reset_index(drop=True)],axis=1)

        df_global.append(df)

    return pd.concat(df_global,ignore_index=True)

def build_row_inputs(self, row):

    time_var = self.model_ins["variables"]["time"]
    state_names = list(self.model_ins["states"].keys())
    volume_name = self.model_ins["volume"]["name"]

    t = row[time_var]

    state = np.array([
        row[var]
        for var in state_names + [volume_name]
    ])

    derivatives = np.array([
        row[f"d{var}dt"]
        for var in state_names
    ])

    return t, state, derivatives