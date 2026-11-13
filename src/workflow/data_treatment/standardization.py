"""
Nanobody-based Antivenom Production with E. coli Reactor Simulation 
Dataset unification for treatment

Author: Juan Camilo Castaño Sanchez
Email: jcastano-san@insa-toulose.fr
Date: 01/09/2026

"""

import pandas as pd
from pathlib import Path
from src.utils.io import load_yaml

class DatasetStandardization:
    def __init__(self, filepath, model_path):

        # self.path = filepath
        model_cfg = load_yaml(model_path)
        rename_dict = {v: k for k, v in model_cfg["model_instructions"]["variables"].items()}

        df = pd.read_excel(filepath, header=0, skiprows=[1])
        df = df.rename(columns=rename_dict)

        self.df = df 

        states = model_cfg["model_instructions"]["states"].keys()
        V = model_cfg["model_instructions"]["volume"]["name"]

        self.y0 = [df[var].iloc[0]for var in [*states, V]]

        self.br_id = Path(filepath).stem
        

class ProcessedDataset:

    def __init__(self, br_id, df, filepath):

        self.path = filepath

        self.br_id = br_id

        self.df = df

        self.t = df["time"].to_numpy()

        self.X = df["X"].to_numpy()
        self.S = df["S"].to_numpy()
        self.P = df["P"].to_numpy()
        self.V = df["V"].to_numpy()
        self.T = df["T"].to_numpy()

        self.data = df