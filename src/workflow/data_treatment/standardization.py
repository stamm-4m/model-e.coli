"""
Nanobody-based Antivenom Production with E. coli Reactor Simulation 
Dataset unification for treatment

Author: Juan Camilo Castaño Sanchez
Email: jcastano-san@insa-toulose.fr
Date: 01/09/2026

"""

import pandas as pd

class DatasetStandardization:
    def __init__(self, filepath):#, sheet="Feuil1"):
        df = pd.read_excel(
            filepath,
            # sheet_name=sheet,
            header=0,
            skiprows=[1]
        )

        df = df.rename(columns={
            "Time": "time",
            "Glucose": "S",
            "Biomass": "X",
            "Protein": "P",
            "Acetate": "A",
            "Citrate": "C",
            "Vreal": "V",
            "Temperature": "T",
            "Induction": "I"
        })

        df["t"] = df["time"]
        self.df = df 

        self.t = df["t"].values

        self.data = {
            "t": df["t"].values,
            "X": df["X"].values,
            "S": df["S"].values,
            "P": df["P"].values,
            "C": df["C"].values,
            "V": df["V"].values,
        }

        self.V = df["V"].values
        self.X = df["X"].values
        self.T = df["T"].values
        self.I = df["I"].values

        self.y0 = [
            df["X"].iloc[0],
            df["S"].iloc[0],
            df["P"].iloc[0],
            df["V"].iloc[0],
        ]

        self.path = filepath

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