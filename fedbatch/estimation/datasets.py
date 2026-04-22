import pandas as pd

class ExperimentDataset:
    def __init__(self, filepath, sheet="Feuil1"):
        df = pd.read_excel(
            filepath,
            sheet_name=sheet,
            header=0,
            skiprows=[1]
        )

        df = df.rename(columns={
            "Time": "time",
            "Glucose": "S",
            "Biomass": "X",
            "Protein": "P",
            "Vreal": "V",
            "Temperature": "T",
            "Induction": "I"
        })

        self.t = df["time"].values

        self.data = {
            "X": df["X"].values,
            "S": df["S"].values,
            "P": df["P"].values,
            "V": df["V"].values,
        }

        self.V = df["V"].values
        self.T = df["T"].values
        self.I = df["I"].values

        self.y0 = [
            df["X"].iloc[0],
            df["S"].iloc[0],
            df["P"].iloc[0],
            df["V"].iloc[0],
        ]

        self.path = filepath

