
import pandas as pd
from glob import glob
from fedbatch.data_analysis.preprocessing import (
    unificar_xls, run_EAD, timeseries_per_run, scatter_fun, agregar_T_ind )

# Import and Unification of data
dataset_files = sorted(glob("data/raw/BR*.xls"))
df = unificar_xls(dataset_files)
df = agregar_T_ind(df)
df.to_csv("data/processed/BR_unified.csv", index=False)

# EAD
save_dir="results/data_analysis/EAD"
run_EAD(df, save_dir)

# Time series overlaped
save_dir="results/data_analysis/time_series"
variable = ['X']
timeseries_per_run(df, variable, save_dir)

# Scatter
save_dir="results/data_analysis/scatter"
variables = ["X", "V", "mu", "qP"]
for i, x in enumerate(variables):
    for y in variables[i+1:]:
        scatter_fun(df, x, y, save_dir)
