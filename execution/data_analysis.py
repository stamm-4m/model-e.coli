
from glob import glob
from fedbatch.data_analysis.preprocessing import (
    unificar_xls, run_EAD, agregar_T_ind )
from fedbatch.data_analysis.data_plots import timeseries_per_run, scatter_fun

# Import and Unification of data
dataset_files = sorted(glob("data/raw/BR*.xls"))
save_dir="results/data_analysis/spline"
df = unificar_xls(dataset_files, save_dir)
df = agregar_T_ind(df)
df.to_csv("data/processed/BR_unified.csv", index=False)
df.to_excel("data/processed/BR_unified.xlsx",index=False,engine="openpyxl")


# EAD
save_dir="results/data_analysis/EAD"
run_EAD(df, save_dir)

# Time series overlaped
save_dir="results/data_analysis/time_series"
variables = ["X", "S", "V", "P", "mu", "qP", "I", "T", "A"] # qp_Old
for variable in variables:
        timeseries_per_run(df, variable, save_dir)

# Scatter
save_dir="results/data_analysis/scatter"
variables = ["X", "S", "V", "P", "mu", "qP", "I", "T", "A"] # qp_Old
for i, x in enumerate(variables):
    for y in variables[i+1:]:
        scatter_fun(df, x, y, save_dir)
