
from glob import glob
from fedbatch.data_analysis.preprocessing import ( unificar_xls, run_EAD )
from fedbatch.data_analysis.data_plots import timeseries_per_run, scatter_fun

# Import and Unification of data
dataset_files = sorted(glob("data/raw/BR*.xls"))
save_dir="results/data_analysis/global/spline" # path for spline plots
yaml_path = "fedbatch/config/default_parameters.yaml"

df_global, df_batch, df_fedbatch, df_induction = unificar_xls(dataset_files, yaml_path, save_dir)

# Save global dataframe
df_global.to_csv("data/processed/BR_unified.csv", index=False)
df_global.to_excel("data/processed/BR_unified.xlsx",index=False,engine="openpyxl")

# -----------------Global Analysis------------------------------
# EAD
variables = ["time","X", "S", "V", "P", "mu", "qP", "I", "T", "A"] # qp_Old

save_dir="results/data_analysis/global/EAD"
run_EAD(df_global, variables, save_dir)

variables = ["X", "S", "V", "P", "mu", "qP", "I", "T", "A"] # qp_Old

# Time series overlaped
save_dir="results/data_analysis/global/time_series"
for variable in variables:
        timeseries_per_run(df_global, variable, save_dir)

# Scatter
save_dir="results/data_analysis/global/scatter"
for i, x in enumerate(variables):
    for y in variables[i+1:]:
        scatter_fun(df_global, x, y, save_dir)

# -----------------Batch Analysis------------------------------
# EAD
variables = ["time", "X", "S", "V", "mu", "T", "A"] # qp_Old # I , P, qP

save_dir="results/data_analysis/batch/EAD"
run_EAD(df_batch, variables, save_dir)

variables = ["X", "S", "V", "mu", "T", "A"] # qp_Old # I , P, qP

# Time series overlaped
save_dir="results/data_analysis/batch/time_series"
for variable in variables:
        timeseries_per_run(df_batch, variable, save_dir)

# Scatter
save_dir="results/data_analysis/batch/scatter"
for i, x in enumerate(variables):
    for y in variables[i+1:]:
        scatter_fun(df_batch, x, y, save_dir)
        
# -----------------Fedbatch Analysis------------------------------
# EAD
variables = ["time", "X", "S", "V", "mu", "T", "A"] # qp_Old # I, P, qP

save_dir="results/data_analysis/fed-batch/EAD"
run_EAD(df_fedbatch, variables, save_dir)

variables = ["X", "S", "V", "mu", "T", "A"] # qp_Old # I, P, qP

# Time series overlaped
save_dir="results/data_analysis/fed-batch/time_series"
for variable in variables:
        timeseries_per_run(df_fedbatch, variable, save_dir)

# Scatter
save_dir="results/data_analysis/fed-batch/scatter"
for i, x in enumerate(variables):
    for y in variables[i+1:]:
        scatter_fun(df_fedbatch, x, y, save_dir)
        
# -----------------Induction Analysis------------------------------
# EAD
variables = ["time", "X", "V", "P", "mu", "qP", "T"] # qp_Old # S, A , I

save_dir="results/data_analysis/induction/EAD"
run_EAD(df_induction, variables, save_dir)

variables = ["X", "V", "P", "mu", "qP", "T"] # qp_Old # S, A , I

# Time series overlaped
save_dir="results/data_analysis/induction/time_series"
for variable in variables:
        timeseries_per_run(df_induction, variable, save_dir)

# Scatter
save_dir="results/data_analysis/induction/scatter"
for i, x in enumerate(variables):
    for y in variables[i+1:]:
        scatter_fun(df_induction, x, y, save_dir)