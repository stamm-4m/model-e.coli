
import pandas as pd
from glob import glob
from fedbatch.data_analysis.preprocessing import unificar_xls, run_EDA, timeseries_per_run, scatter_fun

# Import and Unification of data
dataset_files = sorted(glob("data/raw/BR*.xls"))
df = unificar_xls(dataset_files)
df.to_csv("data/processed/BR_unified.csv", index=False)

# EDA
save_dir="results/data_analysis/plots"
run_EDA(df, save_dir)

# Time series overlaped
variable = ['X']
timeseries_per_run(df, variable, save_dir)

# Time series overlaped
x = 'qP'
y ='V'
scatter_fun(df, x, y, save_dir)