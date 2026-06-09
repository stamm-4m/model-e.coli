
from glob import glob
import pandas as pd
from pathlib import Path
from src.data_analysis.data_treatment.data import ExperimentDataset
from src.data_analysis.data_treatment.outliers import process_all_datasets # , data_unification
from src.data_analysis.data_treatment.derivative import compute_derivatives_for_datasets
from src.data_analysis.data_treatment.processing import processing_data
from src.data_analysis.data_treatment.ead import compute_ead
from src.data_analysis.feature_selection.filter import filter_feature_selection
from src.data_analysis.feature_selection.wrapper import wrapper_feature_selection
from src.data_analysis.cross_validation.cross_val import cross_validation
from src.utils.io import load_yaml

 # use_log

# # =================== Import raw data ===================
# ruta = Path("data/raw")
# dataset_files = sorted(glob("data/raw/BR*.xls"))
# datasets = [ExperimentDataset(f) for f in dataset_files]
# br_id_list = [file.stem for file in ruta.iterdir() if file.suffix == ".xls"]
# variable_list = ["t", "X", "S", "V", "P", "T"]

# # =================== Data treatment =================== 
# _, treat_data = process_all_datasets(datasets = datasets, time_col = "time", variable_list = variable_list, 
#                                                 results_root="results/data_analysis/outliers_and_smoothing", smooth=False)

# # =================== Derivates calculation =================== 
# _, data_sets = compute_derivatives_for_datasets(treat_data, variables=("X", "S", "V", "P"), 
#                                                 results_root="results/data_analysis/derivatives/treat")

# # =================== Computes qP and mu calculation =================== 
# yaml_path = "src/config/default_parameters.yaml"
# df_global, df_induction = processing_data(data_sets, yaml_path, t_ind_exp = True) # type: ignore
# df_induction.to_excel("data/processed/BR_processed_ind.xlsx",index=False,engine="openpyxl")
# df_global.to_excel("data/processed/BR_processed.xlsx",index=False,engine="openpyxl")

# =================== List of ML models (for wrapper feature selection) ===================
backward_models = ["linear", "poisson", "tweedie", "LASSO_b", "Ridge_b", 
                  "elasticnet_b", "rf_b", "gbm_b", "svm_linear"] # "tree", 
permutation_models = ["svm_rbf", "svm_poly", "knn"] 
                    # "mlp", "gpr", "LASSO_p", "Ridge_p", 
                    # "elasticnet_p", "rf_p", "gbm_ps",

# =================== All Features, Targets & Settings ===================
top_n = 1 # number of models to save for each cv execution
i = 0 # (0) only calc target (1) real target  # for i in range(0,1): # (1) onlycalc (2) real 

all_features = ["t", "t_ind", "t_ind_ad",
                "X", "S", "V", "P", "mu", "T", "I", "FS_calc", "dXdt", "dSdt", "dVdt", "Xlag1", "Plag1", 
                "X_calc", "V_calc", "mu_calc",                 "dXdt_calc", "dVdt_calc", "Xlag1_calc"]
 
target = ["qP_calc", "rP_calc", "P"] # "qP", "rP", "P"] 

exclude_features = ["X", "S", "V", "mu", "dXdt", "dSdt", "dVdt", "Xlag1"] # in-line measurments  
vars_ = [v for v in all_features if v not in exclude_features]
vars_P = [v for v in vars_ if v != "P"]

targets_loop = [ (target[2*i  ], vars_ ),
                 (target[2*i+1], vars_ ),
                 (target[2*i+2], vars_P) ]

# =================== Data-frame load ===================
df_global = pd.read_excel(r"data/processed/BR_processed.xlsx")
df_induction = pd.read_excel(r"data/processed/BR_processed_ind.xlsx")

df_global_ind = df_global.copy()
mask = df_global["I"] == 0
df_global_ind.loc[mask, "Run_ID"] = "BR09"

dfs = {"global": df_global, "induction": df_induction, "global_ind": df_global_ind}
config = load_yaml("src/data_analysis/config.yaml")

# =================== Computation ===================
for name, df in dfs.items():
    print(f"\n===== {name.upper()} =====")
    # ---------- paths ----------
    base_ead = f"results/data_analysis/ead/{name}"
    base_fs = f"results/feature_selection/{name}"
    base_cv = f"results/cross_validation/{name}"

    # # =================== EAD =================== 
    # compute_ead(df, vars = vars_ + target[2*i:2*i+1], results_root=f"{base_ead}/qPrP") # qP and rP
    # compute_ead(df, vars = vars_, results_root=f"{base_ead}/P") # P
    
    for y_var, X_vars in targets_loop:
        # # =================== FEATURE SELECTION ===================
        # # ------------------- Filter methods -------------------
        # filter_feature_selection(df, X_vars = X_vars, y_var = y_var, out_dir=f"{base_fs}/filter/{y_var}")                                                                 

        # ------------------- Wrapper -------------------
        cfg = config[name][y_var]
        exclude_features_filter = cfg["exclude"]
        c_var = cfg["c_var"]

        vars_wrapper = [v for v in X_vars if v not in exclude_features_filter] # mu_calc, Plag1, t, t_ind, T can be filter by wrapper methods

        wrapper_feature_selection(df, X_vars = vars_wrapper, y_var = y_var, c_var = c_var, 
                                  model_names_b = backward_models, 
                                  model_names_p = permutation_models, 
                                  out_path=f"{base_fs}/wrapper/{y_var}")

        #  =================== CV =================== 
        cross_validation(df, y_var = y_var, in_dir=f"{base_fs}/wrapper", out_dir=base_cv, top_n = top_n)
        
    print(f"{name} Finished")
