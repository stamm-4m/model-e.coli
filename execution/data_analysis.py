
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

 # use_log
 
# # # ---------- Import raw data ---------- 
# ruta = Path("data/raw")
# dataset_files = sorted(glob("data/raw/BR*.xls"))
# datasets = [ExperimentDataset(f) for f in dataset_files]
# br_id_list = [file.stem for file in ruta.iterdir() if file.suffix == ".xls"]
# variable_list = ["t", "X", "S", "V", "P", "T"]

# # ---------- Data treatment ---------- 
# smooth_data, treat_data = process_all_datasets(datasets = datasets, time_col = "time", variable_list = variable_list, 
#                                                 results_root="results/data_analysis/outliers_and_smoothing", smooth=False)

# # ---------- Derivates calculation ---------- 
# _, data_sets = compute_derivatives_for_datasets(treat_data, variables=("X", "S", "V", "P"), 
#                                                 results_root="results/data_analysis/derivatives/treat")
# # smooth_data = compute_derivatives_for_datasets(smooth_data, variables=("X", "V", "P"), 
# #                                                 results_root="results/data_analysis/derivatives/smooth")

# # ---------- Computes qP and mu calculation and ---------- 
# yaml_path = "src/config/default_parameters.yaml"
# df_global, df_induction = processing_data(data_sets, yaml_path, t_ind_exp = True) # type: ignore
# df_induction.to_excel("data/processed/BR_processed_ind.xlsx",index=False,engine="openpyxl")
# df_global.to_excel("data/processed/BR_processed.xlsx",index=False,engine="openpyxl")

# ----------- ML models --------------
wrapper_models = ["linear", "poisson", "tweedie", "LASSO_b", "Ridge_b", 
                  "elasticnet_b", "rf_b", "gbm_b", "svm_linear"] # "tree", 
permutation_models = ["svm_rbf", "svm_poly", "knn"] # "mlp", "gpr", "LASSO_p", "Ridge_p", "elasticnet_p", "rf_p", "gbm_ps",

# Features & target
all_features = ["t", "t_ind",
                "X", "S", "V", "P", "mu", "T", "I", "FS_calc", "dXdt", "dSdt", "dVdt", "Xlag1", "Plag1", 
                "X_calc", "V_calc", "mu_calc",                 "dXdt_calc", "dVdt_calc", "Xlag1_calc"] 

target = ["qP_calc", "rP_calc", 
          "qP", "rP"] 

exclude_features = ["X", "S", "V", "mu", "dXdt", "dSdt", "dVdt", "Xlag1"] # in-line measurments
                    
vars = [v for v in all_features if v not in exclude_features]

top_n = 1 # number of models to save

# qP and rP
for i in range(0,1): # (1) onlycalc (2) real 
    # # # --------- Global --------------------------------------------------------------------------------------------------------------
    df_global = pd.read_excel(r"data/processed/BR_processed.xlsx")
    vars_ead = vars + target[2*i : 2*i + 2] 

    # # ---------- EAD ---------- 
    compute_ead(df_global, vars = vars_ead, results_root="results/data_analysis/ead/global")

    # # ----------- Feature selection -------------- 
    # ----------- Filter methods --------------
    filter_feature_selection(df_global, X_vars = vars, y_var=target[2*i], out_dir=f"results/feature_selection/global/filter/{target[2*i]}")
    filter_feature_selection(df_global, X_vars = vars, y_var=target[2*i + 1], out_dir=f"results/feature_selection/global/filter/{target[2*i + 1]}")

    exclude_features = ["X_calc", "V_calc", "dXdt_calc", "dVdt_calc", "Xlag1_calc"] # once it is analyzed filter feature selection                                                                   
    vars = [v for v in vars if v not in exclude_features] # mu_calc, Plag1, t, t_ind, T can be filter by wrapper methods
    permanent_features = ["I", "P"] # cant be exclude (heatmap analysis)

    # --------- Wrapper & Embedded methods ------------
    wrapper_feature_selection(df_global, X_vars = vars, y_var=target[2*i], c_var = permanent_features, model_names_w=wrapper_models, 
                        model_names_p=permutation_models, out_path=f"results/feature_selection/global/wrapper/{target[2*i]}")
    wrapper_feature_selection(df_global, X_vars = vars, y_var=target[2*i + 1], c_var = permanent_features, model_names_w=wrapper_models, 
                        model_names_p=permutation_models, out_path=f"results/feature_selection/global/wrapper/{target[2*i + 1]}")

    # # --------- Cross-Validation methods --------------- 
    cross_validation(df_global, y_var=target[2*i],in_dir="results/feature_selection/global/wrapper", 
                    out_dir="results/cross_validation/global", top_n = top_n)
    cross_validation(df_global, y_var=target[2*i + 1], in_dir="results/feature_selection/global/wrapper", 
                    out_dir="results/cross_validation/global", top_n = top_n)
    
# P
# # ---------- EAD ---------- 
compute_ead(df_global, vars = vars, results_root="results/data_analysis/ead/P")

# # ----------- Feature selection -------------- 
# ----------- Filter methods --------------
vars = [v for v in all_features if v not in ("P")]

filter_feature_selection(df_global, X_vars = vars, y_var="P", out_dir=f"results/feature_selection/global/filter/P")
filter_feature_selection(df_global, X_vars = vars, y_var="P", out_dir=f"results/feature_selection/global/filter/P")

# --------- Wrapper & Embedded methods ------------
wrapper_feature_selection(df_global, X_vars = vars, y_var="P", model_names_w=wrapper_models, 
                    model_names_p=permutation_models, out_path=f"results/feature_selection/global/wrapper/P")
wrapper_feature_selection(df_global, X_vars = vars, y_var="P", model_names_w=wrapper_models, 
                    model_names_p=permutation_models, out_path=f"results/feature_selection/global/wrapper/P")

# # --------- Cross-Validation methods --------------- 
cross_validation(df_global, y_var="P",in_dir="results/feature_selection/global/wrapper", 
                out_dir="results/cross_validation/P", top_n = top_n)
cross_validation(df_global, y_var="P", in_dir="results/feature_selection/global/wrapper", 
                out_dir="results/cross_validation/global", top_n = top_n)

    # # # --------- Ind = 0 Training --------------------------------------------------------------------------------------------------------------
    # df_global = pd.read_excel(r"data/processed/BR_processed.xlsx")
    # df_global_ind = df_global.copy()
    # mask = df_global["I"] == 0
    # df_global_ind.loc[mask, "Run_ID"] = "BR09"

    # # # ---------- EAD ---------- 
    # compute_ead(df_global_ind, vars = vars + target[2*i:2*i + 1], results_root="results/data_analysis/ead/global_ind")

    # # # ----------- Feature selection -------------- 
    # # ----------- Filter methods --------------
    # filter_feature_selection(df_global_ind, X_vars = vars, y_var=target[2*i], out_dir=f"results/feature_selection/global_ind/filter/{target[2*i]}")
    # filter_feature_selection(df_global_ind, X_vars = vars, y_var=target[2*i + 1], out_dir=f"results/feature_selection/global_ind/filter/{target[2*i + 1]}")

    # # --------- Wrapper & Embedded methods ------------
    # wrapper_feature_selection(df_global_ind, X_vars = vars, y_var=target[2*i], model_names_w=wrapper_models, 
    #                     model_names_p=permutation_models, out_path=f"results/feature_selection/global_ind/wrapper/{target[2*i]}")
    # wrapper_feature_selection(df_global_ind, X_vars = vars, y_var=target[2*i + 1], model_names_w=wrapper_models, 
    #                     model_names_p=permutation_models, out_path=f"results/feature_selection/global_ind/wrapper/{target[2*i + 1]}")

    # # # --------- Cross-Validation methods --------------- 
    # cross_validation(df_global_ind, y_var=target[2*i],in_dir="results/feature_selection/global_ind/wrapper", 
    #                 out_dir="results/cross_validation/global_ind", top_n = top_n)
    # cross_validation(df_global_ind, y_var=target[2*i + 1], in_dir="results/feature_selection/global_ind/wrapper", 
    #                 out_dir="results/cross_validation/global_ind", top_n = top_n)

    # # # ---------- Induction --------------------------------------------------------------------------------------------------------------
    # df_induction = pd.read_excel(r"data/processed/BR_processed_ind.xlsx")

    # # # ---------- EAD ---------- 
    # compute_ead(df_induction, vars = vars + target[2*i:2*i + 1], results_root="results/data_analysis/ead/induction")

    # # # ----------- Feature selection -------------- 
    # # ----------- Filter methods --------------
    # filter_feature_selection(df_induction, X_vars = vars, y_var=target[2*i], out_dir=f"results/feature_selection/induction/filter/{target[2*i]}")
    # filter_feature_selection(df_induction, X_vars = vars, y_var=target[2*i + 1], out_dir=f"results/feature_selection/induction/filter/{target[2*i + 1]}")

    # # --------- Wrapper & Embedded methods ------------
    # wrapper_feature_selection(df_induction, X_vars = vars, y_var=target[2*i], model_names_w=wrapper_models, 
    #                     model_names_p=permutation_models, out_path=f"results/feature_selection/induction/wrapper/{target[2*i]}")
    # wrapper_feature_selection(df_induction, X_vars = vars, y_var=target[2*i + 1], model_names_w=wrapper_models, 
    #                     model_names_p=permutation_models, out_path=f"results/feature_selection/induction/wrapper/{target[2*i + 1]}")

    # # # --------- Cross-Validation methods --------------- 
    # cross_validation(df_induction, y_var=target[2*i],in_dir="results/feature_selection/induction/wrapper", 
    #                 out_dir="results/cross_validation/induction", top_n = top_n)
    # cross_validation(df_induction, y_var=target[2*i + 1], in_dir="results/feature_selection/induction/wrapper", 
    #                 out_dir="results/cross_validation/induction", top_n = top_n)