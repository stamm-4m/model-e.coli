
import pandas as pd
from pathlib import Path
import logging
import os
from src.utils.io import load_yaml
from src.modelling.run import run_model
from src.modelling.modelling_plots import plot_comparison, plot_multibr_states_parametric # plot_single_model 
from src.data_analysis.cross_validation.plots_cross_validation import plot_all_metrics
from joblib import Parallel, delayed

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s")

# Global parameters # use_log
output_dir_r = Path("results/modelling")
os.makedirs(output_dir_r, exist_ok=True)
cfg = load_yaml("src/config/default_parameters.yaml")
param_names = list(cfg["kinetics"].keys())
full_params = { k: cfg["kinetics"][k]["value"] for k in param_names }
theta = [ cfg["kinetics"][k]["value"] for k in param_names ]

model_configs = {
    # "parametric": None,
    # "global_qP": "results/cross_validation/global/qP_calc/best_model_per_fold",
    # "global_rP": "results/cross_validation/global/rP_calc/best_model_per_fold",
    # "global_ind_qP": "results/cross_validation/global_ind/qP_calc/best_model_per_fold",
    # "global_ind_rP": "results/cross_validation/global_ind/rP_calc/best_model_per_fold",
    # "induction_qP": "results/cross_validation/induction/qP_calc/best_model_per_fold",
    # "induction_rP": "results/cross_validation/induction/rP_calc/best_model_per_fold",

    "global_P": "results/cross_validation/global/P/best_model_per_fold",
}

MODEL_COLORS = {
        "parametric": "black",
        "global_qP": "tab:blue",
        "global_rP": "tab:orange",
        "global_ind_qP": "tab:green",
        "global_ind_rP": "tab:red",
        "induction_qP": "tab:purple",
        "induction_rP": "tab:brown",
        "global_P": "tab:red",
    }

metrics_name = ["R2", "MAE", "MSE", "RMSE", "MAPE", "SCORE", "AIC", "BIC"]
plots = ("boxplot", "heatmap") # "by_run" "ranking"

# --------------------- per fold ----------------------
ensemble_mode = "fold"
output_dir = output_dir_r / ensemble_mode
dense = True

results = Parallel(n_jobs=1, backend="loky")( # -1
    delayed(run_model)(name, path, output_dir, cfg, theta, param_names, full_params, ensemble_mode, dense)
    for name, path in model_configs.items() )

all_global_results = []
all_dataset_results = []
all_predictions = {}
data_plots = {}
datasets_ref = None

for res in results:
    if datasets_ref is None:
        datasets_ref = res["datasets"]
    
    for ds, preds in res["predictions"].items():        
        if ds not in all_predictions:
            all_predictions[ds] = {}

        all_predictions[ds].update(preds)
    
    all_global_results.extend(res["global"])
    all_dataset_results.extend(res["dataset"])
    data_plots.update(res["plots"])


print("Generating comparison plots...")
output_dir = Path(output_dir)
output = output_dir / "comparison" # type: ignore
output = Path(output) 
output.mkdir(parents=True, exist_ok=True)

for dataset in datasets_ref:
    dataset_key = dataset.path
    plot_comparison(dataset,all_predictions[dataset_key],output,MODEL_COLORS)

plot_multibr_states_parametric(datasets_ref, all_predictions, output_dir)
plot_all_metrics(data_plots, metrics_name, output_dir / "metrics", plots)

df_global = pd.DataFrame(all_global_results)
df_dataset = pd.DataFrame(all_dataset_results)

print("Saving Excel results...")
excel_path = output_dir / "metrics_summary_all_models.xlsx"
with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
    df_global.to_excel(writer, sheet_name="global_metrics", index=False)
    df_dataset.to_excel(writer, sheet_name="per_dataset_metrics", index=False)
print(f"Results saved in: {excel_path}")

# --------------------- global ----------------------
ensemble_mode = "global"
output_dir = output_dir_r / ensemble_mode

results = Parallel(n_jobs=1, backend="loky")( # -1
    delayed(run_model)(name, path, output_dir, cfg, theta, param_names, full_params, ensemble_mode, dense)
    for name, path in model_configs.items() )

all_global_results = []
all_dataset_results = []
all_predictions = {}
data_plots = {}
datasets_ref = None

for res in results:
    if datasets_ref is None:
        datasets_ref = res["datasets"]
    
    for ds, preds in res["predictions"].items():        
        if ds not in all_predictions:
            all_predictions[ds] = {}

        all_predictions[ds].update(preds)
    
    all_global_results.extend(res["global"])
    all_dataset_results.extend(res["dataset"])
    data_plots.update(res["plots"])

print("Generating comparison plots for global models...")
output_dir = Path(output_dir)
output = output_dir / "comparison" # type: ignore
output = Path(output) 
output.mkdir(parents=True, exist_ok=True)

for dataset in datasets_ref:
    dataset_key = dataset.path
    plot_comparison(dataset,all_predictions[dataset_key],output,MODEL_COLORS)

plot_multibr_states_parametric(datasets_ref, all_predictions, output_dir)
plot_all_metrics(data_plots, metrics_name, output_dir / "metrics", plots)

df_global = pd.DataFrame(all_global_results)
df_dataset = pd.DataFrame(all_dataset_results)

print("Saving Excel results...")
excel_path = output_dir / "metrics_summary_all_models.xlsx"
with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
    df_global.to_excel(writer, sheet_name="global_metrics", index=False)
    df_dataset.to_excel(writer, sheet_name="per_dataset_metrics", index=False)
print(f"Results saved in: {excel_path}")