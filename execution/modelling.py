
import pandas as pd
from pathlib import Path
import os
from src.utils.io import load_yaml
from src.core.reactor.kinetics import Kinetic_Models
from src.modelling.experiment_factory import build_experiments, run_model_with_parameters
from src.modelling.modelling_plots import plot_comparison, plot_multibr_states_parametric, plot_multi_dataset_model #, plot_single_model 

# Gloabal parameters
output_dir = "results/modelling"
os.makedirs(output_dir, exist_ok=True)
cfg = load_yaml("src/config/default_parameters.yaml")
param_names = list(cfg["kinetics"].keys())
full_params = { k: cfg["kinetics"][k]["value"] for k in param_names }
theta = [ cfg["kinetics"][k]["value"] for k in param_names ]
all_global_results = []
all_dataset_results = []
all_predictions = {}

model_configs = {
    "parametric": None,
    # "global_qP": "results/cross_validation/global/qP/best_model_per_fold_dynamic",
    # "global_rP": "results/cross_validation/global/rP/best_model_per_fold_dynamic",
    "global_ind_qP": "results/cross_validation/global_ind/qP/best_model_per_fold_dynamic",
    "global_ind_rP": "results/cross_validation/global_ind/rP/best_model_per_fold_dynamic",
    "induction_qP": "results/cross_validation/induction/qP/best_model_per_fold_dynamic",
    "induction_rP": "results/cross_validation/induction/rP/best_model_per_fold_dynamic",
}

MODEL_COLORS = {
        "parametric": "black",
        # "global_qP": "tab:blue",
        # "global_rP": "tab:orange",
        "global_ind_qP": "tab:green",
        "global_ind_rP": "tab:red",
        "induction_qP": "tab:purple",
        "induction_rP": "tab:brown",
    }

for model_name, model_path in model_configs.items():
    print(f"\n===== Running model: {model_name} =====")
    if model_name == "parametric":
        kin = Kinetic_Models(hybrid=False)
        model_output_dir = Path(output_dir) / "parametric"
    else:
        kin = Kinetic_Models(hybrid=True, models_folder=model_path)
        parts = Path(model_path).parts
        idx = parts.index("cross_validation")
        subpath = Path(*parts[idx+1:-2])  
        model_output_dir = Path(output_dir) / subpath

    model_output_dir.mkdir(parents=True, exist_ok=True)

    print("Building experiments...")
    datasets, simulators, y0s = build_experiments(cfg, kin, BR09=False)

    # Run model
    print("Running simulation...")
    ( per_dataset_metrics, global_metrics, all_residuals, solutions 
     ) = run_model_with_parameters(                      # FedBatchBalances can operate with real values of V
            datasets=datasets, 
            simulators=simulators, 
            y0s=y0s, 
            kin=kin, 
            theta=theta, 
            param_names=param_names, 
            full_params=full_params, 
            dense = False )
    print("Simulation completed")
    
    for dataset in datasets:
        dataset_key = dataset.path

        if dataset_key not in all_predictions:
            all_predictions[dataset_key] = {}
        sol = solutions[dataset_key]["sol"]

        # Save
        all_predictions[dataset_key][model_name] = {
            "t": sol.t,
            "X": sol.y[0],
            "S": sol.y[1],
            "P": sol.y[2],
            "V": sol.y[3],
            }
        # print(f"      Plotting individual model : {model_name}")
        # plot_single_model(dataset, solutions[dataset_key], model_name, model_output_dir)

    print(f"Plotting combined figure for model: {model_name}")
    plot_multi_dataset_model(datasets, solutions, model_name, model_output_dir)

    # Global metrics
    global_row = {"model": model_name}
    global_row.update(global_metrics) # type: ignore
    all_global_results.append(global_row)

    # Metrics per dataset
    for dataset_path, data in per_dataset_metrics.items():

        dataset_name = dataset_path.split("/")[-1]  # BRXX.xls
        # dataset_name = os.path.basename(dataset.path).replace(".xls","")
        if "P" in data["regression"]:
            metrics = data["regression"]["P"]
            row = { "model": model_name,
                    "dataset": dataset_name,
                    "variable": "P" }
            row.update(metrics)
            all_dataset_results.append(row)

print("Generating comparison plots...")
output = model_output_dir / "comparison" # type: ignore
output_dir = Path(output) 
output_dir.mkdir(parents=True, exist_ok=True)

for dataset in datasets:
    dataset_key = dataset.path
    plot_comparison(dataset,all_predictions[dataset_key],output_dir,MODEL_COLORS)

plot_multibr_states_parametric(datasets, all_predictions, model_output_dir)

df_global = pd.DataFrame(all_global_results)
df_dataset = pd.DataFrame(all_dataset_results)

excel_path = os.path.join(model_output_dir, "metrics_summary_all_models.xlsx")

print("Saving Excel results...")
with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
    df_global.to_excel(writer, sheet_name="global_metrics", index=False)
    df_dataset.to_excel(writer, sheet_name="per_dataset_metrics", index=False)
print(f"Results saved in: {excel_path}")



