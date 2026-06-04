
from pathlib import Path
import logging
from src.core.reactor.kinetics import Kinetic_Models
from src.modelling.experiment_factory import build_experiments, run_model_with_parameters
from src.modelling.modelling_plots import plot_multi_dataset_model 

def run_model(model_name, model_path, output_dir, cfg, theta, param_names, full_params, ensemble_mode):

    local_predictions = {}
    local_global_results = []
    local_dataset_results = []
    local_data_plots = {}

    logging.info(f"{model_name} started")

    if model_name == "parametric":
        kin = Kinetic_Models(hybrid=False)
        model_output_dir = Path(output_dir) / "parametric"
    else:
        kin = Kinetic_Models(hybrid=True, models_folder=model_path, ensemble_mode=ensemble_mode)
        parts = Path(model_path).parts
        subpath = parts[parts.index("cross_validation") + 1]
        # idx = parts.index("cross_validation")
        # subpath = Path(*parts[idx+1:-2])  
        model_output_dir = Path(output_dir) / subpath

    model_output_dir.mkdir(parents=True, exist_ok=True)
    model_label = f"{model_name}_{kin.ensemble_mode}"

    print("Building experiments...")
    datasets, simulators, y0s = build_experiments(cfg, kin, BR09=False)

    print("Running simulation...")
    ( per_dataset_metrics, global_metrics, _, solutions 
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

        if dataset_key not in local_predictions:
            local_predictions[dataset_key] = {}
            
        if dataset_key not in solutions or "sol" not in solutions[dataset_key]:
                continue

        sol = solutions[dataset_key]["sol"]
        # Save
        local_predictions[dataset_key][model_label] = {
            "t": sol.t.copy(),
            "X": sol.y[0].copy(),
            "S": sol.y[1].copy(),
            "P": sol.y[2].copy(),
            "V": sol.y[3].copy(),
            }
        # print(f"      Plotting individual model : {model_name}")
        # plot_single_model(dataset, solutions[dataset_key], model_name, model_output_dir)

    print(f"Plotting combined figure for model: {model_name}")
    plot_multi_dataset_model(datasets, solutions, model_name, model_output_dir)

    # Global metrics
    global_row = {"model": model_label}
    global_row.update(global_metrics) # type: ignore
    local_global_results.append(global_row)

    # Metrics per dataset
    fold_results = []
    for dataset_path, data in per_dataset_metrics.items():
        dataset_name = Path(dataset_path).stem
        # dataset_name = Path(dataset_path).name
        # dataset_name = dataset_name.replace(".xls","")

        if "P" in data["regression"]:
            metrics = data["regression"]["P"]
            row = { "model": model_label,
                    "dataset": dataset_name}
            row.update(metrics)
            local_dataset_results.append(row)

            fold_results.append({
                    "test_groups": dataset_name,
                    "metrics": metrics
                })
        
    local_data_plots[model_name] = {
        "folds": fold_results,
        "summary": None # aggregate_cv_results(fold_results)
    }
    
    logging.info(f"{model_name} completed")

    return {
        "datasets": datasets,
        "predictions": local_predictions,
        "global": local_global_results,
        "dataset": local_dataset_results,
        "plots": local_data_plots
    }