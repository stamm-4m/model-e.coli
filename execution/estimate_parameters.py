
from glob import glob
import time
from scipy.optimize import least_squares
from fedbatch.utils.io import load_yaml, save_yaml
from fedbatch.estimation.datasets import ExperimentDataset
from fedbatch.core.kinetics import Kinetic_Models
from fedbatch.utils.experiment_factory import build_experiments
from fedbatch.estimation.objective import MultiExperimentObjective
from fedbatch.estimation.postprocessing import compute_confidence_intervals
from fedbatch.utils.visualization_correlation_io import plot_parameter_correlation
from fedbatch.utils.visualization_fitting_io import plot_time_profiles_mc, plot_parity_mc #, plot_time_profiles, plot_parity, 
from fedbatch.utils.execute_model_io import run_model_with_parameters
from fedbatch.utils.visualization_residuals_io import qq_plot_residuals
from fedbatch.utils.experiment_factory import build_experiments

start = time.time()

cfg = load_yaml("fedbatch/config/default_parameters.yaml")

#--------------Load Datasets--------------------

dataset_files = sorted( glob("data/processed/BR*.xls") )
datasets = [ExperimentDataset(f) for f in dataset_files]

# Full parameter set from YAML (fixed defaults)
full_params = {
    name: cfg["kinetics"][name]["value"]
    for name in cfg["kinetics"]
}

#--------------Define parameters (theta)----------------- (quit any as u wish)
param_names = [
    "mu_max_p", 
    "mu_max_0", 
    "Ks", 
    "b", 
    # "m", 
    "Y_XS", 
    "alpha", 
    "gamma_1", 
    "Ap_1", 
    "gamma_2", 
    "Ap_2", 
    "sigma"
    ]

#------------------Initial values for parameters estimation----------------
theta0 = []
lower_bounds = []
upper_bounds = []

for name in param_names:
    p0 = cfg["kinetics"][name]["value"]

    if name == "Y_XS":
        lb, ub = 0.01, 0.99
    else:
        lb, ub = 0.001 * p0, 3 * p0

    theta0.append(p0) 
    lower_bounds.append(lb) 
    upper_bounds.append(ub)

#--------------kinetics and experiment--------------------

kin = Kinetic_Models()
datasets, simulators, y0s = build_experiments(cfg, kin)

#--------------Create the objetive--------------------

objective = MultiExperimentObjective(
    datasets=datasets,
    simulators=simulators,
    model=kin,
    y0s=y0s,
    param_names=param_names, 
    full_params=full_params
)

#--------------Run parameter estimation--------------------

result = least_squares(
    objective,
    x0=theta0,
    bounds=(lower_bounds, upper_bounds)
)

#--------------Postprocessing  (CIs, etc.)------------------ 

cov, std, ci = compute_confidence_intervals(result)

# print("Strong parameter correlations:")
# for i in range(len(param_names)):
#     for j in range(i + 1, len(param_names)):
#         if abs(corr[i, j]) > 0.8:
#             print(
#                 f"{param_names[i]} ↔ {param_names[j]} : "
#                 f"{corr[i, j]:.2f}"
#             )

#---------------------Simulated with parameters fitted------------------------
results_dict = {
    "model": {
        "type": "FedBatch kinetic model",
        "parameters": param_names,
        "n_parameters": len(param_names),
    },
    "estimation": {
        "success": bool(result.success),
        "cost": float(result.cost),
        "n_function_evals": int(result.nfev),
    },
    "parameters": {
        name: {
            "estimate": float(value),
            "ci_95": [float(low), float(high)]
        }
        for name, value, (low, high)
        in zip(param_names, result.x, ci)
    }
}

# _, _, y0s = build_experiments(cfg, kin)

per_dataset_metrics, global_metrics, global_ic, all_residuals, solutions = run_model_with_parameters(
    datasets=datasets,
    simulators=simulators,
    y0s=y0s,
    kin=kin,
    theta=result.x,
    param_names=param_names,
    full_params=full_params

)

#------------------Save yaml----------------------------

results_dict["metrics"] = {
    "per_dataset": per_dataset_metrics,
    "global": {
        "regression": global_metrics,
        "information_criteria": global_ic,
    }
}

output_path = "results/estimation/kinetic_fit_results.yaml"
save_yaml(results_dict, output_path)
print(f"Results saved to {output_path}")

end_1 = time.time()

print(f"Tiempo de ejecución: {end_1 - start:.4f} segundos")

#-------------Plots---------------------------

corr = plot_parameter_correlation(
    cov,
    param_names,
    threshold=0.8,
    savepath="results/plots/processed/parameter_correlation.png"
)

# QQ plot
qq_plot_residuals(
        all_residuals,
        title="Q-Q plot — global residuals",
        savepath=f"results/plots/processed/qq_residuals_global.png"
    )

for dataset, simulator in zip(datasets, simulators):
    name = dataset.path.split("/")[-1].replace(".xls", "")

    # plot_time_profiles(
    plot_time_profiles_mc(
        dataset,
        simulator,
        kin,
        result.x,
        cov,
        param_names,
        full_params,  
        n_samples=300,
        savepath=f"results/plots/{name}_mc_time.png",  # savepath=f"results/plots/{name}_time.png"
        cfg_tdense=True
    )

    # plot_parity(
    plot_parity_mc(
        dataset,
        simulator,
        kin,
        result.x,
        cov,
        param_names,
        full_params, 
        n_samples=300,
        savepath=f"results/plots/{name}_mc_parity.png"
        # savepath=f"results/plots/{name}_parity.png"
    )

end_2 = time.time()

print(f"Tiempo de ejecución total: {end_2 - start:.4f} segundos")

# ---------------- Save updated parameters to YAML (kinetics only) ----------------

# Reload original configuration (safe practice)
updated_cfg = load_yaml("fedbatch/config/default_parameters.yaml")

# Update only the kinetics parameter values
for name, value in zip(param_names, result.x):
    if name not in updated_cfg["kinetics"]:
        raise KeyError(f"Kinetic parameter '{name}' not found in YAML config")

    updated_cfg["kinetics"][name]["value"] = float(value)

updated_cfg.setdefault("estimation_metadata", {})
updated_cfg["estimation_metadata"]["date_utc"] = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
updated_cfg["estimation_metadata"]["source"] = "least_squares fit"
# updated_cfg["estimation_metadata"]["n_parameters"] = len(param_names)
updated_cfg["estimation_metadata"]["fitted_parameters"] = list(param_names)
updated_cfg["estimation_metadata"]["fixed_parameters"] = [
    k for k in updated_cfg["kinetics"] if k not in param_names
]


# Save updated configuration
updated_cfg_path = "results/estimation/updated_parameters.yaml"
save_yaml(updated_cfg, updated_cfg_path)

print(f"Updated parameters saved to {updated_cfg_path}")
