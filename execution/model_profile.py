
from fedbatch.utils.io import load_yaml
from fedbatch.core.kinetics import Kinetic_Models
from fedbatch.utils.experiment_factory import build_experiments
from fedbatch.utils.execute_model_io import run_model_with_parameters
from fedbatch.utils.visualization_fitting_io import plot_time_profiles_multi

from fedbatch.utils.excel_io import save_fitting_outputs_to_excel

# cfg = load_yaml("fedbatch/config/default_parameters.yaml")
cfg = load_yaml("fedbatch/config/updated_parameters.yaml")

kin = Kinetic_Models()

datasets, simulators, y0s = build_experiments(cfg, kin)

param_names = list(cfg["kinetics"].keys())

full_params = {
    k: cfg["kinetics"][k]["value"]
    for k in param_names
}

theta = [cfg["kinetics"][k]["value"] for k in param_names]
# theta = [cfg["kinetics"][k]["value"] for k in cfg["kinetics"]]

(
    per_dataset_metrics,
    global_metrics,
    global_ic,
    residuals,
    solutions
) = run_model_with_parameters(
    datasets=datasets,
    simulators=simulators,
    y0s=y0s,
    kin=kin,
    theta=theta,
    param_names=param_names,
    full_params=full_params

)

save_fitting_outputs_to_excel(
    output_path="results/estimation/model_results.xlsx",
    datasets=datasets,
    per_dataset_metrics=per_dataset_metrics,
    global_metrics=global_metrics,
    global_ic=global_ic,
    residuals=residuals,
    solutions=solutions
)

plot_time_profiles_multi(
    datasets=datasets,
    simulators=simulators,
    kin=kin,
    theta=theta,
    param_names=param_names,
    full_params=full_params,
    plot_ci=False,
    save_dir="results/plots/time_profiles"
)


