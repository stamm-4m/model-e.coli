
import numpy as np
from fedbatch.utils.metrics_io import (
    regression_metrics,
    information_criteria,
    information_criteria_from_residuals
)
from scipy.interpolate import interp1d

def run_model_with_parameters(
    datasets,
    simulators,
    y0s,
    kin,
    theta,
    param_names,
    full_params

):
    """
    Run simulations using fitted parameters and compute metrics.

    Returns:
        per_dataset_metrics: dict
        global_metrics: dict
        global_information_criteria: dict
    """

    # Update model with optimal parameters
    
    params = full_params.copy()

    for name, value in zip(param_names, theta):
        params[name] = value

    kin.set_params(params)

    per_dataset_metrics = {}
    all_residuals = []
    solutions = {}

    all_y_exp = []
    all_y_model = []

    for dataset, simulator, y0 in zip(datasets, simulators, y0s):
        sol = simulator.run(
            y0=y0,
            t_span=(dataset.t[0], dataset.t[-1]),
            t_eval=dataset.t
        )
        
        # t_model = sol.t

        # X_fun = interp1d(t_model, sol.y[0, :], kind="linear", fill_value="extrapolate")
        # S_fun = interp1d(t_model, sol.y[1, :], kind="linear", fill_value="extrapolate")
        # P_fun = interp1d(t_model, sol.y[2, :], kind="linear", fill_value="extrapolate")

        # X_model = X_fun(dataset.t)
        # S_model = S_fun(dataset.t)
        # P_model = P_fun(dataset.t)

        X_model, S_model, P_model, _ = sol.y

        # Collect values for global regression metrics
        all_y_exp.append(dataset.data["X"])
        all_y_exp.append(dataset.data["S"])
        all_y_exp.append(dataset.data["P"])

        all_y_model.append(X_model)
        all_y_model.append(S_model)
        all_y_model.append(P_model)

        # Regression metrics per variable
        metrics = {
            "X": regression_metrics(dataset.data["X"], X_model),
            "S": regression_metrics(dataset.data["S"], S_model),
            "P": regression_metrics(dataset.data["P"], P_model),
        }

        # # Regression metrics per variable (consider weights)
        # weights_X = 1 / np.max(dataset.data["X"])**2
        # weights_S = 1 / np.max(dataset.data["S"])**2
        # weights_P = 1 / np.max(dataset.data["P"])**2

        # metrics = {
        #     "X": regression_metrics(dataset.data["X"], X_model, weights_X),
        #     "S": regression_metrics(dataset.data["S"], S_model, weights_S),
        #     "P": regression_metrics(dataset.data["P"], P_model, weights_P),
        # }

        # Residual vector for IC
        residuals = np.concatenate([
            X_model - dataset.data["X"],
            S_model - dataset.data["S"],
            P_model - dataset.data["P"],
        ])

        ic = information_criteria_from_residuals(
            residuals,
            k=len(theta)
        )

        # ic = information_criteria(
        #     y_true=np.zeros_like(residuals),
        #     y_pred=residuals,
        #     k=len(theta)
        # )

        per_dataset_metrics[dataset.path] = {
            "regression": metrics,
            "information_criteria": ic,
        }

        all_residuals.append(residuals)
        solutions[dataset.path] = sol

    # -------- Global metrics --------
    all_y_exp = np.concatenate(all_y_exp)
    all_y_model = np.concatenate(all_y_model)

    global_metrics = regression_metrics(
        y_true=all_y_exp,
        y_pred=all_y_model
    )

    all_residuals = np.concatenate(all_residuals)
    global_ic = information_criteria_from_residuals(
        all_residuals,
        k=len(theta)
        )

    # global_ic = information_criteria(
    #     y_true=np.zeros_like(all_residuals),
    #     y_pred=all_residuals,
    #     k=len(theta)
    # )

    return per_dataset_metrics, global_metrics, global_ic, all_residuals, solutions
