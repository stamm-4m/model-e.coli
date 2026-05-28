
import numpy as np
from glob import glob

from src.data_analysis.data_treatment.data import ExperimentDataset
from src.core.auxiliar.temperature_profile import TemperatureProfile
from src.core.auxiliar.volume_profile import VolumeProfile
from src.core.auxiliar.biomass_profile import BiomassProfile
from src.core.auxiliar.induction_func import InductionProfile
from src.core.auxiliar.feed_factory import create_feed
from src.core.auxiliar.simulator import Simulator
from src.core.auxiliar.initial_conditions import build_initial_state
from src.core.reactor.balances import FedBatchBalances
from src.core.reactor.fedbatch_model import FedBatchModel
from src.utils.io import get_br_id, timer
from src.utils.metrics_io import compute_metrics

@timer
def build_experiments(cfg, kin, BR09=False):
    """
    Build datasets, simulators and initial conditions for all BR experiments.
    """
    if BR09:
        dataset_files = [f for f in sorted(glob("data/raw/BR*.xls"))]
    else:
        dataset_files = [f for f in sorted(glob("data/raw/BR*.xls")) if "BR09" not in f]

    datasets = [ExperimentDataset(f) for f in dataset_files]

    simulators = []
    y0s = []

    for dataset in datasets:
        br_id = get_br_id(dataset)  

        T_profile = TemperatureProfile(dataset.t, dataset.T)

        V_profile = VolumeProfile(dataset.t, dataset.V)

        X_profile = BiomassProfile(dataset.t, dataset.X, V_profile)

        t_ind = cfg["bioreactor"][br_id]["t_ind"]["value"]
        I_profile = InductionProfile(t_ind, br_id)

        feed_cfg = cfg["feeds"][br_id]
        feed_S = create_feed(feed_cfg["feed_S"])
        feed_A = create_feed(feed_cfg["feed_A"])

        balances = FedBatchBalances(
            kinetics=kin,
            Sf=cfg["bioreactor"][br_id]["Sf"]["value"],
            temperature_profile=T_profile,
            volume_profile=V_profile,
            biomass_profile=X_profile,
            induction_profile=I_profile,
            br_id = br_id
        )

        model = FedBatchModel(
            balances=balances,
            feed_S=feed_S,
            feed_A=feed_A
        )

        method = cfg["simulation"]["method_ode"]["type"]
        rtol = cfg["simulation"]["rtol_ode"]["value"]
        atol = cfg["simulation"]["atol_ode"]["value"]
        max_step = cfg["simulation"]["max_step"]["value"]

        sim = Simulator(model, method, rtol, atol, max_step)
        simulators.append(sim)

        y0 = build_initial_state(cfg, br_id, dataset)
        y0s.append(y0)

    return datasets, simulators, y0s

@timer
def run_model_with_parameters( datasets, simulators, y0s, kin, theta, param_names, full_params, dense = False):
    # Update model with optimal parameters

    # if kin.hybrid == False:
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
            y0 = y0,
            t_span = (dataset.t[0], dataset.t[-1]),
            t_eval = dataset.t ) 
        
        X_model, S_model, P_model, V_model = sol.y
        t_vals = sol.t

        if dense:
            sol_dense = simulator.run( 
                y0 = y0,
                t_span = (dataset.t[0], dataset.t[-1]),
                t_eval = np.linspace(dataset.t[0], dataset.t[-1], 200), 
                dense_ouput = True )
        else:
            sol_dense = None
        
    # ----- Compute mu and dVdt from dynamic model --------
        mu_values = []
        dXdt_values = []
        dVdt_values = []

        for i, t in enumerate(t_vals):
            X = X_model[i]
            S = S_model[i]
            T = simulator.model.balances.temperature.F(t)

            ind_F = simulator.model.balances.induction_P.F(t)
            FS = simulator.model.feed_S.F(t)[0]
            FA = simulator.model.feed_A.F(t)[0]

            mu = kin.mu(X, S, T, ind_F)
            mu_values.append(mu)

            state = np.array([X_model[i], S_model[i], P_model[i], V_model[i]])           
            derivatives = simulator.model.balances.dfdt(t, state, FS, FA, ind_F)

            dXdt = derivatives[0]  
            dXdt_values.append(dXdt)

            dVdt = derivatives[3]  # 4th equation
            dVdt_values.append(dVdt)

        mu_values = np.array(mu_values)
        dVdt_values = np.array(dVdt_values)
        dXdt_values = np.array(dXdt_values)

        # Collect values for global regression metrics
        all_y_exp.append(dataset.data["X"])
        all_y_exp.append(dataset.data["S"])
        all_y_exp.append(dataset.data["P"])
        all_y_exp.append(dataset.data["V"])

        all_y_model.append(X_model)
        all_y_model.append(S_model)
        all_y_model.append(P_model)
        all_y_model.append(V_model)

        # Regression metrics per variable
        metrics = {
            "X": compute_metrics(dataset.data["X"], X_model, k=len(theta)),
            "S": compute_metrics(dataset.data["S"], S_model, k=len(theta)),
            "P": compute_metrics(dataset.data["P"], P_model, k=len(theta)),
            "V": compute_metrics(dataset.data["V"], V_model, k=len(theta))
        }

        # Residual vector for IC
        residuals = np.concatenate([
            X_model - dataset.data["X"],
            S_model - dataset.data["S"],
            P_model - dataset.data["P"],
            V_model - dataset.data["V"]
        ])

        per_dataset_metrics[dataset.path] = {
            "regression": metrics
        }

        all_residuals.append(residuals)
        # solutions[dataset.path] = sol
        solutions[dataset.path] = {
            "sol": sol,
            "dense":sol_dense,
            "mu": mu_values,
            "dXdt": dXdt_values,
            "dVdt": dVdt_values,
        }


    # -------- Global metrics --------
    all_y_exp = np.concatenate(all_y_exp)
    all_y_model = np.concatenate(all_y_model)

    global_metrics = compute_metrics(
        y_true=all_y_exp,
        y_pred=all_y_model,
        k=len(theta)
    )

    return per_dataset_metrics, global_metrics, all_residuals, solutions