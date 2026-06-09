
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
from src.utils.io import load_yaml

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

        br_id = get_br_id(dataset) 
        # cfg = load_yaml("src/config/default_parameters.yaml")
        # t_ind = cfg["bioreactor"][br_id]["t_ind"]["value"]
        # if br_id in ("BR07", "BR08"):
        #     t_span = (dataset.t[0], dataset.t[-2])
        #     t_eval = dataset.t[:-1]
        #     t_eval_dense = np.linspace(dataset.t[0], dataset.t[-2], 200)
        # else:
        t_max = dataset.t[-1]
        t_span = (dataset.t[0], dataset.t[-1])
        t_eval = dataset.t
        t_eval_dense = np.linspace(dataset.t[0], dataset.t[-1], 200) 

        sol = simulator.run( 
            y0 = y0,
            t_span = t_span ,
            t_eval = t_eval ) 

        if dense:
            sol_dense = simulator.run( 
                y0 = y0,
                t_span = t_span ,
                t_eval = t_eval_dense )
        else:
            sol_dense = None

        print(f"\nDataset: {dataset.path}")
        # print("sol.success:", sol.success)
        print("message:", sol.message)
        # print("len(t_eval):", len(t_eval))
        # print("len(sol.t):", len(sol.t))

    # ----- Compute mu and dVdt from dynamic model --------
        mu_values = []
        dXdt_values = []
        dVdt_values = []
        P_ML_values = []
        P_ML_values_dense = []

        if dense == True:
            X_model, S_model, P_model, V_model = sol_dense.y
            for i, t in enumerate(sol_dense.t):
                X = X_model[i]
                S = S_model[i]
                T = simulator.model.balances.temperature.F(t)

                ind_F, t_ind = simulator.model.balances.induction_P.F(t)
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
                
                if kin.PMLmodel == True and kin.hybrid == False:

                    features =  {  # "X": X_real, # "S": S, # "V": V_real,
                        "t": t, 
                        "t_ind": t - t_ind,
                        "t_ind_ad": (t - t_ind)/t_max,
                        "T": T,
                        "I": ind_F, # "mu": mu_real, 
                        "FS_calc": FS,          # "dXdt": dX_real,# "dSdt": dSdt, # "dVdt": dV_real,# "Xlag1": self.prev_X_real, 
                        "Xlag1_calc": X_model[i] if i == 0 else X_model[i-1],  
                        "Plag1": 0 if i == 0 else P_ML_values_dense[i-1],
                        "X_calc": X,
                        "V_calc": V_model[i], 
                        "mu_calc": mu, 
                        "dXdt_calc": dXdt, 
                        "dVdt_calc": dVdt}

                    features = {k: np.float64(v) for k, v in features.items()}

                    value = kin.PML_model(features, br_id)
                    P_ML_values_dense.append( np.clip(value, 0, None) )
                    # features["P"] = P_ML

        X_model, S_model, P_model, V_model = sol.y
        t_vals = sol.t
        for i, t in enumerate(t_vals):
            X = X_model[i]
            S = S_model[i]
            T = simulator.model.balances.temperature.F(t)

            ind_F, t_ind = simulator.model.balances.induction_P.F(t)
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
            
            if kin.PMLmodel == True and kin.hybrid == False:

                features =  {  # "X": X_real, # "S": S, # "V": V_real,
                    "t": t, 
                    "t_ind": t - t_ind,
                    "t_ind_ad": (t - t_ind)/t_max,
                    "T": T,
                    "I": ind_F, # "mu": mu_real, 
                    "FS_calc": FS,          # "dXdt": dX_real,# "dSdt": dSdt, # "dVdt": dV_real,# "Xlag1": self.prev_X_real, 
                    "Xlag1_calc": X_model[i] if i == 0 else X_model[i-1],  
                    "Plag1": 0 if i == 0 else P_ML_values[i-1],
                    "X_calc": X,
                    "V_calc": V_model[i], 
                    "mu_calc": mu, 
                    "dXdt_calc": dXdt, 
                    "dVdt_calc": dVdt}

                features = {k: np.float64(v) for k, v in features.items()}

                value = kin.PML_model(features, br_id)
                P_ML_values.append( np.clip(value, 0, None) )
                # features["P"] = P_ML


        mu_values = np.array(mu_values)
        dVdt_values = np.array(dVdt_values)
        dXdt_values = np.array(dXdt_values)
        P_ML_values = np.array(P_ML_values)
        P_ML_values_dense = np.array(P_ML_values_dense)

        # Collect values for global regression metrics
        all_y_exp.append(dataset.data["X"])
        all_y_exp.append(dataset.data["S"])
        all_y_exp.append(dataset.data["P"])
        all_y_exp.append(dataset.data["V"])

        if kin.PMLmodel == True and kin.hybrid == False:
            P_pred = P_ML_values
        else:
            P_pred = P_model

        all_y_model.append(X_model)
        all_y_model.append(S_model)
        all_y_model.append(P_pred)
        all_y_model.append(V_model)

        # Regression metrics per variable
        if len(dataset.data["X"]) != len(X_model):
            print(f"Mismatch en {dataset.path}")
            A = 2

        metrics = {
            "X": compute_metrics(dataset.data["X"], X_model, k=len(theta)),
            "S": compute_metrics(dataset.data["S"], S_model, k=len(theta)),
            "P": compute_metrics(dataset.data["P"], P_pred, k=len(theta)),
            "V": compute_metrics(dataset.data["V"], V_model, k=len(theta))
        }

        # Residual vector for IC
        residuals = np.concatenate([
            X_model - dataset.data["X"],
            S_model - dataset.data["S"],
            P_pred - dataset.data["P"],
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
            "P_ML": P_ML_values,
            "P_ML_dense": P_ML_values_dense,
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