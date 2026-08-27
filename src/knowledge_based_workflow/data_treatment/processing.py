
"""
Nanobody-based Antivenom Production with E. coli Reactor Simulation 

Author: Juan Camilo Castaño Sanchez
Email: jcastano-san@insa-toulose.fr
Date: 01/09/2026

"""

import numpy as np
import pandas as pd
from src.utils.io import load_yaml, get_time_ranges, timer, get_br_id
from src.knowledge_based_workflow.model.core.kinetics import Kinetic_Models
from src.utils.experiment_factory import build_experiments, run_model_with_parameters
from src.knowledge_based_workflow.model.auxiliar.feed_factory import create_feed

# ------------------- Computes qP and mu and unifies dataframes --------------
# @timer
def processing_data(datasets, yaml_path, t_ind_exp = True):
    
    yaml_params = load_yaml(yaml_path)
    df_global = []
    df_induction_all = []

    df_calc = calculate_features(BR09=True)

    for br_id in datasets:

        df = pd.DataFrame(datasets[br_id])

        # if br_id in ["BR07", "BR08"]:
        #     df = df.iloc[:-1]

        # Indicates the dataset name
        df["Run_ID"] = br_id
        df.insert(0, "Run_ID", df.pop("Run_ID"))

        # Indicatates dataset numer and T of induction
        df = add_T_ind(df)
        df.insert(1, "Run_T", df.pop("Run_T"))

        # qP and mu calculation
        _, time_ind = get_time_ranges(yaml_params, br_id)

         # -- Add calculated features --
        calc_features = df_calc[br_id]

        # df_semibatch = df[(df["time"] >= time_sb) & (df["time"] < time_ind)].copy()
        df_induction = df[df["time"] >= time_ind].copy()

        # mu and qp calculation
        if t_ind_exp == True:
            df = calc_mu_qp_rp(df, calc_features, time_ind)
            df_induction = calc_mu_qp_rp(df_induction, calc_features, time_ind)
        else:
            df = calc_mu_qp_rp(df, calc_features, t_ind=None)
            df_induction = calc_mu_qp_rp(df_induction, calc_features, t_ind=None)

        # Final df
        df_global.append(df)
        df_induction_all.append(df_induction)

    # final unification
    df_global_final = pd.concat(df_global, ignore_index=True)
    df_induction_final = pd.concat(df_induction_all, ignore_index=True)

    return df_global_final, df_induction_final 

# -------------------------- mu, qp & rp function ---------------------------------------

def calc_mu_qp_rp(df, calc_features, t_ind=None):

    df = df.sort_values("time").copy()

    n = len(df)

    mu = np.zeros(n)
    qp = np.zeros(n)
    qp_2 = np.zeros(n)
    rp = np.zeros(n)

    t = df["time"].values
    X = df["X"].values
    P = df["P"].values
    V = df["V"].values

    dXdt = df["dXdt"].values
    dVdt = df["dVdt"].values
    dPdt = df["dPdt"].values

    # low_qp = 0 # 1e-6
    # low_rp = 0 # 1e-5

    mu_calc = calc_features["mu"]
    
    if t_ind != None:
        for i in range(n):
            if t[i] < t_ind:
                qp[i] = qp_2[i] = 0
                rp[i] = 0

            else:
                rp[i] = dPdt[i] + (dVdt[i] * P[i] / V[i])
                qp[i] = rp[i] / X[i]
                qp_2[i] = (rp[i] - mu_calc[i] * P)/ X[i]

    else:
        rp    =  dPdt + (dVdt * P / V)  
        qp    = rp / X

    rx    =  dXdt + (dVdt * X / V)      
    mu = (1/X) * ( dXdt ) + (1/V) * ( dVdt )

    # mu = np.clip(mu, 0, None)
    # qp = np.clip(qp, low_qp, None)
    # rp = np.clip(rp, low_rp, None)

    df["mu"] = mu
    df["qP"] = qp
    df["qP_2"] = qp_2
    df["rP"] = rp
    df["rX"] = rx

    return df

# --------------- Add identification column named Run_T function ---------------

def add_T_ind(df,n_ultimos=4):

    last_T = (
        df
        .sort_values("time")
        .groupby("Run_ID")["T"] 
        # .last()
        .apply(lambda s: s.tail(n_ultimos).mean()) # last 4 values
        .round(1)
        .astype(int)  
        .astype(str) 
    )

    # rows asignation
    df["T_ind"] = df["Run_ID"].map(last_T)
    df["Run_T"] = df["Run_ID"].astype(str) + " T = " + df["T_ind"] + "°C " 

    return df 

# --- Calculate features ---
def calculate_features(BR09, in_dir = "src/config/params.yaml"):

    # Same code as mode_profile.py
    cfg = load_yaml(in_dir)

    kin = Kinetic_Models()
    datasets, simulators, y0s = build_experiments(cfg, kin, BR09)
    param_names = list(cfg["kinetics"].keys())
    full_params = { k: cfg["kinetics"][k]["value"] for k in param_names }
    theta = [ cfg["kinetics"][k]["value"] for k in param_names ]
    
    _, _, _, solutions = run_model_with_parameters (
        datasets=datasets, simulators=simulators, y0s=y0s, kin=kin, theta=theta, param_names=param_names, full_params=full_params)
    
    results = {}

    for dataset in datasets:
        br_id = get_br_id(dataset)
        sol_block = solutions[dataset.path]

        results[br_id] = { "mu": sol_block["mu"], }

    return results