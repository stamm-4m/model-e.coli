
import numpy as np
import pandas as pd
# from src.data_analysis.data_treatment.data import ExperimentDataset
from src.utils.io import load_yaml, get_time_ranges, timer, get_br_id
from src.utils.io import load_yaml
from src.core.reactor.kinetics import Kinetic_Models
from src.core.auxiliar.feed_factory import create_feed
from src.modelling.experiment_factory import build_experiments, run_model_with_parameters

# ------------------- Computes qP and mu and unifies dataframes --------------
@timer
def processing_data(datasets, yaml_path, t_ind_exp = True):
    
    yaml_params = load_yaml(yaml_path)
    
    df_global = []
    df_induction_all = []

    df_calc = calculate_features(BR09=True)

    var_calc = ["X", "V", "mu", "dXdt", "dVdt","FS"]

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
        time_sb, time_ind = get_time_ranges(yaml_params, br_id)

        df["t_ind"] = df["t"] - time_ind

        df_semibatch = df[(df["time"] >= time_sb) & (df["time"] < time_ind)].copy()
        df_induction = df[df["time"] >= time_ind].copy()

        # -- Add previous values of X and P as features (lag 1) for Global --
        df["Xlag1"] = df["X"].shift(1)
        df["Plag1"] = df["P"].shift(1)
        df.loc[df.index[0], "Xlag1"] = df["X"].iloc[0]
        df.loc[df.index[0], "Plag1"] = df["P"].iloc[0]

        # -- Add previous values of X and P as features (lag 1) for induction phase --
        df_induction["Xlag1"] = df_induction["X"].shift(1)
        df_induction["Plag1"] = df_induction["P"].shift(1)
        df_induction.loc[df_induction.index[0], "Xlag1"] = df_semibatch["X"].iloc[-1]
        df_induction.loc[df_induction.index[0], "Plag1"] = df_semibatch["P"].iloc[-1]

        # -- Add calculated features --
        calc_features = df_calc[br_id]

        t_df = df["time"].values
        t_ind = df_induction["time"].values
        t_calc = calc_features["time"]

        idx_df = [np.argmin(np.abs(t_calc - t)) for t in t_df] # [np.where(t_calc == t)[0][0] for t in t_df]
        for var in var_calc:
            df[f"{var}_calc"] = calc_features[var][idx_df]
        
        df["Xlag1_calc"] = df["X_calc"].shift(1)
        df.loc[df.index[0], "Xlag1_calc"] = df["X_calc"].iloc[0]

        idx_ind = [np.argmin(np.abs(t_calc - t)) for t in t_ind] #[np.where(t_calc == t)[0][0] for t in t_ind]
        for var in var_calc:
            df_induction[f"{var}_calc"] = calc_features[var][idx_ind]
        
        df_induction["Xlag1_calc"] = df_induction["X_calc"].shift(1)
        last_idx_sb = df[df["time"] < time_ind].index[-1]
        df_induction.loc[df_induction.index[0], "Xlag1_calc"] = df["X_calc"].loc[last_idx_sb]

        # mu and qp calculation
        if t_ind_exp == True:
            df = calc_mu_qp_rp(df, time_ind)
            df_induction = calc_mu_qp_rp(df_induction, time_ind)
        else:
            df = calc_mu_qp_rp(df, t_ind=None)
            df_induction = calc_mu_qp_rp(df_induction, t_ind=None)

        # Final df
        df_global.append(df)
        df_induction_all.append(df_induction)

    # final unification
    df_global_final = pd.concat(df_global, ignore_index=True)
    df_induction_final = pd.concat(df_induction_all, ignore_index=True)

    return df_global_final, df_induction_final 

# -------------------------- mu, qp & rp function ---------------------------------------

def calc_mu_qp_rp(df, t_ind=None):

    df = df.sort_values("time").copy()

    n = len(df)

    mu = np.zeros(n)
    qp = np.zeros(n)
    rp = np.zeros(n)

    mu_calc = np.zeros(n)
    qp_calc = np.zeros(n)
    rp_calc = np.zeros(n)

    t = df["time"].values
    X = df["X"].values
    P = df["P"].values
    V = df["V"].values

    X_calc = df["X_calc"].values
    V_calc = df["V_calc"].values
    
    dXdt = df["dXdt"].values
    dVdt = df["dVdt"].values
    dPdt = df["dPdt"].values

    dXdt_calc = df["dXdt_calc"].values
    dVdt_calc = df["dVdt_calc"].values

    low_qp = 0 # 1e-6
    low_rp = 0 # 1e-5
    
    if t_ind != None:
        for i in range(n):
            if t[i] < t_ind:
                qp[i] = 0
                rp[i] = 0

                qp_calc[i] = 0
                rp_calc[i] = 0
            else:
                rp[i] = dPdt[i] + (dVdt[i] * P[i] / V[i])
                qp[i] = rp[i] / X[i]
                
                rp_calc[i] =  dPdt[i] + (dVdt_calc[i] * P[i] / V_calc[i])
                qp_calc[i] = rp_calc[i] / X_calc[i]

    else:
        rp    =  dPdt + (dVdt * P / V)  
        qp    = rp / X
        
        rp_calc    =  dPdt + (dVdt_calc * P / V_calc) 
        qp_calc    = rp_calc / X_calc
        
    
    mu         = (1/X)      * ( dXdt )      + (1/V)      * ( dVdt )
    mu_calc    = (1/X_calc) * ( dXdt_calc ) + (1/V_calc) * ( dVdt_calc )

    mu = np.clip(mu, 0, None)
    qp = np.clip(qp, low_qp, None)
    rp = np.clip(rp, low_rp, None)

    mu_calc = np.clip(mu_calc, 0, None)
    qp_calc = np.clip(qp_calc, low_qp, None)
    rp_calc = np.clip(rp_calc, low_rp, None)

    df["mu"] = mu
    df["qP"] = qp
    df["rP"] = rp

    df["mu_calc"] = mu_calc
    df["qP_calc"] = qp_calc
    df["rP_calc"] = rp_calc

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
    df["Run_T"] = df["Run_ID"].astype(str) + "_T_" + df["T_ind"]

    return df 

# --- Calculate features ---
def calculate_features(BR09, in_dir = "src/config/default_parameters.yaml"):

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

        sol = sol_block["sol"]
        X, _, _, V = sol.y
        time = sol.t

        feed_cfg = cfg["feeds"][br_id]
        feed_S = create_feed(feed_cfg["feed_S"])
        FS = np.zeros(len(time))
        for i in range(0,len(time)):
            FS[i], _ = feed_S.F(time[i])

        results[br_id] = {
            "time":time,
            "X": X,
            "V": V,
            "mu": sol_block["mu"],
            "dXdt": sol_block["dXdt"],
            "dVdt": sol_block["dVdt"],
            "FS": FS
        }

    return results