
import numpy as np
import pandas as pd
from src.data_analysis.data import ExperimentDataset
from src.utils.io import get_br_id, load_yaml, get_time_ranges

# ------------------- Computes qP and mu and unifies dataframes --------------
 
def processing_data(datasets, yaml_path, t_ind_exp = True):
    
    yaml_params = load_yaml(yaml_path)
    # dataset_files = sorted(glob("data/raw/BR*.xls"))
    
    # datasets = [ExperimentDataset(f) for f in dataset_files]
    
    df_global = []
    df_batch_all = []
    df_semibatch_all = []
    df_induction_all = []

    for br_id in datasets:

        df = pd.DataFrame(datasets[br_id])
        
        # Indicates the dataset name
        df["Run_ID"] = br_id
        df.insert(0, "Run_ID", df.pop("Run_ID"))

        # Indicatates dataset numer and T of induction
        df = add_T_ind(df)
        df.insert(1, "Run_T", df.pop("Run_T"))

        # qP and mu calculation
        time_sb, time_ind = get_time_ranges(yaml_params, br_id)

        # mu and qp calculation
        if t_ind_exp == True:
            df = calc_mu_qp_rp(df, time_ind)
        else:
            df = calc_mu_qp_rp(df, t_ind=None)
        # df = df.sort_values("time").reset_index(drop=True)

        df_batch = df[(df["time"] >= 0) & (df["time"] < time_sb)].copy()
        df_semibatch = df[(df["time"] >= time_sb) & (df["time"] < time_ind)].copy()
        df_induction = df[df["time"] >= time_ind].copy()

        # -- Add previous values of X and P as features (lag 1) for induction phase --
        df_induction["Xlag1"] = df_induction["X"].shift(1)
        df_induction["Plag1"] = df_induction["P"].shift(1)
        df_induction.loc[df_induction.index[0], "Xlag1"] = df_semibatch["X"].iloc[-1]
        df_induction.loc[df_induction.index[0], "Plag1"] = df_semibatch["P"].iloc[-1]

        df_global.append(df)
        df_batch_all.append(df_batch)
        df_semibatch_all.append(df_semibatch)
        df_induction_all.append(df_induction)

    # final unification
    df_global_final = pd.concat(df_global, ignore_index=True)
    df_batch_final = pd.concat(df_batch_all, ignore_index=True)
    df_semibatch_final = pd.concat(df_semibatch_all, ignore_index=True)
    df_induction_final = pd.concat(df_induction_all, ignore_index=True)

    return df_global_final, df_batch_final, df_semibatch_final, df_induction_final 

# -------------------------- mu, qp & rp function ---------------------------------------

def calc_mu_qp_rp(df, t_ind=None):

    df = df.sort_values("time").copy()

    n = len(df)

    mu = np.zeros(n)
    qp = np.zeros(n)
    rp = np.zeros(n)

    t = df["time"].values
    X = df["X"].values
    V = df["V"].values
    P = df["P"].values
    dXdt = df["dXdt"].values
    dVdt = df["dVdt"].values
    dPdt = df["dPdt"].values

    if t_ind != None:
        for i in range(n):
            if t[i] < t_ind:
                qp[i] = 0
                rp[i] = 0
            else:
                # qp[i] = (1/X[i]) * dPdt[i] 
                qp[i] = (1/X[i]) * ( dPdt[i] + (dVdt[i] * P[i] / V[i]) ) 
                rp[i] = dPdt[i] + (dVdt[i] * P[i] / V[i]) 
    else: 
        # qp = (1/X) *  dPdt 
        qp    = (1/X) * ( dPdt + (dVdt * P / V) )
        rp    =  dPdt + (dVdt * P / V) 
    
    mu    = (1/X) * ( dXdt ) + (1/V) * ( dVdt )

    # Clip negative values to zero
    mu = np.clip(mu, 1e-8, None)
    qp = np.clip(qp, 1e-8, None)
    rp = np.clip(rp, 1e-8, None)

    df["mu"] = mu
    df["qP"] = qp
    df["rP"] = rp

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