
# from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline, UnivariateSpline
from fedbatch.estimation.datasets import ExperimentDataset
from fedbatch.simulation.feed_factory import create_feed
from fedbatch.utils.io import get_br_id, load_yaml, get_time_ranges
from fedbatch.data_analysis.data_plots import (
    heatmap_global, heatmap_per_run,pca_global, pca_per_run, boxplot_global, boxplot_por_run
    )

#---------------Global function--------------------------
def run_EAD(df,variables,save_dir=None):
    heatmap_global(df,variables ,save_dir)
    heatmap_per_run(df,variables,save_dir)

    pca_global(df,variables,save_dir)
    pca_per_run(df,variables,save_dir)

    boxplot_global(df,variables,save_dir)
    boxplot_por_run(df,variables,save_dir)

# -------------------process dataset function--------------
def unificar_xls(dataset_files, yaml_path, save_dir = False):
    
    yaml_params = load_yaml(yaml_path)
    # dataset_files = sorted(glob("data/raw/BR*.xls"))
    
    datasets = [ExperimentDataset(f) for f in dataset_files]
    
    df_global = []
    df_batch_all = []
    df_semibatch_all = []
    df_induction_all = []

    for dataset in datasets:

        df = dataset.df.copy()

        # Indicates the dataset name
        br_id = get_br_id(dataset)
        df["Run_ID"] = br_id
        df.insert(0, "Run_ID", df.pop("Run_ID"))
        time_sb, time_ind = get_time_ranges(yaml_params, br_id)
        feed_S = None # create_feed(yaml_params["feeds"][br_id]["feed_S"])

        # mu and qp calculation
        df = calcular_mu_qp(df,time_ind, feed_S, plot_splines=True, save_dir=save_dir, id=br_id)
        # df = df.sort_values("time").reset_index(drop=True)
        df = agregar_T_ind(df)

        df_batch = df[(df["time"] >= 0) & (df["time"] < time_sb)].copy()
        df_semibatch = df[(df["time"] >= time_sb) & (df["time"] < time_ind)].copy()
        df_induction = df[df["time"] >= time_ind].copy()

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


# mu and qp calculation function
def calcular_mu_qp(df,t_ind=None, feed_S=None, 
        spline_type="univariate",   # "cubic" or "univariate"
        plot_splines=False,
        s_smooth=(4, 0.001, 0), # last value for s_P when no log() was 0
        save_dir = False,
        id = None
):
    df = df.sort_values("time").copy()
    V0 = df["V"][0]
    n = len(df)
    mu = np.zeros(n)
    # qp_old = np.zeros(n)
    qp = np.zeros(n)

    t = df["time"].values
    X = df["X"].values
    V = df["V"].values
    P = df["P"].values

    eps = 1e-10  # avoid log(0)
    
    #  Spline fitting 
    if id in ("BR06", "BR07", "BR08"):
        s_smooth = (s_smooth[0], 0.005, s_smooth[2])
    elif id == "BR05":
        s_smooth = (s_smooth[0], 0.0015, s_smooth[2])

    spline_X_cubic = CubicSpline(t, X, bc_type="natural")
    spline_V_cubic = CubicSpline(t, V, bc_type="natural")
    spline_P_cubic = CubicSpline(t, P, bc_type="natural")
    # spline_P_log_cubic = CubicSpline(t, np.log(P + eps), bc_type="natural")
    # spline_P_cubic = np.exp(spline_P_log_cubic(t))

    spline_X_uni = UnivariateSpline(t, X, s=s_smooth[0])
    spline_V_uni = UnivariateSpline(t, V, s=s_smooth[1])
    spline_P_uni = UnivariateSpline(t, P, s=s_smooth[2])
    # spline_P_log_uni = UnivariateSpline(t, np.log(P + eps), s=s_smooth[2])
    # spline_P_uni = np.exp(spline_P_log_uni(t))

    if spline_type == "cubic":
        dXdt = spline_X_cubic.derivative()(t)
        dPdt = spline_P_cubic.derivative()(t)
        # dPdt = spline_P_cubic * spline_P_log_cubic.derivative()(t)
    elif spline_type == "univariate":
        dXdt = spline_X_uni.derivative()(t)
        dPdt = spline_P_uni.derivative()(t)
        # dPdt = spline_P_uni * spline_P_log_uni.derivative()(t)
    else:
        raise ValueError("spline_type must be 'cubic' or 'univariate")
    
    i=0
    h = t[i+1] - t[i]
    h2 = t[i+2] - t[i]
    alpha = h2/h
    dXdt[0] = (1/h) * (1/alpha) * (1/(1-alpha)) * ( X[i+2] - (alpha**2 * X[i+1]) - (1-alpha**2) * X[i] )
    
    # X_s = spline_X_uni(t)
    # V_s = spline_V_uni(t)
    # P_s = spline_P_uni(t)
    if  feed_S != None:
        sol = solve_ivp(V_model,t_span=(0, t[-1]),y0=[V0],t_eval=t,args=(feed_S,))
        V = sol.y[0]
        dVdt = np.zeros(n)
        for i in range(n):
            FS, _ = feed_S.F(t[i])
            dVdt[i] = FS
    else: 
        if spline_type == "cubic":
            dVdt = spline_V_cubic.derivative()(t)
        elif spline_type == "univariate":
            dVdt = spline_V_uni.derivative()(t)
        else:
            raise ValueError("spline_type must be 'cubic' or 'univariate")

    if t_ind != None:
        for i in range(n):
            if t[i] < t_ind:
                qp[i] = 0
            else:
                qp[i] = (1/X[i]) * ( dPdt[i] + (dVdt[i] * P[i] / V[i]) ) # - (mu*P_s) )
    else: 
        # qp_old = (1/X_s) * ( dPdt )
        qp    = (1/X) * ( dPdt + (dVdt * P / V) )
    
    mu    = (1/X) * ( dXdt ) + (1/V) * ( dVdt )

    if plot_splines:
        _plot_spline_comparison(
            t, X, V, P,
            spline_X_cubic, spline_V_cubic, spline_P_cubic,
            spline_X_uni, spline_V_uni, spline_P_uni, 
            n_dense=500, save_dir=save_dir, id=id, log_P=False
        )

    # X = spline_X_uni(t)
    # V = spline_V_uni(t)
    # P = spline_P_uni(t)

    # for i in range(n):
    # # first row
    #     if i == 0:
    #         h = t[i+1] - t[i]
    #         h2 = t[i+2] - t[i]
    #         alpha = h2/h

    #         dXdt = (1/h) * (1/alpha) * (1/(1-alpha)) * ( X[i+2] - (alpha**2 * X[i+1]) - (1-alpha**2) * X[i] )
    #         dVdt = (1/h) * (1/alpha) * (1/(1-alpha)) * ( V[i+2] - (alpha**2 * V[i+1]) - (1-alpha**2) * V[i] )
    #         dPdt = (1/h) * (1/alpha) * (1/(1-alpha)) * ( P[i+2] - (alpha**2 * P[i+1]) - (1-alpha**2) * P[i] )

    # # last row
    #     elif i == n - 1:
    #         h =  t[i] - t[i-1]
    #         h2 = t[i] - t[i-2]
    #         alpha = h2/h

    #         dXdt = (1/h) * (1/alpha) * (1/(alpha-1)) * ( (alpha**2-1) * X[i] - (alpha**2 * X[i-1]) + X[i-2] )
    #         dVdt = (1/h) * (1/alpha) * (1/(alpha-1)) * ( (alpha**2-1) * V[i] - (alpha**2 * V[i-1]) + V[i-2] )
    #         dPdt = (1/h) * (1/alpha) * (1/(alpha-1)) * ( (alpha**2-1) * P[i] - (alpha**2 * P[i-1]) + P[i-2] )
    # # others
    #     else:
    #         h = t[i+1] - t[i]
    #         h2 = t[i] - t[i-1]
    #         alpha = h2/h

    #         dXdt = (1/h) * (1/alpha) * (1/(1+alpha)) * ( (alpha**2 * X[i+1]) + (1-alpha**2) * X[i] - X[i-1] )
    #         dVdt = (1/h) * (1/alpha) * (1/(1+alpha)) * ( (alpha**2 * V[i+1]) + (1-alpha**2) * V[i] - V[i-1] )
    #         dPdt = (1/h) * (1/alpha) * (1/(1+alpha)) * ( (alpha**2 * P[i+1]) + (1-alpha**2) * P[i] - P[i-1] )

    #     mu[i]     = (1/X[i]) * ( dXdt ) + (1/V[i]) * ( dVdt )
    #     # qp_old[i] = (1/X[i]) * ( dPdt )
    #     qp[i]     = (1/X[i]) * ( dPdt + (dVdt * P[i] / V[i]) ) # - (mu[i]*P[i]) )

    df["mu"] = mu
    # df["qP_old"] = qp_old
    df["qP"] = qp

    return df

def V_model(t,V,feed_S):
    if feed_S is not None:
        FS, ind_F = feed_S.F(t)
        dVdt = FS
    else:
        dVdt = 0.0

    return dVdt

def agregar_T_ind(df,n_ultimos=4):

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

    # runT_map = (
    #         df[["Run_ID", "T_ind", "Run_T"]]
    #         .drop_duplicates()
    #         .set_index("Run_ID")
    #     )

    return df # , runT_map


# def asignar_Run_T(df, runT_map):

#     df = df.copy()

#     df["T_ind"] = df["Run_ID"].map(runT_map["T_ind"])
#     df["Run_T"] = df["Run_ID"].map(runT_map["Run_T"])

#     return df


def _plot_spline_comparison(
    t, X, V, P,
    sXc, sVc, sP_logc,
    sXu, sVu, sP_logu, 
    n_dense=500, save_dir=False, id=None, log_P=False
):
    t_dense = np.linspace(t.min(), t.max(), n_dense)
    
    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

    # variables = [
    #     ("X", X, sXc, sXu),
    #     ("V", V, sVc, sVu),
    #     ("P", P, sP_logc, sP_logu),
    # ]

    # for ax, (name, data, cubic_spline, uni_spline) in zip(axes, variables):
    #     ax.scatter(t, data, color="black", s=20, label="Data")
    #     ax.plot(t_dense, cubic_spline(t_dense), color="blue", linewidth=1.5,
    #             label="CubicSpline", alpha = 0.7)
    #     ax.plot(t_dense, uni_spline(t_dense), color="red", linestyle="--", linewidth=1.5,
    #             label="UnivariateSpline", alpha = 0.7)

    #     ax.set_ylabel(name)
    #     ax.grid(True)
    #     ax.legend()

    # axes[-1].set_xlabel("Time")
    # fig.suptitle("Spline fitting comparison", fontsize=14)
    # plt.tight_layout()

    # X
    axes[0].scatter(t, X, color="black", s=20, label="Data")
    axes[0].plot(t_dense, sXc(t_dense), color="blue", label="CubicSpline")
    axes[0].plot(t_dense, sXu(t_dense), color="red", linestyle="--",
                 label="UnivariateSpline")
    axes[0].set_ylabel("X")
    axes[0].legend()
    axes[0].grid(True)

    # V
    axes[1].scatter(t, V, color="black", s=20, label="Data")
    axes[1].plot(t_dense, sVc(t_dense), color="blue")
    axes[1].plot(t_dense, sVu(t_dense), color="red", linestyle="--")
    axes[1].set_ylabel("V")
    axes[1].grid(True)

    # P (log‑spline → exp for plotting)
    axes[2].scatter(t, P, color="black", s=20, label="Data")

    if log_P:
        axes[2].plot(
            t_dense,
            np.exp(sP_logc(t_dense)),
            color="blue",
            label="CubicSpline (log)"
        )
        axes[2].plot(
            t_dense,
            np.exp(sP_logu(t_dense)),
            color="red",
            linestyle="--",
            label="UnivariateSpline (log)"
        )
    else:
        axes[2].plot(t_dense, sP_logc(t_dense), color="blue")
        axes[2].plot(t_dense, sP_logu(t_dense), color="red", linestyle="--")

    axes[2].set_ylabel("P")
    axes[2].set_xlabel("Time")
    axes[2].legend()
    axes[2].grid(True)

    fig.suptitle("Spline fitting comparison", fontsize=14)
    plt.tight_layout()

    if save_dir:
            savepath = f"{save_dir}/spline_{id}.png"
            plt.savefig(savepath, dpi=300, bbox_inches="tight")

    plt.close(fig)

    # plt.show()
