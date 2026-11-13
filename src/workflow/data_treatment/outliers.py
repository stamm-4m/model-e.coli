"""
Nanobody-based Antivenom Production with E. coli Reactor Simulation 

Author: Juan Camilo Castaño Sanchez
Email: jcastano-san@insa-toulose.fr
Date: 01/09/2026

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.utils.io import save_yaml, get_br_id, timer
from src.utils.metrics_io import compute_metrics, safe_mape, fun_mase
from src.workflow.data_treatment.plots_outlier_derivative import plot_outlier_diagnostics, plot_all_derivatives
# from src.workflow.data_treatment.processing import add_T_ind
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
from statsmodels.nonparametric.smoothers_lowess import lowess
from skmisc.loess import loess
from scipy.interpolate import UnivariateSpline
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression
 
# @timer

def outliers_and_smoothing(datasets, time_col="time", variable_list=None, ignore_vars=None, 
                           results_root="results", smooth=True, states_variables=None, non_states_cfg=None):
    """
    For each dataset, it processes the data by detecting outliers, replacing them with appropriate values following methodology, 
    and optionally smoothing the data. It saves metrics and plots for each variable in the dataset.

    Inputs:
    - datasets: List of DatasetStandardization objects to be processed.
    - time_col: Name of the column representing time in the datasets.
    - variable_list: List of variable names to be processed. If None, all variables in the dataset will be processed.
    - results_root: Root directory where results (metrics and plots) will be saved.
    - smooth: Boolean indicating whether to apply smoothing to the data after outlier replacement.
    """
    all_metrics = {}

    smoothed_datasets = {}
    replaced_datasets = {}

    for dataset in datasets:
        
        br_id = dataset.br_id
        print(f"Processing dataset: {br_id}")

        # if br_id in ("BR06", "BR07", "BR08"):
        #     # dataset = dataset.drop(dataset.index[-1])
        #     dataset.df.drop(dataset.df.index[-1], inplace=True)

        br_results_dir = f"{results_root}/{br_id}"
        all_metrics[br_id] = {}

        smoothed_datasets[br_id] = {time_col: dataset.df[time_col].to_numpy()}
        replaced_datasets[br_id] = {time_col: dataset.df[time_col].to_numpy()}

        # -------- Process each variable (X, S, P, V) -------- 
        # if variable_list == None:
        variable_list = dataset.df.keys()

        plots_dict = {}

        for variable_col in variable_list:

            if variable_col in ignore_vars:
                continue

            variable_results_dir = f"{br_results_dir}/{variable_col}"

            # -------- Treatment data execution --------
            x_smooth, x_replaced, dxdt_smooth, dxdt_replaced, outliers, metrics, time = treat_data(
                    df=dataset.df, time_col=time_col,
                    variable_col=variable_col,file_id=br_id,
                    results_dir=variable_results_dir,smooth=smooth, 
                    state_variables = states_variables, 
                    non_states_cfg = non_states_cfg # type: ignore
                )
            
            # -------- Save metrics --------
            all_metrics[br_id][variable_col] = metrics
            # if metrics["outliers"]["present"]:
            #     all_metrics[br_id][variable_col] = metrics

            # -------- Save smoothed data --------
            smoothed_datasets[br_id][variable_col] = x_smooth
            smoothed_datasets[br_id][f"d{variable_col}dt"] = dxdt_smooth
            smoothed_datasets[br_id][f"d2{variable_col}dt2"] = np.gradient(dxdt_smooth, time, edge_order=2) 

            replaced_datasets[br_id][variable_col] = x_replaced
            replaced_datasets[br_id][f"d{variable_col}dt"] = dxdt_replaced
            replaced_datasets[br_id][f"d2{variable_col}dt2"] = np.gradient(dxdt_replaced, time, edge_order=2) 

            plots_dict[variable_col] = {
                "signal": {
                    "replaced": x_replaced,
                    "smooth": x_smooth,
                },
                "derivatives": {
                    "replaced": dxdt_replaced,
                    "smooth": dxdt_smooth,
                },
                "second_derivatives": {
                    "replaced": np.gradient(dxdt_replaced, time, edge_order=2) ,
                    "smooth": np.gradient(dxdt_smooth, time, edge_order=2) ,
                },
            }
        
        plot_variables = list(states_variables.keys())
        if plot_variables != None:
            plot_all_derivatives(t=time,results=plots_dict,variables=plot_variables,br_id=br_id, out_dir=br_results_dir)

    # -------- Save global metrics --------  
    save_yaml( all_metrics,
        filepath=f"{results_root}/summary.yaml"
    )

    print("\n Processing finished. \n")

    return smoothed_datasets, replaced_datasets


def treat_data(df, time_col, variable_col, file_id=None, results_dir=False,
               window_outlier=5, sg_order=4, sg_window=11, smooth=True, 
               state_variables = None, non_states_cfg=None) : # , sg_window=11
    
    """
    Treats the data for a specific variable in the dataset by detecting outliers, replacing them with appropriate values, 
    and optionally smoothing the data. It saves metrics and plots for the variable.

    Inputs:
    - df: DataFrame containing the dataset to be processed.
    - time_col: Name of the column representing time in the dataset.
    - variable_col: Name of the variable column to be processed.
    - file_id: Identifier for the dataset (used for saving results). If None, the dataset will not be saved with a specific identifier.
    - results_dir: Directory where results (metrics and plots) will be saved. If False, results will not be saved.
    - window_outlier: Window size for outlier detection and replacement.
    - sg_order: Order of the Savitzky-Golay filter for smoothing.
    - sg_window: Window size for the Savitzky-Golay filter for smoothing.
    - smooth: Boolean indicating whether to apply smoothing to the data after outlier replacement.
    """

    time = df[time_col].to_numpy()
    x = df[variable_col].to_numpy()
    eps = 1e-12

    # --- Outlier detection (mov median) ---
    outliers = movmedian_outliers(x)

    outlier_indices = np.where(outliers)[0]
    has_outliers = len(outlier_indices) > 0

    # --- Candidate replacements ---
    s = pd.Series(x)

    model = loess(time, x, span=window_outlier/len(x), degree=2)
    model.fit()
    loess_vals = np.asarray(model.outputs.fitted_values)

    candidates = {
        "movmean": s.rolling(window = window_outlier, center=True, min_periods=1).mean().to_numpy(),
        "movmedian": s.rolling(window = window_outlier, center=True, min_periods=1).median().to_numpy(),
        "gaussian": gaussian_filter1d(x, sigma= (window_outlier-1)/6, mode='reflect'), # sigma=2 it is needed to calculate sigma for a movile window
        "lowess": lowess(x, time, frac=window_outlier/len(x), it=3, return_sorted=False),
        "loess": loess_vals,
        "sgolay": savgol_filter(x, window_outlier, polyorder = 2), # window_outlier = 11
        "mean": (
            s.rolling(window = window_outlier, center=True, min_periods=1).mean().to_numpy() +
            s.rolling(window = window_outlier, center=True, min_periods=1).median().to_numpy() +
            gaussian_filter1d(x, sigma= (window_outlier-1)/6, mode='reflect') +
            lowess(x, time, frac= window_outlier/len(x), it=3, return_sorted=False) + 
            loess_vals +
            savgol_filter(x, window_outlier, polyorder = 2) 
            ) / 6
    }

    # --- Outlier replacement ---
    x_replaced = x.copy()
    selected_method_per_outlier = {}

    for idx in np.where(outliers)[0]:
        if idx == len(x) - 1: # último elemento del vector
            best_method = "mean" # "mean rlowess sgolay movmedian"
        else:
            diffs = {
                m: abs(candidates[m][idx] - x[idx])
                for m in candidates
                if not np.isnan(candidates[m][idx])
            }
            best_method = max(diffs, key=diffs.get) 

        x_replaced[idx] = candidates[best_method][idx]
        selected_method_per_outlier[idx] = best_method

    dxdt_replaced = np.gradient(x_replaced, time, edge_order=2) 

    # --- Special Outlier replacement (Biomass and Protein titers) --- end of the code if necessary

    # --- Smoothing (Savitzky–Golay) ---
    state_variables_list = list(state_variables.keys())

    if smooth and variable_col in state_variables_list:
        # --- Smoothing with own function ---
        idx_start = None
        idx_end = None
        x_in=x_replaced
        t_in=time

        state_cfg = (state_variables.get(variable_col, {}))
        smooth_crit = (state_cfg.get("smooth_crit", {}))
        monotonicity = smooth_crit.get("monotonicity", None)
        positive_start = smooth_crit.get("positive_start",False)
        positive_end = smooth_crit.get("positive_end",False)

        iso = IsotonicRegression(increasing = (monotonicity == "increasing"))

        if positive_start:
            mask = x_replaced > 0
            if mask.any():
                idx_start = np.where(mask)[0][0]
                x_in = x_replaced[idx_start-1:]
                t_in = time[idx_start-1:]
        
        if positive_end:
            mask = x_replaced > 1e-3
            if mask.any():
                idx_end = np.where(mask)[0][-2]
                x_in = x_replaced[:idx_end+2]
                t_in = time[:idx_end+2]
        
        if monotonicity == "increasing":
            iso = IsotonicRegression(increasing=True)
            x_iso = iso.fit_transform(t_in,x_in)
        elif monotonicity == "decreasing":
            iso = IsotonicRegression(increasing=False)
            x_iso = iso.fit_transform(t_in,x_in)
        else:
            x_iso = x_in

        output_dic = smooth_and_differentiate(t=t_in, x=x_iso, smooth_crit=smooth_crit,  
                                              var=variable_col, x_replaced=x_replaced)

        x_smooth = x_replaced.copy()
        dxdt_smooth = dxdt_replaced.copy()

        if idx_start is not None:
            x_smooth[idx_start:] = output_dic["x_smooth"][1:]
            dxdt_smooth[idx_start:] = output_dic["dxdt"][1:]
        elif idx_end is not None:
            x_smooth[:idx_end+1] = output_dic["x_smooth"][:-1]
            dxdt_smooth[:idx_end+1] = output_dic["dxdt"][:-1]
        else:
            x_smooth = output_dic["x_smooth"]
            dxdt_smooth = output_dic["dxdt"]
        
    else:
        cfg = non_states_cfg["default"]
        x_smooth = savgol_filter(x_replaced,cfg["window"],cfg["order"])
        dxdt_smooth = np.gradient(x_smooth,time, edge_order=2 )
        
    # --- Metrics for smoothing and outlier replacement ---
    # mape = np.mean(np.abs((x - x_smooth) / (x + eps))) * 100    
    mape = safe_mape(x, x_smooth)
    # mase = fun_mase(x, x_smooth, x)

    # --- Metrics dictionary ---
    metrics = {
        "file": file_id,
        "variable": variable_col,

        "outliers": {
            "present": has_outliers,
            "count": int(len(outlier_indices)),
            "indices": outlier_indices.tolist(),
            "replacement_methods": {
                int(idx): selected_method_per_outlier[idx]
                for idx in selected_method_per_outlier
            }
        },
        # "special_outliers": {
        #     "applied": variable_col in ("X", "P"),
        #     "details": special_outliers
        # },
        "statistics": {
            "raw": {
                "mean": float(np.mean(x)),
                "std": float(np.std(x))
            },
            "treated": {
                "info": "if 'smooth = True' refers to smoothed data",
                "mean": float(np.mean(x_smooth)) if x_smooth is not None else float(np.mean(x_replaced)),
                "std": float(np.std(x_smooth)) if x_smooth is not None else float(np.std(x_replaced))
            },
            "MAPE_raw_vs_treated": float(mape),
            # "MASE_in-sample_vs_treated": float(mase) #,
        # "outlier_ratio": float(outliers.mean())
        }
    }

    # --- Save metrics in yaml file ---
    if results_dir and has_outliers: # (has_outliers or special_outliers):
        filepath = f"{results_dir}/{file_id}_{variable_col}_metrics.yaml"
        save_yaml( metrics, filepath = filepath )

    # --- Plots ---
    plot_outlier_diagnostics(
                time=time,x=x,outliers=outliers,candidates=candidates,metrics=metrics,
                selected_method_per_outlier=selected_method_per_outlier,
                x_replaced=x_replaced,x_smooth=x_smooth,
                save_dir=results_dir,prefix=f"{variable_col}",has_outliers=has_outliers
            )
    

    return x_smooth, x_replaced, dxdt_smooth, dxdt_replaced, outliers, metrics, time

# -------- Outliers function detection based on mobile window median ------- **

def movmedian_outliers(x, window=5, thresh=3):    
    """
    movmedian_outliers function was created based on Hampel filter with adaptative window at edges
    
    Hampel clásico thresh 3 -Hampel (1974)-
    Z-score clásico 3 -regla empírica- 
    Modified Z-score thresh 3.5 -Iglewicz & Hoaglin (1993)-
    """
    
    x = np.asarray(x)
    n = len(x)
    k = window // 2
    med_local = np.zeros(n)
    mad_local = np.zeros(n)
    
    for i in range(n):
        start = max(0, i - k)
        end = min(n, i + k + 1)

        w = x[start:end]
        
        med = np.median(w)
        mad = np.median(np.abs(w - med))

        med_local[i] = med
        mad_local[i] = mad

    c = 1 / 0.67449 # 1.4826 ( 75th percentile of a standard normal distribution \sigma )
    mad_local[mad_local == 0] = np.nan
    z = np.abs(x - med_local) / (c * mad_local)

    outliers = z > thresh

    # --------- NaN to False ---------
    outliers = np.nan_to_num(outliers, nan=False)

    return outliers 

def smooth_and_differentiate(t, x, smooth_crit=None, plot=False, 
                             var = None, x_replaced=None, bounds=None):
    """
    Smooth x(t) and compute its derivatives.

    Parameters:
    - t : array-like Independent variable values. Must be strictly increasing.
    x : array-like Signal values to be smoothed.
    bounds : tuple, default=(3.0, 5.0)
        Search interval for the smoothing parameter s.
        The default interval corresponds to the one used for
        variable X in the original file.
    monotonicity : str, optional
        If "increasing", penalizes negative first derivatives.
        If "decreasing", penalizes positive first derivatives.
        If None, no monotonicity constraint is applied.
    plot : bool, default=True If True, creates a figure showing x, the smoothed signal,
        and its derivatives.

    Returns:
    dict:
        - x_smooth : smoothed signal from the penalized spline.
        - dxdt     : selected first derivative, computed as the average of the numerical gradient and spline derivative.
        - d2xdt2   : selected second derivative following the same logic.
        - s        : optimal smoothing parameter.
        - figure   : matplotlib figure, or None if plot=False.
    """

    # Selection of s using the same conceptual criterion
    # as the original objective function.

    monotonicity = smooth_crit.get("monotonicity", None)

    if smooth_crit is None:
        smooth_crit = {}
    
    bounds = smooth_crit.get("bounds", None)

    if x_replaced is None:
        x_replaced = x

    if bounds is None:
        bounds = ( 0.01*np.var(x)*len(x), 10*np.var(x)*len(x) )
    else:
        bounds = tuple(bounds)

    optimization = minimize_scalar(
        spline_objective_function,
        bounds=bounds,
        method="bounded",
        args=(t, x, monotonicity),
    )

    s_opt = float(optimization.x)

    # Final spline fitting using the same iterative penalty
    spline = UnivariateSpline(t, x, s=s_opt)
    x_smooth = spline(t)
    
    # --- Final constrains ---
    constraints = smooth_crit.get("constraints",{})

    if constraints.get("initial_condition",False):
        x_smooth[0] = x_replaced[0]

    if constraints.get("lower_bound_initial", False):
        x_smooth = np.maximum(x_smooth,x_replaced[0])

    if constraints.get("final_maximum",False):
        i_max = np.argmax(x_smooth)
        if i_max != len(x_smooth)-1:
            x_smooth[-1] = x_smooth[i_max]
        
    x_smooth = np.clip(x_smooth, 0, None)
    dxdt_smooth = np.gradient(x_smooth, t, edge_order=2)
    d2xdt2_smooth = np.gradient(dxdt_smooth, t, edge_order=2)

    fig = plot_derivatives(t=t, x=x, x_smooth=x_smooth, dxdt=dxdt_smooth, d2xdt2=d2xdt2_smooth )

    return {
        "x_smooth": x_smooth,
        "dxdt": dxdt_smooth,
        "d2xdt2": d2xdt2_smooth,
        "s": s_opt,
        "optimization": {
            "success": bool(optimization.success),
            "fun": float(optimization.fun),
            "nfev": int(optimization.nfev),
        },
        "spline": spline,
        "figure": fig,
    }

def spline_objective_function(s, t, x, monotonicity = None):
    """
    Original objective function used to select s.

    Combines:
    1. Difference between the numerical derivative and the spline derivative.
    2. Penalty for oscillations in the second-derivative sign.
    """
    spline = UnivariateSpline(t, x, s=s)
    x_pred = spline(t)

    dxdt_pred = np.gradient(x_pred, t, edge_order=2)
    d2xdt2 = np.gradient(dxdt_pred, t, edge_order=2)

    loss_negative = np.sum(np.minimum(x_pred, 0.0)**2)

    if monotonicity == "increasing":
        loss_monotonicity = np.sum(
            np.abs(np.minimum(dxdt_pred, 0.0)) # **2
        )

    elif monotonicity == "decreasing":
        loss_monotonicity = np.sum(
            np.maximum(dxdt_pred, 0.0) # **2
        )

    else:
        loss_monotonicity = 0.0

    loss_curvature = np.mean(np.abs(d2xdt2))

    signs = np.sign(d2xdt2)
    
    loss_oscillation = np.sum( np.abs(np.diff(signs)))

    loss = ( 1e8 * loss_monotonicity + 1e3 * loss_curvature + 1e3 * loss_oscillation + 1e5 * loss_negative) # oscillation_count
    # loss = ( loss_fit + 1000.0 * loss_monotonicity + 1.0 * loss_smoothness + 1.0 * loss_oscillation + 1000.0 * loss_negative) 

    return float(loss)

def plot_derivatives(t,x,x_smooth,dxdt,d2xdt2):
    """Generate figure with original signal, smoothing and derivatives."""
    fig, axes = plt.subplots( 3,1,figsize=(10, 10), sharex=True,)

    axes[0].plot(t, x, "o-", label="original x", alpha=0.7)
    axes[0].plot(t, x_smooth, "-", linewidth=2, label="smoothed x")
    axes[0].set_ylabel("x")
    axes[0].set_title("Smoothing and derivatives")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, dxdt, "-", label="dx/dt")
    axes[1].set_ylabel("dx/dt")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t, d2xdt2, "-", label="d²x/dt²")
    axes[2].set_xlabel("t")
    axes[2].set_ylabel("d²x/dt²")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()

    # if show:
    #     # plt.show()
    plt.close(fig)

    return fig