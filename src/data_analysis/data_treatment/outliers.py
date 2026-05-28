
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
from src.data_analysis.data_treatment.plots_outlier_derivative import plot_outlier_diagnostics
from src.utils.io import save_yaml, get_br_id, timer
from statsmodels.nonparametric.smoothers_lowess import lowess
from src.data_analysis.data_treatment.processing import add_T_ind
 
# def data_unification(datasets, files_names):

#     df_global = []

#     for file_name in files_names: # ******
#         df = pd.DataFrame(datasets[file_name])
#         df["Run_ID"] = file_name
#         df.insert(0, "Run_ID", df.pop("Run_ID"))
#         df_global.append(df)
    
#     df_global_final = pd.concat(df_global, ignore_index=True)

#     return df_global_final

@timer
def process_all_datasets(datasets, time_col="time", variable_list=None, results_root="results", smooth=True):

    all_metrics = {}
    smoothed_datasets = {}
    replaced_datasets = {}

    for dataset in datasets:
        
        br_id = get_br_id(dataset)
        print(f"Processing dataset: {br_id}")

        br_results_dir = f"{results_root}/{br_id}"
        all_metrics[br_id] = {}

        smoothed_datasets[br_id] = {
            "time": dataset.df[time_col].to_numpy()
        }

        replaced_datasets[br_id] = {
            "time": dataset.df[time_col].to_numpy()
        }

        # -------- Process each variable (X, S, P, V) -------- 
        if variable_list == None:
            variable_list = dataset.data.keys()

        for variable_col in variable_list:
            # print(f"  └─ Signal: {variable_col}")

            variable_results_dir = f"{br_results_dir}/{variable_col}"

            # -------- Treatment data execution --------
            x_smooth, x_replaced, outliers, metrics = treat_data(
                    df=dataset.df, time_col=time_col,
                    variable_col=variable_col,file_id=br_id,
                    results_dir=variable_results_dir,smooth=smooth # type: ignore
                )

            # -------- Save metrics --------
            all_metrics[br_id][variable_col] = metrics
            # if metrics["outliers"]["present"]:
            #     all_metrics[br_id][variable_col] = metrics

            # -------- Save smoothed data --------
            smoothed_datasets[br_id][variable_col] = x_smooth
            replaced_datasets[br_id][variable_col] = x_replaced

            smoothed_datasets[br_id]["I"] = dataset.df["I"].to_numpy()
            replaced_datasets[br_id]["I"]  = dataset.df["I"].to_numpy()

    # -------- Save global metrics --------  
    save_yaml(
        all_metrics,
        filepath=f"{results_root}/summary.yaml"
    )

    print("\n Processing finished. \n")

    return smoothed_datasets, replaced_datasets


def treat_data(df,time_col,variable_col,file_id=None,results_dir=False,
               window_outlier=5,sg_order=4,sg_window=11,smooth=True): # ,sg_window=11

    time = df[time_col].to_numpy()
    x = df[variable_col].to_numpy()
    eps = 1e-12

    # --- Outlier detection (mov median) ---
    outliers = movmedian_outliers(x)

    outlier_indices = np.where(outliers)[0]
    has_outliers = len(outlier_indices) > 0

    # --- Candidate replacements ---
    candidates = {
        "movmean": pd.Series(x).rolling(window = window_outlier, center=True).mean().to_numpy(),
        "movmedian": pd.Series(x).rolling(window = window_outlier, center=True).median().to_numpy(),
        "gaussian": gaussian_filter1d(x, sigma= (window_outlier-1)/6), # sigma=2 it is needed to calculate sigma for a movile window
        "rlowess": lowess(x, np.arange(len(x)), frac= window_outlier/len(x), return_sorted=False),
        "sgolay": savgol_filter(x, window_outlier, sg_order), # window_outlier = 11
        "mean_methods": (
            lowess(x, np.arange(len(x)), frac= window_outlier/len(x), return_sorted=False) + 
            savgol_filter(x, window_outlier, sg_order) + 
            pd.Series(x).rolling(window = window_outlier, center=True).median().to_numpy() 
            ) / 3
    }

    x_replaced = x.copy()
    selected_method_per_outlier = {}

    for idx in np.where(outliers)[0]:
        # diffs = {
        #     m: abs(candidates[m][idx] - x[idx])
        #     for m in candidates
        #     if not np.isnan(candidates[m][idx])
        # }
        # best_method = max(diffs, key=diffs.get) 
        best_method = "mean_methods" # rlowess sgolay movmedian
        x_replaced[idx] = candidates[best_method][idx]
        selected_method_per_outlier[idx] = best_method 

    special_outliers = {}

    if variable_col in ("X", "P") and len(x) > 0:

        i0 = 0
        i_end = len(x) - 1
        i_min = int(np.argmin(x_replaced))
        i_max = int(np.argmax(x_replaced))

        apply_first = i_min != i0
        apply_last  = i_max != i_end

        # --- FIRST–MIN pair ---
        if apply_first:
            x0_old   = float(x_replaced[i0])
            xmin_old = float(x_replaced[i_min])

            first_avg = 0.5 * (x0_old + xmin_old)

            x_replaced[i0]    = first_avg
            x_replaced[i_min] = first_avg

            special_outliers["first_min_pair"] = {
                "first_index": i0,
                "min_index": i_min,
                "original_values": {
                    "first": x0_old,
                    "min": xmin_old
                },
                "final_value": float(first_avg)
            }

        # --- LAST–MAX pair ---
        if apply_last:
            xend_old = float(x_replaced[i_end])
            xmax_old = float(x_replaced[i_max])

            last_avg = 0.5 * (xend_old + xmax_old)

            x_replaced[i_end] = last_avg
            x_replaced[i_max] = last_avg

            special_outliers["last_max_pair"] = {
                "last_index": i_end,
                "max_index": i_max,
                "original_values": {
                    "last": xend_old,
                    "max": xmax_old
                },
                "final_value": float(last_avg)
            }

    if smooth == True:
        # --- Smoothing Savitzky–Golay ---
        x_smooth = savgol_filter(x_replaced, sg_window, sg_order) # window_outlier // sg_window

        # eps = 1e-12
        # x_smooth = np.exp(savgol_filter(np.log(x_replaced + eps), sg_window, sg_order)) # Log to avoid negative numbers

        for _ in range(5):
            x_smooth = np.maximum(x_smooth, 0)
            x_smooth = savgol_filter(x_smooth, sg_window, sg_order) # Iterative to avoid negative numbers # window_outlier
        x_smooth = np.maximum(x_smooth, 0)
        
        # --- Metrics ---
        mape = np.mean(np.abs((x - x_smooth) / (x + eps))) * 100
        # mape = np.mean(np.abs((x - x_smooth) / x)) * 100

    else:
        x_smooth = None
        mape = np.mean(np.abs((x - x_replaced) / (x + eps))) * 100
    
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
        "special_outliers": {
            "applied": variable_col in ("X", "P"),
            "details": special_outliers
        },
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
            "MAPE_raw_vs_treated": float(mape) #,
        # "outlier_ratio": float(outliers.mean())
        }
    }

    # --- Save yaml file ---
    if results_dir and (has_outliers or special_outliers):
        save_yaml(metrics,
                filepath=f"{results_dir}/{file_id}_{variable_col}_metrics.yaml"
            )

    # --- Plots ---
    plot_outlier_diagnostics(
                time=time,x=x,outliers=outliers,candidates=candidates,metrics=metrics,
                selected_method_per_outlier=selected_method_per_outlier,
                x_replaced=x_replaced,x_smooth=x_smooth,
                save_dir=results_dir,prefix=f"{variable_col}",has_outliers=has_outliers
            )

    return x_smooth, x_replaced, outliers, metrics

# -------- Outliers function detection based on mobile window median ------- **

def movmedian_outliers(x, window=5, thresh=3.5): #  (3 unal // thresh=3.5 Tukey, Iglewicz & Hoaglin) *** Search Hampel / Tukey
    x = np.asarray(x)

    med_local = pd.Series(x).rolling(window, center=True).median() # Local MEDian 
    mad_local = (np.abs(x - med_local)).rolling(window, center=True).median() # type: ignore # Local Median Absolute Deviation

    med_global = np.median(x)
    mad_global = np.median(np.abs(x - med_global))
    c = 1 / 0.67449 # 1.4826 (the 75th percentile of a standard normal distribution \sigma )

    z = np.abs(x - med_local) / (c * mad_local) 
    # MAD_{Gaussian} ​= \sigma * 0.67449
    # Z-score = ( x - mean ) / sigma
    # Z-score = ( x - med ) / MAD_{Gaussian}

    # # ---------- movmedian inside adn global MAD only at edges ----------
    # edge = np.isnan(z)
    # z[edge] = np.abs(x[edge] - med_global) / (1.4826 * mad_global)

    outliers = z > thresh

    # --------- borders as non‑outliers ---------
    outliers = outliers.fillna(False).to_numpy()

    return outliers 


