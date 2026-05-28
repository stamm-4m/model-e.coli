
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# def regression_metrics(y_true, y_pred):
#     y_true = np.asarray(y_true)
#     y_pred = np.asarray(y_pred)

#     residuals = y_true - y_pred
#     n = y_true.size

#     # --- Basic errors ---
#     mse = np.mean(residuals**2)
#     rmse = np.sqrt(mse)
#     mae = np.mean(np.abs(residuals))
#     medae = np.median(np.abs(residuals))

#     # --- Bias ---
#     me = np.mean(residuals)          # Mean Error
#     mbe = me                         # alias 

#     # --- Normalized RMSE (scale-free) ---
#     y_range = y_true.max() - y_true.min()
#     nrmse = rmse / y_range if y_range > 0 else np.nan

#     # --- R² ---
#     ss_res = np.sum(residuals**2)
#     ss_tot = np.sum((y_true - np.mean(y_true))**2)
#     r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

#     # --- MAPE (careful with zeros) ---
#     eps = 1e-12
#     mape = np.mean(np.abs(residuals / (y_true + eps))) * 100

#     # --- Residual spread ---
#     resid_std = np.std(residuals, ddof=1)

#     return {
#         "RMSE": float(rmse),
#         "NRMSE": float(nrmse),
#         "MAE": float(mae),
#         "MedAE": float(medae),
#         "ME": float(me),
#         "MBE": float(mbe),
#         "MAPE_%": float(mape),
#         "R2": float(r2),
#         "Residual_STD": float(resid_std),
#         "N": int(n),
#     }


# def regression_metrics_weight(y_true, y_pred, weights=None):
#     y_true = np.asarray(y_true)
#     y_pred = np.asarray(y_pred)

#     residuals = y_true - y_pred
#     n = y_true.size

#     if weights is None:
#         weights = np.ones_like(residuals)

#     weights = np.asarray(weights)
#     weights = weights / np.mean(weights)  # normalize weights

#     # --- Weighted errors ---
#     w_mse = np.mean(weights * residuals**2)
#     w_rmse = np.sqrt(w_mse)
#     w_mae = np.mean(weights * np.abs(residuals))

#     # --- Unweighted (for reference) ---
#     rmse = np.sqrt(np.mean(residuals**2))
#     mae = np.mean(np.abs(residuals))

#     # --- Bias ---
#     mbe = np.average(residuals, weights=weights)

#     # --- R² (weighted) ---
#     y_bar = np.average(y_true, weights=weights)
#     ss_res = np.sum(weights * residuals**2)
#     ss_tot = np.sum(weights * (y_true - y_bar)**2)
#     r2_w = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

#     return {
#         "RMSE": float(rmse),
#         "MAE": float(mae),
#         "wRMSE": float(w_rmse),
#         "wMAE": float(w_mae),
#         "MBE": float(mbe),
#         "R2_weighted": float(r2_w),
#         "N": int(n),
#     }

# def information_criteria(y_true, y_pred, k):
#     residuals = y_true - y_pred
#     rss = np.sum(residuals**2)
#     N = len(y_true)

#     aic = N * np.log(rss / N) + 2 * k
#     bic = N * np.log(rss / N) + k * np.log(N)

#     return {
#         "AIC": aic,
#         "BIC": bic
#     }

# def information_criteria_from_residuals(residuals, k):
#     residuals = np.asarray(residuals)
#     rss = np.sum(residuals**2)
#     rss = max(rss, 1e-12)  # numerical safety
#     N = residuals.size

#     aic = N * np.log(rss / N) + 2 * k
#     bic = N * np.log(rss / N) + k * np.log(N)

#     return {
#         "AIC": float(aic),
#         "BIC": float(bic)
#     }

# ------------ Metrics ---------------
def compute_metrics(y_true, y_pred, k):

    n = len(y_true)
    alpha=0.7 
    penalty=2.0

    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    # mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    mape = safe_mape(y_true, y_pred)

    # AIC / BIC
    aic = n * np.log(mse) + 2 * k
    bic = n * np.log(mse) + k * np.log(n)
    
    # Base score (normalize: RMSE smaller is better, R2 larger is better)
    score = alpha * mse + (1 - alpha) * (1 - r2)
    if r2 < 0:
        score *= penalty

    return {
        "R2": float(r2),
        "MAE": float(mae),
        "MSE": float(mse),
        "RMSE": float(rmse),
        "MAPE": float(mape),
        "SCORE": float(score),
        "AIC": float(aic),
        "BIC": float(bic)
    }

# ------------- MAPE calculation ---------------

def safe_mape(y_true, y_pred):

    mask = y_true != 0

    if np.sum(mask) == 0:
        return np.nan  # all zero

    return np.mean(
        np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])
    ) * 100



def save_metrics_tables_excel(cv_results, filepath):

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
        for model_name, model_data in cv_results.items():
            rows = {}
            for fold in model_data["folds"]:
                run_id = fold["test_groups"][0]
                for metric, value in fold["metrics"].items():
                    if metric not in rows:
                        rows[metric] = {}
                    rows[metric][run_id] = value

            df = pd.DataFrame(rows).T
            df["MEAN"] = df.mean(axis=1)
            cols = sorted([c for c in df.columns if c != "MEAN"]) + ["MEAN"]
            df = df[cols]
            df.to_excel(writer, sheet_name=model_name[:31]) 

