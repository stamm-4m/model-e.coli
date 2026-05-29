
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ------------ Metrics ---------------
def compute_metrics(y_true, y_pred, k, k_norm=0):

    n = len(y_pred)
    alpha = 0.7 
    penalty = 2.0
    lambda_k = 0.01 

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
    score = alpha * mse + (1 - alpha) * (1 - r2) + lambda_k * k /20

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

