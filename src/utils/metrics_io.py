
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ------------ Metrics ---------------
def compute_metrics(y_true, y_pred):

    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    mape = safe_mape(y_true, y_pred)
    
    return {
        "R2": float(r2),
        "MAE": float(mae),
        "MSE": float(mse),
        "RMSE": float(rmse),
        "MAPE": float(mape),
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

