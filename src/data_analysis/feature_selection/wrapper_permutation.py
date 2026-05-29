
import matplotlib
matplotlib.use("Agg")

import numpy as np
from pathlib import Path
import pandas as pd

from src.data_analysis.feature_selection.wrapper import wrapper_feature_selection 
from src.data_analysis.feature_selection.permutation import permutation_feature_selection
from src.data_analysis.feature_selection.plots_wrapper_permutation import (
    plot_metric_comparison, plot_feature_heatmap_from_summary, plot_metrics_heatmap_from_summary) # 
from src.utils.io import save_yaml, to_python_type, models_dict, save_wrapper_summary_table, timer

# from sklearn.model_selection import cross_val_predict
# from sklearn.model_selection import GroupKFold

# ----------- Global function ------------
@timer
def WnE_feature_selection(df, X_vars, y_var, model_names_w, model_names_p, out_path):

    print(f"Starting RFE for {y_var} by wrapper models ... \n")
    out_dir = Path(out_path)
    all_results = {}
    summary = {}
    metrics = ["R2", "MAE", "MSE", "RMSE", "MAPE", "SCORE", "AIC", "BIC"]

# ----------- Recursive Feature Selection (RFE) (Wrapper methods) ---------------
    models_wrapper = models_dict(model_names_w)
    
    summary, all_results = wrapper_feature_selection(df, X_vars, y_var, models_wrapper, summary, all_results)
    print(f"\nRFE for wrapper models finished \n")

    wrapper_results = {k: v for k, v in all_results.items() if k in model_names_w}

# ----------- Feature selection based on permutation importance ---------------
    models_permutation = models_dict(model_names_p)

    summary, all_results = permutation_feature_selection(df, X_vars, y_var, models_permutation, summary, all_results)
    print(f"\nPermutation feature selection finished \n")

    permutation_results = {k: v for k, v in all_results.items() if k in model_names_p}

# ----------- Plot -----------------------
    for m in metrics:
        plot_metric_comparison(wrapper_results,metric_name=m,out_dir=out_dir / "metrics", model_type = "wrapper")
        plot_metric_comparison(permutation_results,metric_name=m,out_dir=out_dir / "metrics", model_type = "permutation")
    
# ----------- Metrics ---------------
    best_score = np.inf
    best_global = None

    for model_name, info in summary.items():

        score = info["metrics"]["AIC"]

        if score < best_score:
            best_score = score
            best_global = model_name

# ----------- YAML ---------------
    yaml_data = {
        "target": y_var,
        "best_model": best_global,
        "best_info": summary[best_global],
        "models": summary
    }
    
    yaml_data = to_python_type(yaml_data)

    save_yaml(yaml_data, Path(out_path)/"wrapper_summary.yaml")

    # Table and metrics of features 
    excel_path = Path(out_path) / "metrics_global.xlsx"
    save_wrapper_summary_table(yaml_data, excel_path)
    df = pd.read_excel(excel_path)
    plot_metrics_heatmap_from_summary(df, Path(out_path))
    plot_feature_heatmap_from_summary(yaml_data, Path(out_path))

   # return 




