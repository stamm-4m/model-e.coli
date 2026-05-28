
import numpy as np

from sklearn.feature_selection import RFE
from sklearn.model_selection import GridSearchCV
from sklearn.base import clone

from src.utils.io import select_optimal_model_feature, custom_group_split, get_param_grid
from src.utils.metrics_io import compute_metrics

def wrapper_feature_selection(df, X_vars, y_var, models, summary, all_results):

    for name, model in models.items():

        results = rfe_analysis(df, X_vars, y_var, model, model_name=name)

        print(f"Features selection for {name} model done")

        all_results[name] = results

        best, best_score = select_optimal_model_feature(results)

        # Saving the best model
        # X_sel = df[best["features"]].values
        # y = df[y_var].values
        # model.fit(X_sel, y)
        # save_model(model, name, out_dir/"models")

        summary[name] = {
            "n_features": len(best["features"]),
            "features": best["features"],
            "metrics": best["metrics"]
        }

    return summary, all_results

# ---------------- RFE analysis for linear, tree and SVM---------------

def rfe_analysis(df, X_vars, y_var, model, model_name):

    y = df[y_var].values

    results = []

    # ---------------- Hyperparameter tuning ----------------
    param_grid = get_param_grid(model_name)
    groups = df["Run_ID"].values
    inner_cv = list(custom_group_split(groups, fixed_group="BR09"))

    # importance_getter = get_importance_getter(best_model)
    if model_name in ("svm_linear", "LASSO_w", "Ridge_w", "elasticnet_w"):
        importance_getter = lambda est: est.named_steps["model"].coef_
    elif model_name in ("tree", "rf_w", "gbm_w"):
        importance_getter = "feature_importances_"
    elif model_name == "linear":
        importance_getter = "coef_"
    else:
        importance_getter = "auto"  

    selected_vars = list(X_vars)

    # ---------------- RFE loop ----------------
    k = 1
    while len(selected_vars) >= 1:
        X_sel = df[selected_vars].values

        # Tune
        grid = GridSearchCV(model, param_grid, cv=inner_cv,  # cv=5, 
                            scoring="neg_mean_squared_error",
                            n_jobs=-1)
        grid.fit(X_sel, y, groups=groups)
        current_model = grid.best_estimator_

        # Fit model with best hyperparameters
        # current_model.fit(X_sel, y)
        y_pred = np.zeros_like(y, dtype=float)

        for train_idx, test_idx in custom_group_split(groups, "BR09"):
            model_clone = clone(current_model)
            model_clone.fit(X_sel[train_idx], y[train_idx])
            y_pred[test_idx] = model_clone.predict(X_sel[test_idx])

        mask = groups != "BR09"
        metrics = compute_metrics(y[mask], y_pred[mask], k)

        results.append({
            "k": k,
            "features": selected_vars,
            "metrics": metrics,
            "best_params": current_model.get_params()
        })

        # ------ Stop condition --------
        if len(selected_vars) == 1:
            break
        
        # ---- RFE ----
        X_rfe = X_sel[mask]
        y_rfe = y[mask]
        rfe = RFE(estimator = current_model, 
            n_features_to_select = len(selected_vars) - 1,
            importance_getter = importance_getter)
        rfe.fit(X_rfe, y_rfe)
        selected_vars = [v for v, s in zip(selected_vars, rfe.support_) if s]
        
        k += 1

    return results
