
import numpy as np

from sklearn.feature_selection import RFE
from sklearn.model_selection import GridSearchCV
from sklearn.base import clone

from src.utils.io import select_optimal_model_feature, custom_group_split, get_param_grid
from src.utils.metrics_io import compute_metrics

def backward_feature_selection(df, X_vars, y_var, c_var, models, summary, all_results):

    for name, model in models.items():

        results = backward_feature_analysis(df, X_vars, y_var, c_var, model, model_name=name)

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

def backward_feature_analysis(df, X_vars, y_var, c_var, model, model_name):

    y_orig = df[y_var].values

    if model_name in ("poisson", "tweedie"):
        y = y_orig
    else:
        y = np.log1p(y_orig)

    results = []

    # ---------------- Hyperparameter tuning ----------------
    param_grid = get_param_grid(model_name)
    groups = df["Run_ID"].values
    inner_cv = list(custom_group_split(groups, fixed_group=("BR09")))

    # importance_getter = get_importance_getter(best_model)
    if model_name in ("svm_linear", "poisson", "tweedie", "LASSO_b", "Ridge_b", "elasticnet_b"):
        importance_getter = lambda est: np.abs(est.named_steps["model"].coef_).ravel()
    elif model_name in ("tree", "rf_b", "gbm_b"):
        importance_getter = "feature_importances_"
    elif model_name == "linear":
        importance_getter = lambda est: np.abs(est.coef_).ravel()
        # importance_getter = "coef_"
    else:
        importance_getter = "auto"  

    selected_vars = list(X_vars)

    # ---------------- RFE loop ----------------
    # while len(selected_vars) >= 1:
    # while any(v not in c_var for v in selected_vars):
    
    while True:

        if len(selected_vars) == 0:
            break  
        
        X_sel = df[selected_vars].values

        # Tune
        grid = GridSearchCV(model, param_grid, cv=inner_cv,  # cv=5, 
                            scoring="neg_mean_squared_error",
                            n_jobs=-1)
        grid.fit(X_sel, y, groups=groups)
        current_model = grid.best_estimator_

        # Fit model with best hyperparameters
        y_pred = np.zeros_like(y, dtype=float)

        for train_idx, test_idx in custom_group_split(groups, "BR09"):
            model_clone = clone(current_model)
            model_clone.fit(X_sel[train_idx], y[train_idx])
            y_pred[test_idx] = model_clone.predict(X_sel[test_idx])

        use_log = model_name not in ("poisson", "tweedie")
        if use_log:
            y_pred = np.expm1(y_pred)

        mask = groups != "BR09"
        metrics = compute_metrics(y_orig[mask], y_pred[mask],len(selected_vars))

        results.append({
            "k": len(selected_vars),
            "features": selected_vars.copy(),
            "metrics": metrics,
            "best_params": current_model.get_params()
        })

        # ------ Stop condition --------
        # if len(selected_vars) == 1:
        #     break
        
        # ---- RFE ----
        # X_rfe = X_sel[mask]
        # y_rfe = y[mask]
        # rfe = RFE(estimator = current_model, 
        #     n_features_to_select = len(selected_vars) - 1,
        #     importance_getter = importance_getter)
        # rfe.fit(X_rfe, y_rfe)
        # selected_vars = [v for v, s in zip(selected_vars, rfe.support_) if s]

        # Backward elimination
        if callable(importance_getter):
            importances = importance_getter(current_model)
        elif importance_getter == "auto":
            if hasattr(current_model, "coef_"):
                importances = np.abs(current_model.coef_).ravel()
            else:
                importances = current_model.feature_importances_
        else:
            importances = getattr(current_model, importance_getter)

        feat_imp = list(zip(selected_vars, importances))
        removable = [(v, imp) for v, imp in feat_imp if v not in c_var]

        if set(selected_vars) == set(c_var):
            break

        if len(selected_vars) == 1 and len(c_var) == 0:
            break

        if len(removable) == 0:
            break

        var_to_remove = sorted(removable, key=lambda x: x[1])[0][0]
        selected_vars.remove(var_to_remove)

    return results
