
import numpy as np

from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV
from sklearn.base import clone

from src.utils.io import select_optimal_model_feature, custom_group_split, get_param_grid # , stringify_params
from src.utils.metrics_io import compute_metrics

def permutation_feature_selection(df, X_vars, y_var, models, summary, all_results):

    for name, model in models.items():

        # if name in ("mlp", "gpr", "svm_rbf", "svm_poly"):
        #     importance_type = "permutation"
        # else:
        #     importance_type = "model"
        importance_type = "permutation"

        results = permutation_importance_analysis(df, X_vars, y_var, model, 
                                model_name = name, 
                                importance_type = importance_type) 
                                
        print(f"Features selection for {name} model done")

        all_results[name] = results

        best, best_score = select_optimal_model_feature(results)

        summary[name] = {
            "n_features": len(best["features"]), # ["k"]
            "features": best["features"],
            "metrics": best["metrics"]
        }

    return summary, all_results

# ------------- permutation importance for  RF, GBM and MLP ---------------
def permutation_importance_analysis(df, X_vars, y_var, model, model_name, importance_type):

    y = df[y_var].values
    results = []

    # -------- Initial hyperparameter tuning --------
    param_grid = get_param_grid(model_name)
    groups = df["Run_ID"].values
    inner_cv = list(custom_group_split(groups, fixed_group="BR09"))

    # -------- Backward elimination loop --------
    remaining = list(X_vars)

    # for k in range(1, len(X_vars) + 1):
    while len(remaining) >= 1:
        X_sel = df[remaining].values

        # Tune at each subset
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
        metrics = compute_metrics(y[mask],y_pred[mask], len(remaining))
        # metrics = compute_metrics(y, y_pred, len(remaining))

        # -------- Importance ----------
        if hasattr(current_model, "named_steps"):
            # final_model = list(best_model.named_steps.values())[-1]
            final_model = current_model.named_steps["model"]
            params = current_model.named_steps["model"].get_params()
        else:
            final_model = current_model
            params = current_model.get_params()
        
        kernel_str = str(getattr(final_model, "kernel", None))

        # if importance_type == "model":
        #     if hasattr(final_model, "feature_importances_"):
        #         importances = final_model.feature_importances_
        #     elif hasattr(final_model, "coef_"):
        #         importances = np.abs(final_model.coef_).ravel()
            
        # elif importance_type == "permutation":
        result = permutation_importance(current_model, X_sel[mask], y[mask], n_repeats=5,
                                        random_state=42, n_jobs=-1)
        importances = result.importances_mean 
        importances_std = result.importances_std

        # -------- Ranking --------
        ranked = sorted( zip(remaining, importances), key=lambda x: x[1], reverse=True )
        
        results.append({
            "k": len(remaining),
            "features": remaining.copy(),
            "metrics": metrics,
            "ranking": ranked,
            "importances": dict(zip(remaining, importances)),
            "importances std": dict(zip(remaining, importances_std)),
            # "best_params": best_model.get_params(),
            # "best_params": stringify_params(params),
            "kernel": kernel_str,
            "alpha": final_model.alpha if hasattr(final_model, "alpha") else None
        })

        # Remove the least important feature
        worst_feature = ranked[-1][0]
        remaining.remove(worst_feature)

    return results

