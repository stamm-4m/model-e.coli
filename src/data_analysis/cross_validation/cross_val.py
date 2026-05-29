
import numpy as np
from pathlib import Path
import shutil

from sklearn.model_selection import GridSearchCV
from sklearn.base import clone

from src.utils.metrics_io import compute_metrics, save_metrics_tables_excel
from src.utils.io import (load_yaml, to_python_type, save_yaml, 
                custom_group_split, save_model, get_param_grid,
                select_best_model, select_optimal_model_feature, models_dict, timer) #, get_param_grid_importance , make_positive_model
from src.data_analysis.cross_validation.plots_cross_validation import plot_all_metrics, plot_predictions_by_model, plot_residuals_by_model

# from sklearn.model_selection import GroupKFold
# from sklearn.model_selection import cross_val_predict
# from sklearn.model_selection import RandomizedSearchCV
@timer
# --------- Cross - Validation global function ---------------
def cross_validation(df,y_var,in_dir,out_dir):

    print(f"Starting cross validation ... \n")

    out_dir = Path(out_dir)
    in_dir = Path(in_dir)

    config = load_yaml(in_dir / y_var / "wrapper_summary.yaml")

    # Model 
    model_names = config["models"].keys()
    # model_names = ["linear", "LASSO_w", "Ridge_w", "elasticnet_w", "tree", 
    #                "rf_w", "gbm_w", "svm_linear", "LASSO_p", "Ridge_p", 
    #                "elasticnet_p","svm_rbf", "svm_poly", "rf_p", 
    #                "gbm_p", "knn"] # "mlp", "gpr"
    
    model_objects = models_dict(model_names)

    # Build dict models + features
    models = {}

    for model_name, info in config["models"].items():
        if model_name not in model_objects: 
            continue

        models[model_name] = {
            "model": model_objects[model_name],
            "features": info["features"]
        }

    # CV
    print(f"Starting CV for {y_var} by all models ...")
    cv_results, predictions = evaluate_models_leave_one_run(df=df, models_dict=models, y_var=y_var, run_col="Run_ID")

    # ------------- Table --------------------
    save_metrics_tables_excel(cv_results, out_dir / "metrics_summary.xlsx")

    # SAVE results
    cv_results = to_python_type(cv_results)
    out_dir = Path(out_dir)
    target_dir = out_dir / y_var
    save_yaml(cv_results, target_dir / "cv_results_full.yaml")

    # Best model ("Cross-validated model deployment")
    best_model_name, best_score = select_best_model(cv_results)
    print(f"\nBest model: {best_model_name} ({best_score})")
    # train_and_save_best_model_per_fold(df, models, best_model_name, y_var, out_dir)
    train_and_save_best_model_per_fold_dynamic(df, models, y_var, out_dir)

    # Plots
    metrics = ["R2", "MAE", "MSE", "RMSE", "MAPE", "SCORE", "AIC", "BIC"]
    plots = ("boxplot", "ranking", "heatmap") # "by_run"
    plot_all_metrics(cv_results, metrics, target_dir / "metrics", plots)
    plot_predictions_by_model(predictions, target_dir / "predictions")
    plot_residuals_by_model(predictions, target_dir / "residuals")

    print(f"CV for {y_var} done \n")

    return cv_results

# --------- Cross-Validation specific folds ---------------

def evaluate_models_leave_one_run(df, models_dict, y_var, run_col):

    results = {}
    predictions = {}
    # runs = df[run_col].unique()
    groups = df[run_col].values


    for model_name, model_info in models_dict.items():

        base_model = model_info["model"]
        features = model_info["features"]
  
        X_all = df[features].values
        y_all = df[y_var].values
        time_all = df["time"].values
        
        # gkf = GroupKFold(n_splits=df[run_col].nunique())  

        fold_results = []
        predictions[model_name] = []

        # Choose grid by model
        param_grid = get_param_grid(model_name)

        # for fold, (train_idx, test_idx) in enumerate(gkf.split(X_all, y_all, groups)):
        for fold, (train_idx, test_idx) in enumerate(custom_group_split(groups, fixed_group="BR09")):

            X_train = X_all[train_idx]
            y_train = y_all[train_idx]

            X_test = X_all[test_idx]
            y_test = y_all[test_idx]

            test_runs = np.unique(groups[test_idx])
            time_test = time_all[test_idx]

            # Clone model
            model = clone(base_model)

            # -------- HYPERPARAMETER TUNING -------- 
            grid = None
            if param_grid:
                # inner_cv = GroupKFold(n_splits=5)
                # inner_cv = list(custom_group_split(groups[train_idx])) 
                inner_cv = list(custom_group_split(groups[train_idx], fixed_group="BR09"))
                grid = GridSearchCV(model, param_grid, cv=inner_cv,  
                                    scoring="neg_mean_squared_error",
                                    n_jobs=-1)
                # grid = RandomizedSearchCV(model,param_distributions=param_grid,
                #     n_iter=30,cv=inner_cv,scoring="neg_mean_squared_error",n_jobs=-1)
                grid.fit(X_train, y_train, groups=groups[train_idx])
                best_model = grid.best_estimator_
            else:
                best_model = model.fit(X_train, y_train)

            # -------- Predict --------
            y_pred = best_model.predict(X_test)
            # y_pred = np.maximum(0, y_pred)
            # y_pred = cross_val_predict(best_model, X_test, y_test,cv=inner_cv,groups=groups[train_idx])

            # # Inverse scaling
            # if model_name in ("gpr", "mlp", "sv_linear", "sv_rbf", "svm_poly"):  
            #     scaler = best_model.named_steps["scaler"]
            #     y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1)).ravel() #####

            k = len(features) 
            metrics = compute_metrics(y_test, y_pred, k)

            # best_params = grid.best_params_ if grid else {}

            fold_results.append({
                "fold": fold,
                "test_groups": list(test_runs),
                "metrics": metrics,
                # "best_params": best_params, # best_model.get_params(),
                # "best_score": grid.best_score_ if grid is not None else np.nan,
                # "cv_results": grid.cv_results_ if grid is not None else {}
            })

            predictions[model_name].append({
                "fold": fold,
                "test_groups": list(test_runs),
                "time": time_test,
                "y_test": y_test,
                "y_pred": y_pred
            })

        
        # summary_params = []

        # for f in fold_results:
        #     summary_params.append(f["best_params"])

        results[model_name] = {
            "folds": fold_results,
            # "best_params_per_fold": summary_params,
            "summary": aggregate_cv_results(fold_results)
        }

    return results, predictions


# --------- Cross-Validation add mean and std ---------------
def aggregate_cv_results(fold_results):
    metrics_keys = fold_results[0]["metrics"].keys()

    summary = {}
    for key in metrics_keys:
        values = [fold["metrics"][key] for fold in fold_results]
        summary[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values))
        }

    return summary


def train_and_save_best_model_per_fold(df, models, best_model_name, y_var, out_dir):

    model_info = models[best_model_name]
    base_model = model_info["model"]
    features = model_info["features"]

    X_all = df[features].values
    y_all = df[y_var].values
    groups = df["Run_ID"].values

    param_grid = get_param_grid(best_model_name)

    model_dir = Path(out_dir) / y_var / "best_model_folds"
    model_dir.mkdir(parents=True, exist_ok=True)

    for fold, (train_idx, test_idx) in enumerate(custom_group_split(groups, fixed_group="BR09")):

        X_train = X_all[train_idx]
        y_train = y_all[train_idx]
        test_group = np.unique(groups[test_idx])[0]
        model = clone(base_model)
        inner_cv = list(custom_group_split(groups[train_idx], fixed_group="BR09"))

        grid = GridSearchCV(model,param_grid,cv=inner_cv,
            scoring="neg_mean_squared_error",n_jobs=-1)
        grid.fit(X_train, y_train, groups=groups[train_idx])

        best_model = grid.best_estimator_
        best_params = grid.best_params_

        # SAVE MODEL
        model_name = f"{best_model_name}_{test_group}"
        save_model(best_model, model_name, model_dir)
        save_yaml(best_params, model_dir / f"{model_name}_params.yaml")
        save_yaml({
            "model": best_model_name,
            "test_group": test_group,
            "features": features
        }, model_dir / f"{model_name}_metadata.yaml")


def train_and_save_best_model_per_fold_dynamic(df, models, y_var, out_dir):

    model_dir = Path(out_dir) / y_var / "best_model_per_fold_dynamic"
    if model_dir.exists():
        shutil.rmtree(model_dir)

    model_dir.mkdir(parents=True, exist_ok=True)

    groups = df["Run_ID"].values

    for fold, (train_idx, test_idx) in enumerate(custom_group_split(groups, fixed_group="BR09")):

        results = []

        X_train_full = df.iloc[train_idx]
        y_train = X_train_full[y_var].values

        test_group = np.unique(groups[test_idx])[0]

        best_fold_score = np.inf
        best_fold_model = None
        best_fold_name = None
        best_fold_params = None
        best_features = None

        print(f"\nFold {fold} — Test group: {test_group}")

        # LOOP OVER ALL MODELS
        for model_name, model_info in models.items():

            base_model = model_info["model"]
            features = model_info["features"]

            X_train = X_train_full[features].values

            param_grid = get_param_grid(model_name)

            inner_groups = groups[train_idx]
            inner_cv = list(custom_group_split(inner_groups, fixed_group="BR09"))

            model = clone(base_model)

            grid = GridSearchCV(model,param_grid,cv=inner_cv,scoring="neg_mean_squared_error",n_jobs=-1)

            grid.fit(X_train, y_train, groups=inner_groups)

            best_model = grid.best_estimator_

            # ---- Evaluate on validation folds (approx proxy) ----
            
            X_test = df.iloc[test_idx][features].values
            y_test = df.iloc[test_idx][y_var].values
            y_pred = best_model.predict(X_test)

            metrics = compute_metrics(y_test, y_pred, k=len(features))

            results.append({
                "model_name": model_name,
                "model": grid.best_estimator_,
                "params": grid.best_params_,
                "features": features,
                "metrics": metrics})
            
        best, best_fold_score = select_optimal_model_feature(results)

        best_fold_model = best["model"]
        best_fold_name = best["model_name"]
        best_fold_params = best["params"]
        best_features = best["features"]

        # SAVE BEST MODEL OF THIS FOLD
        model_name = f"{best_fold_name}_{test_group}"

        save_model(best_fold_model, model_name, model_dir)

        save_yaml(best_fold_params, model_dir / f"{model_name}_params.yaml")

        save_yaml({
            "model": best_fold_name,
            "test_group": test_group,
            "features": best_features,
            "score": float(best_fold_score)
        }, model_dir / f"{model_name}_metadata.yaml")

        print(f"Best model for {test_group}: {best_fold_name}")
