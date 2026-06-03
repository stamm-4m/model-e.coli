
import yaml
from pathlib import Path
import os
import numpy as np
import joblib
import pandas as pd
import time

# from sklearn.compose import TransformedTargetRegressor
# from sklearn.preprocessing import PowerTransformer


def load_yaml(path: str | Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)
    
def load_models_from_folder(base_path):

    base_path = Path(base_path)
    config = load_yaml(base_path / "wrapper_summary.yaml")
    models = {}

    for model_name, info in config["models"].items():

        model_path = base_path / "models" / f"{model_name}.pkl"

        models[model_name] = {
            "model": joblib.load(model_path),
            "features": info["features"]
        }
        
    return models, config


def save_yaml(data, filepath):
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    safe_data = to_python_type(data)
    with open(filepath, "w") as f:
        yaml.safe_dump(safe_data, f, sort_keys=False)


def get_time_ranges(yaml_dict: dict, br_id: str):
    try:
        time_sb = yaml_dict["bioreactor"][br_id]["t_sb"]["value"]
        time_ind = yaml_dict["bioreactor"][br_id]["t_ind"]["value"]
    except KeyError as e:
        raise KeyError(f"Missing time parameters for {br_id}: {e}")

    if time_sb >= time_ind:
        raise ValueError(
            f"Invalid time ranges for {br_id}: "
            f"t_sb ({time_sb}) >= t_ind ({time_ind})"
        )

    return time_sb, time_ind

def get_br_id(dataset):
    """
    Extract BR02, BR03, ... from dataset filename.
    """
    name = os.path.basename(dataset.path)
    return name.replace(".xls", "")


def to_python_type(obj):
    if isinstance(obj, dict):
        return {
            to_python_type(k): to_python_type(v)
            for k, v in obj.items()
        }
    elif isinstance(obj, list):
        return [to_python_type(v) for v in obj]

    elif isinstance(obj, tuple):
        return tuple(to_python_type(v) for v in obj)

    elif isinstance(obj, np.ndarray):
        return obj.tolist()

    elif isinstance(obj, np.generic):
        return obj.item()
    else:
        try:
            yaml.safe_dump(obj)
            return obj
        except Exception:
            return str(obj)

# ------------- ML Model Saving ---------------
def save_model(model, name, out_dir):

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, out_dir / f"{name}.pkl")

# ------------- Model Selection ---------------s

def select_optimal_model_feature(results):
    best_score = np.inf
    for r in results:
        # score = r["metrics"]["SCORE"]
        score = r["metrics"]["AIC"]
        if score < best_score:
            best_score = score
            best = r

    return best, best_score

# --------- Select best model ---------------


def select_best_models(cv_results, top_n=3):

    def ranking(item):
        _, info = item

        folds = info["folds"]

        r2_values = [f["metrics"]["R2"] for f in folds if "R2" in f["metrics"]]
        aic_values = [f["metrics"]["AIC"] for f in folds if "AIC" in f["metrics"]]

        if not r2_values or not aic_values:
            return (np.inf, np.inf, np.inf)

        r2_values = np.array(r2_values)

        r2_positive_count = np.sum(r2_values > 0)
        aic_mean = np.mean(aic_values)

        r2_penalty = np.sum(np.minimum(0, r2_values))

        return (-r2_positive_count, aic_mean, -r2_penalty)

    sorted_models = sorted(cv_results.items(), key=ranking)

    top_models = []
    for model_name, info in sorted_models[:top_n]:

        folds = info["folds"]

        r2_values = [f["metrics"]["R2"] for f in folds if "R2" in f["metrics"]]
        aic_values = [f["metrics"]["AIC"] for f in folds if "AIC" in f["metrics"]]

        r2_positive_count = np.sum(np.array(r2_values) > 0)
        aic_mean = np.mean(aic_values)

        top_models.append({
            "model_name": model_name,
            "r2_positive": r2_positive_count,
            "aic": aic_mean
        })

    return top_models

# ----Splitter -----
def custom_group_split(groups, fixed_group=None):

    groups = np.array(groups)
    unique_groups = sorted(set(groups))

    if fixed_group is None:
        test_groups = unique_groups  # Use all
    else:
        test_groups = [g for g in unique_groups if g != fixed_group]

    for test_g in test_groups:
        test_idx = np.where(groups == test_g)[0]
        train_idx = np.where(groups != test_g)[0]

        yield train_idx, test_idx


def stringify_params(params):
    clean = {}
    for k, v in params.items():
        try:
            yaml.safe_dump(v)  # test
            clean[k] = v
        except:
            clean[k] = str(v)
    return clean


# def make_positive_model(model):
#     return TransformedTargetRegressor(
#         regressor=model,
#         transformer=PowerTransformer(method='yeo-johnson')
#     )

# ----------- Hyperparameters optimization ----------

def get_param_grid(model_name):
    if model_name == "linear":
        return {}  # No hyperparams 
    
    elif model_name in ("LASSO_p", "LASSO_w"):
        return {
            "model__alpha": np.logspace(-4, 1, 20),  # 1e-4 → 10
            "model__max_iter": [10000]
        }
    
    elif model_name in ("Ridge_p", "Ridge_w"):
        return {
            "model__alpha": np.logspace(-4, 2, 20)  # 1e-4 → 100
        }
    
    elif model_name in ("elasticnet_w", "elasticnet_p"):
        return {
            "model__alpha": np.logspace(-4, 1, 10),
            "model__l1_ratio": [0.1, 0.5, 0.7, 0.9, 1.0],
            "model__max_iter": [20000] 
        }

    elif model_name == "tree":
        return {
             "max_depth": [3, 4, 5, 6, None],
             "min_samples_split": [2, 5, 10]
            # "max_depth": [2, 3, 4, 5, 6, 8, 10, None],
            # "min_samples_split": [2, 5, 10, 20],
            # "min_samples_leaf": [1, 2, 5]
        }

    elif model_name == "svm_linear":
        return{
                "model__C": [0.1, 1, 10, 100],
                "model__epsilon": [0.01, 0.1, 1]
                # "C": [0.01, 0.1, 1, 10, 100, 500],
                # "epsilon": [1e-4, 1e-3, 1e-2, 0.1, 1],
                # "gamma": ["scale", 0.1, 0.01, 0.001]
            }
    
    elif model_name == "svm_rbf":
        return {
                "model__C": [0.1, 1, 10, 100],
                "model__gamma": ["scale", "auto", 0.01, 0.001],
                "model__epsilon": [0.01, 0.1, 1]
            }

    elif model_name == "svm_poly":
        return{
                "model__C": [0.1, 1, 10],
                "model__degree": [2, 3, 4],
                "model__gamma": ["scale", "auto", 0.01, 0.001],
                "model__epsilon": [0.01, 0.1, 1]
            }

    elif model_name in ( "rf_w", "rf_p"):
        return {
            "n_estimators": [100, 200],
            "max_depth": [None, 5, 10],
            "min_samples_split": [2, 5],
            "max_features": ["sqrt", "log2"]
            # "n_estimators": [100, 200, 500],
            # "max_depth": [None, 5, 10, 20],
            # "min_samples_split": [2, 5, 10],
            # "min_samples_leaf": [1, 2, 5]
        }

    elif model_name in ("gbm_w", "gbm_p"):
        return {
            "n_estimators": [100, 200],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [3, 4, 5],
            "subsample": [0.7, 1.0]
            # "n_estimators": [100, 200, 500],
            # "learning_rate": [0.01, 0.05, 0.1],
            # "max_depth": [2, 3, 5],
            # "subsample": [0.7, 1.0]
        }
    
    elif model_name == "gpr":
        
        from sklearn.gaussian_process.kernels import (
            RBF, Matern, RationalQuadratic, WhiteKernel, ConstantKernel)

        return {
            "model__kernel": [
            #     ConstantKernel(1.0, (1e-6, 1e3)) *
            #     RBF(1.0, (1e-3, 1e3)) +
            #     WhiteKernel(1e-3, (1e-6, 1e1)),

                # ConstantKernel(1.0, (1e-6, 1e3)) *
                ConstantKernel(1.0, (1e-4, 1e1)) *
                # Matern(1.0, (1e-3, 1e3), nu=1.5) +
                Matern(2e-2, (1e-3, 0.1), nu=1.5) # +
                # WhiteKernel(1e-3, (1e-6, 1e1)),

            #     ConstantKernel(1.0, (1e-6, 1e3)) *
            #     Matern(1.0, (1e-3, 1e3), nu=2.5) +
            #     WhiteKernel(1e-3, (1e-6, 1e1)),

            #     ConstantKernel(1.0, (1e-6, 1e3)) *
            #     RationalQuadratic(length_scale=1.0, alpha=1.0) +
            #     WhiteKernel(1e-3, (1e-6, 1e1))
            # ],
            # "model__kernel": [
            #     ConstantKernel(1.0, (1e-4, 10.0)) * 
            #     RBF(1.0, (1e-3, 100.0))# + 
            #     # WhiteKernel(1e-3, (1e-7, 1.0))
            ],
            "model__alpha": [1e-4, 1e-2]
        }

    elif model_name == "mlp":
        return {
            "model__hidden_layer_sizes": [(50,), (100,)],#, (50,50)], #(100,50)
            "model__alpha": [0.01, 0.1, 1.0], # [1e-5, 1e-4, 1e-3],
            # "model__learning_rate_init": [1e-4, 1e-3],
            # "model__max_iter": [2000],
            "model__early_stopping": [True]
        }
    
    elif model_name == "knn":
        return {
            "model__n_neighbors": [3, 5, 7, 10],
            "model__weights": ["uniform", "distance"],
            "model__p": [1, 2]  # Manhattan / Euclidiana distance
        }


    else:
        return {}


def models_dict(model_names):

    from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet #, SGDRegressor
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.svm import SVR
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler # , MinMaxScaler
    from sklearn.neighbors import KNeighborsRegressor

    model_definitions  = {
        "linear": LinearRegression(positive=True),
        "LASSO_w": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Lasso(random_state=42))]),
        "Ridge_w": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(random_state=42))]),
        "elasticnet_w": Pipeline([
            ("scaler", StandardScaler()),
            ("model", ElasticNet(random_state=42))]),
        "tree": DecisionTreeRegressor(random_state=42),
        "rf_w": RandomForestRegressor(random_state=42),
        "gpm_w": GradientBoostingRegressor(random_state=42),
        "svm_linear": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVR(kernel="linear"))]),
        "LASSO_p": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Lasso(random_state=42))]),
        "Ridge_p": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(random_state=42))]),
        "elasticnet_p": Pipeline([
            ("scaler", StandardScaler()),
            ("model", ElasticNet(random_state=42))]),
        "svm_rbf": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVR(kernel = 'rbf'))]),
        "svm_poly": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVR(kernel = 'poly'))]),
        "gpr": Pipeline([
            ("scaler", StandardScaler()),
            ("model", GaussianProcessRegressor(random_state=42))]),
        "rf_p": RandomForestRegressor(random_state=42),
        "gbm_p": GradientBoostingRegressor(random_state=42),
        "mlp": Pipeline([
            ("scaler", StandardScaler()),
            ("model", MLPRegressor(random_state=42))]),
        "knn": Pipeline([
            ("scaler", StandardScaler()),  
            ("model", KNeighborsRegressor())]),
    }

    return {
            name: model_definitions[name] for name in model_names if name in model_definitions
        }

def save_wrapper_summary_table(yaml_data, filepath):

    rows = []

    for model_name, info in yaml_data["models"].items():

        row = {
            "model": model_name,
            "n_features": info.get("n_features", np.nan),
            "features": ", ".join(info.get("features", []))  
        }

        # métricas
        for metric, value in info["metrics"].items():
            row[metric] = value

        rows.append(row)

    df = pd.DataFrame(rows)

    # orden columnas
    base_cols = ["model", "n_features", "features"]
    metric_cols = [c for c in ["R2","MAE","MSE","RMSE","MAPE","SCORE","AIC","BIC"] if c in df.columns]

    df = df[base_cols + metric_cols]

    # ordenar (ej: RMSE)
    df = df.sort_values("RMSE")

    # guardar excel
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="summary", index=False)


def timer(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__}: {end-start:.2f} sec")
        return result
    return wrapper
