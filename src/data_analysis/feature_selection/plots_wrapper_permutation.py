
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import seaborn as sns
from pathlib import Path
from src.utils.io import select_optimal_model_feature

# ------------- Plotting metrics by model ---------------
def plot_metric_comparison(all_results, metric_name, out_dir, model_type=None):

    palette = {
        "linear": "#071620",
        "LASSO_w": "#1a568f",
        "Ridge_w": "#479cd8",
        "elasticnet_w": "#728ea1",
        "tree": "#28d628",
        "rf_w": "#08580c",
        "gbm_w": "#598056",
        "svm_linear": "#f2f700",
        "LASSO_p": "#004b91",
        "Ridge_p": "#007fda",
        "elasticnet_p": "#728ea1",
        "svm_rbf": "#91872E",
        "svm_poly": "#a15100",
        "gpr": "#7C1C7C",
        "rf_p": "#cf52ee",
        "gbm_p": "#db1c1c",
        "mlp": "#e37777",
        "knn":"#856767",
    }

    plt.figure(figsize=(8, 5))

    # ranking
    model_scores = {
        m: np.mean([r["metrics"][metric_name] for r in res])
        for m, res in all_results.items()
    }

    ascending = metric_name in ["RMSE", "MSE", "MAE", "MAPE", "SCORE"]

    top_models = sorted(model_scores, key=model_scores.get, reverse=not ascending)[:3]

    for model_name, results in all_results.items():

        ks = np.array([r["k"] for r in results])
        values = np.array([r["metrics"][metric_name] for r in results])

        is_top = model_name in top_models

        plt.plot(
            ks,
            values,
            marker="o",
            markersize=5,
            linestyle="-" if is_top else "--",
            linewidth=1.2 if is_top else 1,
            alpha=1.0 if is_top else 0.3,
            color=palette.get(model_name, "gray") if is_top else "gray",
            label=model_name if is_top else None
        )

        # marcar mejor punto
        best, best_score = select_optimal_model_feature(results)
        plt.scatter(
            best["k"],
            best["metrics"][metric_name],
            s=50 if is_top else 30,
            edgecolor="black",
            zorder=3, color = "#45d46b"
        )

    plt.gca().invert_xaxis()

    plt.xlabel("# Features")
    plt.ylabel(metric_name)
   
    title_suffix = f" ({model_type.capitalize()})" if model_type else ""
    plt.title(f"{metric_name} vs #Features{title_suffix}")

    plt.legend(title=f"Top {model_type} models", bbox_to_anchor=(1.02, 1))

    plt.grid(alpha=0.3)
    plt.tight_layout()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.savefig(out_dir / f"{metric_name}_{model_type}_comparison.png", dpi=200)
    plt.close()


def plot_metrics_heatmap_from_summary(df, out_path):

    metrics_cols = ["R2", "SCORE"] #, "MAE", "MSE", "RMSE", "MAPE"
    metrics_cols = [c for c in metrics_cols if c in df.columns]

    plt.figure(figsize=(10, 6))

    sns.heatmap(
        df.set_index("model")[metrics_cols],
        cmap="coolwarm",
        cbar=False,
        annot=True,
        fmt=".3f"
    )

    plt.title("Metrics heatmap (models)")
    plt.tight_layout()

    plt.savefig(out_path / "metrics_heatmap.png", dpi=200)
    plt.close()


def plot_feature_heatmap_from_summary(yaml_data, out_path):

    all_features = set()
    for m in yaml_data["models"].values():
        all_features.update(m.get("features", []))

    all_features = sorted(list(all_features))

    # construir matriz binaria
    data = []

    for model_name, info in yaml_data["models"].items():
    # for model_name, info in yaml_data["best_info"].items():

        features = info.get("features", [])

        row = {
            f: 1 if f in features else 0
            for f in all_features
        }

        row["model"] = model_name
        data.append(row)

    df = pd.DataFrame(data).set_index("model")

    # plot
    plt.figure(figsize=(10, 6))

    sns.heatmap(
        df,
        cmap="Greens",
        cbar=False, # True,
        linewidths=0.5,
        linecolor="gray"
    )

    plt.title("Feature selection heatmap")
    plt.xlabel("Features")
    plt.ylabel("Model")

    plt.tight_layout()
    plt.savefig(out_path / "features_heatmap.png", dpi=200)
    plt.close()

    return df
