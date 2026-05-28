
import seaborn as sns
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import math
import numpy as np
from src.utils.metrics_io import compute_metrics

# --------- Cross-Validation plots ---------------
# These functions can be moved to a separate file if needed, but for now they are here for simplicity

def plot_all_metrics(cv_results, metrics, out_dir, plots=("by_run", "boxplot", "ranking", "heatmap")):

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

    marker = {
        "linear": "o", "LASSO_w": "o", "Ridge_w": "o", 
        "elasticnet_w": "o", "tree": "o",
        "rf_w": "o", "gbm_w": "o", "svm_linear": "o",
        
        "LASSO_p": "d", "Ridge_p": "d", "elasticnet_p": "d",
        "svm_rbf": "d", "svm_poly": "d", "gpr": "d",
        "rf_p": "d", "gbm_p": "d", "mlp": "d", "knn": "d"
    }
    
    for metric in metrics:
        if "by_run" in plots:
            plot_metric_by_run(cv_results, metric, palette, marker, out_dir)
        if "boxplot" in plots:
            plot_metric_boxplot(cv_results, metric, out_dir)
        if "ranking" in plots:
            plot_model_ranking(cv_results, metric, out_dir)
        if "heatmap" in plots:
            plot_heatmap_models_runs(cv_results, metric, out_dir)
            

# --------- Line plot of metric per run for each model ---------------
def plot_metric_by_run(cv_results, metric, palette, marker, out_dir=None):

    plt.figure(figsize=(8, 5))

    for model_name, data in cv_results.items():

        runs = [fold["test_groups"][0] for fold in data["folds"]]
        values = [fold["metrics"][metric] for fold in data["folds"]]

        plt.plot(runs, values, marker=marker[model_name], color=palette[model_name], label=model_name)

    plt.xlabel("Run ID")
    plt.ylabel(metric)
    plt.title(f"{metric} per Run (Leave-One-Run-Out)")
    plt.xticks(rotation=45)

    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")

    plt.grid(True, alpha=0.3)

    if out_dir != None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(Path(out_dir) / f"{metric}_per_run.png", dpi=150, bbox_inches="tight")
        plt.close()

# -------- Heatmap of metric per model and run ---------------
def plot_heatmap_models_runs(cv_results, metric, out_dir=None):

    data = []

    runs = None

    for model_name, res in cv_results.items():

        row = []
        runs = [f["test_groups"][0] for f in res["folds"]]

        for fold in res["folds"]:
            row.append(fold["metrics"][metric])

        data.append(row)

    df_heat = pd.DataFrame(data, index=cv_results.keys(), columns=runs)

    plt.figure(figsize=(8, 5))
    sns.heatmap(df_heat, annot=True, cmap="viridis")

    plt.title(f"{metric} heatmap (model vs test dataset)")

    if out_dir != None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(Path(out_dir) / f"{metric}_heatmap.png", dpi=150)
        plt.close()

# --------- Model ranking by mean metric ---------------

def plot_model_ranking(cv_results, metric, out_dir=None):

    models = []
    values = []
    stds = []

    for model_name, data in cv_results.items():
        models.append(model_name)
        values.append(data["summary"][metric]["mean"])
        stds.append(data["summary"][metric]["std"])

    if metric =="R2":
        sorted_idx = np.argsort(values)[::-1]
    else:
        sorted_idx = np.argsort(values)

    models = [models[i] for i in sorted_idx]
    values = [values[i] for i in sorted_idx]
    stds =   [stds[i] for i in sorted_idx]

    plt.figure(figsize=(10, 6))
    bars = plt.barh(models, values, xerr=stds, capsize=4, label="Mean ± std")

    for bar, value in zip(bars, values):
        plt.text(
            bar.get_width() + (0.01 * max(abs(np.array(values)))),  # dynamic offset
            bar.get_y() + bar.get_height()/2,
            f"{value:.2f}",
            va='center'
        )

    plt.xlabel(metric)
    plt.ylabel("Model")
    plt.title(f"Model ranking ({metric} CV mean across test folds)")
    plt.subplots_adjust(left=0.25)

    # Grid 
    plt.grid(axis='x', linestyle='--', alpha=0.4)

    if out_dir is not None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(Path(out_dir) / f"{metric}_ranking.png", dpi=150, bbox_inches='tight')
        plt.close()


# --------- Boxplot of metric distribution per model ---------------
def plot_metric_boxplot(cv_results, metric, out_dir=None):

    rows = []
    for model_name, data in cv_results.items():
        for fold in data["folds"]:
            rows.append({"model": model_name,"value": fold["metrics"][metric],
                "dataset": ",".join(map(str, fold["test_groups"]))})

    df = pd.DataFrame(rows)

    # -------------------------Detect bad datasets globally -------------------------
    Q1 = df["value"].quantile(0.25)
    Q3 = df["value"].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    bad_datasets = df[df["value"] < lower]["dataset"].value_counts()

    if not bad_datasets.empty:
        print("\nBAD DATASETS (frequent outliers):")
        for dataset, count in bad_datasets.head(10).items():
            print(f"  - {dataset}: {count} times")
    else:
        print("\nNo significant outliers detected")


    # ------------------------- Stability ranking -------------------------
    stability = df.groupby("model")["value"].std().sort_values()
    most_stable = stability.index[:3]  
    
    with plt.rc_context():
    # ------------------------- Plot (Seaborn style) -------------------------
        sns.set_theme(style="whitegrid", context="talk")
        plt.figure(figsize=(12, 7))
        ax = sns.boxplot( data=df, x="model", y="value")
        plt.xticks(rotation=45, ha="right")

        # ------------------------- Highlight stable models -------------------------
        for tick, label in zip(ax.get_xticks(), ax.get_xticklabels()):
            model = label.get_text()
            if model in most_stable:
                label.set_color("green")
                label.set_fontweight("bold")

        # ------------------------- Annotate outliers -------------------------
        # models = sorted(df["model"].unique())
        # for i, model in enumerate(models):
        models = [tick.get_text() for tick in ax.get_xticklabels()]
        for i, model in enumerate(models):
            subset = df[df["model"] == model]
            q1 = subset["value"].quantile(0.25)
            q3 = subset["value"].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outliers = subset[(subset["value"] < lower) | (subset["value"] > upper)]

            for _, row in outliers.iterrows():
                plt.text(i,row["value"]+ 0.02 * df["value"].std(),row["dataset"],fontsize=7,color="red",ha="center", alpha=0.8)

    # Labels
    plt.title(f"{metric} distribution by model\n(CV across test datasets)")
    plt.xlabel("Model")
    plt.ylabel(metric)
    # plt.legend(title="Model type")
    # plt.legend()
    plt.tight_layout()

    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(Path(out_dir) / f"{metric}_boxplot_advanced.png",
                    dpi=200, bbox_inches="tight")
        plt.close()

# --------- Plot predictions by model and run ---------------
def plot_predictions_by_model(predictions, out_dir=None):

    for model_name, runs in predictions.items():
        n = len(runs)
        cols = 3
        rows = math.ceil(n / cols)

        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
        axes = axes.flatten()

        for i, run_info in enumerate(runs):
            ax = axes[i]
            y_test = run_info["y_test"]
            y_pred = run_info["y_pred"]
            run_id = run_info["test_groups"][0]

            # LOCAL min/max
            vals = np.concatenate([y_test, y_pred])
            min_val = np.min(vals)
            max_val = np.max(vals)

            # add margin
            margin = 0.05 * (max_val - min_val + 1e-8)
            min_val -= margin
            max_val += margin

            # ------------------------- Scatter -------------------------
            ax.scatter(y_test, y_pred,alpha=0.6,edgecolor="k",linewidth=0.5,s=40)

            ax.plot([min_val, max_val],
                    [min_val, max_val],
                    "r--", linewidth=1.5, label="Ideal")

            # ------------------------- Metrics -------------------------
            metrics = compute_metrics(y_test, y_pred, 1) #
            r2 = metrics["R2"]
            rmse = metrics["RMSE"]

            if r2 < 0:
                ax.set_facecolor("#ffe6e6")  # fondo rojo suave

            ax.text(0.05, 0.90,f"R²={r2:.2f}\nRMSE={rmse:.2f}",
                    transform=ax.transAxes,fontsize=9,
                    bbox=dict(facecolor='white', alpha=0.7))

            ax.set_xlim(min_val, max_val)
            ax.set_ylim(min_val, max_val)#
            ax.set_title(f"{run_id}", fontsize=11)
            ax.set_aspect('equal', adjustable='box')
            ax.grid(alpha=0.3)

            if i % cols == 0:
                ax.set_ylabel("y_pred")
            else:
                ax.set_ylabel("")

            if i >= (rows - 1) * cols:
                ax.set_xlabel("y_test")
            else:
                ax.set_xlabel("")

        for j in range(i+1, len(axes)):
            axes[j].axis("off")

        fig.suptitle(f"Predictions vs True Values — {model_name}",fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        if out_dir is not None:
            out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(out_dir / f"{model_name}_predictions.png",
                        dpi=300, bbox_inches="tight")
            plt.close()
    
        # --------- Time series plot ---------------
        fig_ts, axes_ts = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        axes_ts = axes_ts.flatten()

        for i, run_info in enumerate(runs):
            ax = axes_ts[i]

            y_test = run_info["y_test"]
            y_pred = run_info["y_pred"]

            # If you have time, use it. Otherwise use index
            if "time" in run_info:
                x = run_info["time"]
            else:
                x = np.arange(len(y_test))

            # ax.plot(x, y_test, label="y_true", linewidth=2)
            # ax.plot(x, y_pred, label="y_pred", linestyle="--")
            ax.scatter(x,y_test,alpha=0.6, label="y_true",edgecolor="k",linewidth=0.5,s=40)
            ax.scatter(x, y_pred,alpha=0.6, label="y_true",edgecolor="k",linewidth=0.5,s=40)

            ax.set_title(f"{run_info['test_groups'][0]}", fontsize=11)
            ax.grid(alpha=0.3)

            if i % cols == 0:
                ax.set_ylabel("Value")
            else:
                ax.set_ylabel("")

            if i >= (rows - 1) * cols:
                ax.set_xlabel("Time [h]")
            else:
                ax.set_xlabel("")

            ax.legend(fontsize=8)

        # Turn off unused axes
        for j in range(i+1, len(axes_ts)):
            axes_ts[j].axis("off")

        fig_ts.suptitle(f"Time Series - {model_name}", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        if out_dir is not None:
            plt.savefig(out_dir / f"{model_name}_timeseries.png",
                        dpi=300, bbox_inches="tight")
            plt.close()


# ------- residuals plots -----------
def plot_residuals_by_model(predictions, out_dir=None):

    for model_name, runs in predictions.items():

        n = len(runs)
        cols = 3
        rows = math.ceil(n / cols)

        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        axes = axes.flatten()

        for i, run_info in enumerate(runs):

            ax = axes[i]

            y_test = run_info["y_test"]
            y_pred = run_info["y_pred"]
            run_id = run_info["test_groups"][0]

            residuals = y_test - y_pred

            # Scatter
            ax.scatter(y_pred, residuals, alpha=0.6, edgecolor="k")

            # Línea horizontal (ideal)
            ax.axhline(0, color="red", linestyle="--")

            # Métricas rápidas
            mean_res = np.mean(residuals)
            std_res = np.std(residuals)

            ax.text(
                0.05, 0.9,
                f"μ={mean_res:.2f}\n $sigma$={std_res:.2f}",
                transform=ax.transAxes,
                fontsize=9,
                bbox=dict(facecolor="white", alpha=0.7)
            )

            ax.set_title(f"{run_id}")
            ax.set_xlabel("y_pred")
            ax.set_ylabel("Residual (y - y_pred)")
            ax.grid(alpha=0.3)

        # ocultar vacíos
        for j in range(i+1, len(axes)):
            axes[j].axis("off")

        fig.suptitle(f"Residual plots — {model_name}", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        if out_dir:
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            plt.savefig(Path(out_dir) / f"{model_name}_residuals.png",
                        dpi=300, bbox_inches="tight")
            plt.close()



# def plot_metric_boxplot(cv_results, metric, palette, marker, out_dir=None):

#     rows = []

#     for model_name, data in cv_results.items():
#         for fold in data["folds"]:
#             rows.append({"model": model_name, "value": fold["metrics"][metric],
#                 "dataset": ",".join(map(str, fold["test_groups"]))})

#     df_plot = pd.DataFrame(rows)
#     plt.figure(figsize=(10, 6))

#     # Boxplot
#     ax = df_plot.boxplot(by="model", column="value", grid=False)
#     plt.xticks(rotation=45, ha="right")

# # ------------------------- Detect outliers (IQR) -------------------------
#     for i, model in enumerate(df_plot["model"].unique(), start=1):
#         subset = df_plot[df_plot["model"] == model]
#         values = subset["value"]
#         q1 = values.quantile(0.25)
#         q3 = values.quantile(0.75)
#         iqr = q3 - q1
#         lower = q1 - 1.5 * iqr
#         upper = q3 + 1.5 * iqr
#         outliers = subset[(values < lower) | (values > upper)]
# # -------------------------Annotate outliers -------------------------
#         for _, row in outliers.iterrows():
#             plt.text(i,row["value"],row["dataset"],fontsize=8,color="red",ha="center")

#     # Titles
#     plt.title(f"{metric} distribution by model")
#     plt.suptitle("")
#     plt.xlabel("Model")
#     plt.ylabel(metric)
#     plt.tight_layout()

#     if out_dir is not None:
#         Path(out_dir).mkdir(parents=True, exist_ok=True)
#         plt.savefig(Path(out_dir) / f"{metric}_boxplot.png", dpi=150, bbox_inches="tight")
#         plt.close()