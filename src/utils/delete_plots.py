import os
import matplotlib.pyplot as plt
from pathlib import Path
from src.utils.io import get_br_id

def plot_single_model(dataset, solution, model_name, output_dir):

    # dataset_name = dataset.path.split("/")[-1].replace(".xls", "")
    dataset_name = Path(dataset.path).stem

    t_exp = dataset.t
    P_exp = dataset.data["P"]

    sol = solution["sol"]
    t_model = sol.t
    P_model = sol.y[2]

    plt.figure()

    plt.plot(t_exp, P_exp, "o", label="Experimental")
    plt.plot(t_model, P_model, "-", label=f"{model_name}")

    plt.xlabel("Time")
    plt.ylabel("P")
    plt.title(f"{dataset_name} - {model_name}")
    plt.legend()

    filepath = os.path.join(output_dir, f"{dataset_name}_{model_name}_P.png")
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(filepath)
    plt.close()


def plot_multi_dataset_model(datasets, solutions, model_name, output_dir, P_ML=None, dense=False):

    n = len(datasets)

    # Crear figura con subplots (2 filas x 3 columnas si hay 6 experimentos)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 8))
    axes = axes.flatten()

    for i, dataset in enumerate(datasets):
        ax = axes[i]

        dataset_key = dataset.path
        dataset_name = Path(dataset.path).stem

        # br_id = get_br_id(dataset) 
        # if br_id in ("BR07", "BR08"):
        #     t_exp = dataset.t[:-1]
        #     P_exp = dataset.data["P"][:-1]
        # else:
        t_exp = dataset.t
        P_exp = dataset.data["P"]

        if dense:
            sol = solutions[dataset_key]["dense"]
        else:
            sol = solutions[dataset_key]["sol"]
        
        t_model = sol.t
        if P_ML is not None and P_ML.get(dataset_key) is not None:
            P_model = P_ML[dataset_key]
        else:
            P_model = sol.y[2]

        if t_model is None or P_model is None:
            print(f"[ERROR] {model_name} - {dataset_name}: missing data")
            continue

        ax.plot(t_exp, P_exp, "o", label="Exp")
        ax.plot(t_model, P_model, "-", label=model_name)

        ax.set_title(dataset_name)
        ax.set_xlabel("Time")
        ax.set_ylabel("P")

    # Ocultar axes extra si hay menos de 6
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    # Leyenda global
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")

    fig.suptitle(f"{model_name} - All datasets", fontsize=16)

    plt.tight_layout()

    # Guardar
    filepath = Path(output_dir) / f"{model_name}_all_datasets.png"
    filepath.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(filepath, dpi=200)
    plt.close()


def plot_comparison(dataset, predictions, output_dir, MODEL_COLORS):
    
    # dataset_name = os.path.basename(dataset.path).replace(".xls","")
    dataset_name = Path(dataset.path).stem

    plt.figure(figsize=(6,5))

    # experimental
    plt.scatter(dataset.t, dataset.data["P"], color="black", label="Exp", zorder=3)

    for model_name in sorted(predictions):

        if model_name == "parametric":
            continue

        pred = predictions[model_name]

        plt.plot(
            pred["t"],
            pred["P"],
            label=model_name,
            color=MODEL_COLORS.get(model_name, None),
            alpha=0.9,
            linewidth=2
        )

    plt.xlabel("Time")
    plt.ylabel("P")
    plt.title(dataset_name)

    plt.legend(fontsize=8)
    plt.tight_layout()

    filepath = os.path.join(output_dir, f"{dataset_name}_comparison_P.png")
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(filepath)
    plt.close()


def plot_multibr_states_parametric(datasets, all_predictions, output_dir):

    variables = ["X", "S", "V"]

    n_rows = len(variables)
    n_cols = len(datasets)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4*n_cols, 3*n_rows),
        sharex=True
    )

    for col, dataset in enumerate(datasets):

        dataset_key = dataset.path
        dataset_name = os.path.basename(dataset.path).replace(".xls","")

        for row, var in enumerate(variables):

            ax = axes[row, col] if n_cols > 1 else axes[row]

            # Experimental
            ax.scatter(
                dataset.t,
                dataset.data[var],
                color="black",
                s=12, alpha=0.8,
                label="Exp" if (row == 0 and col == 0) else None
            )

            # SOLO modelo paramétrico
            if "parametric" in all_predictions[dataset_key]:

                pred = all_predictions[dataset_key]["parametric"]

                ax.plot(
                    pred["t"],
                    pred[var],
                    color="red",
                    linewidth=2,
                    label="Parametric" if (row == 0 and col == 0) else None
                )

            # títulos
            if row == 0:
                ax.set_title(dataset_name)

            if col == 0:
                ax.set_ylabel(var)

            ax.grid(alpha=0.3)

    # leyenda global
    handles, labels = axes[0, 0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        fontsize=9
    )

    plt.tight_layout(rect=[0, 0, 1, 0.93])

    filepath = os.path.join(output_dir, "multibr_XSV_parametric.png")
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(filepath)
    plt.close()


