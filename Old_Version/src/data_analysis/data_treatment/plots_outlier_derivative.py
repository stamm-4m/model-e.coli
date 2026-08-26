
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

# ---------- Plot outlier treatment for dynamic data -----------------

def plot_outlier_diagnostics(
    time, x, outliers, candidates, metrics,
    selected_method_per_outlier,
    x_replaced, x_smooth,
    method_style=None,
    save_dir=None, prefix="", has_outliers=False
):
    
    has_special_outliers = (
        metrics.get("special_outliers", {}).get("applied", False)
        and len(metrics.get("special_outliers", {}).get("details", {})) > 0
    )

    if method_style is None:
        METHOD_STYLE = {
            "movmean":   {"marker": "+", "color": "orange"},
            "movmedian":{"marker": "*", "color": "gold"},
            "gaussian": {"marker": "s", "color": "purple"},
            "rlowess":  {"marker": "D", "color": "green"},
            "sgolay":   {"marker": "v", "color": "red"},
        }
    else:
        METHOD_STYLE = method_style

    # ---------- FIG 1: RAW + OUTLIERS ----------
    if has_outliers:
        fig1, ax1 = plt.subplots(figsize=(10, 4))

        ax1.plot(time, x, "o", label="Raw data", markersize=4, alpha=0.4)
        ax1.plot(time[outliers], x[outliers], "x", color="red",
                label="Outlier", markersize=7)

        # Candidate methods
        for method, y_cand in candidates.items():
            style = METHOD_STYLE.get(method, {"marker": "o", "color": "black"})
            ax1.scatter(
                time[outliers],
                y_cand[outliers],
                marker=style["marker"],
                color=style["color"],
                s=40,
                label=method
            )

        ax1.set_title("Outliers and possible replacements")
        ax1.set_xlabel("Time")
        ax1.set_ylabel(prefix)
        ax1.legend(ncol=2)
        ax1.grid(True, alpha=0.3)

        fig1.tight_layout()

        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            fig1.savefig(f"{save_dir}/{prefix}_outlier_diagnosis.png",
                        dpi=300, bbox_inches="tight")
        plt.close(fig1)

    if has_outliers or has_special_outliers:
    # ---------- FIG 2: REPLACED + SMOOTHED ----------
        fig2, ax2 = plt.subplots(figsize=(10, 4))

        ax2.plot(time, x, "o", label="Raw data", alpha=0.4, markersize=4)

        special_indices = set()
        if metrics["special_outliers"]["applied"]:
            for block in metrics["special_outliers"]["details"].values():
                special_indices.update(
                    i for i in [
                        block.get("first_index"),
                        block.get("min_index"),
                        block.get("last_index"),
                        block.get("max_index")
                    ] if i is not None
                )

        for idx, method in selected_method_per_outlier.items():
            if idx in special_indices:
                continue
            style = METHOD_STYLE.get(method, {"marker": "o", "color": "black"})
            ax2.scatter(time[idx], x_replaced[idx],
                        marker=style["marker"],
                        color=style["color"],
                        s=40, alpha=0.5,
                        label=f"Method: {method}")
        if x_smooth is not None:
            ax2.plot(time, x_smooth, linewidth=2,
                    color="blue", linestyle="--",
                    label="Savitzky-Golay smoothing")

        # Special outliers
        already_labeled = False
        if metrics["special_outliers"]["applied"]:
            for block in metrics["special_outliers"]["details"].values():
                for idx in [block.get("first_index"), block.get("min_index"),
                            block.get("last_index"), block.get("max_index")]:
                    if idx is not None:
                        label = "Special outlier" if not already_labeled else None
                        ax2.plot(
                            time[idx],
                            x_replaced[idx],"x",
                            # s=50,
                            color="black",
                            label=label, markersize=7
                            # zorder=2
                        )
                        already_labeled = True

        handles, labels = ax2.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax2.legend(by_label.values(), by_label.keys(), ncol=2)

        ax2.set_title("Outlier replacement and smoothing")
        ax2.set_xlabel("Time")
        ax2.set_ylabel(prefix)
        ax2.grid(True, alpha=0.3)

        fig2.tight_layout()

        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            fig2.savefig(f"{save_dir}/{prefix}_replacement_smoothing.png",
                        dpi=300, bbox_inches="tight")

        plt.close(fig2)


# -------------- Derivatives plots ---------------------------

def plot_all_derivatives(t,results,variables,br_id,out_dir):
# results: output of compute_derivatives()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        nrows=3, ncols=len(variables),
        figsize=(5 * len(variables), 11),
        sharex="col"
    )
    for j, var in enumerate(variables):

        if var == "P":
            idx_min = results[var]["idx"]["start"]
            t_dense = np.linspace(t[idx_min], t.max(), 500)
        elif var == "S":
            idx_max = results[var]["idx"]["end"]
            t_dense = np.linspace(t.min(), t[idx_max], 500)
        else:
            t_dense = np.linspace(t.min(), t.max(), 500)

        f  = results[var]["f"]
        df  = results[var]["df"]
        d2f = results[var]["d2f"]

        # spline_cubic = results[var]["splines"]["Cubic"]
        spline_uni = results[var]["splines"]["Univariate"]

        # ---------- First row: Data ----------
        ax = axes[0, j]
        # ax.scatter(t, f["taylor"], color="black", s=25, zorder=2, label="Taylor")
        ax.scatter(t, f["grad"], color="grey", s=25, zorder=2, label="Gradient")
        # ax.scatter(t, f["cubic"], s=20, alpha=1, zorder=2, color="green")
        ax.scatter(t, f["uni"], s=20, zorder=2, color="red", label="UnivariateSpline")
        # ax.scatter(t, f["mean"], s=20, zorder=2, marker="D", color="purple")
        # ax.plot(t, f["mean"], marker="D", color="purple",linewidth=1, label="Mean Grad & UnivariateSpline")
        # ax.plot(t_dense, spline_cubic(t_dense), label="CubicSpline", linewidth=1.4, alpha=0.6, zorder=3, color="green")
        if br_id != "BR09":
            ax.plot(t_dense, spline_uni(t_dense), linestyle="--", linewidth=1.3, zorder=4, color="red")
        ax.set_title(f"{var}", fontsize=15)
        # axes[0].set_xlabel("time")
        # axes[0].legend()
        ax.grid(True, alpha=0.3)
        if j == 0:
            ax.set_ylabel("f", fontsize=13)

        # ---------- Second row: First derivative ----------
        ax = axes[1, j]
        # ax.scatter(t, df["taylor"], color="black", s=20)
        ax.scatter(t, df["grad"], color="grey", s=20)
        # ax.scatter(t, df["cubic"], s=20, color="green")
        ax.scatter(t, df["uni"], s=20, color="red")
        # ax.scatter(t, df["mean"], s=20, marker="D", color="purple")
        ax.plot(t, df["mean"], marker="D", color="purple",linewidth=1, label="Mean Grad & UnivariateSpline")
        # ax.plot(t, df["taylor"], color="black",linewidth=1)
        ax.plot(t, df["grad"], color="grey",linewidth=1)
        # ax.plot(t_dense, spline_cubic.derivative()(t_dense), label="CubicSpline", linewidth=1.4, alpha=0.6, zorder=3, color="green")
        if br_id != "BR09":
            ax.plot(t_dense, spline_uni.derivative()(t_dense), linestyle="--", linewidth=1.3, zorder=4, color="red")
        # axes[1].set_xlabel("time")
        # axes[1].set_title(f"{br_id} - {variable}")
        # axes[1].legend()
        ax.grid(True, alpha=0.3)
        if j == 0:
            ax.set_ylabel("f'", fontsize=13)

        # ---------- Third row: Second derivative ----------
        ax = axes[2, j]
        # ax.scatter(t, d2f["taylor"], color="black", s=20)
        ax.scatter(t, d2f["grad"], color="grey", s=20)
        # ax.scatter(t, d2f["cubic"], s=20, color="green")
        ax.scatter(t, d2f["uni"], s=20, color="red")
        # ax.scatter(t, d2f["mean"], s=20, marker="D", color="purple")
        ax.plot(t, d2f["mean"], marker="D", color="purple",linewidth=1)
        # ax.plot(t, d2f["taylor"], color="black",linewidth=1)
        ax.plot(t, d2f["grad"], color="grey",linewidth=1)
        # ax.plot(t_dense, spline_cubic.derivative(2)(t_dense), label="CubicSpline", linewidth=1.4, alpha=0.6, zorder=3, color="green")
        if br_id != "BR09":
            ax.plot(t_dense, spline_uni.derivative(2)(t_dense), linestyle="--", linewidth=1.3, zorder=4, color="red")
        ax.set_xlabel("time", fontsize=12)
        # axes[2].legend()
        ax.grid(True, alpha=0.3)
        if j== 0 :
            ax.set_ylabel("f''", fontsize=13)
    
    # ---------- Global title ----------
    fig.suptitle(f"{br_id} data and derivatives",fontsize=17)#, y=0.97 )

    # ---------- Global legend ----------
    fig.legend(
            ["Gradient points", "UnivariateSpline","Mean Grad & UnivariateSpline"],
            loc="upper center",
            bbox_to_anchor=(0.5, 0.95),
            ncol=4,
            fontsize=13,
            frameon=True)
    
    plt.tight_layout( rect=[0, 0, 1, 0.93] )

    # fig.subplots_adjust(top=0.88, hspace=0.18, wspace=0.25)

    plt.savefig(out_dir / "all_derivatives.png", dpi=150)
    plt.close()

