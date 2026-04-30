
# import pandas as pd
import numpy as np
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

#------------- Heatmaps ------------------

def heatmap_global(df, variables ,save_dir=None, threshold=0.8):
    num_cols = get_numeric_columns(df, variables)
    corr = df[num_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=False, cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Heatmap Global Correlation")

    for i in range(len(corr.columns)):
            for j in range(len(corr.columns)):
                if i != j and abs(corr.iloc[i, j]) > threshold:
                    ax.text(
                        j + 0.5,
                        i + 0.5,
                        f"{corr.iloc[i, j]:.2f}",
                        ha="center",
                        va="center",
                        color="black",
                        fontsize=9,
                        fontweight="bold"
                    )

    plt.tight_layout()

    if save_dir:
            savepath = f"{save_dir}/Heatmap/heatmap_global.png"
            plt.savefig(savepath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    #plt.show()

def heatmap_per_run(df, variables ,save_dir=None, threshold=0.8):
    num_cols = get_numeric_columns(df, variables)

    for run_id, sub in df.groupby("Run_T"):
        corr = sub[num_cols].corr()

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, annot=False, cmap="coolwarm", center=0, ax=ax)
        ax.set_title(f"Heatmap {run_id}")

        for i in range(len(corr.columns)):
                    for j in range(len(corr.columns)):
                        if i != j and abs(corr.iloc[i, j]) > threshold:
                            ax.text(
                                j + 0.5,
                                i + 0.5,
                                f"{corr.iloc[i, j]:.2f}",
                                ha="center",
                                va="center",
                                color="black",
                                fontsize=9,
                                fontweight="bold"
                            )

        plt.tight_layout()

        if save_dir:
            savepath = f"{save_dir}/Heatmap/heatmap_{run_id}.png"
            plt.savefig(savepath, dpi=300, bbox_inches="tight")
        plt.close(fig)
        #plt.show()

#------------- PCA ------------------

def pca_global(df, variables,save_dir=None):
    num_cols = get_numeric_columns(df, variables)

    df = df.dropna()
    df = df.sort_values("time")

    time = df["time"] 
    num_cols = [c for c in num_cols if c != "time"]
    X = df[num_cols]
    

    X_scaled = StandardScaler().fit_transform(X)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(8, 6))
    # ax.plot(X_pca[:, 0], X_pca[:, 1], "--k", alpha=0.4) # time trajectory

    sc = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=time, cmap="viridis", alpha=0.7)
    plt.colorbar(sc, ax=ax, label="Time")

    draw_biplot_vectors(ax, pca, num_cols, scale=3)

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA Global (colored by time + Biplot)")
    ax.grid(True)

    if save_dir:
            scree_plot(
                 pca,
                 savepath = f"{save_dir}/PCA/PCA_global_scree.png",
                 title="Global PCA - Scree plot"
                 )

            savepath = f"{save_dir}/PCA/PCA_global.png"
            plt.savefig(savepath, dpi=300, bbox_inches="tight")
    plt.close()
    # plt.show()
    print("Explained variance ratio (PCA):", pca.explained_variance_ratio_)

def pca_per_run(df, variables,save_dir=None):
    num_cols = get_numeric_columns(df, variables)

    for run_id, sub in df.groupby("Run_T"):
        sub = sub[num_cols].dropna()
        if len(sub) < 2:
            continue
        
        sub = sub.sort_values("time") # Line trajectory

        time = sub["time"]
        num_cols_new = [c for c in num_cols if c != "time"]
        X = sub[num_cols_new]

        X_scaled = StandardScaler().fit_transform(X)

        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        fig, ax = plt.subplots(figsize=(7, 6))
        
        ax.plot(X_pca[:, 0], X_pca[:, 1], "--k", alpha=0.4) # time trajectory

        sc = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=time, cmap="plasma", alpha = 0.8)
        plt.colorbar(sc, ax=ax, label="time")

        draw_biplot_vectors(ax, pca, num_cols_new, scale=3)

        ax.set_title(f"PCA {run_id} (colored by time + Biplot)")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(True)

        if save_dir:
            scree_plot(
                pca,
                savepath=f"{save_dir}/PCA/PCA_{run_id}_scree.png",
                title=f"PCA {run_id} - Scree plot"
            )

            savepath = f"{save_dir}/PCA/PCA_{run_id}.png"
            plt.savefig(savepath, dpi=300, bbox_inches="tight")
        plt.close()
        # plt.show()

# ----------------Scree plot in PCA--------------------
def scree_plot(pca, savepath=None, title = "Scree plot"):
    explained_var = pca.explained_variance_ratio_
    cum_var = np.cumsum(explained_var)

    fig, ax = plt.subplots(figsize=(6, 4))
    x = range(1, len(explained_var) + 1)

    ax.bar(x, explained_var, alpha=0.7, label="Individual")
    ax.plot(x, cum_var, marker="o", label="Cumulative")

    plt.xlabel("Principal Component")
    plt.ylabel("Explained Variance Ratio")
    plt.title(title)
    plt.grid(True)
    plt.legend()

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    if savepath:
        plt.savefig(savepath, dpi=300, bbox_inches="tight")
    plt.close()

# ---------------- Biplot (PCA)--------------------

def draw_biplot_vectors(ax, pca, feature_names, scale=3):
    loadings = pca.components_.T

    for i, var in enumerate(feature_names):
        ax.arrow(
            0, 0,
            loadings[i, 0] * scale,
            loadings[i, 1] * scale,
            color="red",
            alpha=0.7,
            head_width=0.05,
            length_includes_head=True
        )
        ax.text(
            loadings[i, 0] * scale * 1.1,
            loadings[i, 1] * scale * 1.1,
            var,
            color="red",
            fontsize=9
        )

#------------- Boxplots ------------------

def boxplot_global(df, variables,save_dir=None):
    num_cols = get_numeric_columns(df, variables)

    for col in num_cols:
        df[col].plot(kind="box", figsize=(10, 6))
        plt.title(f"Boxplot Global of {col}")
        plt.xticks(rotation=45)
        plt.tight_layout()
        if save_dir:
                savepath = f"{save_dir}/Boxplot/boxplot_global_{col}.png"
                plt.savefig(savepath, dpi=300, bbox_inches="tight")
        plt.close()
            # plt.show()


def boxplot_por_run(df, variables, save_dir=None):
    num_cols = get_numeric_columns(df, variables)

    for col in num_cols:
        plt.figure(figsize=(8, 5))
        sns.boxplot(x="Run_T", y=col, data=df)
        plt.title(f"Boxplot of {col}")
        plt.xticks(rotation=45)
        plt.tight_layout()
        if save_dir:
            savepath = f"{save_dir}/Boxplot/boxplot_{col}.png"
            plt.savefig(savepath, dpi=300, bbox_inches="tight")
        plt.close()
        # plt.show()

# --------- función para obtener las columnas numericas -----------

def get_numeric_columns(df, variables):
    # return df.select_dtypes(include=[np.number]).columns.tolist()
    return [
        col for col in variables
        if col in df.columns # and pd.api.types.is_numeric_dtype(df[col])
    ]


# ------------- Otras funciones que pueden ser utiles--------------

# -------------  Series temporales superpuestas --------------
def timeseries_per_run(df, variable, save_dir=None):

    # plt.figure(figsize=(8, 5))
    fig, ax = plt.subplots(figsize=(8, 5))

    for run_id, sub in df.groupby("Run_T"):   
        ax.scatter(sub["time"], sub[variable], label = run_id, s=20, alpha=0.7)
        # ax.plot(sub["time"], sub[variable], label=run_id)
    
    ax.set_title(f"{variable} vs time")
    ax.set_xlabel("time (h)")
    ax.set_ylabel(variable)
    ax.legend()
    ax.grid(True)
    plt.tight_layout()

    if save_dir:
            savepath = f"{save_dir}/timeseries_{variable}.png"
            plt.savefig(savepath, dpi=300, bbox_inches="tight")

    plt.close(fig)

    # Ejemplo uso:
    # timeseries_per_run(df, "X")

# ------------- Scatter para pares de variables --------------
def scatter_fun(df, x, y, save_dir=None):
    # plt.figure(figsize=(6, 5))

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(
        data=df,
        x=x,
        y=y,
        hue="Run_T",
        ax=ax
    )

    ax.set_title(f"{y} vs {x}")
    ax.grid(True)
    plt.tight_layout()

    if save_dir:
            savepath = f"{save_dir}/scatter_{x}_{y}.png"
            plt.savefig(savepath, dpi=300, bbox_inches="tight")

    plt.close(fig)

    # Ejemplo uso:
    # scatter_ode(df, "V", "qP")

# ----------Otro boxplot----------------------
    
def boxplots_by_phase(
    dfs_by_phase,
    numeric_features,
    output_dir
):
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for phase, df in dfs_by_phase.items():
        fig, ax = plt.subplots(figsize=(12, 6))

        df[numeric_features].boxplot(ax=ax)
        ax.set_title(f"Boxplots - {phase}")
        ax.set_ylabel("Value")
        ax.tick_params(axis="x", rotation=45)

        plt.tight_layout()
        fig.savefig(f"{output_dir}/boxplot_{phase}.png")
        plt.close(fig)
