
# from glob import glob
import numpy as np
import pandas as pd
from fedbatch.estimation.datasets import ExperimentDataset
from fedbatch.utils.io import get_br_id
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def run_EAD(df,save_dir=None):
    heatmap_global(df,save_dir)
    heatmap_per_run(df,save_dir)

    pca_global(df,save_dir)
    pca_per_run(df,save_dir)

    boxplot_global(df,save_dir)
    boxplot_por_run(df,save_dir)


def unificar_xls(dataset_files):
    
    # dataset_files = sorted(glob("data/raw/BR*.xls"))
    
    datasets = [ExperimentDataset(f) for f in dataset_files]
    dfs = []

    for dataset in datasets:

        df = dataset.df.copy()

        # Indicar el origen
        br_id = get_br_id(dataset)
        df["Run_ID"] = br_id
        df.insert(0, "Run_ID", df.pop("Run_ID"))

        # Calculo de mu & qp
        df = calcular_mu_qp(df)

        # # Ordenar por tiempo
        # df = df.sort_values("time").reset_index(drop=True)

        dfs.append(df)

    # Unificación final
    df_final = pd.concat(dfs, ignore_index=True)

    return df_final


def calcular_mu_qp(df):
    df = df.sort_values("time").copy()
    n = len(df)

    mu = np.zeros(n)
    qp_old = np.zeros(n)
    qp = np.zeros(n)

    dXdt = 0
    dVdt = 0
    dPdt = 0

    t = df["time"].values
    X = df["X"].values
    V = df["V"].values
    P = df["P"].values

    for i in range(n):
    # Primera fila 
        if i == 0:
            h = t[i+1] - t[i]
            h2 = t[i+2] - t[i]
            alpha = h2/h

            dXdt = (1/h) * (1/alpha) * (1/(1-alpha)) * ( X[i+2] - (alpha**2 * X[i+1]) - (1-alpha**2) * X[i] )
            dVdt = (1/h) * (1/alpha) * (1/(1-alpha)) * ( V[i+2] - (alpha**2 * V[i+1]) - (1-alpha**2) * V[i] )
            dPdt = (1/h) * (1/alpha) * (1/(1-alpha)) * ( P[i+2] - (alpha**2 * P[i+1]) - (1-alpha**2) * P[i] )

    # Última fila 
        elif i == n - 1:
            h =  t[i] - t[i-1]
            h2 = t[i] - t[i-2]
            alpha = h2/h

            dXdt = (1/h) * (1/alpha) * (1/(alpha-1)) * ( (alpha**2-1) * X[i] - (alpha**2 * X[i-1]) + X[i-2] )
            dVdt = (1/h) * (1/alpha) * (1/(alpha-1)) * ( (alpha**2-1) * V[i] - (alpha**2 * V[i-1]) + V[i-2] )
            dPdt = (1/h) * (1/alpha) * (1/(alpha-1)) * ( (alpha**2-1) * P[i] - (alpha**2 * P[i-1]) + P[i-2] )
    # Datos intermedios
        else:
            h = t[i+1] - t[i]
            h2 = t[i] - t[i-1]
            alpha = h2/h

            dXdt = (1/h) * (1/alpha) * (1/(1+alpha)) * ( (alpha**2 * X[i+1]) + (1-alpha**2) * X[i] - X[i-1] )
            dVdt = (1/h) * (1/alpha) * (1/(1+alpha)) * ( (alpha**2 * V[i+1]) + (1-alpha**2) * V[i] - V[i-1] )
            dPdt = (1/h) * (1/alpha) * (1/(1+alpha)) * ( (alpha**2 * P[i+1]) + (1-alpha**2) * P[i] - P[i-1] )

        mu[i]     = (1/X[i]) * ( dXdt ) + (1/V[i]) * ( dVdt )
        qp_old[i] = (1/X[i]) * ( dPdt )
        qp[i]     = (1/X[i]) * ( dPdt + (dVdt * P[i] / V[i]) - (mu[i]*P[i]) )

    df["mu"] = mu
    df["qP_old"] = qp_old
    df["qP"] = qp

    return df


def agregar_T_ind(df,n_ultimos=4):

    # Asegurar orden temporal
    last_T = (
        df
        .sort_values("time")
        .groupby("Run_ID")["T"] 
        # .last()
        .apply(lambda s: s.tail(n_ultimos).mean()) # últimos 4 valores
        .round(1)
        .astype(str)  # convertir a str
    )

    # Asignar a todas las filas
    df["T_ind"] = df["Run_ID"].map(last_T)
    df["Run_T"] = df["Run_ID"].astype(str) + "_T_" + df["T_ind"]

    return df

#------------- Heatmaps ------------------

def heatmap_global(df,save_dir=None, threshold=0.8):
    num_cols = get_numeric_columns(df)
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
            savepath = f"{save_dir}/heatmap_global.png"
            plt.savefig(savepath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    #plt.show()

def heatmap_per_run(df,save_dir=None, threshold=0.8):
    num_cols = get_numeric_columns(df)

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
            savepath = f"{save_dir}/heatmap_{run_id}.png"
            plt.savefig(savepath, dpi=300, bbox_inches="tight")
        plt.close(fig)
        #plt.show()

#------------- PCA ------------------

def pca_global(df,save_dir=None):
    num_cols = get_numeric_columns(df)

    X = df[num_cols].dropna()
    X_scaled = StandardScaler().fit_transform(X)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.5)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA Global")
    plt.grid(True)

    if save_dir:
            savepath = f"{save_dir}/PCA_global.png"
            plt.savefig(savepath, dpi=300, bbox_inches="tight")
    plt.close()
    # plt.show()

    print("*PCA* Varianza explicada:", pca.explained_variance_ratio_)

def pca_per_run(df,save_dir=None):
    num_cols = get_numeric_columns(df)

    for run_id, sub in df.groupby("Run_T"):
        sub = sub[num_cols].dropna()
        if len(sub) < 2:
            continue

        X_scaled = StandardScaler().fit_transform(sub)

        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        plt.figure(figsize=(6, 5))
        plt.scatter(X_pca[:, 0], X_pca[:, 1])
        plt.title(f"PCA {run_id}")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.grid(True)
        if save_dir:
            savepath = f"{save_dir}/PCA_{run_id}.png"
            plt.savefig(savepath, dpi=300, bbox_inches="tight")
        plt.close()
        # plt.show()

#------------- Boxplots ------------------

def boxplot_global(df,save_dir=None):
    num_cols = get_numeric_columns(df)

    df[num_cols].plot(kind="box", figsize=(10, 6))
    plt.title("Boxplot Global")
    plt.xticks(rotation=45)
    plt.tight_layout()
    if save_dir:
            savepath = f"{save_dir}/boxplot_global.png"
            plt.savefig(savepath, dpi=300, bbox_inches="tight")
    plt.close()
        # plt.show()


def boxplot_por_run(df,save_dir=None):
    num_cols = get_numeric_columns(df)

    for col in num_cols:
        plt.figure(figsize=(8, 5))
        sns.boxplot(x="Run_T", y=col, data=df)
        plt.title(f"Boxplot of {col}")
        plt.xticks(rotation=45)
        plt.tight_layout()
        if save_dir:
            savepath = f"{save_dir}/boxplot_{col}.png"
            plt.savefig(savepath, dpi=300, bbox_inches="tight")
        plt.close()
        # plt.show()

# --------- función para obtener las columnas numericas -----------

def get_numeric_columns(df):
    return df.select_dtypes(include=[np.number]).columns.tolist()

# ------------- Otras funciones que pueden ser utiles--------------

# -------------  Series temporales superpuestas --------------
def timeseries_per_run(df, variable, save_dir=None):

    # plt.figure(figsize=(8, 5))
    fig, ax = plt.subplots(figsize=(8, 5))

    for run_id, sub in df.groupby("Run_T"):   
        ax.scatter(sub["time"], sub[variable], label = run_id, s=20, alpha=0.7)
        # ax.plot(sub["time"], sub[variable], label=run_id)
    
    ax.set_title(f"{variable} vs tiempo")
    ax.set_xlabel("Time")
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

