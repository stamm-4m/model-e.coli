
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from src.utils.io import timer

@timer
def compute_ead(df, vars, results_root):
    
    print(f"Starting EAD analysis ... \n")

    results_root = Path(results_root)
    results_root.mkdir(parents=True, exist_ok=True)
    # -------- PCA Plots ----------
    out_dir = results_root/"PCA_global.png"

    exclude = {}
    vars_pca = [v for v in vars if v not in exclude]

    pca_time_biplot( df.sort_values("time"),vars_pca, title="PCA Global", out_dir=out_dir)

    for run_id, df_run in df.groupby("Run_ID"):
        out_dir = results_root/f"PCA_{run_id}.png"
        pca_time_biplot( df_run.sort_values("time"), vars_pca, title=f"PCA {run_id} (colored by time + Biplot)", out_dir=out_dir)

    # -------- Heatmap Plots -----------

    exclude = {}
    vars_hm = [v for v in vars if v not in exclude]

    corr_global = df[vars_hm].corr()

    annot = corr_global.copy()
    annot = annot.map(
        lambda v: f"{v:.2f}" if abs(v) >= 0.7 else ""
    )

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_global, annot=annot, fmt="", cmap="coolwarm", center=0)
    plt.title("Heatmap global")
    plt.tight_layout()

    out_dir = results_root/"HM_global.png"
    plt.savefig(out_dir, dpi=150)
    plt.close()

    for run_id, df_run in df.groupby("Run_ID"):

        corr = df_run[vars_hm].corr(method='spearman')
        annot = corr.copy()
        annot = annot.map(
            lambda v: f"{v:.2f}" if abs(v) >= 0.7 else ""
        )
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr, annot=annot, fmt="", cmap="coolwarm", center=0)
        plt.title(f"Heatmap {run_id}")
        plt.tight_layout()

        out_dir = results_root/f"HM_{run_id}.png"
        plt.savefig(out_dir, dpi=150)
        plt.close()

    # --------Boxplots Plots -----------

    exclude = {"X", "I", "T", "S", "V", "P","dXdt", "dSdt", "dVdt","Plag1","Xlag1"} # "dPdt","qP","rP","mu""
    vars_bp = [v for v in vars if v not in exclude]
    
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df[vars_bp])
    plt.xticks(rotation=45)
    plt.title("Boxplot global")
    plt.tight_layout()

    out_dir = results_root/"BP_global.png"
    plt.savefig(out_dir, dpi=150)
    plt.close()

    for run_id, df_run in df.groupby("Run_ID"):
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df_run[vars_bp])
        plt.xticks(rotation=45)
        plt.title(f"Boxplot {run_id}")
        plt.tight_layout()

        out_dir = results_root/f"BP_{run_id}.png"
        plt.savefig(out_dir, dpi=150)
        plt.close()
    
    # ----------- qP, rP, & mu vs time-----------
    
    plt.figure(figsize=(10, 5))
    
    for run_T, df_run in df.groupby("Run_T"):
        df_run = df_run.sort_values("time")

        plt.scatter(df_run["time"],df_run["qP"] * 10,label=f"{run_T} - qP * 10",alpha=0.7)

        plt.scatter(df_run["time"],df_run["rP"],label=f"{run_T} - rP",alpha=0.7,marker="x")

        # plt.scatter(df["time"], df["mu"], label="mu")

    plt.legend(title="Run_T", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.xlabel("time")
    plt.ylabel("value")
    plt.title("Time series global (colored by Run_T)")
    plt.tight_layout()

    out_dir = f"{results_root}/time_series_global.png"
    plt.savefig(out_dir, dpi=150)
    plt.close()

    for run_id, df_run in df.groupby("Run_ID"):
        df_run = df_run.sort_values("time")

        plt.figure(figsize=(10, 5))
        plt.scatter(df_run["time"], df_run["qP"]*10, label="qP * 10")
        plt.scatter(df_run["time"], df_run["rP"], label="rP")
        # plt.scatter(df_run["time"], df_run["mu"], label="mu")
        plt.legend()
        plt.xlabel("time")
        plt.ylabel("value")
        plt.title(f"Time series {run_id}")
        plt.tight_layout()

        out_dir = f"{results_root}/time_series_{run_id}.png"
        plt.savefig(out_dir, dpi=150)
        plt.close()

    # ----------- qP, rP, & mu vs T -----------
    
    plt.figure(figsize=(10, 5))
    
    for run_T, df_run in df.groupby("Run_T"):
        df_run = df_run.sort_values("T")

        plt.scatter(df_run["T"],df_run["qP"] * 10,label=f"{run_T} - qP * 10",alpha=0.7)

        plt.scatter(df_run["T"],df_run["rP"],label=f"{run_T} - rP",alpha=0.7,marker="x")

        # plt.scatter(df["T"], df["mu"], label="mu")

    plt.legend(title="Run_T", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.xlabel("T")
    plt.ylabel("value")
    plt.title("Temperature series global (colored by Run_T)")
    plt.tight_layout()

    out_dir = f"{results_root}/Temperature_series_global.png"
    plt.savefig(out_dir, dpi=150)
    plt.close()

    for run_id, df_run in df.groupby("Run_ID"):
        df_run = df_run.sort_values("T")

        plt.figure(figsize=(10, 5))
        plt.scatter(df_run["T"], df_run["qP"]*10, label="qP * 10")
        plt.scatter(df_run["T"], df_run["rP"], label="rP")
        # plt.scatter(df_run["T"], df_run["mu"], label="mu")
        plt.legend()
        plt.xlabel("T")
        plt.ylabel("value")
        plt.title(f"Temperature series {run_id}")
        plt.tight_layout()

        out_dir = f"{results_root}/Temperature_series_{run_id}.png"
        plt.savefig(out_dir, dpi=150)
        plt.close()

    return

# -------- PCA Plots ----------

def pca_time_biplot(df_sub, vars_pca, title, out_dir):
    X = df_sub[vars_pca].values
    time = df_sub["time"].values

    # Escalado
    Xs = StandardScaler().fit_transform(X)

    pca = PCA(n_components=2)
    scores = pca.fit_transform(Xs)

    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

    plt.figure(figsize=(8, 7))

    # Puntos coloreados por tiempo
    sc = plt.scatter(
        scores[:, 0], scores[:, 1],
        c=time, cmap="plasma", s=60, edgecolor="k"
    )

    # Conexión temporal
    if title != "PCA Global":
        plt.plot(scores[:, 0], scores[:, 1],
             linestyle="--", color="gray", alpha=0.6)

    # Vectores (biplot)
    for i, var in enumerate(vars_pca):
        plt.arrow(0, 0,
                  loadings[i, 0]*2,
                  loadings[i, 1]*2,
                  color="red", width=0.01)
        plt.text(loadings[i, 0]*2.2,
                 loadings[i, 1]*2.2,
                 var, color="red", fontsize=10)

    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    plt.title(title)
    plt.colorbar(sc, label="time")

    plt.grid(True)
    plt.tight_layout()

    plt.savefig(out_dir, dpi=150)
    plt.close()