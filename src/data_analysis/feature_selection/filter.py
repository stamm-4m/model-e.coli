
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_regression
import dcor
from src.utils.io import save_yaml, timer
import networkx as nx
from src.data_analysis.feature_selection.plots_filter import (
    plot_cmi_comparison, plot_fs_bars, plot_fs_heatmap, plot_redundancy_graph)

@timer
def filter_feature_selection(df, X_vars, y_var, out_dir):

    print(f"Starting filter feature selection analysis ... \n")

    fs = (
        spearman_fs(df, X_vars, y_var)
        .merge(eta_fs(df, X_vars, y_var), on="feature")
        .merge(dcor_fs(df, X_vars, y_var), on="feature")
        .merge(mi_fs(df, X_vars, y_var), on="feature")
    )

    fs.sort_values("mutual_info", ascending=False)

    fs_dict = fs.set_index("feature").to_dict(orient="index")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    save_yaml(fs_dict,out_dir/"feature.yaml")

    plot_fs_bars(fs, out_dir)
    plot_fs_heatmap(fs, out_dir)

    cmi_matrix = cmi_all_vs_all(df, X_vars, y_var)
    
    plot_cmi_comparison(cmi_matrix, X_vars, out_dir)

    G = build_redundancy_graph(fs, cmi_matrix)
    plot_redundancy_graph(G, fs, cmi_matrix, out_dir)

    # to_remove = []

    # for node in G.nodes:
    #     if G.out_degree(node) > 2:  # heuristics
    #         to_remove.append(node)

    # # Nodes without strong redundancy
    # core = [n for n in G.nodes if G.in_degree(n) > G.out_degree(n)]

# ----------------Spearman---------------------------
def spearman_fs(df, X_vars, y_var):
    results = []

    for x in X_vars:
        rho, p = spearmanr(df[x], df[y_var])
        results.append({"feature": x, "spearman": rho, "p_value": p})

    return pd.DataFrame(results)

# ---------------- eta squared ---------------------------
def eta_fs(df, X_vars, y_var):
    return pd.DataFrame({
        "feature": X_vars,
        "eta2": [eta_squared(df[x], df[y_var]) for x in X_vars]
    })

def eta_squared(x, y, bins=10):
    y_bins = pd.qcut(y, bins, duplicates="drop")
    grand_mean = x.mean()
    
    ss_between = sum(
        len(x[y_bins == b]) * (x[y_bins == b].mean() - grand_mean)**2
        for b in y_bins.unique()
    )
    
    ss_total = sum((x - grand_mean)**2)

    if ss_total == 0:
        return np.nan
    return ss_between / ss_total

# ---------------- distance correlation ---------------------------
def dcor_fs(df, X_vars, y_var):
    return pd.DataFrame({
        "feature": X_vars,
        "dcor": [
            dcor.distance_correlation(
                df[x].astype(float).values, 
                df[y_var].astype(float).values
                ) 
                for x in X_vars
            ]
    })

# ---------------- mutual information ---------------------------
def mi_fs(df, X_vars, y_var):
    mi = mutual_info_regression(
        df[X_vars].values,
        df[y_var].values,
        random_state=42
    )

    return pd.DataFrame({
        "feature": X_vars,
        "mutual_info": mi
    })

# ---------------- conditional mutual information ---------------------------
def conditional_mutual_information(df, x, y, z):
    X_xz = df[[x, z]].values
    X_z = df[[z]].values
    Y = df[y].values

    mi_xz = mutual_info_regression(X_xz, Y, random_state=42).sum()
    mi_z = mutual_info_regression(X_z, Y, random_state=42).sum()

    return max(0.0, mi_xz - mi_z)


def cmi_all_vs_all(df, features, target):
    """
    Rows: conditioning feature (fixed)
    Columns: evaluated feature
    Values: I(feature ; target | conditioning_feature)
    """
    cmi_matrix = pd.DataFrame(
        index=features,
        columns=features,
        dtype=float
    )

    for f_cond in features:
        for f_eval in features:
            if f_eval == f_cond:
                cmi_matrix.loc[f_cond, f_eval] = np.nan
            else:
                cmi_matrix.loc[f_cond, f_eval] = conditional_mutual_information(
                    df, f_eval, target, f_cond
                )

    return cmi_matrix


def build_redundancy_graph(fs_df, cmi_matrix, threshold=0.1):
    G = nx.DiGraph()

    for f in fs_df["feature"]:
        G.add_node(f)

    for fi in fs_df["feature"]:
        mi_fi = fs_df.loc[fs_df["feature"] == fi, "mutual_info"].values[0]

        if mi_fi < 1e-6:
            continue

        for fj in fs_df["feature"]:
            if fi == fj:
                continue

            cmi = cmi_matrix.loc[fj, fi]

            # Normalized redundancy
            redundancy = max(0.0, (mi_fi - cmi) / (mi_fi + 1e-10))

            if redundancy > threshold:
                G.add_edge(fj, fi, weight=redundancy)

    return G

