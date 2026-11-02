
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_regression
import dcor
from src.utils.io import save_yaml, timer
import networkx as nx
from src.utils.plots_filter import (
    plot_cmi_comparison, plot_fs_bars, plot_fs_heatmap, plot_redundancy_graph)

import numpy as np

def mrmr_selection(fr_df, redundancy_graph=None, cmi=None, mi=None, n_features=5, method='cmi_based'): 
    """
    Perform Minimum Redundancy Maximum Relevance (mRMR) feature selection.
    method="cmi_based"      # proposed method based on conditional mutual information
    method="cmim"           # Fleuret 2004
    method="mrmr_mid"       # Peng et al. 2005
    method="mrmr_miq"       # Peng et al. 2005
    """

    # Normalize mutual information
    fr_df = fr_df.copy()

    selected = []
    selection_info = []

    first_feature = ( fr_df.sort_values( "mutual_info", ascending=False ) .iloc[0]["feature"] )

    selected.append(first_feature)

    first_relevance = ( fr_df.loc[ fr_df["feature"] == first_feature, "mutual_info" ].iloc[0] )

    selection_info.append({
        "feature": first_feature,
        "relevance": first_relevance,
        "redundancy": 0.0,
        "score": first_relevance, 
        "method": method
    })

    remaining = [ f for f in fr_df["feature"] if f not in selected]

    while len(selected) < min(n_features, len(fr_df)):

        candidate_info = {}
        for feat in remaining:
            relevance = ( fr_df.loc[ fr_df["feature"] == feat, "mutual_info" ].iloc[0] )

            if method == 'cmi_based':
                redundancies = []
                for sel in selected:
                    if redundancy_graph.has_edge(sel, feat):
                        redundancies.append(redundancy_graph[sel][feat]["redundancy"])
                redundancy = (np.mean(redundancies) if redundancies else 0.0)
                score = relevance * (1 - redundancy)

            elif method == "cmim":
                cmi_values = []
                for sel in selected:
                    cmi_values.append(cmi.loc[sel, feat])
                score = np.min(cmi_values) if cmi_values else relevance
                redundancy = np.nan

            elif method == 'peng_mid':
                mi_values = []
                for sel in selected:
                    mi_values.append(mi.loc[sel, feat])
                redundancy = (np.mean(mi_values) if mi_values else 0.0)
                score = relevance - redundancy

            elif method == 'peng_miq':
                mi_values = []
                for sel in selected:
                    mi_values.append(mi.loc[sel, feat])
                redundancy = (np.mean(mi_values) if mi_values else 0.0)
                score = relevance / (redundancy + 1e-10)

            else:
                raise ValueError(f"Unknown method '{method}'" "Choose 'cmi' or 'peng'")

            candidate_info[feat] = {
                "relevance": relevance,
                "redundancy": redundancy,
                "score": score
            }

        best_feature = max(candidate_info, key=lambda f: candidate_info[f]["score"])
        selected.append(best_feature)
        selection_info.append({
            "feature": best_feature,
            **candidate_info[best_feature], 
            "method": method
        })
        remaining.remove(best_feature)

    return pd.DataFrame(selection_info)


def filter_feature_selection(df, X_vars, y_var, out_dir):

    # print(f"Starting filter feature selection analysis ... \n")

    fs_results = (
        spearman_fs(df, X_vars, y_var)
        .merge(eta_fs(df, X_vars, y_var), on="feature")
        .merge(dcor_fs(df, X_vars, y_var), on="feature")
        .merge(mi_fs(df, X_vars, y_var), on="feature")
    )

    fs_results.sort_values("mutual_info", ascending=False)

    fs_dict = fs_results.set_index("feature").to_dict(orient="index")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    save_yaml(fs_dict,out_dir/"feature.yaml")

    plot_fs_bars(fs_results, out_dir)
    plot_fs_heatmap(fs_results, out_dir)

    mi_matrix = mi_features_all_vs_all(df, X_vars)
    cmi_matrix = cmi_all_vs_all(df, X_vars, y_var)

    plot_cmi_comparison(cmi_matrix, X_vars, out_dir)

    G = build_redundancy_graph(fs_results, cmi_matrix)
    plot_redundancy_graph(G, fs_results, cmi_matrix, out_dir)

    # to_remove = []

    # for node in G.nodes:
    #     if G.out_degree(node) > 2:  # heuristics
    #         to_remove.append(node)

    # # Nodes without strong redundancy
    # core = [n for n in G.nodes if G.in_degree(n) > G.out_degree(n)]
    return { "feature_relevance_df": fs_results, "cmi": cmi_matrix, "mi": mi_matrix, "graph": G }

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

def mi_features_all_vs_all(df, features):
    """
    Rows: reference feature
    Columns: evaluated feature
    Values: I(feature_i ; feature_j)
    """

    mi_matrix = pd.DataFrame(index=features, columns=features, dtype=float)
    for i, fi in enumerate(features):
        for j, fj in enumerate(features[i:], i):
            if i == j:
                mi_matrix.loc[fi, fj] = np.nan
            else:
                mi_val = mutual_info_regression(df[[fj]].values, df[fi].values, random_state=42)[0]
                mi_matrix.loc[fi, fj] = mi_val
                mi_matrix.loc[fj, fi] = mi_val
    return mi_matrix


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
                G.add_edge(fj, fi, redundancy=redundancy, cmi=cmi)

    return G

