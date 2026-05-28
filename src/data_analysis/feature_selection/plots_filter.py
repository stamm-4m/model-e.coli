
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import networkx as nx
import numpy as np

# ----------------- Figures --------------------------
def plot_fs_bars(fs_df, out_dir):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    metrics = ["spearman", "eta2", "dcor", "mutual_info"]
    axes = axes.flatten()

    for ax, m in zip(axes, metrics):
        sns.barplot(
            data=fs_df.sort_values(m, ascending=False),
            x=m, ax=ax, y="feature"
        )
        ax.set_ylabel("")
        # ax.set_title(m)

    plt.tight_layout()
    plt.savefig(out_dir / "feature_selection_bars.png", dpi=150)
    plt.close()

def plot_fs_heatmap(fs_df, out_dir):
    
    metrics_for_heatmap = [
        "spearman",
        "p_value_sig",
        "eta2",
        "dcor",
        "mutual_info"
    ]

    alpha = 0.05
    fs_df = fs_df.copy()
    fs_df["p_value_sig"] = np.maximum(0.0,-np.log10(fs_df["p_value"] / alpha))

    fs_hm = fs_df.set_index("feature")[metrics_for_heatmap]
    fs_norm = (fs_hm - fs_hm.min()) / (fs_hm.max() - fs_hm.min())

    plt.figure(figsize=(8, 6))
    sns.heatmap(fs_norm, annot=True, cmap="RdYlGn")#"viridis")
    plt.title("Feature selection — normalized scores")
    plt.tight_layout()
    plt.savefig(out_dir / "feature_selection_heatmap.png", dpi=150)
    plt.close()


def plot_cmi_comparison(cmi_matrix, fixed_features, out_dir):

    rows = []
    for f_cond in fixed_features:
        for f_eval in cmi_matrix.columns:
            if f_eval == f_cond:
                continue
            rows.append({
                "Conditioning": f_cond,
                "Feature": f_eval,
                "CMI": cmi_matrix.loc[f_cond, f_eval]
            })

    df_plot = pd.DataFrame(rows)

    plt.figure(figsize=(7, 5), constrained_layout=True)
    ax = sns.barplot(
            data=df_plot,
            x="CMI",
            y="Feature",
            hue="Conditioning",
            palette="Paired"
        )

    # plt.xlabel("Conditional Mutual Information")
    # sns.move_legend(ax, "upper left", bbox_to_anchor=(1,1))
    ax.set_ylabel("Feature")
    ax.set_xlabel("CMI")
    plt.title("CMI for different conditioning features")

    plt.legend(
            title="Conditioning",
            fontsize=8,          
            title_fontsize=9,    
            loc="upper left",    
            bbox_to_anchor=(1.02, 1),  # outside
            borderaxespad=0
        )

    # plt.tight_layout()

    plt.savefig(out_dir / "CMI_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

def plot_redundancy_graph(G, fs_df, cmi_matrix, out_path):

    threshold = 0.1

    # ---- Build filtered graph (keep direction) ----
    H = nx.DiGraph()
    H.add_nodes_from(G.nodes())

    for u, v in G.edges():
        if (
            G[u][v]["weight"] > threshold
            and cmi_matrix.loc[u, v] > 1e-6
        ):
            H.add_edge(u, v, weight=G[u][v]["weight"])

    # ---- Node importance ----
    mi_values = fs_df.set_index("feature")["mutual_info"]

    nodes = sorted(H.nodes(), key=lambda n: mi_values.get(n, 0), reverse=True)

    # ---- Circular layout ordered ----
    angles = np.linspace(0, 2 * np.pi, len(nodes), endpoint=False)
    pos = {
        node: np.array([np.cos(a), np.sin(a)])
        for node, a in zip(nodes, angles)
    }

    # ---- Edge weights ----
    if len(H.edges()) > 0:
        weights = np.array([H[u][v]["weight"] for u, v in H.edges()])
        cmi_vals = [cmi_matrix.loc[u, v] for u, v in H.edges()]
    else:
        weights = np.array([0])
        cmi_vals = [0]

    # ---- Normalize for color ----
    norm = mpl.colors.Normalize(vmin=min(cmi_vals), vmax=max(cmi_vals))
    cmap = plt.cm.RdYlGn_r

    edge_colors = [cmap(norm(cmi_matrix.loc[u, v])) for u, v in H.edges()]
    widths = [1 + 3 * w for w in weights]

    # ---- Node properties ----
    node_sizes = [100 + 2000 * mi_values.get(n, 0.01) for n in H.nodes()]
    node_colors = [mi_values.get(n, 0) for n in H.nodes()]

    # ---- Labels ----
    node_labels = {
        n: f"{n}\n({mi_values.get(n,0):.2f})"
        for n in H.nodes()
    }

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(7, 6))

    nx.draw(
        H, pos, ax=ax,
        labels=node_labels,
        node_size=node_sizes,
        node_color=node_colors,
        cmap=plt.cm.Blues,
        edge_color=edge_colors,
        alpha=0.85,
        width=widths,
        arrows=True,
        arrowstyle='-|>',
        arrowsize=15
    )

    # ---- Edge labels ----
    edge_labels = {
        (u, v): f"{w:.2f}"
        for (u, v), w in nx.get_edge_attributes(H, "weight").items()
    }

    nx.draw_networkx_edge_labels(
        H, pos,
        edge_labels=edge_labels,
        font_size=8,
        bbox=dict(facecolor='none', edgecolor='none')
    )

    # ---- Colorbar (CMI) ----
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Conditional Information (CMI)")

    ax.set_title("Directed redundancy graph (CMI-informed)", pad=20)
    ax.axis("off")

    fig.savefig(out_path / "redundancy.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
