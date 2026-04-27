
import numpy as np
import matplotlib.pyplot as plt
from fedbatch.utils.io import get_br_id
from scipy.interpolate import interp1d

# -------------------------------------------------
# Monte-Carlo simulation
# -------------------------------------------------
def monte_carlo_simulation(
    simulator,
    dataset,
    kin,
    theta_mean,
    cov,
    param_names,
    full_params,
    n_samples=200,
    random_state=42,
    cfg_tdense = False
):
    rng = np.random.default_rng(random_state)
    
    if cfg_tdense:
        t_eval = np.linspace(dataset.t[0], dataset.t[-1], 100)
    else:
        t_eval = dataset.t
    
    n_states = len(dataset.y0)
    n_time = len(t_eval)

    # Sample parameter vectors
    theta_samples = rng.multivariate_normal(
        mean=theta_mean,
        cov=cov,
        size=n_samples
    )

    trajectories = []
    
    n_fail_solver = 0
    n_fail_nan = 0
    n_fail_shape = 0
    n_success = 0

    for theta in theta_samples:
        
        theta = np.maximum(theta, 1e-8)
        
        params = full_params.copy()
        for name, value in zip(param_names, theta):
            params[name] = value

        kin.set_params(params)

        try: 
            sol = simulator.run(
                y0=dataset.y0,
                t_span=(dataset.t[0], dataset.t[-1]),
                t_eval=t_eval
            )
        except Exception as e:
            n_fail_solver += 1
            continue

        if sol.y is None:
            n_fail_solver += 1
            continue

        if not np.all(np.isfinite(sol.y)):
            n_fail_nan += 1
            continue

        if sol.y.shape[0] != n_states:
            n_fail_shape += 1
            continue

        # # Keep only valid solutions
        # if ( sol.y.shape == (n_states, n_time) and np.all(np.isfinite(sol.y)) ):
        #     trajectories.append(sol.y)

        trajectories.append(sol.y)
        n_success += 1

    # Restore optimal parameters
    params = full_params.copy()
    for name, value in zip(param_names, theta_mean):
        params[name] = value

    kin.set_params(params)


    print("MC diagnostics:")
    print("  success:", n_success)
    print("  solver failures:", n_fail_solver)
    print("  NaN/Inf failures:", n_fail_nan)
    print("  shape failures:", n_fail_shape)

    if len(trajectories) == 0:
            raise RuntimeError(
                "No valid Monte‑Carlo trajectories were generated "
                "(solver unstable around optimum or covariance too large)."
            )

    return np.stack(trajectories, axis=0)  
    # shape: (Nmc, n_states, n_time)

# -------------------------------------------------
# Helper: simulate prediction ± confidence envelope
# -------------------------------------------------
def simulate_with_ci(simulator, dataset, kin, theta, cov, n_std=2):
    """
    Approximate confidence envelope using parameter covariance.
    """
    y_pred = simulator.run(
        y0=dataset.y0,
        t_span=(dataset.t[0], dataset.t[-1]),
        t_eval=dataset.t
    ).y

    # Linearized uncertainty (very standard approximation)
    std_theta = np.sqrt(np.diag(cov))
    delta = n_std * std_theta

    y_upper = []
    y_lower = []

    for sign in [+1, -1]:
        kin.set_params(theta + sign * delta)
        sol = simulator.run(
            y0=dataset.y0,
            t_span=(dataset.t[0], dataset.t[-1]),
            t_eval=dataset.t
        )
        if sign == +1:
            y_upper = sol.y
        else:
            y_lower = sol.y

    # Restore nominal parameters
    kin.set_params(theta)

    return y_pred, y_lower, y_upper


# -------------------------------------------------
# Time‑profile plots
# -------------------------------------------------


def plot_time_profiles(
    dataset,
    simulator,
    kin,
    theta,
    param_names,
    full_params,
    cov=None,
    plot_ci=False,
    n_std=2,
    n_points_model=500,
    savepath=None
):
    
    params = full_params.copy()
    for name, value in zip(param_names, theta):
        params[name] = value

    kin.set_params(params)

    t_exp = dataset.t
    t_dense = np.linspace(t_exp[0], t_exp[-1], n_points_model)

    sol_dense = simulator.run(
        y0=dataset.y0,
        t_span=(t_dense[0], t_dense[-1]),
        t_eval=t_dense
    )
    
    # t_model = sol_dense.t

    # # Interpoladores para los estados
    # interp_states = [
        # interp1d(t_model, sol_dense.y[i, :],
                # kind="linear", fill_value="extrapolate")
        # for i in range(sol_dense.y.shape[0])
    # ]

    # # Evaluar en t_dense
    # y_pred = np.vstack([f(t_dense) for f in interp_states])
    y_pred = sol_dense.y

    T_fun = interp1d(
        sol_dense.t,
        np.array([simulator.model.balances.temperature.F(t)
                for t in sol_dense.t]),
        fill_value="extrapolate"
    )

    T = T_fun(t_dense)
    # T = np.array([simulator.model.balances.temperature.F(t)
    #                 for t in sol_dense.t])

    if plot_ci:
        if cov is None:
            raise ValueError("plot_ci=True requires cov")

        std_theta = np.sqrt(np.diag(cov))
        delta = n_std * std_theta

        kin.set_params(theta + delta)
        y_hi = simulator.run(
            y0=dataset.y0,
            t_span=(t_dense[0], t_dense[-1]),
            t_eval=t_dense
        ).y

        kin.set_params(theta - delta)
        y_lo = simulator.run(
            y0=dataset.y0,
            t_span=(t_dense[0], t_dense[-1]),
            t_eval=t_dense
        ).y

        kin.set_params(theta)

    labels = ["X", "S", "P", "V"]
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    fig, axes = plt.subplots(2, 3, figsize=(10, 8), sharex=True)
    axes = axes.ravel()

    for i, (label, color) in enumerate(zip(labels, colors)):

        # --- Experimental data only if available ---
        if label in dataset.data:
            axes[i].scatter(
                t_exp,
                dataset.data[label],
                color=color,
                s=30,
                label=f"{label} exp"
            )

        # --- Model ---
        axes[i].plot(
            t_dense,
            y_pred[i],
            color=color,
            lw=2,
            label=f"{label} model"
        )

        if plot_ci:
            axes[i].fill_between(
                t_dense,
                y_lo[i],
                y_hi[i],
                color=color,
                alpha=0.25,
                label=f"{n_std}σ CI"
            )

        axes[i].set_ylabel(label)
        axes[i].legend()
        axes[i].grid(True)
  
# ---- Plot T (último subplot) ----
    ax_T = axes[4]

    if dataset.T is not None:
        ax_T.scatter(
            t_exp,
            dataset.T,
            color="tab:purple",
            s=30,
            label="T exp"
        )
#   Model temperature (interpolated)
    ax_T.plot(
        t_dense,
        T,
        color="tab:purple",
        lw=2,
        label="T model"
    )

    ax_T.set_ylabel("T")
    ax_T.legend()
    ax_T.grid(True)

    # ---- Labels finales ----
    for ax in axes[-3:]:
        ax.set_xlabel("Time")

    fig.suptitle(f"Fit — {dataset.path}", fontsize=14)

    if savepath:
        plt.savefig(savepath, dpi=300, bbox_inches="tight")

    # plt.show()


# -------------------------------------------------
# Time-profile plots with monte-carlo
# -------------------------------------------------
def plot_time_profiles_mc(
    dataset,
    simulator,
    kin,
    theta,
    cov,
    param_names,
    full_params,
    n_samples=200,
    ci=(2.5, 97.5),
    savepath=None,
    cfg_tdense=False
):
    mc_traj = monte_carlo_simulation(
        simulator,
        dataset,
        kin,
        theta,
        cov,
        param_names=param_names,
        full_params=full_params,
        n_samples=n_samples,
        cfg_tdense=cfg_tdense
    )
    
    if cfg_tdense:
        t_eval = np.linspace(dataset.t[0], dataset.t[-1], 100)
    else:
        t_eval = dataset.t

    # Nominal prediction
    sol = simulator.run(
        y0=dataset.y0,
        t_span=(dataset.t[0], dataset.t[-1]),
        t_eval=t_eval
    )

    labels = ["X", "S", "P"]
    colors = ["tab:blue", "tab:orange", "tab:green"]

    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

    for i, (label, color) in enumerate(zip(labels, colors)):
        lower = np.percentile(mc_traj[:, i, :], ci[0], axis=0)
        upper = np.percentile(mc_traj[:, i, :], ci[1], axis=0)

        axes[i].scatter(
            dataset.t, dataset.data[label],
            color=color, s=30, label=f"{label} exp"
        )

        axes[i].plot(
            t_eval, sol.y[i],
            color=color, lw=2.5, zorder = 3, 
            label=f"{label} model"
        )

        axes[i].fill_between(
            t_eval, lower, upper,
            color=color, alpha=0.15,
            label=f"{ci[1]-ci[0]:.0f}% MC envelope"
        )

# --- Adjust view to data range (ONLY visualization) ---
        y_data = dataset.data[label]
        y_model = sol.y[i]

        y_min = min(y_data.min(), y_model.min())
        y_max = max(y_data.max(), y_model.max())

        margin = 0.15 * (y_max - y_min + 1e-9)

        axes[i].set_ylim(y_min - margin, y_max + margin)

        axes[i].set_ylabel(label)
        axes[i].legend()
        axes[i].grid(True)

    # for ax in axes:
    #     ax.autoscale(enable=False)

    axes[-1].set_xlabel("Time")
    fig.suptitle(f"Monte-Carlo fit — {dataset.path}")

    if savepath:
        plt.savefig(savepath, dpi=300, bbox_inches="tight")

    # plt.show()

# -------------------------------------------------
# Parity plots (exp vs pred)
# -------------------------------------------------
def plot_parity(dataset, simulator, kin, theta, savepath=None):
    sol = simulator.run(
        y0=dataset.y0,
        t_span=(dataset.t[0], dataset.t[-1]),
        t_eval=dataset.t
    )

    labels = ["X", "S", "P"]
    colors = ["tab:blue", "tab:orange", "tab:green"]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    for i, (label, color) in enumerate(zip(labels, colors)):
        y_exp = dataset.data[label]
        y_pred = sol.y[i]

        axes[i].scatter(y_exp, y_pred, color=color)
        lims = [
            min(y_exp.min(), y_pred.min()),
            max(y_exp.max(), y_pred.max())
        ]
        axes[i].plot(lims, lims, "k--")
        axes[i].set_xlabel(f"{label} exp")
        axes[i].set_ylabel(f"{label} pred")
        axes[i].grid(True)

    fig.suptitle(f"Parity plots for {dataset.path}")

    if savepath:
        plt.savefig(savepath, dpi=300, bbox_inches="tight")

    # plt.show()

# -------------------------------------------------
# Parity plots (exp vs pred) with monte carlo
# -------------------------------------------------

def plot_parity_mc(
    dataset,
    simulator,
    kin,
    theta,
    cov,
    param_names,
    full_params,
    n_samples=200,
    savepath=None
):
    mc_traj = monte_carlo_simulation(
        simulator,
        dataset,
        kin,
        theta,
        cov,
        param_names=param_names,
        full_params=full_params,
        n_samples=n_samples
    )

    labels = ["X", "S", "P"]
    colors = ["tab:blue", "tab:orange", "tab:green"]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    for i, (label, color) in enumerate(zip(labels, colors)):
        y_exp = dataset.data[label]
        y_mc = mc_traj[:, i, :].reshape(-1)

        n_valid = mc_traj.shape[0]  # ✅ actual number of MC runs

        axes[i].scatter(
            np.tile(y_exp, n_valid),
            y_mc,
            color=color,
            alpha=0.05,
            zorder=1
        )

# --- Zoom parity plot to experimental range (visualisation only) ---
        y_min = y_exp.min()
        y_max = y_exp.max()
        margin = 0.15 * (y_max - y_min + 1e-9)
        lims = [y_min - margin, y_max + margin]

        axes[i].set_xlim(lims)
        axes[i].set_ylim(lims)

        # lims = [
        #     min(y_exp.min(), y_mc.min()),
        #     max(y_exp.max(), y_mc.max())
        # ]

        # 1:1 reference line
        axes[i].plot(lims, lims, "k--", zorder=3)
        axes[i].set_xlabel(f"{label} exp")
        axes[i].set_ylabel(f"{label} MC pred")
        axes[i].grid(True)

    fig.suptitle(f"Monte-Carlo parity — {dataset.path}")

    if savepath:
        plt.savefig(savepath, dpi=300, bbox_inches="tight")

    # plt.show()


def plot_time_profiles_multi(
    datasets,
    simulators,
    kin,
    theta,
    param_names,
    full_params,
    cov=None,
    plot_ci=False,
    save_dir=None
):
    """
    Plot time profiles for multiple datasets (one figure per dataset).
    """

    for dataset, simulator in zip(datasets, simulators):

        br_id = get_br_id(dataset)

        plot_time_profiles(
            dataset=dataset,
            simulator=simulator,
            kin=kin,
            theta=theta,
            param_names=param_names,
            full_params=full_params,
            cov=cov,
            plot_ci=plot_ci,
            n_points_model=1000,
            savepath=(
                f"{save_dir}/time_profiles_{br_id}.png"
                if save_dir else None
            )
        )


def plot_time_profiles_mc_multi(
    datasets,
    simulators,
    kin,
    theta,
    cov,
    n_samples=200,
    ci=(2.5, 97.5),
    save_dir=None
):
    """
    Monte‑Carlo time profiles for multiple datasets.
    """

    for dataset, simulator in zip(datasets, simulators):

        br_id = get_br_id(dataset)

        plot_time_profiles_mc(
            dataset=dataset,
            simulator=simulator,
            kin=kin,
            theta=theta,
            cov=cov,
            n_samples=n_samples,
            ci=ci,
            savepath=(
                f"{save_dir}/time_profiles_mc_{br_id}.png"
                if save_dir else None
            ),
            cfg_tdense=True
        )
