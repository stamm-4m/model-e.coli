# src/workflows/parameter_estimation.py

from glob import glob
from scipy.optimize import least_squares
from src.utils.io import load_yaml, save_yaml
from src.utils.experiment_factory import build_experiment, run_model_with_parameters
from src.knowledge_based_workflow.data_treatment.standardization import DatasetStandardization
from src.knowledge_based_workflow.model.core.kinetics import Kinetic_Models
from src.utils.fitting_tools import MultiExperimentObjective, compute_confidence_intervals
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

class ParameterEstimationWorkflow:

    def __init__(self, config_path="src/config/params.yaml", dataset_pattern="data/processed/BR_processed.xls",param_names=["mu_max_p","mu_max_0"]):

        self.config_path = config_path
        self.dataset_pattern = dataset_pattern

        self.cfg = None

        self.datasets = None
        self.simulators = None
        self.y0s = None

        self.kin = None

        self.full_params = None
        self.param_names = param_names

        self.theta0 = None
        self.lower_bounds = None
        self.upper_bounds = None

        self.result = None
        self.cov = None
        self.std = None
        self.ci = None

    def load_configuration(self):
        self.cfg = load_yaml(self.config_path)

    def load_datasets(self):
        dataset_files = sorted( glob(self.dataset_pattern) )
        self.datasets =  [DatasetStandardization(f) for f in dataset_files ]

    def prepare_parameters(self):

        self.full_params = { name: self.cfg["kinetics"][name]["value"] for name in self.cfg["kinetics"] }

        theta0 = []
        lower_bounds = []
        upper_bounds = []

        for name in self.param_names:

            p0 = self.cfg["kinetics"][name]["value"]

            if name == "Y_XS":
                lb, ub = 0.01, 0.99
            else:
                lb, ub = 0.001 * p0, 3 * p0

            theta0.append(p0)
            lower_bounds.append(lb)
            upper_bounds.append(ub)

        self.theta0 = theta0
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds

    # =====================================================
    # Model and experiment
    # =====================================================

    def build_experiment(self):

        self.kin = Kinetic_Models()
        self.datasets, self.simulators, self.y0s = build_experiment(self.cfg, self.kin)

    # =====================================================
    # Parameter estimation
    # =====================================================

    def fit(self):

        objective = MultiExperimentObjective(
            datasets=self.datasets,
            simulators=self.simulators,
            kin=self.kin,
            y0s=self.y0s,
            param_names=self.param_names,
            full_params=self.full_params
        )

        self.result = least_squares( objective, x0=self.theta0,
            bounds=( self.lower_bounds, self.upper_bounds )
        )

        self.cov, self.std, self.ci = compute_confidence_intervals( self.result )

    # =====================================================
    # Model validation
    # =====================================================

    def evaluate(self):

        (
            self.per_dataset_metrics,
            self.global_metrics,
            self.all_residuals,
            self.solutions
        ) = run_model_with_parameters(
            datasets=self.datasets,
            simulators=self.simulators,
            y0s=self.y0s,
            kin=self.kin,
            theta=self.result.x,
            param_names=self.param_names,
            full_params=self.full_params
        )

    # =====================================================
    # Save estimation results
    # =====================================================

    def save_estimation_results(self, output_path="results/parametric_model/kinetic_fit_results.yaml"):

        results_dict = {
            "model": {
                "type": "FedBatch kinetic model",
                "parameters": self.param_names,
                "n_parameters": len(self.param_names)
            },
            "estimation": {
                "success": bool(self.result.success),
                "cost": float(self.result.cost),
                "n_function_evals": int(self.result.nfev)
            },
            "parameters": {
                name: {
                    "estimate": float(value),
                    "ci_95": [
                        float(low),
                        float(high)]
                }
                for name, value, (low, high)
                in zip(
                    self.param_names,
                    self.result.x,
                    self.ci
                )
            }
        }

        results_dict["metrics"] = {
            "per_dataset":
                self.per_dataset_metrics,
            "global": {
                "regression":
                    self.global_metrics
            }
        }

        save_yaml( results_dict,output_path)

        print( f"Results saved to {output_path}")

    # =====================================================
    # Save updated parameters
    # =====================================================

    def save_updated_parameters(self, output_path="results/estimation/updated_parameters.yaml"):

        updated_cfg = load_yaml( self.config_path )

        for name, value in zip( self.param_names, self.result.x):
            if name not in updated_cfg["kinetics"]:
                raise KeyError( f"Parameter {name} " f"not found in YAML" )
            updated_cfg["kinetics"][name]["value"] = float(value)

        updated_cfg.setdefault("estimation_metadata", {})
        updated_cfg["estimation_metadata"]["source"] = ( "least_squares fit" )
        updated_cfg["estimation_metadata"][ "fitted_parameters" ] = list(self.param_names)
        updated_cfg["estimation_metadata"][ "fixed_parameters" ] = [ name for name in updated_cfg["kinetics"] if name not in self.param_names  ]

        save_yaml(updated_cfg, output_path )

        print( f"Updated parameters saved to {output_path}")

    # =====================================================
    # Plot fitting
    # =====================================================

    def plot_fits_with_ci(self, n_mc=200, output_path="results/parametric_model"):

        # params_opt = self.full_params.copy()

        # for name, value in zip(self.param_names, self.result.x):
        #     params_opt[name] = value
        #     self.kin.set_params(params_opt)

        fig, axes = plt.subplots( 2, len(self.datasets), figsize=(4*len(self.datasets),8), sharex=False , sharey="row")

        if len(self.datasets) == 1:
            axes = np.array(axes).reshape(2, 1)

        # theta_samples = np.random.multivariate_normal( mean=self.result.x, cov=self.cov, size=n_mc)
        
        for i, (ds, sim, y0) in enumerate( zip( self.datasets,   self.simulators, self.y0s ) ):

            # self.kin.set_params(params_opt)
            # sol_nom = sim.run(t_span=(ds.t[0], ds.t[-1]), y0=y0,t_eval=ds.t )

            # X_mc = []
            # S_mc = []

            # for theta_mc in theta_samples:
                
            #     pars = self.full_params.copy()
            #     for name, value in zip(self.param_names, theta_mc):
            #         pars[name] = value

            #     self.kin.set_params(pars)
            #     sol = sim.run( t_span=(ds.t[0], ds.t[-1]),  y0=y0,  t_eval=ds.t )

            #     if sol.success and sol.y.shape[1] == len(ds.t):
            #         X_mc.append(sol.y[0,:])
            #         S_mc.append(sol.y[1,:])
                
            #     if len(X_mc) == 0:
            #         continue

            # X_mc = np.asarray(X_mc)
            # x_low = np.percentile(X_mc, 2.5, axis=0)
            # x_high = np.percentile(X_mc, 97.5, axis=0)

            # S_mc = np.asarray(S_mc)
            # s_low = np.percentile(S_mc, 2.5, axis=0)
            # s_high = np.percentile(S_mc, 97.5, axis=0)

            X_nom, x_low, x_high = self.prediction_band(ds,sim,y0,state_idx=0)
            S_nom, s_low, s_high = self.prediction_band(ds,sim,y0,state_idx=1)

            ax = axes[0,i]
            ax.scatter( ds.t, ds.data["X"], color="k", s=15 )
            # ax.plot(  ds.t, sol_nom.y[0,:], color="C0" )
            ax.plot(  ds.t, X_nom, color="C0" )
            ax.fill_between( ds.t, x_low, x_high, color="C0", alpha=0.3 )
            # ax.set_title( Path(ds.filepath).stem )

            ax = axes[1,i]
            ax.scatter( ds.t, ds.data["S"], color="k", s=15 )
            # ax.plot(  ds.t, sol_nom.y[1,:], color="C1" )]
            ax.plot(  ds.t, S_nom, color="C1" )
            ax.fill_between( ds.t, s_low, s_high, color="C1", alpha=0.3 )

            tmax = ds.t.max()
            axes[0,i].set_xlim(0, tmax)
            axes[1,i].set
            axes[1,i].set_xlabel("Time [h]")

            if i == 0:
                axes[0,i].legend( ["Fit", "95% CI"],loc="best")
                axes[0,i].set_ylabel("X [g/L]")
                axes[1,i].set_ylabel("S [g/L]")

            axes[0,i].set_title(f"BR{i+1:02d}")

        fig.tight_layout()

        fig.savefig( f"{output_path}/fits_with_ci.png", dpi=300, bbox_inches="tight" )

        plt.close(fig)

        return
    
    def prediction_band( self,  ds, sim,  y0, state_idx,  rel_step = 1e-4 ):
        
        # -------------------------
        # Optimal parameters
        # -------------------------
        params_opt = self.full_params.copy()

        for name, value in zip( self.param_names, self.result.x ):
            params_opt[name] = value

        self.kin.set_params(params_opt)

        t_eval_dense = np.linspace(ds.t[0], ds.t[-1], 200) 

        sol_out = sim.run( t_span=(ds.t[0], ds.t[-1]),  y0=y0, t_eval=ds.t, dense_output=True)
        sol_nom = sol_out.sol(t_eval_dense)
        y_nom = sol_nom.y[state_idx, :]

        n_t = 200 # len(ds.t)
        n_p = len(self.param_names)

        J = np.zeros((n_t, n_p))

        # -------------------------
        # Finite-difference sensitivities
        # -------------------------
        for j, (name, theta_j) in enumerate(  zip(self.param_names, self.result.x) ):

            delta = rel_step * max(  abs(theta_j), 1e-8 )
            theta_plus = self.result.x.copy()
            theta_minus = self.result.x.copy()
            theta_plus[j] += delta
            theta_minus[j] -= delta

            # + delta
            pars = self.full_params.copy()
            for n, v in zip( self.param_names, theta_plus ):
                pars[n] = v

            self.kin.set_params(pars)
            sol_out_plus = sim.run( t_span=(ds.t[0], ds.t[-1]), y0=y0,  t_eval=ds.t, dense_output=True )
            sol_plus = sol_out_plus.sol(t_eval_dense)

            # - delta
            pars = self.full_params.copy()
            for n, v in zip( self.param_names,  theta_minus  ):
                pars[n] = v

            self.kin.set_params(pars)
            sol_out_minus = sim.run(  t_span=(ds.t[0], ds.t[-1]), y0=y0,  t_eval=ds.t, dense_output=True )
            sol_minus = sol_out_minus.sol(t_eval_dense)

            if (  sol_plus.success  and sol_minus.success ):
                J[:, j] = ( sol_plus.y[state_idx, :]  - sol_minus.y[state_idx, :]  ) / (2 * delta)

        # -------------------------
        # Covariance propagation
        # -------------------------
        var_y = np.einsum( "ij,jk,ik->i", J, self.cov, J )
        var_y = np.maximum(var_y, 0)
        std_y = np.sqrt(var_y)
        y_low = y_nom - 1.96 * std_y
        y_high = y_nom + 1.96 * std_y

        self.kin.set_params(params_opt)

        return (  y_nom,  y_low,  y_high )

    # =====================================================
    # Run complete workflow
    # =====================================================

    def run(self):

        print("Loading configuration...")
        self.load_configuration()

        print("Loading datasets...")
        self.load_datasets()

        print("Preparing parameters...")
        self.prepare_parameters()

        print("Building experiment...")
        self.build_experiment()

        print("Running estimation...")
        self.fit()

        print("Running validation...")
        self.evaluate()

        print("Plotting...")
        self.plot_fits_with_ci()

        print("Saving results...")
        self.save_estimation_results()

        print("Saving updated parameters...")
        self.save_updated_parameters()

        print("Done.")

        return self.result