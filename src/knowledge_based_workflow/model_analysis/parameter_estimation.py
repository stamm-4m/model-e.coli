# src/workflows/parameter_estimation.py

from glob import glob
from scipy.optimize import least_squares
from src.utils.io import load_yaml, save_yaml
from src.utils.experiment_factory import build_experiment, run_model_with_parameters
from src.knowledge_based_workflow.data_treatment.standardization import DatasetStandardization
from src.knowledge_based_workflow.model.core.kinetics import Kinetic_Models
from src.utils.fitting_tools import MultiExperimentObjective, compute_confidence_intervals

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
            self.global_ic,
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
                    self.global_metrics,
                "information_criteria":
                    self.global_ic
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

        print("Saving results...")
        self.save_estimation_results()

        print("Saving updated parameters...")
        self.save_updated_parameters()

        print("Done.")

        return self.result