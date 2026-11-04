
import pandas as pd
from src.utils.io import models_dict # , save_wrapper_summary_table, timer, save_yaml, to_python_type
from src.utils.hybrid_models_utils import objective_function
import numpy as np
import pandas as pd

class HybridModelSelectionWorkflow:
    def __init__(self, data_path="data/processed/BR_processed.xlsx", features_objective_path="results/filtered_features", global_target=None,
                 machine_learning_models=None, ignore_data = None, tuning_algorithm="grid_search", n_trainings=200):
        
        self.data_path = data_path
        self.features_objective_path = features_objective_path
        self.global_target = global_target

        self.ignore_data = ignore_data
        self.machine_learning_models = models_dict(machine_learning_models)

        self.summary = {}

        self.tuning_algorithm = tuning_algorithm
        self.n_trainings = n_trainings

    def load_datasets(self):
        self.df = pd.read_excel(self.data_path, header=0)
        if self.ignore_data is not None:
            self.df = self.df[self.df["Run_ID"]!=self.ignore_data[0]].copy()

    def get_targets(self):
        self.df_mrmr = pd.read_excel(f"{self.features_objective_path}/mrmr_selected_features.xlsx", header=0)
        self.targets = self.df_mrmr["target"].unique().tolist()
        return self.targets

    def get_features(self, target):
        self.features = (self.df_mrmr.loc[self.df_mrmr["target"] == target,"feature"].tolist())
        return self.features

    def hyperparameter_tuning(self, model_factory, eval_features, target):

        algorithm = self.tuning_algorithm

        if algorithm == "grid_search":
            # model = model_factory()
            # grid = GridSearchCV(model, param_grid, cv=inner_cv,  # cv=5, 
            #                     scoring="neg_mean_squared_error",
            #                     n_jobs=-1)
            # grid.fit(X_sel, y, groups=groups)
            # current_model = grid.best_estimator_
            # hyperparameters = current_model.get_params()
            hyperparameters = {}
        elif algorithm == "random_search":
            hyperparameters = {}
        elif algorithm == "bayesian_optimization":
            out = objective_function(hyperparameters, model_factory, eval_features, target, self.df)
            hyperparameters = {}

        # Containers
        tuning_criteria = out[0]
        performance_criteria = out[1]
        global_metrics = out[2]
        features_list = out[3]
        hyper_params = out[4]

        return ( tuning_criteria, performance_criteria,  global_metrics, features_list, hyper_params )

    def save_results(self):
        # Implement results saving logic here
        pass

    def run(self):

        print("Loading datasets...")
        self.load_datasets()

        print("Loading targets...")
        targets = self.get_targets()

        for target in targets:

            self.summary[target] = {}
            all_features = self.get_features(target)

            for model_name, model_factory in self.machine_learning_models.items():
                
                print(f"Running algorithm for {model_name}...")
                self.summary[target][model_name] = {}
                eval_features = all_features.copy()

                while len(eval_features) > 1:

                    (tuning_criteria, performance_criteria, global_metrics, features_list, hyper_params
                     ) = self.hyperparameter_tuning(model_factory, eval_features, target)

                    n_features = len(eval_features)

                    self.summary[target][model_name][str(n_features)] = {
                                                                            "features_list": features_list,
                                                                            "global_metrics": global_metrics,
                                                                            "hyper_params": hyper_params,
                                                                            "tuning_criteria": tuning_criteria,
                                                                            "performance_criteria": performance_criteria
                                                                        }
                    
                    eval_features = (features_list["names"][:-1]).copy() # worst_feature
                    # eval_features = features_list["selected_features"][:-1].copy()

        print("Saving results...")
        self.save_results()

        print("Hybrid Model Comparison workflow completed successfully.")
        # return self.summary

    