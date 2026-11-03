
import pandas as pd
from src.utils.io import models_dict # , save_wrapper_summary_table, timer, save_yaml, to_python_type


class HybridModelSelectionWorkflow:
    def __init__(self, data_path="data/processed/BR_processed.xlsx", features_objective_path="results/filtered_features", 
                 machine_learning_models=None, ignore_data = None, tuning_algorithm="grid_search"):
        
        self.data_path = data_path
        self.features_objective_path = features_objective_path

        self.ignore_data = ignore_data
        self.machine_learning_models = models_dict(machine_learning_models)

        self.summary = None

        self.tuning_algorithm = tuning_algorithm 

    def load_datasets(self):
        self.df = pd.read_excel(self.data_path, header=0)
        if self.ignore_data is not None:
            self.df = self.df[self.df["Run_ID"]!=self.ignore_data[0]].copy()

    def get_targets(self):
        self.df_mrmr = pd.read_excel(f"{self.features_objective_path}/mrmr_selected_features.xlsx", header=0)
        self.targets = self.df_mrmr[self.df_mrmr.columns["target"]].unique().copy()
        return self.targets

    def get_features(self, target):
        self.features = self.df_mrmr[self.df_mrmr.columns["target"]==target]["feature"].copy()
        return self.features

    def hyperparameter_tuning(self, model, eval_features, target):
        algorithm = self.tuning_algorithm
        # Implement hyperparameter tuning logic here
        pass

    def save_results(self):
        # Implement results saving logic here
        pass

    def run(self):

        print("Loading datasets...")
        self.load_datasets()

        print("Loading targets...")
        targets = self.get_targets()

        for target in targets:
            all_features = self.get_features(target)
            eval_features = all_features

            for name, model in self.machine_learning_models.items():

                print(f"Running algorithm for {name}...")

                for i in range(len(all_features)-1):
                    
                    tuning_criteria, performance_criteria, global_metrics, features_list, hyper_params = self.hyperparameter_tuning(model, eval_features, target)
                    eval_features = features_list["names"][:-1]

                    self.summary[target][name][f"{len(eval_features)-1}"] = {
                                                                            "features_list": features_list,
                                                                            "global_metrics": global_metrics,
                                                                            "hyper_params": hyper_params,
                                                                            "tuning_criteria": tuning_criteria,
                                                                            "performance_criteria": performance_criteria
                                                                        }

        print("Saving results...")
        self.save_results()

        print("Hybrid Model Comparison workflow completed successfully.")
        # return self.summary