"""
Nanobody-based Antivenom Production with E. coli Reactor Simulation 

Adapted from:
Corrales, D. C., Villela, S. M. A., Bouhaouala-Zahar, B., Cescut, J., Daboussi, F., 
O'donohue, M., (...) & Aceves-Lara, C. A. (2024). Dynamic Hybrid Model for Nanobody-based 
Antivenom Production (scorpion antivenon) with E. coli CH10-12 and E. coli NbF12-10. 
Computer Aided Chemical Engineering, 53, pp. 145-150. 

Main — Simulation runner
 
Authors:
    - Juan Camilo Castaño Sanchez           <jcastano-san@insa-toulouse.fr>
    - David Camilo Corrales                 <David-Camilo.Corrales-Munoz@inrae.fr>

Date: 09/2026
 
Configuration files expected at:
    configs/
    ├── parameters.yaml     ← 
    ├── scenario.yaml            ← active scenario selector + parameter overrides
    └── simulation.yaml          ← ODE solver settings, time horizon, output options
"""

from src.utils.io import load_yaml
from workflow.data_treatment.data_analysis import DataAnalysisWorkflow
from src.workflow.model_analysis.parameter_estimation import ParameterEstimationWorkflow
# from src.workflow.model_analysis.sensitivity_analysis import SensitivityAnalysisWorkflow
from workflow.hybrid_selection.feature_filter import FeaturesFilterWorkflow
from workflow.hybrid_selection.hybrid_model import HybridModelSelectionWorkflow

# def main():
#     # ============================================================
#     # 0. Configuration paths
#     # ============================================================
#     PARAMS_FILE = "configs/adm1_parameters.yaml"
#     STATES_FILE = "configs/Initial_states.yaml"
#     INFLUENT_FILE = "configs/Influent.yaml"
#     SCENARIO_FILE = "configs/Scenario.yaml"
#     SIMULATION_FILE = "configs/Simulation.yaml"

#     # ============================================================
#     # 1. Utility helpers
#     # ============================================================
#     ...

def main():

    print("Starting project workflow...")
    
    cfg = load_yaml("src/config/project.yaml")
        
    if cfg["workflow"]["data_analysis"]:

        DataAnalysisWorkflow(
            raw_data_path = cfg["paths"]["raw_data"],
            yaml_path = cfg["paths"]["parameters"],
            ead_path = cfg["paths"]["results"]["ead"]
        ).run()

    if cfg["workflow"]["parameter_estimation"]:

        ParameterEstimationWorkflow(
            config_path=cfg["paths"]["parameters"], 
            param_names=cfg["estimation"]["parameters"]  
        ).run()

    # if cfg["workflow"]["sensitivity_analysis"]:
    #     SensitivityAnalysisWorkflow().run()

    if cfg["workflow"]["features_filter"]: 
        FeaturesFilterWorkflow(
            data_path=cfg["paths"]["processed_data"],
            filter_path=cfg["paths"]["results"]["filtered_features"],
            obj_param=cfg["features_filter"]["objective_parameter"],
            ignore_data=cfg["features_filter"]["ignore_data"], 
            ignore_features=cfg["features_filter"]["ignore_features"], 
            method=cfg["features_filter"]["method"], 
            n_features=cfg["features_filter"]["n_features"]
        ).run()

    if cfg["workflow"]["hybrid_model_selection"]:
        HybridModelSelectionWorkflow(
            data_path=cfg["paths"]["processed_data"],
            features_objective_path=cfg["paths"]["features_objective"],
            global_target=cfg["sensitivity_analysis"]["global_target"],
            ignore_data=cfg["hybrid_model_selection"]["ignore_data"],
            tuning_algorithm=cfg["hybrid_model_selection"]["tuning_algorithm"], 
            n_trainings=cfg["hybrid_model_selection"]["n_trainings"],
            machine_learning_models=cfg["hybrid_model_selection"]["machine_learning_models"]
        ).run()

    # if cfg["workflow"]["simulation"]:
    #     SimulationWorkflow().run()

if __name__ == "__main__":
    main()