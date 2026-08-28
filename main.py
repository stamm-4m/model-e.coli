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
from src.knowledge_based_workflow.model_analysis.data_analysis import DataAnalysisWorkflow
from src.knowledge_based_workflow.model_analysis.parameter_estimation import ParameterEstimationWorkflow

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

        param_names = cfg["estimation"]["fitted_parameters"]  
        ParameterEstimationWorkflow(
            config_path=cfg["paths"]["parameters"], 
            param_names=param_names
        ).run()

    # if cfg["workflow"]["simulation"]:
    #     SimulationWorkflow().run()

if __name__ == "__main__":
    main()