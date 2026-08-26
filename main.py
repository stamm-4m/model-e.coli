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

from pathlib import Path
from src.knowledge_based_workflow.data_treatment.data_processing import DataTreatment

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

path_raw_files = Path("data/raw")
yaml_path="src/config/params.yaml"

treat_data = DataTreatment(path_raw_files)
df_global, df_induction = treat_data.data_frame(yaml_path)
