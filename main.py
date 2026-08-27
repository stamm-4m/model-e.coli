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

import pandas as pd
from pathlib import Path
from src.knowledge_based_workflow.data_treatment.treatment_main import DataTreatment
from src.knowledge_based_workflow.data_treatment.ead import compute_ead
from src.utils.io import load_yaml

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
yaml_path = "src/config/params.yaml"
config = load_yaml(yaml_path)
vars = ["X", "S", "P", "V", "mu", "qP", "qP_2", "rX", "rP", "dXdt", "dSdt", "dPdt", "dVdt", "T", "I"]

# =================== Data-frame generation ===================
treat_data = DataTreatment(path_raw_files)
df_global, df_induction = treat_data.data_frame(yaml_path)

# =================== Data-frame load ===================
# df_global = pd.read_excel(r"data/processed/BR_processed.xlsx")
# df_induction = pd.read_excel(r"data/processed/BR_processed_ind.xlsx")

dfs = {"global": df_global, "induction": df_induction}

# =================== Computation ===================
for name, df in dfs.items():
    print(f"\n===== {name.upper()} =====")
    # ---------- paths ----------
    base_ead = f"results/data_analysis/ead/{name}"
    # =================== EAD =================== 
    compute_ead(df, vars, results_root=f"{base_ead}")
