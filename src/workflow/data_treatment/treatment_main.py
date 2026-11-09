"""
Nanobody-based Antivenom Production with E. coli Reactor Simulation 
Class: DataTreatment
This class handles the data treatment process for the nanobody-based antivenom production simulation. 
It standardizes datasets, processes outliers, computes derivatives, and prepares the data for further analysis.

Author: Juan Camilo Castaño Sanchez
Email: jcastano-san@insa-toulose.fr
Date: 01/09/2026
"""

from glob import glob
from pathlib import Path
from src.utils.io import load_yaml
from src.workflow.data_treatment.standardization import DatasetStandardization
from src.workflow.data_treatment.outliers import outliers_and_smoothing # , data_unification
from src.workflow.data_treatment.processing import processing_data

class DataTreatment:
    def __init__(self, data_path, model_path, ignore_vars=None, states_variables=None, non_states_cfg=None):
        
        """
        Generates a DataTreatment object that handles the data treatment process
        """
        # =================== Data Standardization step =================== 
        self.data_path = data_path
        self.ignore_vars = ignore_vars
        self.dataset_files = sorted(glob(f"{data_path}/BR*.xls"))
        self.datasets = [DatasetStandardization(f, model_path) for f in self.dataset_files]

        # ===================  =================== 
        self.br_id_list = [file.stem for file in Path(data_path).iterdir() if file.suffix == ".xls"]
        variables = list(load_yaml(model_path)["model_instructions"]["variables"].keys())
        independent = (load_yaml(model_path)["model_instructions"]["independent"][0])
        variables_list = (v for v in variables if v != independent)
        self.variable_list = variables_list

        # =================== Data treatment (outlier treatment, smoothing data,  derivative computation) =================== 
        smoothed_data, _ = outliers_and_smoothing(datasets = self.datasets, time_col = independent, variable_list = self.variable_list, 
                                                  ignore_vars = self.ignore_vars, results_root="results/data_analysis/treatment", 
                                                  smooth=True, states_variables=states_variables, non_states_cfg=non_states_cfg)
        
        self.data_sets = smoothed_data

        # =================== Computes qP and mu calculation ===================  
    def data_frame(self, model_path = "src/config/model.yaml"):  # yaml_path = "src/config/default_parameters.yaml"

        df_global = processing_data(datasets = self.data_sets, model_path = model_path) 
        df_global.to_excel("data/processed/BR_processed.xlsx",index=False,engine="openpyxl")
        
        return df_global
    
    