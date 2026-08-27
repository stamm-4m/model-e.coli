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
import pandas as pd
from pathlib import Path
from src.knowledge_based_workflow.data_treatment.standardization import DatasetStandardization
from src.knowledge_based_workflow.data_treatment.outliers import outliers_and_smoothing # , data_unification
from src.knowledge_based_workflow.data_treatment.processing import processing_data

class DataTreatment:
    def __init__(self, path):
        
        """
        Generates a DataTreatment object that handles the data treatment process
        """
        # =================== Data Standardization step =================== 
        self.path = path
        self.dataset_files = sorted(glob(f"{path}/BR*.xls"))
        self.datasets = [DatasetStandardization(f) for f in self.dataset_files]

        # ===================  =================== 
        self.br_id_list = [file.stem for file in Path(path).iterdir() if file.suffix == ".xls"]
        self.variable_list = ["X", "S", "V", "P", "T"]

        # =================== Data treatment (outlier treatment, smoothing data,  derivative computation) =================== 
        self.data_sets, _ = outliers_and_smoothing(datasets = self.datasets, time_col = "time", variable_list = self.variable_list, 
                                                            results_root="results/data_analysis/treatment", smooth=True)

        # =================== Computes qP and mu calculation ===================  
    def data_frame(self, yaml_path: str = "src/config/params.yaml"):  # yaml_path = "src/config/default_parameters.yaml"

        df_global, df_induction = processing_data(self.data_sets, yaml_path, t_ind_exp = True) 
        df_induction.to_excel("data/processed/BR_processed_ind.xlsx",index=False,engine="openpyxl")
        df_global.to_excel("data/processed/BR_processed.xlsx",index=False,engine="openpyxl")
        
        return df_global, df_induction
    
    