


from glob import glob
import pandas as pd
from pathlib import Path
from src.knowledge_based_workflow.data_treatment.standardization import DatasetStandardization
from src.knowledge_based_workflow.data_treatment.outliers import process_all_datasets # , data_unification
from src.knowledge_based_workflow.data_treatment.derivative import compute_derivatives_for_datasets
from src.knowledge_based_workflow.data_treatment.processing import processing_data
from src.knowledge_based_workflow.data_treatment.ead import compute_ead


class DataTreatment:
    def __init__(self, path):
        self.path = path
        self.dataset_files = sorted(glob(f"{path}/BR*.xls"))
        self.datasets = [DatasetStandardization(f) for f in self.dataset_files]
        self.br_id_list = [file.stem for file in Path(path).iterdir() if file.suffix == ".xls"]
        self.variable_list = ["t", "X", "S", "V", "P", "T"]

        # =================== Data treatment =================== 
        smoothed_data, treat_data = process_all_datasets(datasets = self.datasets, time_col = "time", variable_list = self.variable_list, 
                                                        results_root="results/data_analysis/outliers_and_smoothing", smooth=False)
        # =================== Derivates calculation =================== 
        _, self.data_sets = compute_derivatives_for_datasets(smoothed_data, variables=("X", "S", "V", "P"), 
                                                        results_root="results/data_analysis/derivatives/treat")

        # =================== Computes qP and mu calculation ===================  
    def data_frame(self, yaml_path: str = "src/config/default_parameters.yaml"):         # yaml_path = "src/config/default_parameters.yaml"

        df_global, df_induction = processing_data(self.data_sets, yaml_path, t_ind_exp = True) 
        df_induction.to_excel("data/processed/BR_processed_ind.xlsx",index=False,engine="openpyxl")
        df_global.to_excel("data/processed/BR_processed.xlsx",index=False,engine="openpyxl")
        
        return df_global, df_induction