
import joblib
import yaml
import numpy as np
import os
from pathlib import Path


class DataDrivenModel:

    def __init__(self, hybrid=False, models_folder=None, models_folder_P=None, ensemble_mode="fold", PMLmodel = False):

        self.params = {}
        
        self.models = {}
        self.feature_orders = {}
        self.models_name = {}

        self.models_P = {}
        self.feature_orders_P = {}
        self.models_name_P = {}

        self.use_qp = False
        self.use_rp = False
        self.use_induction = False

        self.hybrid = hybrid
        self.ensemble_mode = ensemble_mode

        self.PMLmodel = PMLmodel
        self.use_PML = False

        # self.models = {}
        # self.feature_orders = {}
        # self.models_name = {}

        # self.models_P = {}
        # self.feature_orders_P = {}
        # self.models_name_P = {}

        # self.use_qp = False
        # self.use_rp = False

        if hybrid:

            folder_lower = models_folder.lower()
            if "qp" in folder_lower:
                self.use_qp = True
            if "rp" in folder_lower:
                self.use_rp = True
            if "ind" in folder_lower:
                self.use_induction = True
            
            models_path = Path(models_folder)

            for subdir in models_path.iterdir():

                if subdir.is_dir():
                    br_id = subdir.name
                    pkl_files = list(subdir.glob("*.pkl"))
                    self.models[br_id] = []
                    self.feature_orders[br_id] = []
                    self.models_name[br_id] = []

                    for file in pkl_files:
                        model_path = file
                        model = joblib.load(model_path)
                        self.models[br_id].append(model)

                        metadata_path = model_path.with_name(
                            model_path.stem + "_metadata.yaml")

                        with open(metadata_path, "r") as f:
                            meta = yaml.safe_load(f)

                        self.feature_orders[br_id].append(meta["features"])
                        self.models_name[br_id].append(meta["model"])


    def __call__(self, features):

        features = {k: np.float64(v) for k, v in features.items()}
    
        if self.kinetics.use_rp:
            rP = self.kinetics.rp_hybrid(features, self.br_id)
        elif self.kinetics.use_qp:
            qp = self.kinetics.qp_hybrid(features, self.br_id)
            rP = (qp * X)                
        else:
            raise ValueError("No se detectó ni qP ni rP en la ruta del modelo")
        
        rP = np.clip(rP, 0, 10)

        return rP
    
    # ---------------------------------------------------
    #               hybrid kinetic parameters
    # ---------------------------------------------------
    
    def qp_hybrid(self, features, br_id):
        if self.use_induction and features.get("I", 1) == 0:
                return 0
        
        low_value = 1e-6 # 0
        qp = self._predict_hybrid(features, br_id, low_value)

        return qp    
                                 
    def rp_hybrid(self, features, br_id):

        if self.use_induction and features.get("I", 1) == 0:
            return 0
        
        low_value = 1e-5 # 0
        rp = self._predict_hybrid(features, br_id, low_value)

        return rp