
import joblib
import yaml
import numpy as np
import os
from pathlib import Path

class Kinetic_Models:

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

        if PMLmodel:

            folder_lower = models_folder_P.lower()
            if "ind" in folder_lower:
                self.use_induction = True
            
            models_path = Path(models_folder_P)

            for subdir in models_path.iterdir():

                if subdir.is_dir():
                    br_id = subdir.name
                    pkl_files = list(subdir.glob("*.pkl"))
                    self.models_P[br_id] = []
                    self.feature_orders_P[br_id] = []
                    self.models_name_P[br_id] = []

                    for file in pkl_files:
                        model_path = file
                        model = joblib.load(model_path)
                        self.models_P[br_id].append(model)

                        metadata_path = model_path.with_name(
                            model_path.stem + "_metadata.yaml")

                        with open(metadata_path, "r") as f:
                            meta = yaml.safe_load(f)

                        self.feature_orders_P[br_id].append(meta["features"])
                        self.models_name_P[br_id].append(meta["model"])
        

    def set_params(self, params):

        if not isinstance(params, dict):
            raise TypeError("set_params expects a dict {param_name: value}")
        
        self.params.update(params)

        self.params.update(params)

        self.mu_max_p = self.params["mu_max_p"]
        self.mu_max_0 = self.params["mu_max_0"]
        self.Ks = self.params["Ks"]
        self.b = self.params["b"]
        self.m = self.params["m"]
        self.Y_XS = self.params["Y_XS"]
        self.alpha = self.params["alpha"]
        self.gamma_1 = self.params["gamma_1"]
        self.Ap_1 = self.params["Ap_1"]
        self.gamma_2 = self.params["gamma_2"]
        self.Ap_2 = self.params["Ap_2"]
        self.sigma = self.params["sigma"]


    # Monod kinetic
    def mu_max_fun(self, T, ind_F):
        if ind_F == 0:
            mu_max = self.mu_max_0
        else:
            mu_max = self.mu_max_p * T + self.b
        return mu_max
    
    def mu(self, X, S, T, ind_F):
        mu_max = self.mu_max_fun(T, ind_F)
        mu = mu_max * S / (self.Ks + S)
        return mu

    # Modified Luedeking Piret kinetic
    def qp(self, X, S, T, induction):
        
        if self.PMLmodel:
            return 0

        if induction == 0:
            qp = 0
        else:
            mu = self.mu(X, S, T, induction)
            beta = ( 
                ( self.gamma_1 * np.exp(- self.Ap_1 / T)) -
                ( self.gamma_2 * np.exp(- self.Ap_2 / T)) +
                  self.sigma
                )
            qp = - (- self.alpha * mu + beta )
            # qp = abs(- self.alpha * mu + beta )
        return      qp                                      # [mg Nb * L / (g X * h)]
    
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
    
    def PML_model(self, features, br_id):

        if self.use_induction and features.get("I", 1) == 0:
            return 0
        
        low_value = 0
        P = self._predict_ML(features, br_id, low_value)

        return P

    def _predict_hybrid(self, features, br_id, low_value):
        preds = []
        if self.ensemble_mode == "fold":
            if br_id in self.models:
                models_fold = self.models[br_id]
                features_fold = self.feature_orders[br_id]
                models_name = self.models_name[br_id]

                for model, order, name in zip(models_fold, features_fold, models_name):
                    x = self._build_input(features, order)
                    pred = model.predict(x)[0]
                    use_log = name not in ("poisson", "tweedie")
                    if use_log:
                        pred = np.expm1(pred)

                    preds.append(pred)

        elif self.ensemble_mode == "global":
            if "global" in self.models:
                models_global = self.models["global"]
                features_global = self.feature_orders["global"]
                models_name = self.models_name["global"]

                for model, order, name in zip(models_global, features_global, models_name):
                    x = self._build_input(features, order)
                    pred = model.predict(x)[0]
                    use_log = name not in ("poisson", "tweedie")
                    if use_log:
                        pred = np.expm1(pred)
                    preds.append(pred)

        if not preds:
            return 0.0
        
        value = np.mean(preds)
        # weights = np.array([2.0 if p > 0 else 1.0 for p in preds])
        # value = np.sum(weights * preds) / np.sum(weights)

        value = max(value, low_value)

        return value
    
    def _predict_ML(self, features, br_id, low_value):
        preds = []
        if self.ensemble_mode == "fold":
            if br_id in self.models_P:
                models_fold = self.models_P[br_id]
                features_fold = self.feature_orders_P[br_id]
                models_name = self.models_name_P[br_id]

                for model, order, name in zip(models_fold, features_fold, models_name):
                    x = self._build_input(features, order)
                    pred = model.predict(x)[0]
                    use_log = name not in ("poisson", "tweedie")
                    if use_log:
                        pred = np.expm1(pred)

                    preds.append(pred)

        elif self.ensemble_mode == "global":
            if "global" in self.models_P:
                models_global = self.models_P["global"]
                features_global = self.feature_orders_P["global"]
                models_name = self.models_name_P["global"]

                for model, order, name in zip(models_global, features_global, models_name):
                    x = self._build_input(features, order)
                    pred = model.predict(x)[0]
                    use_log = name not in ("poisson", "tweedie")
                    if use_log:
                        pred = np.expm1(pred)
                    preds.append(pred)

        if not preds:
            return 0.0
        
        value = np.mean(preds)
        # weights = np.array([2.0 if p > 0 else 1.0 for p in preds])
        # value = np.sum(weights * preds) / np.sum(weights)

        value = max(value, low_value)
        value = min(value, 2.5)

        return value

    def _build_input(self, features, feature_order):
        try:
            x = [features[f] for f in feature_order]
        except KeyError as e:
            raise ValueError(f"Feature missing: {e}")

        return np.array(x).reshape(1, -1)
