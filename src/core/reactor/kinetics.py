
import joblib
import yaml
import numpy as np
import os

class Kinetic_Models:

    def __init__(self, hybrid=False, models_folder=None):

        self.params = {}
        self.models = {}
        self.feature_orders = {}

        self.use_qp = False
        self.use_rp = False
        self.use_induction = False

        self.hybrid = hybrid

        if hybrid:

            folder_lower = models_folder.lower()
            if "qp" in folder_lower:
                self.use_qp = True
            if "rp" in folder_lower:
                self.use_rp = True
            if "ind" in folder_lower:
                self.use_induction = True

            for file in os.listdir(models_folder):

                if file.endswith(".pkl"):
                    br_id = file.split("_")[-1].replace(".pkl", "")

                    model_path = os.path.join(models_folder, file)
                    metadata_path = os.path.join(
                        models_folder,
                        file.replace(".pkl", "_metadata.yaml")
                    )

                    self.models[br_id] = joblib.load(model_path)

                    with open(metadata_path, "r") as f:
                        meta = yaml.safe_load(f)

                    self.feature_orders[br_id] = meta["features"]


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
        
        if self.use_induction:
            if features["I"] == 0:
                return 0

        x = self._build_input(features, br_id)
        qp = self.models[br_id].predict(x)
        out = np.maximum(1e-8, qp[0])

        return      out      
                                   
    
    def rp_hybrid(self, features, br_id):
        
        if self.use_induction:
            if features["I"] == 0:
                return 0

        x = self._build_input(features, br_id)
        rp = self.models[br_id].predict(x)
        out = np.maximum(1e-8, rp[0])

        return     out
                                      

    def _build_input(self, features, br_id):
        order = self.feature_orders[br_id]
        try:
            x = [features[f] for f in order]
        except KeyError as e:
            raise ValueError(f"Feature missing for model {br_id}: {e}")

        return np.array(x).reshape(1, -1)
