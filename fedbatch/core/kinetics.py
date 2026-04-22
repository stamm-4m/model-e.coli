import numpy as np

class Kinetic_Models:

    def __init__(self):
        self.params = {}

    def set_params(self, params):

        if not isinstance(params, dict):
            raise TypeError("set_params expects a dict {param_name: value}")
        
        self.params.update(params)

        """
        params: dict {param_name: value}
        """
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