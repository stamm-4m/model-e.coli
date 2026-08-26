import numpy as np

class FedBatchBalances:
    def __init__(self, kinetics, Sf, temperature_profile, volume_profile, biomass_profile, induction_profile, br_id):
        self.kinetics = kinetics
        self.Sf = Sf
        self.temperature = temperature_profile
        self.volume = volume_profile
        self.biomass = biomass_profile
        self.induction_P = induction_profile
        self.br_id = br_id
        self.history = []        
        self.lag_dt = 0.05  # (0.05 a 0.2)        
        self.cached_lag = None
        self.last_t_features = None

    def dfdt(self, t, state, FS, FA, ind_F):
        X, S, P, V = state # T also

        # P = max(P, -1e-8)

        T = self.temperature.F(t)

        # V_real, dV_real = self.volume.F(t)
        # X_real, dX_real, mu_real = self.biomass.F(t)

        induction, t_ind = self.induction_P.F(t)

        dVdt = FS + FA 
        # dVdt = dV_real

        mu = self.kinetics.mu(X, S, T, ind_F) 
        mu = max(mu,0)

        dXdt = mu * X - (dVdt * X / V)
        # dXdt = mu * X - (dV_real * X / V_real)

        Y_XS = self.kinetics.Y_XS 
        m = self.kinetics.m

        dSdt = - (mu/Y_XS + m) * X + (dVdt * (self.Sf - S) / V)
        # dSdt = - (mu/Y_XS + m) * X + (dV_real * (self.Sf - S) / V_real)
        
        if self.kinetics.hybrid:
            t_max = self.biomass.t_max

            # self.history.append((t, X, X_real, P, V, mu))
            # self.history.sort(key=lambda e: e[0])
            tol_t = 1e-8
            if len(self.history) == 0:
                self.history.append((t, X, P))
            elif  t > self.history[-1][0] + tol_t:
                self.history.append((t, X, P))
            else:
                pass

            if len(self.history) > 200:
                self.history.pop(0)

            if self.last_t_features is None or abs(t - self.last_t_features) > 1e-10:
                self.cached_lag = self.get_lagged_values(t)

                V_real, dV_real = self.volume.F(t)
                X_real, dX_real, mu_real = self.biomass.F(t)
                self.cached_features_base = (V_real, dV_real, X_real, dX_real, mu_real)

                self.last_t_features = t

            V_real, dV_real, X_real, dX_real, mu_real = self.cached_features_base

            X_lag, P_lag = self.cached_lag
            # X_lag, P_lag = self.get_lagged_values(t)

            if X_lag is None:
                X_lag = X
                P_lag = P

            features =  {  # exclude features ["S", "dSdt", "X", "dXdt","dXdt_calc", "Xlag1", "dVdt", "dVdt_calc", "mu"]
                # "X": X_real, 
                # "S": S, 
                # "V": V_real,
                "t": t, 
                "t_ind": t-t_ind,
                "t_ind_ad": (t-t_ind)/(t_max-t_ind),
                "P": P,
                "T": T,
                "I": induction,

                # "mu": mu_real, 
                "FS_calc": FS,          
                # "dXdt": dX_real,
                # "dSdt": dSdt,
                # "dVdt": dV_real,

                # "Xlag1": self.prev_X_real, 
                "Xlag1_calc": X_lag,  
                "Plag1": P_lag,

                "X_calc": X,
                "V_calc": V, 
                "mu_calc": mu, 
                "dXdt_calc": dXdt, 
                "dVdt_calc": dVdt
            }

            features = {k: np.float64(v) for k, v in features.items()}

            if self.kinetics.PMLmodel:
                P_ML = self.kinetics.PML_model(features, self.br_id)
                P_ML = np.clip(P_ML, 0, None)
                # features["P"] = P_ML
                features["P"] = np.mean([P, P_ML])
            
            if self.kinetics.use_rp:
                rP = self.kinetics.rp_hybrid(features, self.br_id)
            elif self.kinetics.use_qp:
                qp = self.kinetics.qp_hybrid(features, self.br_id)
                rP = (qp * X)                
            else:
                raise ValueError("No se detectó ni qP ni rP en la ruta del modelo")

            rP = np.clip(rP, 0, 10)

        else:
            qp = self.kinetics.qp(X, S, T, induction)
            rP = (qp * X)

        dPdt = rP - (dVdt * P / V) # + (mu * P)
        # dPdt = rP - (dV_real * P / V_real) # + (mu * P)

        return np.array([dXdt, dSdt, dPdt, dVdt])
    
    
    def get_lagged_values(self, t):

        if len(self.history) < 2:
            return self.history[-1][1], self.history[-1][2]
        
        t_lag = t - self.lag_dt
        if t_lag <= self.history[0][0]:
            return self.history[0][1], self.history[0][2]

        for i in range(len(self.history) - 1):
            t0, X0, P0 = self.history[i]
            t1, X1, P1 = self.history[i + 1]

            if t0 <= t_lag <= t1:
                if t1 == t0:
                    continue
                w = (t_lag - t0) / (t1 - t0)
                X_lag = X0 + w * (X1 - X0)
                P_lag = P0 + w * (P1 - P0)
                return X_lag, P_lag

        return self.history[-1][1], self.history[-1][2]