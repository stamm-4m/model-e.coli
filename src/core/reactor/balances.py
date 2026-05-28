import numpy as np

class FedBatchBalances:
    def __init__(self, kinetics, Sf, temperature_profile, volume_profile, induction_profile, br_id):
        self.kinetics = kinetics
        self.Sf = Sf
        self.temperature = temperature_profile
        self.volume = volume_profile
        self.induction_P = induction_profile
        self.br_id = br_id


    def dfdt(self, t, state, FS, FA, ind_F):
        X, S, P, V = state # T also
        
        T = self.temperature.F(t)

        V_real, dV_real = self.volume.F(t)

        induction = self.induction_P.F(t)

        dVdt = FS + FA 
        # dVdt = dV_real

        mu = self.kinetics.mu(X, S, T, ind_F) # mu = max(mu,0)

        dXdt = mu * X - (dVdt * X / V)
        # dXdt = mu * X - (dV_real * X / V_real)

        Y_XS = self.kinetics.Y_XS 
        m = self.kinetics.m

        dSdt = - (mu/Y_XS + m) * X + (dVdt * (self.Sf - S) / V)
        # dSdt = - (mu/Y_XS + m) * X + (dV_real * (self.Sf - S) / V_real)
        
        if self.kinetics.hybrid:

            if not hasattr(self, "prev_X"):
                self.prev_X = X
                self.prev_P = P

            features =  {  # revisar
                "X": X, 
                "S": S, 
                "V": V_real,
                "P": P,
                "T": T,
                "I": induction,
                "mu": mu,          # real?
                "dXdt": dXdt,
                "dSdt": dSdt,
                "dVdt": dV_real,
                "Xlag1": self.prev_X,  
                "Plag1": self.prev_P,
                "X_calc": X,
                "V_calc": V, 
                "mu_calc": mu, 
                "dXdt_calc": dXdt, 
                "dVdt_calc": dVdt
            }

            self.prev_X = X
            self.prev_P = P

            if self.kinetics.use_rp:
                rP = self.kinetics.rp_hybrid(features, self.br_id, induction)
            elif self.kinetics.use_qp:
                qp = self.kinetics.qp_hybrid(features, self.br_id, induction)
                rP = (qp * X)                
            else:
                raise ValueError("No se detectó ni qP ni rP en la ruta del modelo")

        else:
            qp = self.kinetics.qp(X, S, T, induction)
            rP = (qp * X)
            
        dPdt = rP - (dVdt * P / V) # + (mu * P)
        # dPdt = rP - (dV_real * P / V_real) # + (mu * P)

        return np.array([dXdt, dSdt, dPdt, dVdt])