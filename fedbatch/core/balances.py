import numpy as np

class FedBatchBalances:
    def __init__(self, kinetics, Sf, temperature_profile, induction_profile):
        self.kinetics = kinetics
        self.Sf = Sf
        self.temperature = temperature_profile
        self.induction_P = induction_profile

    def dfdt(self, t, state, FS, FA, ind_F):
        X, S, P, V = state # T also

        # X = max(X,0)
        # S = max(S,0)
        # P = max(P,0)
        # V = max(V,0)
        
        T = self.temperature.F(t)   
        induction = self.induction_P.F(t)

        # Kinetic
        mu = self.kinetics.mu(X, S, T, ind_F)
        # mu = max(mu,0)

        qp = self.kinetics.qp(X, S, T, induction)

        Y_XS = self.kinetics.Y_XS 
        m = self.kinetics.m

        # Mass Balances
        dVdt = FS + FA

        dXdt =              mu * X - (dVdt * X / V)
        dSdt = - (mu/Y_XS + m) * X + (dVdt * (self.Sf - S) / V)
        dPdt =  (qp * X) - (dVdt * P / V) # + (mu * P)

        # Energy Balance
        # dTdt = 0
        # dTdt = - (V * sum_Hrxn_rx + UA * (T-Ta)) / mCp # for batch reactor

        return np.array([dXdt, dSdt, dPdt, dVdt])