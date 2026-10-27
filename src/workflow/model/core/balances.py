"""
Nanobody-based Antivenom Production with E. coli Reactor Simulation 
Class: Balance equations for fed-batch reactor

Author: Juan Camilo Castaño Sanchez
Email: jcastano-san@insa-toulose.fr
Date: 01/09/2026

"""

import numpy as np

class FedBatchBalances:
    def __init__(self, kinetics, temperature_profile, induction_profile, Sf, feed_S, feed_A, br_id, rP_scenario=0, hybrid=False, DataDrivenModel=None):
        self.kinetics = kinetics
        self.Sf = Sf
        self.temperature = temperature_profile
        self.induction = induction_profile
        self.br_id = br_id
        self.feed_S = feed_S
        self.feed_A = feed_A
        self.rP_scenario = rP_scenario
        self.hybrid = hybrid
        self.DataDrivenModel = DataDrivenModel

    def parametric_rates(self, state, T, induction, ind_F):
        X, S, P, V = state 
        
        mu = self.kinetics.mu(S, T, ind_F) 
        mu = max(mu,0)
        qp = self.kinetics.qp(S, T, induction, ind_F)

        Y_XS = self.kinetics.Y_XS 
        m = self.kinetics.m

        rX = mu * X
        rS = - (mu/Y_XS + m) * X

        if self.rP_scenario == 1:
            rP = (qp * X)  + (mu * P)
        else:
            rP = (qp * X)
        
        return rX, rS, rP
    
    def ODEs(self, t, state):

        X, S, P, V = state 
        T = self.temperature(t)

        FS, ind_F = self.feed_S(t)
        induction, _ = self.induction(t)
        FA, _ = self.feed_A(t)
        FA = FA * 1e-6

        rX, rS, rP = self.parametric_rates(state, T, induction, ind_F)

        if self.hybrid:
            features =  { 
                "X": X,
                "S": S,
                "V": V,
                "T": T,
                "I": induction,
                "mu": rX/X, 
                "FS": FS }
            rP = rP # self.DataDrivenModel(features)  

        dVdt = FS + FA 
        dXdt = rX - (dVdt * X / V)
        dSdt = rS + (dVdt * (self.Sf - S) / V)
        dPdt = rP - (dVdt * P / V) 

        return np.array([dXdt, dSdt, dPdt, dVdt])
    