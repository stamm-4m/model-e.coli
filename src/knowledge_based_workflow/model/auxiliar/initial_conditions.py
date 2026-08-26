"""
Nanobody-based Antivenom Production with E. coli Reactor Simulation 

Returns the initial state of the system based on the configuration provided in the `cfg` dictionary or in dataset.

Author: Juan Camilo Castaño Sanchez
Email: jcastano-san@insa-toulose.fr
Date: 01/09/2026
"""

import numpy as np 

class DatasetInitialState:
    def __init__(self, dataset):
        self.dataset = dataset

    def __call__(self):
        X0 = self.dataset.y0[0]
        S0 = self.dataset.y0[1]
        P0 = self.dataset.y0[2]
        V0 = self.dataset.y0[3]

        return np.array([X0, S0, P0, V0], dtype=float)

class ConfigInitialState:
    def __init__(self, cfg, br_id):
        self.cfg = cfg 
        self.br_id = br_id

    def __call__(self):
        X0 = self.cfg["bioreactor"][self.br_id]["X0"]["value"]  
        S0 = self.cfg["bioreactor"][self.br_id]["S0"]["value"]  
        P0 = self.cfg["bioreactor"][self.br_id]["P0"]["value"]  
        V0 = self.cfg["bioreactor"][self.br_id]["V0"]["value"]  

        return np.array([X0, S0, P0, V0], dtype=float)
