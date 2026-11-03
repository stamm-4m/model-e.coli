"""
Nanobody-based Antivenom Production with E. coli Reactor Simulation 

Returns the induction profile based on the configuration provided in the `cfg` dictionary.

Author: Juan Camilo Castaño Sanchez
Email: jcastano-san@insa-toulose.fr
Date: 01/09/2026
"""

class InductionProfile:
    def __init__(self, t_ind, br_id):
        self.t_ind = t_ind
        self.br_id = br_id

    def __call__(self, t):
        if self.br_id in ( "BR01",  "BR09"):
            return float(0), self.t_ind
        else:
            return float(t >= self.t_ind), self.t_ind

