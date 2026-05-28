
class InductionProfile:
    def __init__(self, t_ind, br_id):
        self.t_ind = t_ind
        self.br_id = br_id

    def F(self, t):
        if self.br_id == "BR09":
            return float(0)
        else:
            return float(t >= self.t_ind)

