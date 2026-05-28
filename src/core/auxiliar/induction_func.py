
class InductionProfile:
    def __init__(self, t_ind):
        self.t_ind = t_ind

    def F(self, t):
        return float(t >= self.t_ind)

