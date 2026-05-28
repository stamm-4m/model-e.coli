
class FedBatchModel:
    def __init__(self, balances, feed_S, feed_A): # induction_signal
        self.balances = balances
        self.feed_S = feed_S
        self.feed_A = feed_A
        # self.induction_signal = induction_signal

    def dfdt(self, t, y):
        FS, ind_F = self.feed_S.F(t)
        FA, _ = self.feed_A.F(t)
        FA = FA * 1e-6
        return self.balances.dfdt(t, y, FS, FA, ind_F)
    
