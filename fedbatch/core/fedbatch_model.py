
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
    
    # def get_discontinuity_times(self, t0, tf):
        # times = set()

        # # Feed S
        # for t_start, t_end, *_ in self.feed_S.intervals:
            # times.add(t_start)
            # times.add(t_end)

        # # Feed A
        # for t_start, t_end, *_ in self.feed_A.intervals:
            # times.add(t_start)
            # times.add(t_end)

        # # # Induction
        # # if hasattr(self.induction_signal, "t_ind"):
        # #     times.add(self.induction_signal.t_ind)

        # # Keep only relevant times
        # return sorted(t for t in times if t0 < t < tf)
