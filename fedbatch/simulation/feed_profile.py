import numpy as np

class BaseFeed:
    def F(self, t):
        raise NotImplementedError

class ConstantFeed(BaseFeed):
    def __init__(self, F0):
        self.F_const = F0

    def F(self, t):
        return self.F_const

class LinearFeed(BaseFeed):
    def __init__(self, F0, slope):
        self.F0 = F0
        self.slope = slope

    def F(self, t):
        return self.F0 + self.slope * t

class ExponentialFeed:
    def __init__(self, F0, k):
        self.F0 = F0
        self.k = k
        self.type = type

    def F(self, t):
        return self.F0 * np.exp(self.k * t)
    
class OnOffFeed(BaseFeed):
    def __init__(self, intervals):
        self.intervals = intervals
        """
        intervals: list of tuples (t_start, t_end, flow_rate)
        Example:
        [
            (0.0, 5.0, 10.0),   # F = 10 from t=0 to t=5
            (5.0, 8.0, 3.0),    # F = 3 from t=5 to t=8
            (10.0, 15.0, 7.0)   # F = 7 from t=10 to t=15
        ]
        """
    def F(self, t):
        for t_start, t_end, flow_rate in self.intervals:
            if t_start <= t <= t_end:
                return flow_rate
        return 0.0  # OFF outside all intervals

class OnOffFeed_Linear(BaseFeed):
    def __init__(self, intervals):
        """
        intervals: list of tuples (t_start, t_end, slope, intercept)

        Example:
        [
            (0.0, 3.4, 0.0,     0.0),      # OFF
            (3.4, 8.0, 0.013,  0.0015),    # Fs = 0.013*t + 0.0015
            (8.0, 20.0, 0.0001, 0.0015)    # Fs = 0.0001*t + 0.0015
        ]
        """
        self.intervals = intervals

    def F(self, t):
        n = len(self.intervals)
        for i, (t_start, t_end, slope, intercept) in enumerate(self.intervals):
            if t_start <= t <= t_end:
                is_last = (i == n - 1)
                return (slope * t + intercept), float(is_last == True)
        return 0.0, 0.0  # OFF outside all intervals 
