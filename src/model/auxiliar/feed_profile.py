
"""
Nanobody-based Antivenom Production with E. coli Reactor Simulation 

Several classes feed profile are defined here, including:
- ConstantFeed: constant feed rate
- LinearFeed: linear feed rate
- ExponentialFeed: exponential feed rate
- OnOffFeed: feed rate that switches between ON and OFF at specified intervals
- OnOffFeed_Linear: feed rate that switches between ON and OFF at specified intervals, with linear variation during ON periods

Author: Juan Camilo Castaño Sanchez
Email: jcastano-san@insa-toulose.fr
Date: 01/09/2026
"""

import numpy as np

class BaseFeed:
    def __call__(self, t):
        raise NotImplementedError

class ConstantFeed(BaseFeed):
    def __init__(self, F0):
        self.F_const = F0

    def __call__(self, t):
        return self.F_const

class LinearFeed(BaseFeed):
    def __init__(self, F0, slope):
        self.F0 = F0
        self.slope = slope

    def __call__(self, t):
        return self.F0 + self.slope * t

class ExponentialFeed:
    def __init__(self, F0, k):
        self.F0 = F0
        self.k = k
        self.type = type

    def __call__(self, t):
        return self.F0 * np.exp(self.k * t)
    
class OnOffFeed(BaseFeed):
    def __init__(self, intervals):
        self.intervals = intervals
        
    def __call__(self, t):
        for t_start, t_end, flow_rate in self.intervals:
            if t_start <= t <= t_end:
                return flow_rate
        return 0.0  # OFF outside all intervals

class OnOffFeed_Linear(BaseFeed):
    def __init__(self, intervals):
        self.intervals = intervals

    def __call__(self, t):
        n = len(self.intervals)
        for i, (t_start, t_end, slope, intercept) in enumerate(self.intervals):
            if t_start <= t <= t_end:
                is_last = (i == n - 1)
                return (slope * t + intercept), float(is_last == True)
        return 0.0, 0.0  # OFF outside all intervals 
