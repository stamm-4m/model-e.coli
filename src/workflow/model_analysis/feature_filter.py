
class FeaturesFilterWorkflow:
    def __init__(self, data_path="data/processed/BR_processed.xlsx", param_names=None):
        self.data_path = data_path
        self.param_names = param_names
        self.vars = ["X", "S", "V", "mu", "rX", "dXdt", "dSdt", "dVdt", "T", "I"]

    def run(self):
        pass
