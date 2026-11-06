
from src.utils.io import load_yaml
from src.workflow.data_treatment.treatment_main import DataTreatment
from src.workflow.data_treatment.ead import compute_ead

class DataAnalysisWorkflow:
    def __init__(self, raw_data_path="data/raw", params_path="src/config/params.yaml", 
                 model_path="src/config/model.yaml", ead_path="results/data_analysis/ead",
                 ignore_variables=None, states_variables=None, non_states_cfg=None):

        self.raw_data_path = raw_data_path
        self.model_path = model_path
        self.params_path = params_path
        self.ead_path = ead_path
        self.model_ins = load_yaml(model_path)["model_instructions"]
        self.vars = self._build_variable_list()
        self.ignore_vars = ignore_variables
        self.states_variables = states_variables
        self.non_states_cfg = non_states_cfg

    def run(self):

        print("Running data treatment...")
        treat_data = DataTreatment( self.raw_data_path, self.model_path, self.ignore_vars, 
                                   self.states_variables, self.non_states_cfg)
        df_global, df_induction = ( treat_data.data_frame( self.params_path ) )
        dfs = { "global": df_global, "induction": df_induction }
        print("Computing EAD...")
        for name, df in dfs.items():
            compute_ead(  df, self.vars, results_root=f"{self.ead_path}/{name}" )
        print("Data analysis completed.")

    def _build_variable_list(self):

        cfg = self.model_ins["all_variables"]
        vars_ = []
        ignore = set(cfg.get("ignore", []))
        # States + derivatives + rates
        if "states" in cfg["variables"]:
            for state, state_cfg in self.model_ins["states"].items():
                vars_.append(state)
                vars_.append(f"d{state}dt")
                vars_.append(state_cfg["generation"])
        # Inputs
        if "inputs" in cfg["variables"]:
            vars_.extend(self.model_ins["inputs"]["consider"].keys())

        # Kinetics
        if "kinetic_functions" in cfg["variables"]:
            vars_.extend(self.model_ins["kinetic_functions"].keys())

        # Volume
        if "V" in cfg["variables"]:
            V = self.model_ins["volume"]["name"]
            vars_.append(V)
            vars_.append(f"d{V}dt")

        # Remove ignored variables
        vars_ = [v for v in dict.fromkeys(vars_)if v not in ignore]

        return vars_


# path_raw_files = Path("data/raw")
# yaml_path = "src/config/params.yaml"
# config = load_yaml(yaml_path)
# vars = ["X", "S", "P", "V", "mu", "qP", "qP_2", "rX", "rP", "dXdt", "dSdt", "dPdt", "dVdt", "T", "I"]

# # =================== Data-frame generation ===================
# treat_data = DataTreatment(path_raw_files)
# df_global, df_induction = treat_data.data_frame(yaml_path)

# # =================== Data-frame load ===================
# # df_global = pd.read_excel(r"data/processed/BR_processed.xlsx")
# # df_induction = pd.read_excel(r"data/processed/BR_processed_ind.xlsx")

# dfs = {"global": df_global, "induction": df_induction}

# # DataAnalysisWorkflow().run()

# # =================== Computation ===================
# for name, df in dfs.items():
#     print(f"\n===== {name.upper()} =====")
#     # ---------- paths ----------
#     base_ead = f"results/data_analysis/ead/{name}"
#     # =================== EAD =================== 
#     compute_ead(df, vars, results_root=f"{base_ead}")