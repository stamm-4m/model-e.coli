from src.knowledge_based_workflow.data_treatment.treatment_main import DataTreatment
from src.knowledge_based_workflow.data_treatment.ead import compute_ead

class DataAnalysisWorkflow:
    def __init__( self, raw_data_path="data/raw", yaml_path="src/config/params.yaml", ead_path="results/data_analysis/ead"):

        self.raw_data_path = raw_data_path
        self.yaml_path = yaml_path
        self.ead_path = ead_path

        self.vars = ["X", "S", "P", "V", "mu", "qP", "qP_2", "rX", "rP", "dXdt", "dSdt", "dPdt", "dVdt", "T", "I"]

    def run(self):

        print("Running data treatment...")
        treat_data = DataTreatment( self.raw_data_path )

        df_global, df_induction = ( treat_data.data_frame( self.yaml_path ) )

        dfs = { "global": df_global, "induction": df_induction }

        print("Computing EAD...")

        for name, df in dfs.items():
            compute_ead(  df, self.vars, results_root=f"{self.ead_path}/{name}" )

        print("Data analysis completed.")


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