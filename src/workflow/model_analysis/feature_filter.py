
import pandas as pd
from pathlib import Path
from src.utils.filter import filter_feature_selection, mrmr_selection

class FeaturesFilterWorkflow:
    def __init__(self, data_path="data/processed/BR_processed.xlsx", filter_path="results/filtered_features", obj_param=None, ignore_data=None, ignore_features=None):
        
        self.data_path = Path(data_path)
        self.filter_path = Path(filter_path)

        self.obj_param = obj_param
        self.ignore_data = ignore_data

        self.ignore_features = ignore_features if ignore_features is not None else ["Run_ID", "Run_T", "time"]

        self.df = None
        self.features = None

        self.selection_results = {}
        self.selected_features = {}

    def load_datasets(self):

        self.df = pd.read_excel(self.data_path, header=0)

        if self.ignore_data is not None:
            self.df = self.df[self.df["Run_ID"]!=self.ignore_data].copy()

    def find_candidate_features(self):

        excluded = set( self.obj_param + self.ignore_features )
        self.features = [ c for c in self.df.columns if c not in excluded ]

        return self.features

    def compute_relevance(self):

        for param in self.obj_param:
            target_dir = (  self.filter_path / param )
            result = filter_feature_selection(self.df, self.features, param, target_dir)
            self.selection_results[param] = result

    def run_mrmr(self, n_features=10):

        self.selected_features = {}
        for param in self.obj_param:
            fs = self.selection_results[param]["fs"]
            cmi = self.selection_results[param]["cmi"]
            selected = mrmr_selection(fs_df=fs, cmi_matrix=cmi, n_features=n_features)
            self.selected_features[param] = selected

        return self.selected_features

    def save_results(self):

        rows = []
        for target, feats in self.selected_features.items():
            for rank, feat in enumerate( feats, start=1 ):
                rows.append(
                    { "target": target,
                       "rank": rank,
                       "feature": feat
                    }
                )

        summary = pd.DataFrame(rows)
        summary.to_excel(self.filter_path / "mrmr_selected_features.xlsx", index=False)

        return summary

    def run(self, n_features=10):

        print("Loading datasets...")
        self.load_datasets()

        print("Finding candidate features...")
        self.find_candidate_features()

        print("Running filter selection...")
        self.compute_relevance()

        print("Running mRMR...")
        self.run_mrmr(n_features=n_features)

        print("Saving results...")
        # summary = self.save_results()
        self.save_results()

        print("Feature filtering workflow completed successfully.")
        # return summary