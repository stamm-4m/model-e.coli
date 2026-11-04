import pandas as pd
import numpy as np
from utils.metrics_io import compute_metrics
from sklearn.inspection import permutation_importance

def objective_function(hyperparameters, model_factory, eval_features, target, df):

        feature_ranks = []
        rmse_train_list = []
        rmse_valid_list = []
        performance_criteria = {}
        all_y_valid = []
        all_y_pred = []

        # LOOCV by BR experiment
        groups = sorted(df["Run_ID"].unique())

        for validation_group in groups:
            train_df = df[df["Run_ID"] != validation_group].copy()
            valid_df = df[df["Run_ID"] == validation_group].copy()

            X_train = train_df[eval_features]
            X_valid = valid_df[eval_features]
            y_train = train_df[target]
            y_valid = valid_df[target]

            # Train model
            model = model_factory()
            model.fit( X_train, y_train )

            y_train_pred = model.predict(X_train)
            y_valid_pred = model.predict(X_valid)

            # Metrics train
            train_metrics = compute_metrics(y_train,y_train_pred)
            valid_metrics = compute_metrics(y_valid,y_valid_pred)

            # Save split metrics
            rmse_train_list.append(train_metrics["RMSE"])
            rmse_valid_list.append(valid_metrics["RMSE"])
            
            performance_criteria[validation_group] = {
                "train": train_metrics,
                "validation": valid_metrics,
            }

            # Feature importance
            try:
                importance = permutation_importance(model, X_valid,y_valid,n_repeats=20,random_state=42,scoring="neg_root_mean_squared_error")
                importance_df = pd.DataFrame({
                        "feature": eval_features,
                        "importance": importance.importances_mean,
                    })

            except Exception:
                importance_df = pd.DataFrame({
                    "feature": eval_features,
                    "importance": np.ones( len(eval_features) ),
                })

            importance_df = (importance_df.sort_values("importance", ascending=False).reset_index(drop=True))
            importance_df["rank"] = (importance_df.index + 1)
            feature_ranks.append(importance_df[["feature", "rank"]].set_index("feature")["rank"])

            # Store global predictions
            all_y_valid.extend(y_valid.tolist())
            all_y_pred.extend(y_valid_pred.tolist())

        # Global metrics
        all_y_valid = np.asarray(all_y_valid)
        all_y_pred = np.asarray(all_y_pred)

        global_metrics = compute_metrics(all_y_valid, all_y_pred)

        # Global feature ranking
        rank_df = pd.concat(feature_ranks, axis=1)
        mean_rank = rank_df.mean(axis=1)
        mean_rank = (mean_rank.sort_values())

        features_list = {
            "names": mean_rank.index.tolist(),
            "mean_rank": mean_rank.values.tolist(),
            "worst_feature": mean_rank.index[-1],
        }

        # Hyperparameters
        try:
            hyper_params = model.get_params()
        except Exception:
            hyper_params = {}

        # Criteria for hyperparameter tuning and feature selection
        tuning_criteria = (
            np.sum( np.abs(np.array(rmse_valid_list) - np.array(rmse_train_list))) +
            np.sum(rmse_valid_list) 
        )

        return tuning_criteria, performance_criteria,  global_metrics, features_list, hyper_params
