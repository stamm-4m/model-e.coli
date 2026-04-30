
import pandas as pd
from glob import glob
from fedbatch.data_analysis.preprocessing import ( 
    unificar_xls, run_EAD, phase_numeric_summary, mean_ratio_by_phase, within_phase_temporal_drift,
    drift_to_dict, mean_ratios_to_dict, feature_target_corr_by_phase,
    influence_check, outliers_by_phase, outlier_summary_to_dict, influence_to_dict,
    global_influence_to_dict, global_outliers_to_dict, global_outliers, # , global_influence_check
    stability_bootstrap, stability_summary, stability_to_yaml
    )
from fedbatch.data_analysis.data_plots import timeseries_per_run, scatter_fun, boxplots_by_phase
from fedbatch.utils.io import save_yaml

# Import and Unification of data
dataset_files = sorted(glob("data/raw/BR*.xls"))
save_dir="results/data_analysis/global/spline" # path for spline plots
yaml_path = "fedbatch/config/default_parameters.yaml"

df_global, df_batch, df_fedbatch, df_induction = unificar_xls(dataset_files, yaml_path, save_dir)

# # Save global dataframe
# df_global.to_csv("data/processed/BR_unified.csv", index=False)
# df_global.to_excel("data/processed/BR_unified.xlsx",index=False,engine="openpyxl")

# # -----------------Global Analysis based on EAD (1) ------------------------------
# # EAD
# variables = ["time","X", "S", "V", "P", "mu", "qP", "I", "T", "A"] # qp_Old

# save_dir="results/data_analysis/global/EAD"
# run_EAD(df_global, variables, save_dir)

# variables = ["X", "S", "V", "P", "mu", "qP", "I", "T", "A"] # qp_Old

# # Time series overlaped
# save_dir="results/data_analysis/global/time_series"
# for variable in variables:
#         timeseries_per_run(df_global, variable, save_dir)

# # Scatter
# save_dir="results/data_analysis/global/scatter"
# for i, x in enumerate(variables):
#     for y in variables[i+1:]:
#         scatter_fun(df_global, x, y, save_dir)

# # -----------------Batch Analysis------------------------------
# # EAD
# variables = ["time", "X", "S", "V", "mu", "T", "A"] # qp_Old # I , P, qP

# save_dir="results/data_analysis/batch/EAD"
# run_EAD(df_batch, variables, save_dir)

# variables = ["X", "S", "V", "mu", "T", "A"] # qp_Old # I , P, qP

# # Time series overlaped
# save_dir="results/data_analysis/batch/time_series"
# for variable in variables:
#         timeseries_per_run(df_batch, variable, save_dir)

# # Scatter
# save_dir="results/data_analysis/batch/scatter"
# for i, x in enumerate(variables):
#     for y in variables[i+1:]:
#         scatter_fun(df_batch, x, y, save_dir)
        
# # -----------------Fedbatch Analysis------------------------------
# # EAD
# variables = ["time", "X", "S", "V", "mu", "T", "A"] # qp_Old # I, P, qP

# save_dir="results/data_analysis/fed-batch/EAD"
# run_EAD(df_fedbatch, variables, save_dir)

# variables = ["X", "S", "V", "mu", "T", "A"] # qp_Old # I, P, qP

# # Time series overlaped
# save_dir="results/data_analysis/fed-batch/time_series"
# for variable in variables:
#         timeseries_per_run(df_fedbatch, variable, save_dir)

# # Scatter
# save_dir="results/data_analysis/fed-batch/scatter"
# for i, x in enumerate(variables):
#     for y in variables[i+1:]:
#         scatter_fun(df_fedbatch, x, y, save_dir)
        
# # -----------------Induction Analysis------------------------------
# # EAD
# variables = ["time", "X", "V", "P", "mu", "qP", "T"] # qp_Old # S, A , I

# save_dir="results/data_analysis/induction/EAD"
# run_EAD(df_induction, variables, save_dir)

# variables = ["X", "V", "P", "mu", "qP", "T"] # qp_Old # S, A , I

# # Time series overlaped
# save_dir="results/data_analysis/induction/time_series"
# for variable in variables:
#         timeseries_per_run(df_induction, variable, save_dir)

# # Scatter
# save_dir="results/data_analysis/induction/scatter"
# for i, x in enumerate(variables):
#     for y in variables[i+1:]:
#         scatter_fun(df_induction, x, y, save_dir)

# # ----------  Phases' analysis and temporality (2) ---------------------

dfs_by_phase = {
    "phase_A": df_batch,
    "phase_B": df_fedbatch,
    "phase_C": df_induction,
}

numeric_features = [
    "X", "S", "V", "P", "mu", "qP", "T", "A", "I"
]

phase_summary = phase_numeric_summary(dfs_by_phase, numeric_features)

mean_ratios_A = mean_ratio_by_phase(phase_summary, ref_phase="phase_A")
mean_ratios_B = mean_ratio_by_phase(phase_summary, ref_phase="phase_B")
mean_ratios_C = mean_ratio_by_phase(phase_summary, ref_phase="phase_C")

mean_ratios_dict = {
    "ref_phase_A": mean_ratios_to_dict(mean_ratios_A),
    "ref_phase_B": mean_ratios_to_dict(mean_ratios_B),
    "ref_phase_C": mean_ratios_to_dict(mean_ratios_C),
}

# drift_A = within_phase_temporal_drift(df_batch, "time", numeric_features)
# drift_B = within_phase_temporal_drift(df_fedbatch, "time", numeric_features)
# drift_C = within_phase_temporal_drift(df_induction, "time", numeric_features)

drift_G = within_phase_temporal_drift(df_global, "time", numeric_features)

drift_dict = {
    "phase_Global": drift_to_dict(drift_G)
#     "phase_A": drift_to_dict(drift_A),
#     "phase_B": drift_to_dict(drift_B),
#     "phase_C": drift_to_dict(drift_C),
}

phase_cv = (
    phase_summary
    .pivot(index="feature", columns="phase", values="cv")
    .round(3)
    .to_dict()
)

# PHASE_DEPENDENT_RATIO_THRESHOLD = 1.5
# STABILITY_CV_THRESHOLD = 0.3

# phase_dependent_features = sorted({
#     feature
#     for ref_phase, ratios in mean_ratios_dict.items()
#     for feature, ratio in ratios.items()
#     if ratio > PHASE_DEPENDENT_RATIO_THRESHOLD
#        or ratio < 1 / PHASE_DEPENDENT_RATIO_THRESHOLD
# })

# globally_stable_features = sorted({
#     feature
#     for feature, cvs in phase_cv.items()
#     if max(cvs.values()) < STABILITY_CV_THRESHOLD
# })

phase_yaml_data = {
    "time_col": "time",
    "phases": list(dfs_by_phase.keys()),
    "numeric_features": numeric_features,
    "stability": {
        "within_phase_cv": phase_cv 
        # CV [0 1] high (> 0.5) → posible internal inestability
        # CV [0 1] high (< 0.2) → estable
        # CV different between phases → phase dependent feature 
        },
    "mean_ratios": mean_ratios_dict,
        # 1 comparable between phases // 1.3 ~ 2 relevant changes // 2< and 0.5> structural changes
    "temporal_drift": {
        "rolling_window": 50,
        "cv_by_phase": drift_dict,
        },
        # < 0.1    : Stable (without drift)
        # 0.1 - 0.3: soft Drift 
        # 0.3 - 0.6: moderate Drift 
        # 0.6 - 1.0: strong Drift 
        # > 1.0    : Sub-phases or regimes mixture 
#     "phase_dependent_features": phase_dependent_features,
#     "globally_stable_features": globally_stable_features
}

# save_yaml(phase_yaml_data,"results/data_analysis/global/yaml_files/phase_analysis.yaml")

boxplots_by_phase(dfs_by_phase, numeric_features, "results/data_analysis/global")

corr_df = feature_target_corr_by_phase(
    dfs_by_phase = {"phase_C": df_induction},
    numeric_features = numeric_features,
    target="qP"
)

corr_by_phase = (
    corr_df
    .set_index(["phase", "feature"])["correlation"]
    .round(3)
    .unstack(level=0)
    .to_dict()
)

phase_yaml_data["feature_target_correlation"] = {
    "target": "qP",
    "by_phase": corr_by_phase
}

save_yaml(phase_yaml_data,"results/data_analysis/global/yaml_files/phase_analysis.yaml")

# ----------------------------- Outliers  (3) -------------------------------
# -----Global --------
global_outlier_summary = global_outliers(df_global, numeric_features)

records = []
for feature in numeric_features:
    if feature == "qP":
        continue

    corr_all, corr_clean = influence_check(df_global,feature,target="qP")

    records.append({
        "feature": feature,
        "corr_all": corr_all,
        "corr_no_outliers": corr_clean,
        "delta_corr": corr_clean - corr_all
    })

global_influence_df = pd.DataFrame(records)

global_outliers_yaml = global_outliers_to_dict(global_outlier_summary)

global_influence_yaml = global_influence_to_dict(global_influence_df)

outlier_global_yaml = {
    "step": "outliers_and_influence_global",
    "scope": "global",
    "outlier_method": {
        "method": "IQR",
        "k": 1.5,
        "defined_per": "global",
    },
    "outliers": global_outliers_yaml,
    "influence_on_target": {
        "target": "qP",
        "global": global_influence_yaml,
    },
}

save_yaml(outlier_global_yaml,"results/data_analysis/global/yaml_files/global_outliers_analysis.yaml")

# ------------per phase ------------------

outlier_summary = outliers_by_phase(dfs_by_phase, numeric_features)

records = []
# for phase, df in dfs_by_phase.items():
df = df_induction
phase = "phase_C"
for feature in numeric_features:
        if feature == "qP":  # target
            continue

        corr_all, corr_clean = influence_check(
            df,
            feature,
            target="qP"
        )

        records.append({
            "phase": phase,
            "feature": feature,
            "corr_all": corr_all,
            "corr_no_outliers": corr_clean,
            "delta_corr": corr_clean - corr_all
        })

influence_df = pd.DataFrame(records)

outliers_yaml = outlier_summary_to_dict(outlier_summary)
influence_yaml = influence_to_dict(influence_df)


outliers_yaml_data = {
    "step": "outliers_and_influence",
    "outlier_method": {
        "method": "IQR",
        "k": 1.5,
        "defined_per": "phase",
    },
    "outliers": outliers_yaml,
    "influence_on_target": {
        "target": "qP",
        "by_phase": influence_yaml,
    },
}

save_yaml(outliers_yaml_data,"results/data_analysis/global/yaml_files/outliers_analysis.yaml")

# -------------------- Runs stability (4) -------------

stability_df = stability_bootstrap(
    df=df_global,
    features=numeric_features,
    target="qP",
    n_runs=200,
    sample_frac=0.8,
)

summary_df_global = stability_summary(stability_df)
stability_yaml = stability_to_yaml(summary_df_global)

gloabl_yaml_data = {
    "step": "stability_between_runs",
    "phase": "Global",
    "target": "qP",
    "method": "bootstrap",
    "n_runs": 200,
    "sample_fraction": 0.8,
    "stability_metrics": stability_yaml,
}

save_yaml(gloabl_yaml_data,"results/data_analysis/global/yaml_files/stability_yaml_global.yaml")

stability_df = stability_bootstrap(
    df=dfs_by_phase["phase_C"],
    features=numeric_features,
    target="qP",
    n_runs=200,
    sample_frac=0.8,
)

summary_df_induction = stability_summary(stability_df)
stability_yaml = stability_to_yaml(summary_df_induction)

induction_yaml_data = {
    "step": "stability_between_runs",
    "phase": "phase_C",
    "target": "qP",
    "method": "bootstrap",
    "n_runs": 200,
    "sample_fraction": 0.8,
    "stability_metrics": stability_yaml,
}

save_yaml(induction_yaml_data,"results/data_analysis/global/yaml_files/stability_yaml_induction.yaml")

# ------------------- Domain filter -----------------------
# KEEP : T, I, X
# MAYBE: mu (depends on X and S) indirect relation with qP // redundant
#        P (!): DEPENDS STRONGLY ON EXTREM VALUES OF THE FEATURE (outlier analysis)
#               Do P and qP depend on other factor behind ?
# NOT KEEP: A, S, V (pase indicators or control variables)
# 
# See summary on domain_filter.yaml on results/global/yaml_files folder
