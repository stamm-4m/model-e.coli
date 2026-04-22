
import pandas as pd
from fedbatch.utils.io import get_br_id

def per_dataset_metrics_to_df(per_dataset_metrics):
    rows = []

    for dataset, content in per_dataset_metrics.items():
        reg = content["regression"]
        ic = content["information_criteria"]

        for var, metrics in reg.items():
            row = {
                "dataset": dataset,
                "variable": var,
                **metrics,
                **ic
            }
            rows.append(row)

    return pd.DataFrame(rows)


def dict_to_single_row_df(d, index_name="global"):
    return pd.DataFrame([d], index=[index_name])


def solution_to_df(sol, state_names=("X", "S", "P")):
    data = {"time": sol.t}
    for i, name in enumerate(state_names):
        data[name] = sol.y[i]
    return pd.DataFrame(data)


def save_fitting_outputs_to_excel(
    output_path,
    datasets,
    per_dataset_metrics,
    global_metrics,
    global_ic,
    residuals,
    solutions,
    state_names=("X", "S", "P")
):
    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:

        # ---- Global metrics ----
        dict_to_single_row_df(global_metrics).to_excel(
            writer, sheet_name="global_metrics"
        )

        dict_to_single_row_df(global_ic).to_excel(
            writer, sheet_name="global_ic"
        )

        # ---- Per-dataset metrics ----
        per_dataset_df = per_dataset_metrics_to_df(per_dataset_metrics)
        per_dataset_df.to_excel(
            writer, sheet_name="per_dataset_metrics", index=False
        )

        # ---- Residuals ----
        pd.DataFrame({"residual": residuals}).to_excel(
            writer, sheet_name="residuals", index=False
        )

        # ---- Solutions (one sheet per dataset) ----
        for dataset, sol in zip(datasets, solutions.values()):
            df_sol = solution_to_df(sol, state_names)
            br_id = get_br_id(dataset)  
            sheet = f"solution_{br_id}"
            df_sol.to_excel(writer, sheet_name=sheet, index=False)
