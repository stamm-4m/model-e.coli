
from glob import glob
from scipy.interpolate import interp1d
import pandas as pd
from src.utils.io import load_yaml

def I_profile(t, br_id, var):
    params = load_yaml("src/config/params.yaml")
    t_ind = params["bioreactor"][br_id]["t_ind"]["value"]
    return float(t >= t_ind)

def I_mu_profile(t, br_id, var):
    params = load_yaml("src/config/params.yaml")
    intervals = params["feeds"][br_id][var]["value"]
    n = len(intervals)
    for i, (t_start, t_end, slope, intercept) in enumerate(intervals):
        if t_start <= t <= t_end:
            is_last = (i == n - 1)
            return float(is_last == True)
    return 0.0

def feed_profile(t, br_id, var):
    params = load_yaml("src/config/params.yaml")
    intervals = params["feeds"][br_id][var]["value"]
    n = len(intervals)
    for i, (t_start, t_end, slope, intercept) in enumerate(intervals):
        if t_start <= t <= t_end:
            return (slope * t + intercept)
    return 0.0

def feed_concentration_profile(t, br_id, var):
    params = load_yaml("src/config/params.yaml")
    return params["bioreactor"][br_id][f"{var}f"]["value"]

def var_profile(t, br_id, var, raw=True):
    cfg = load_yaml("src/config/project.yaml")
    model_inst = load_yaml("src/config/model.yaml")
    if raw == True:
        path = cfg["paths"]["raw_data"]
        filepath = glob(f"{path}/{br_id}.xls")
        df = pd.read_excel(filepath,header=0,skiprows=[1])
        exp= df[model_inst["model_instructions"]["variables"][var]]
    else:
        filepath = glob(cfg["paths"]["processed_data"])
        df_all = pd.read_excel(filepath,header=0,skiprows=[1])
        df = df_all.loc[df_all["Run ID"]==br_id].copy()
        exp= df[var]
    t_exp= df["Time"]
    function = interp1d(t_exp, exp, fill_value="extrapolate", bounds_error=False)
    value = function(t)
    return round(float(value), 3)