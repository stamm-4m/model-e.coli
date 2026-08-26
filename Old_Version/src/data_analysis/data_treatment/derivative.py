
import numpy as np
from src.utils.io import load_yaml
from src.data_analysis.data_treatment.plots_outlier_derivative import plot_all_derivatives
from src.utils.io import save_yaml, to_python_type, timer
from scipy.interpolate import CubicSpline, UnivariateSpline, make_splrep, splev
from scipy.optimize import minimize_scalar

@timer
def compute_derivatives_for_datasets(
    datasets,
    variables,
    results_root, # "results/derivatives/treat"
):
    print(f"\nDerivatives estimation.\n")

    derivative_datasets = {}

    for br_id, data in datasets.items():

        print(f"Derivatives for {br_id}")

        t = data["time"]
        derivative_datasets[br_id] = {"time": t}
        yaml_summary = {}
        results_per_variable = {}

        for var in variables:
            f = data[var]

            # ---- Compute derivatives ----
            results = compute_derivatives(t, f, br_id, var)
            results_per_variable[var] = results
            
            all_method_metrics = {}

            for method_name, df_method in results["df"].items():
                d2f_method = results["d2f"][method_name]

                all_method_metrics[method_name] = derivative_metrics(df_method, d2f_method)

            # candidate_methods = ("grad", "cubic", "uni")
            
            # best_method = min( all_method_metrics, # candidate_methods,
            #                 key=lambda m: all_method_metrics[m]["smoothness_d2f_var"])
            
            best_method = "mean"
            
            s_info = results["s"] if best_method == "uni" else None

            # ---- Store selected method in dataset ----
            df_sel  = results["df"][best_method]# [method_to_store]
            d2f_sel = results["d2f"][best_method]# [method_to_store]

            if var in ('X' , 'P'):
                df_out = np.maximum(df_sel, 0) # np.clip(df_sel,0,None) # np.maximum(df_sel, 0)
            elif var == 'S': 
                df_out = np.minimum(df_sel, 0) # np.clip(df_sel,None,0) # np.minimum(df_sel, 0)
            else: 
                df_out = df_sel
                
            data[f"d{var}dt"] = df_out
            derivative_datasets[br_id][f"d{var}_dt"] = df_out
            derivative_datasets[br_id][f"d2{var}_dt2"] = d2f_sel
            # data[f"d2{var}dt2"] = d2f_sel  

            # ---- Metrics + s values for YAML ----
            yaml_summary[var] = {
                "method_used": best_method, # method_to_store,
                "s_values": s_info, 
                "first_derivative_metrics": all_method_metrics
            }
        # ---- Plot ----
        plot_dir = f"{results_root}/{br_id}"
        plot_all_derivatives(t=t,results=results_per_variable,variables=variables,br_id=br_id,out_dir=plot_dir)

        save_yaml(
            to_python_type(yaml_summary),
            filepath=f"{results_root}/{br_id}/derivatives_summary.yaml"
        )

    print("\nDerivative computation finished.\n")

    return derivative_datasets, datasets


def compute_derivatives(t, f, br_id, var):
    
    if var == "X":
        bound = (3, 5)
    elif var == "S":
        bound = (0.01,1)    
    elif var == "V":
        bound = (0.0001,0.01)
    elif var == "P":
        bound = (0.00001,0.001)

    (   f_taylor, f_grad, f_scubic, f_suni,
        df_taylor, df_grad, df_scubic, df_suni, df_mean, 
        d2f_taylor, d2f_grad, d2f_scubic, d2f_suni, d2f_mean, 
        spline_F_cubic, spline_F_uni, s_uni, i_start, i_end
    ) = derivative(t, f, bound, var)

    return {
        "f": {
            "taylor": f_taylor,
            "grad": f_grad,
            "cubic": f_scubic,
            "uni": f_suni,
            "mean": f_taylor,
        },
        "df": {
            "taylor": df_taylor,
            "grad": df_grad,
            "cubic": df_scubic,
            "uni": df_suni,
            "mean": df_mean,
        },
        "d2f": {
            "taylor": d2f_taylor,
            "grad": d2f_grad,
            "cubic": d2f_scubic,
            "uni": d2f_suni,
            "mean": d2f_mean,
        },
        "splines": {
            "Cubic": spline_F_cubic,
            "Univariate": spline_F_uni,
        },
        "s": {
            "uni": s_uni,
        },
        "idx": {
            "start": i_start, 
            "end": i_end,
        }
    }

# --------------------
# calculates the first and second derivative using different methods
# --------------------

def derivative(t,f, bounds, var):
    
    df_taylor, d2f_taylor = fun_taylor(t, f)

    df_grad = np.gradient(f, t, edge_order=2)
    # d2f_grad = np.gradient(df_grad, t, edge_order=2)
    d2f_grad, _ = fun_taylor(t, df_grad)

    # for _ in range(5):
    #         x_smooth = np.maximum(x_smooth, 0)
    #         x_smooth = savgol_filter(x_smooth, sg_window, sg_order) # Iterative to avoid negative numbers # window_outlier
    #     x_smooth = np.maximum(x_smooth, 0)

    i_start = 0 
    i_zero = len(f)

    if var != "S":

        if var == "P":
            idx = np.where(f > 0)[0]
            if idx.size == 0:
                # Zero values → fallback
                zeros = np.zeros_like(f)
                return (
                    f, f, zeros, zeros,              # f_scubic, f_suni
                    df_grad, df_grad, zeros, zeros, zeros,   # df_*
                    d2f_grad, d2f_grad, zeros, zeros, zeros, # d2f_*
                    None, None, None, 0, len(f)-1
                )
            else:
                i_start = int(idx[0])

        # Include one point before (anchor to zero)
            i_spline = max(i_start - 1, 0)
            t_sub = t[i_spline:]
            f_sub = f[i_spline:]
        else:
            i_spline = 0
            t_sub = t
            f_sub = f

        f_scubic = f
        f_scubic_sub = f_sub.copy()
        for _ in range(5):
            f_scubic = np.maximum(f_scubic_sub, 0)
            spline_F_cubic = CubicSpline(t_sub, f_scubic_sub, bc_type="natural")
            f_scubic_sub = spline_F_cubic(t_sub)
            
        f_scubic = np.zeros_like(f)
        f_scubic[i_spline:] = spline_F_cubic(t_sub)

        df_scubic = np.zeros_like(f)
        df_scubic[i_spline:] = spline_F_cubic.derivative()(t_sub)

        d2f_scubic = np.zeros_like(f)
        d2f_scubic[i_spline:] = spline_F_cubic.derivative(2)(t_sub)


        res = minimize_scalar(fun_obj_uni, 
                            bounds=bounds,
                            method="bounded",
                            args=(t_sub, f_sub))
        s_calc_uni = res.x # type: ignore

        spline_F_uni = UnivariateSpline(t_sub, f_sub, s=s_calc_uni)
        
        f_suni = np.zeros_like(f)
        f_suni[i_spline:] = spline_F_uni(t_sub)

        df_suni = np.zeros_like(f)
        df_suni[i_spline:] = spline_F_uni.derivative()(t_sub)

        d2f_suni = np.zeros_like(f)
        d2f_suni[i_spline:] = spline_F_uni.derivative(2)(t_sub)

        df_mean = np.mean([df_grad, df_suni], axis=0)
        d2f_mean = np.mean([d2f_grad, d2f_suni], axis=0)

        df_mean[i_spline], df_mean[0], df_mean[-1] = df_grad[i_spline], df_grad[0], df_grad[-1]
        d2f_mean[i_spline], d2f_mean[0], d2f_mean[-1]  = d2f_grad[i_spline], d2f_grad[0], d2f_grad[-1]

    else:
        i_zero = trailing_zeros_start(f) + 1
        t_sub = t[:i_zero]
        f_sub = f[:i_zero]
        if i_zero is not None and i_zero > 0:
            i_spline = i_zero 
        else:
            i_spline = len(f) 

        f_scubic_sub = f_sub.copy()
        for _ in range(5):
            f_scubic = np.maximum(f_scubic_sub, 0)
            spline_F_cubic = CubicSpline(t_sub, f_scubic_sub, bc_type="natural")
            f_scubic_sub = spline_F_cubic(t_sub)
            
        f_scubic = np.zeros_like(f)
        f_scubic[:i_spline] = spline_F_cubic(t_sub)

        df_scubic = np.zeros_like(f)
        df_scubic[:i_spline] = spline_F_cubic.derivative()(t_sub)

        d2f_scubic = np.zeros_like(f)
        d2f_scubic[:i_spline] = spline_F_cubic.derivative(2)(t_sub)


        res = minimize_scalar(fun_obj_uni, 
                            bounds=bounds,
                            method="bounded",
                            args=(t_sub, f_sub))
        s_calc_uni = res.x # type: ignore

        spline_F_uni = UnivariateSpline(t_sub, f_sub, s=s_calc_uni)
        
        f_suni = np.zeros_like(f)
        f_suni[:i_spline] = spline_F_uni(t_sub)

        df_suni = np.zeros_like(f)
        df_suni[:i_spline] = spline_F_uni.derivative()(t_sub)

        d2f_suni = np.zeros_like(f)
        d2f_suni[:i_spline] = spline_F_uni.derivative(2)(t_sub)

        df_mean = np.mean([df_grad, df_suni], axis=0)
        d2f_mean = np.mean([d2f_grad, d2f_suni], axis=0)

        df_mean[i_spline-1:i_spline], df_mean[0], df_mean[-1] = df_grad[i_spline-1:i_spline], df_grad[0], df_grad[-1]
        d2f_mean[i_spline-1:i_spline], d2f_mean[0], d2f_mean[-1]  = d2f_grad[i_spline-1:i_spline], d2f_grad[0], d2f_grad[-1]

    return (f, f, f_scubic, f_suni,
            df_taylor, df_grad, df_scubic, df_suni, df_mean,
            d2f_taylor, d2f_grad, d2f_scubic, d2f_suni, d2f_mean,
            spline_F_cubic, spline_F_uni, s_calc_uni, max(i_start - 1, 0), i_zero - 1)

# --------------------
# calculates the first and second derivative in function of taylor series for variable spaces step
# --------------------

def fun_taylor(t,f):
    
    n = len(t)
    dfdt = np.zeros(n)
    d2fdt2 = np.zeros(n)

    for i in range(n):
    # first row
        if i == 0:
            h = t[i+1] - t[i]
            h2 = t[i+2] - t[i]
            alpha = h2/h
            dfdt[i] = (1/h) * (1/alpha) * (1/(1-alpha)) * ( f[i+2] - (alpha**2 * f[i+1]) - (1-alpha**2) * f[i] )
            d2fdt2[i] = (2/h**2) * (1/alpha) * (1/(1-alpha)) * ( - f[i+2] + (alpha * f[i+1]) + (1-alpha) * f[i] )

    # last row
        elif i == n - 1:
            h =  t[i] - t[i-1]
            h2 = t[i] - t[i-2]
            alpha = h2/h
            dfdt[i] = (1/h) * (1/alpha) * (1/(alpha-1)) * ( (alpha**2 - 1) * f[i] - (alpha**2 * f[i-1]) + f[i-2] )
            d2fdt2[i] = (2/h**2) * (1/alpha) * (1/(alpha-1)) * ( (alpha-1) * f[i] - (alpha * f[i-1]) + f[i-2] )

    # others
        else:
            h = t[i+1] - t[i]
            h2 = t[i] - t[i-1]
            alpha = h2/h
            dfdt[i] = (1/h) * (1/alpha) * (1/(1+alpha)) * ( (alpha**2 * f[i+1]) + (1-alpha**2) * f[i] - f[i-1] )
            d2fdt2[i] = (2/h**2) * (1/alpha) * (1/(1+alpha)) * ( (alpha * f[i+1]) - (1+alpha) * f[i] + f[i-1] )

    return dfdt, d2fdt2

# --------- Tarjet function to set s value on splines ----------

def fun_obj_uni(s,t,f):

    f_pred = f
    for _ in range(5):
        f_pred = np.maximum(f_pred, 0)
        spline_F_uni = UnivariateSpline(t, f_pred, s=s)
        f_pred = spline_F_uni(t)

    df = np.gradient(f, t, edge_order=2)
    df_pred = spline_F_uni.derivative(1)(t)

    d2f_suni = spline_F_uni.derivative(2)(t)

    # out = np.sum((f-f_pred)**2)
    out = np.sum((df-df_pred)**2)

    n = len(t)

    for i in range(3,n):
        if (np.sign(d2f_suni[i]) > np.sign(d2f_suni[i-1])) & (np.sign(d2f_suni[i-1]) < np.sign(d2f_suni[i-2])) & (np.sign(d2f_suni[i-2]) > np.sign(d2f_suni[i-3])):
            out += 0.1 * out
        elif (np.sign(d2f_suni[i]) < np.sign(d2f_suni[i-1])) & (np.sign(d2f_suni[i-1]) > np.sign(d2f_suni[i-2])) & (np.sign(d2f_suni[i-2]) < np.sign(d2f_suni[i-3])):
            out += 0.1 * out

    return out

# Metric for each derivative method

def derivative_metrics(df, d2f):
    return {
        "mean": float(np.mean(df)),
        "std": float(np.std(df)),
        "max_abs": float(np.max(np.abs(df))),
        "smoothness_d2f_var": float(np.var(d2f)),
    }

# Auxiliar functions

def closest_lower(t, t0):
    t = np.asarray(t)
    mask = t <= t0
    if not np.any(mask):
        raise ValueError("t0 is smaller than all values in t")
    return t[mask].max()

def trailing_zeros_start(f, tol=0.0):
    f = np.asarray(f)

    for i in range(len(f)):
        if np.all(np.abs(f[i:]) <= tol):
            return i
    return 0
