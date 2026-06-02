# Dynamic Model of *E. coli* Nanobody Antivenom Production
---
## Overview

This repository contains a **Python-based dynamic model** describing the growth of *Escherichia coli* and the **production of nanobody (Nb) antivenoms** in a bioprocess context.

The project was developed as part of a **first-year engineering / biotechnology Python project**, with the objective of:
- Implementing mass-balance–based dynamic models
- Simulating microbial growth and product formation
- Practicing scientific programming, version control, and model documentation

The present code is based on and adapted from a pre-existing implementation developed by David-Camilo in the context of the work presented in:

Corrales, D. C., Villela, S. M. A., Bouhaouala-Zahar, B., Cescut, J., Daboussi, F., O’donohue, M., ... & Aceves-Lara, C. A. (2024). Dynamic Hybrid Model for Nanobody-based Antivenom Production (scorpion antivenon) with E. coli CH10-12 and E. coli NbF12-10. Computer Aided Chemical Engineering, 53, pp. 145-150. 

---

## Repository Structure
```
dynamic_model_coli_Nb_Antivenom_01/
├── data/                                       
│   ├── processed/
│   │   ├── BR_processed_ind.xlsx
│   │   └── BR_processed.xlsx       
│   └── raw/
│       ├── unused/
│       │   └── BR06.xlsx    
│       ├── BR02.xlsx
│       ├── (...).xlsx
│       └── BR09.xlsx       
├── execution/                
│   ├── data_analysis.py                
│   └── modelling.py               
├── fedbatch/                   
│   ├── config/
│   ├── core/
│   │   ├── balances.py 
│   │   ├── kinetics.py 
│   │   └── fedbatch_model.py 
│   ├── estimation/
│   │   ├── datasets.py 
│   │   ├── objective.py 
│   │   └── postprocessing.py 
│   ├── simulation/
│   │   ├── feed_factory.py 
│   │   ├── feed_profile.py
│   │   ├── induction_func.py 
│   │   ├── initial_conditions.py 
│   │   ├── temperature_profile.py 
│   │   └── simulator.py       
│   └── utils/
│       ├── excel_io.py 
│       ├── execute_model_io.py
│       ├── experiment_factory.py 
│       ├── io.py 
│       ├── metric_io.py 
│       ├── io.py 
│       ├── vizualization_correlation_io.py 
│       ├── vizualization_residuals_io.py 
│       └── vizualization_fitting_io.py      
├── results/                                    
│   ├── data_analysis/
│   │   ├── outliers_and_smoothing/
│   │   │   ├── BR02/
│   │   │   │   ├── T_outlier_diagnosis.png
│   │   │   │   └── T_replacement_smoothing.png
│   │   │   ├── (...)/
│   │   │   ├── BR09/
│   │   │   └── summary.yaml
│   │   ├── derivatives/
│   │   │   ├── treat/
│   │   │   │   ├── BR02/
│   │   │   │   │   ├── all_derivatives.png
│   │   │   │   │   └── derivatives_summary.yaml
│   │   │   │   ├── (...)/
│   │   │   │   └── BR09/
│   │   │   └── smooth/ (empty)
│   │   └── ead/
│   │       ├── induction/
│   │       │   ├── BP_BR02.png (Boxplot)
│   │       │   ├── (...).png
│   │       │   ├── BP_BR09.png
│   │       │   ├── BP_global.png
│   │       │   ├── HM_BR02.png (Heatmap)
│   │       │   ├── (...).png
│   │       │   ├── HM_BR09.png
│   │       │   ├── HM_global.png
│   │       │   ├── PCA_BR02.png
│   │       │   ├── (...).png
│   │       │   ├── PCA_BR09.png
│   │       │   ├── PCA_global.png
│   │       │   ├── Temperature_series_BR02.png (for qP and rP)
│   │       │   ├── (...).png
│   │       │   ├── Temperature_series_BR09.png
│   │       │   ├── Temperature_series_global.png
│   │       │   ├── time_series_BR02.png (for qP and rP)
│   │       │   ├── (...).png
│   │       │   ├── time_BR09.png
│   │       │   └── time_global.png
│   │       ├── global_ind/
│   │       └── global/
│   ├── feature_selection/
│   │   ├── induction/
│   │   │   ├── filter/ (filter methdos)
│   │   │   │   ├── qp/
│   │   │   │   │   ├── CMI_comparison.png
│   │   │   │   │   ├── redundancy.png
│   │   │   │   │   ├── feature_selection_heatmap.png
│   │   │   │   │   ├── feature_selection_bars.png
│   │   │   │   │   └── feature.yaml
│   │   │   │   └── rp/
│   │   │   └── wnp/ (wrapper and permutation selection methods)
│   │   │       ├── qp/
│   │   │       │   ├── metrics/ (RMSE R2 MSE MAPE MAE AIC BIC SCORE)
│   │   │       │   │   ├── RMSE_wrapper_comparison.png
│   │   │       │   │   ├── RMSE_permutation_comparison.png
│   │   │       │   │   ├── (...)_wrapper_comparison.png 
│   │   │       │   │   └── (...)_permutation_comparison.png 
│   │   │       │   ├── feature_heatmap.png
│   │   │       │   ├── metrics_heatmap.png
│   │   │       │   ├── metrics_global.xlsx
│   │   │       │   └── wrapper_summary.yaml
│   │   │       └── rp/
│   │   ├── global_ind/
│   │   └── global/ 
│   ├── cross_validation/
│   │   ├── induction/
│   │   │   ├── qp/
│   │   │   │   ├── best_model_per_fold_dynamic/ 
│   │   │   │   │   ├── svm_linear_BR02_metadata.yaml
│   │   │   │   │   ├── svm_linear_BR02_params.yaml
│   │   │   │   │   ├── svm_linear_BR02.pkl
│   │   │   │   │   ├── (...)_metadata.yaml
│   │   │   │   │   ├── (...)_params.yaml
│   │   │   │   │   └── (...).pkl
│   │   │   │   ├── metrics/ (RMSE R2 MSE MAPE MAE AIC BIC SCORE)
│   │   │   │   │   ├── RMSE_boxplot_advanced.png
│   │   │   │   │   ├── RMSE_heatmap.png
│   │   │   │   │   ├── (...)_boxplot_advanced.png
│   │   │   │   │   └── (...)_heatmap.png
│   │   │   │   ├── predictions/ (linear elasticnet_w LASSO_w Ridge_w svm_linear svm_polu svm_rbf rf_w knn)
│   │   │   │   │   ├── linear_predictions.png
│   │   │   │   │   ├── linear_timeseries.png
│   │   │   │   │   ├── (...)_predictions.png
│   │   │   │   │   └── (...)_timeseries.png
│   │   │   │   ├── residuals/ (linear elasticnet_w LASSO_w Ridge_w svm_linear svm_polu svm_rbf rf_w knn)
│   │   │   │   │   ├── linear_residuals.png
│   │   │   │   │   └── (...)_residuals.png
│   │   │   │   └── cv_results_full.yaml
│   │   │   ├── rp/
│   │   │   └── metrics_summary.xlsx
│   │   ├── global_ind/
│   │   └── global/      
│   └── modelling/
│       ├── parametric/
│       │   └── parametric_all_datasets.png
│       ├── induction/
│       │   ├── induction_qp_all_datasets.png
│       │   └── induction_rp_all_datasets.png
│       ├── global_ind/
│       ├── global/
│       ├── comparison/
│       │   ├── BR02_comparison_P.png
│       │   ├── (...).png
│       │   └── BR08_comparison_P.png
│       ├── metrics/ (RMSE R2 MSE MAPE MAE AIC BIC SCORE)
│       │   ├── RMSE_boxplot_advanced.png
│       │   ├── RMSE_heatmap.png
│       │   ├── (...)_boxplot_advanced.png 
│       │   └── (...)_heatmap.png 
│       ├── metrics_summary_all_mdoels.xlsx         
│       └── multibr_XSV_parametric.png
├── LICENSE               
├── README.md                                   
├── requirements.txt                            
└── .gitignore                                  
```
---
## Model Description

The model is based on ordinary differential equations (ODEs):

$$
\frac{dX}{dt} = \mu X - \frac{dV}{dt} \frac{X}{V}
$$

$$
\frac{dS}{dt} = -\left(\frac{\mu}{Y_{XS}} + m\right) X + \frac{dV}{dt} \left(\frac{S_{in}-S}{V}\right)
$$

$$
\frac{dP}{dt} = q_P\ X - \frac{dV}{dt} \frac{P}{V}
$$

$$
\frac{dV}{dt} = F_{S_{in}}
$$

$$
\mu = \frac{ \mu_{max} S }{K_S + S}
$$

$$
\mu_{\max} =
\begin{cases}
\mu_{\max,0} 
& \text{if  } F_{S_{in}} \neq F_{S,\text{ind}} \\
\mu_{\max}'\, T + b 
& \text{if  } F_{S_{in}} = F_{S,\text{ind}}
\end{cases}
$$

$$
q_{P} =
\begin{cases}
0 & \text{if } t < t_{\text{ind}} \\
\alpha\mu - \gamma_1 e^{\frac{-A_1}{T}} + \gamma_2 e^{\frac{-A_2}{T}} - \sigma & \text{if } t \ge t_{\text{ind}}
\end{cases}
$$

Where:
- $X$ is the biomass concentration
- $S$ is the substrate concentration
- $P$ is the nanobody concentration
- $\mu$ is the specific growth rate
- $q_P$ is the specific product formation rate

---
### Usage
 - For dynamic model simulation run script:
```bash
model_profile.py
```
 - For estimate the model parameters run script:
```bash
estimate_parameters.py
```

