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
model-e.coli/
├── data/                                       
│   ├── processed/     
│   └── raw/
│       ├── unused/
│       │   └── BR06.xlsx    
│       ├── BR02.xlsx
│       ├── (...).xlsx
│       └── BR09.xlsx       
├── execution/                
│   ├── data_analysis.py                
│   └── modelling.py               
├── src/                   
│   ├── config/
│   ├── core/
│   │   ├── auxiliar/ 
│   │   │   ├── feed_factory.py 
│   │   │   ├── feed_profile.py
│   │   │   ├── induction_func.py 
│   │   │   ├── initial_conditions.py 
│   │   │   ├── biomass_profile.py 
│   │   │   ├── volume_profile.py
│   │   │   ├── temperature_profile.py
│   │   │   └── simulator.py     
│   │   └── reactor/ 
│   │       ├── balances.py 
│   │       ├── kinetics.py 
│   │       └── fedbatch_model.py 
│   ├── data_analysis/
│   │   ├── data_treatment/
│   │   │   ├── data.py 
│   │   │   ├── derivative.py
│   │   │   ├── ead.py 
│   │   │   ├── outliers.py 
│   │   │   ├── processing.py 
│   │   │   └── plots_outlier_derivative.py    
│   │   ├── feature_selection/
│   │   │   ├── filter.py 
│   │   │   ├── wrapper.py
│   │   │   ├── permutation.py
│   │   │   ├── wrapper_permutation.py  
│   │   │   ├── plots_filter.py
│   │   │   └── plots_wrapper_permutation.py    
│   │   └── cross_validation/
│   │       ├── cross_val.py 
│   │       └── plots_cross_validation.py    
│   ├── modelling/
│   │   ├── experiment_factory.py 
│   │   └── modelling_plots.py       
│   └── utils/
│       ├── metric_io.py 
│       └── io.py  
├── results/                                    
│   ├── data_analysis/
│   │   ├── outliers_and_smoothing/
│   │   │   ├── BR02/
│   │   │   ├── (...)/
│   │   │   ├── BR09/
│   │   │   └── summary.yaml
│   │   ├── derivatives/
│   │   │   ├── treat/
│   │   │   │   ├── BR02/
│   │   │   │   ├── (...)/
│   │   │   │   └── BR09/
│   │   │   └── smooth/ (empty)
│   │   └── ead/
│   │       ├── induction/
│   │       ├── global_ind/
│   │       └── global/
│   ├── feature_selection/
│   │   ├── induction/
│   │   │   ├── filter/ (filter methdos)
│   │   │   │   ├── qp/
│   │   │   │   └── rp/
│   │   │   └── wnp/ (wrapper and permutation selection methods)
│   │   ├── global_ind/
│   │   └── global/ 
│   ├── cross_validation/
│   │   ├── induction/
│   │   │   ├── qp/
│   │   │   │   ├── best_model_per_fold_dynamic/ 
│   │   │   │   ├── metrics/ 
│   │   │   │   ├── predictions/ 
│   │   │   │   ├── residuals/ 
│   │   │   │   └── cv_results_full.yaml
│   │   │   ├── rp/
│   │   │   └── metrics_summary.xlsx
│   │   ├── global_ind/
│   │   └── global/      
│   └── modelling/
│       ├── parametric/
│       ├── induction/
│       ├── global_ind/
│       ├── global/
│       ├── comparison/
│       ├── metrics/ 
│       ├── metrics_summary_all_mdoels.xlsx         
│       └── multibr_XSV_parametric.png
├── LICENSE               
├── README.md                                   
├── requirements.txt
├── .gitattributes                             
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

