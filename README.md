# Dynamic Model of *E. coli* Nanobody Antivenom Production

## Overview

This repository contains a **Python-based dynamic model** describing the growth of *Escherichia coli* and the **production of nanobody (Nb) antivenoms** in a bioprocess context.

The project was developed as part of a **first-year engineering / biotechnology Python project**, with the objective of:
- Implementing mass-balance–based dynamic models
- Simulating microbial growth and product formation
- Practicing scientific programming, version control, and model documentation

The present code is based on and adapted from a pre-existing implementation developed by Irene and Juan-Camilo in the context of the work presented in:

Flavio Manenti, Gintaras V. Reklaitis (Eds.), *Proceedings of the 34th European Symposium on Computer Aided Process Engineering / 15th International Symposium on Process Systems Engineering (ESCAPE34/PSE24)*, June 2–6, 2024, Florence, Italy.  
**Dynamic Hybrid Model for Nanobody-based Antivenom Production (scorpion antivenom) with *E. coli* CH10-12 and *E. coli* NbF12-10.**  
© 2024 Elsevier B.V. All rights reserved.


## Repository Structure
```
dynamic_model_coli_Nb_Antivenom_01/
│
├── data/                                       # Experimental data
│   ├── processed/         
│   └── raw/   
├── execution/                
│   ├── estimate_parameters.py                
│   └── model_profile.py               
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
│   │   ├── excel_io.py 
│   │   ├── execute_model_io.py
│   │   ├── experiment_factory.py 
│   │   ├── io.py 
│   │   ├── metric_io.py 
│   │   ├── io.py 
│   │   ├── vizualization_correlation_io.py 
│   │   ├── vizualization_residuals_io.py 
│   │   └── vizualization_fitting_io.py      
├── results/                                    # Simulation outputs (plots, tables)
│   ├── estimation/         
│   └── plots/
│   │   ├── processed/         
│   │   └── time_profiles/
├── LICENSE               
├── README.md                                   # Project documentation
├── requirements.txt                            # Python dependencies
└── .gitignore                                  # Files ignored by Git
```
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

## Installation

### Prerequisites
- Python ≥ 3.9
- Git

### Clone the repository
```bash
git clone https://github.com/jucam9810/dynamic_model_coli_Nb_Antivenom_01.git
cd dynamic_model_coli_Nb_Antivenom_01
```

### Create a virtual environment
```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

### Install dependencies
```bash
pip install -r requirements.txt
```

### Usage
 - For dynamic model simulation run script:
```bash
model_profile.py
```
 - For estimate the model parameters run script:
```bash
estimate_parameters.py
```

