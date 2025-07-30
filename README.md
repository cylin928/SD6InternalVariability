# The Role of Internal Variability in Shaping Groundwater Policy Effectiveness

This repository contains the complete codebase for the research paper "The Role of Internal Variability in Shaping Groundwater Policy Effectiveness", which investigates how internal variability influence the transferability and effectiveness of groundwater conservation policies.

## Overview

TBA

## Prerequisites

- Python 3.10+ (tested with Python 3.11 and 3.12)
- R (for spatial scenario generation)

### Required Python Packages

The main dependency is the `py_champ` package, which implements the coupled human-natural system model. Additional packages are listed in `pyproject.toml`.

## Installation

1. **Clone the repository:**
```bash
git clone https://github.com/philip928lin/SD6InternalVariability.git
cd SD6InternalVariability
```

2. **Install PyCHAMP:**
Follow the detailed installation instructions at [PyCHAMP Installation Guide](https://github.com/philip928lin/PyCHAMP).

3. **Install this package and dependencies:**
```bash
pip install -e .
```

## Repository Structure

```
├── code/                           # Main analysis scripts
│   ├── calibration.py             # Model calibration using PSO
│   ├── create_input_pkl.py        # Input data preparation
│   ├── anova_*.py                 # ANOVA analysis scripts
│   ├── oat_*.py                   # One-at-a-time sensitivity analysis
│   ├── scenario_generation_*.R    # Spatial scenario generation in R
│   ├── plotting.py                # Plotting utilities
│   └── utils.py                   # General utilities
├── data/                          # Input datasets
│   ├── Data_SD6_2012_2022.csv    # Historical SD6 data
│   ├── KFMA_crop_income.csv      # Crop price data
│   ├── prec_*.csv                # Precipitation data
│   └── SD6_grid_info*.csv        # Spatial grid information
├── figures/                       # Generated figures and plotting code
│   ├── code_for_plotting/         # Scripts to reproduce paper figures
│   └── *.jpg                     # Generated figure files
├── inputs/                        # Processed model inputs
├── models/                        # Calibrated model files
├── outputs/                       # Simulation results
│   ├── ANOVA/                     # ANOVA experiment outputs
│   └── OAT_*/                     # OAT experiment outputs
├── scenarios/                     # Generated scenario files
├── pyproject.toml                 # Package configuration
└── README.md                      # This file
```

### Key Scripts

- `calibration.py`: Performs model calibration using historical data
- `oat_run_with_joblib.py`: Runs one-at-a-time sensitivity analysis
- `anova_run.py`: Executes ANOVA experiments with multiple scenarios
- `anova_analysis.py`: Analyzes ANOVA results

## Reproducing Results

To reproduce the main results from the paper:

1. **Figure 1 (Calibration Results):**
```bash
python figures/code_for_plotting/fig1_elements.py
```

2. **Figure 3 (OAT Heatmaps):**
```bash
python figures/code_for_plotting/fig3_S3_oat_heatmaps.py
```

3. **Figures 4-5 (ANOVA Results):**
```bash
python figures/code_for_plotting/fig4_5_anova.py
```

4. **Supplementary Figures:**
```bash
python figures/code_for_plotting/figS1_S2_B_derivation.py
python figures/code_for_plotting/figS5_anova.py
```

## Citation

If you use this code or data in your research, please cite:

## License

This project is licensed under the terms specified in the LICENSE file.

## Acknowledgments

This work was supported by the National Science Foundation Grant No. RISE-2108196 (‘DISES: Toward resilient and adaptive community-driven management of groundwater dependent agricultural systems’) and the Foundation for Food and Agriculture Research Grant No. FF-NIA19-0000000084. Any opinions, findings, and conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the National Science Foundation or the Foundation for Food and Agriculture Research.