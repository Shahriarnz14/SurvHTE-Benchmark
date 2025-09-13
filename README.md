# SurvHTE-Bench: A Benchmark for Heterogeneous Treatment Effect Estimation in Survival Analysis

This repository contains implementations and evaluations of causal inference methods for survival data. The codebase supports various meta-learners, double machine learning approaches, and specialized survival-based methods for estimating heterogeneous treatment effects in the presence of censoring.

## Repository Structure

```
├── models_causal_impute/          # Implementation of imputation-based meta-learners
├── models_causal_survival/        # Specialized survival causal effects models
├── models_causal_survival_meta/   # Meta-learners adapted for survival analysis
├── real_data/                     # Real-world datasets (ACTG, Twin)
├── results/                       # Storage for experiment results
├── scripts/                       # Experiment runner scripts
├── synthetic_data/                # Synthetic data generators and datasets
├── notebooks/                     # Analysis notebooks and visualizations
└── environment.yml                # Conda environment specification
```

### Key Modules

- **models_causal_impute**: Implements imputation-based meta-learners that first impute censored outcomes and then apply standard causal inference methods.
  - `meta_learners.py`: T-Learner, S-Learner, X-Learner, DR-Learner
  - `dml_learners.py`: Double ML, Causal Forest 
  - `survival_eval_impute.py`: Various imputation strategies (IPCW-T, Pseudo-obs, Margin)
  
- **models_causal_survival_meta**: Implements meta-learners directly adapted for survival analysis
  - `meta_learners_survival.py`: Survival T-Learner, Survival S-Learner, Matching Learner
  - `survival_base.py`: Base class for survival models with hyperparameter tuning

- **models_causal_survival**: Specialized causal survival models (referred as `Direct-survival CATE models` in the paper)
  - `causal_survival_forest.py`: Implementation of Causal Survival Forests

## Data Sources

### Synthetic Data

The synthetic data used in experiments is generated using the `data.py` script, which creates various scenarios with different treatment effect patterns, censoring mechanisms, and confounding structures. These datasets include:

- RCT scenarios with different treatment proportions (`RCT_0_5.h5` and `RCT_0_05.h5`)
- Observational scenarios with confounding (`e_X.h5`)
- Scenarios with unobserved confounders (`e_X_U.h5`)
- Scenarios with/without overlap (`e_X_no_overlap`)
- Scenarios with informative censoring (`e_X_info_censor.h5`, `e_X_U_info_censor.h5`, and `e_X_no_overlap_info_censor.h5`)

The generated data is stored in the `synthetic_data/` directory and includes:
- `.h5` files containing the generated data for each scenario
- `idx_split.csv` file with 10 different training/testing splits for reproducibility
- `generate_synthetic_data.ipynb` notebook for generating the synthetic data for different causal configuration and survival scenarios

The data is also available at Harvard Dataverse: [SurvHTE-Bench](https://doi.org/10.7910/DVN/VLJO28)

### Real Data

The real-world datasets are stored in the `real_data/` directory:

1. **ACTG HIV Clinical Trial Data**: 
   - `ACTG_175_HIV1.csv`, `ACTG_175_HIV2.csv`, `ACTG_175_HIV3.csv`: Different versions of the ACTG 175 dataset
   - `idx_split_HIV1.csv`, `idx_split_HIV2.csv`, `idx_split_HIV3.csv`: Train/test splits

2. **Twin Mortality Data**:
   - `twin.csv`, `twin30.csv`, `twin180.csv`: Different variations of the twin mortality dataset
   - `idx_split_twin.csv`: Train/test splits

## Installation

### Prerequisites

- Python 3.9+
- Conda

### Environment Setup

To set up the required environment:

```bash
# Clone the repository
git clone https://github.com/Shahriarnz14/SurvHTE-Bench.git
cd SurvHTE-Bench

# Create and activate conda environment
conda env create -f environment.yml
conda activate causal_survival_db
```

The environment includes packages for:
- Core ML: scikit-learn, xgboost, pytorch
- Survival analysis: scikit-survival, lifelines, pycox
- Causal inference: econml
- R integration via rpy2 (for Causal Survival Forest method)

## Running Experiments

The repository includes various scripts to run experiments across different methods and datasets.

### Experiments with Outcome Imputation-Based Methods

#### Meta-learners after imputation:

```bash
# Run all meta-learners with a specific imputation method on synthetic data
./scripts/run_meta_learners_impute.sh t_learner 5000

# Run a specific meta-learner with a specific imputation method on synthetic data
./scripts/run_meta_learners_impute_single.sh t_learner 5000 Margin

# Run on ACTG data
./scripts/actg/run_meta_learners_impute_actg.sh

# Run on ACTG "long" data (alternative format)
./scripts/actgL/run_meta_learners_impute_actgL.sh

# Run on Twin data
./scripts/twin/run_meta_learners_impute_twin.sh
```

#### Double ML and Causal Forest Experiments

```bash
# Run DML learners on synthetic data
./scripts/run_dml_learners_impute.sh double_ml 5000

# Run a specific DML learner with a specific imputation method
./scripts/run_dml_learners_impute_single.sh double_ml 5000 Pseudo_obs

# Run on ACTG data
./scripts/actg/run_dml_learners_impute_actg.sh

# Run on ACTG "long" data
./scripts/actgL/run_dml_learners_impute_actgL_cf.sh
./scripts/actgL/run_dml_learners_impute_actgL_doubleml.sh

# Run on Twin data
./scripts/twin/run_dml_learners_impute_twin.sh
```

### Survival-adapted meta-learners:

```bash
# Run survival meta-learners on synthetic data
./scripts/run_meta_learners_survival.sh s_learner_survival 5000

# Run on ACTG data
./scripts/actg/run_meta_learners_survival_actg.sh

# Run on ACTG "long" data
./scripts/actgL/run_meta_learners_survival_actgL.sh

# Run on Twin data
./scripts/twin/run_meta_learners_survival_twin.sh
```

### Direct-survival CATE models:
Use the notebooks provided to run Causal Survival Forest:
```
# Run on synthetic data
notebooks/causal_survival_forest.ipynb

# Run on ACTG data
notebooks/causal_survival_forest_actg.ipynb
notesbook/causal_survival_forest_actgL.ipynb

# Run on Twin data
notebooks/causal_survival_forest_twin.ipynb
```

## Running Individual Scripts

You can also run the individual Python scripts directly for more control:

```bash
# Meta-learners with imputation on synthetic data
python run_meta_learner_impute.py --num_repeats 10 --train_size 5000 --test_size 5000 --meta_learner t_learner --impute_method Pseudo_obs --load_imputed --imputed_path synthetic_data/imputed_times_lookup.pkl

# DML learners on real data
python run_dml_learner_impute_actg.py --num_repeats 10 --train_size 0.75 --dml_learner causal_forest --impute_method Pseudo_obs --load_imputed --imputed_path real_data/imputed_times_lookup.pkl
```

## Saved Imputations
The previously ran imputation of the synthetic data is available at [imputed_times_lookup.pkl](https://drive.google.com/file/d/18LyjPWb-SOz2QmCinHGOlyzgiMGNp-WL). This file should be placed in `synthetic_data/` directory.

## Result Analysis

The results of experiments are saved as pickle files in the `results/` directory, organized by dataset type (synthetic or real), model category, and specific method. These can be loaded and analyzed using the notebooks in the `notebooks/` directory.

## Acknowledgments

* This code builds on several open-source packages including EconML, scikit-survival, and PyCox
* The ACTG 175 clinical trial data is provided by the AIDS Clinical Trials Group (Data available at [AIDS Clinical Trials Group Study 175](https://archive.ics.uci.edu/dataset/890/aids+clinical+trials+group+study+175))
* The Twin mortality data is derived from the Twin birth registry of NBER (Subset obtained from [GANITE](https://github.com/YorkNishi999/ganite_pytorch/blob/main/data/Twin_data.csv.gz))
