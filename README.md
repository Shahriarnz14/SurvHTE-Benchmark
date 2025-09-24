# SurvHTE-Bench: A Benchmark for Heterogeneous Treatment Effect Estimation in Survival Analysis

This repository provides code for SurvHTE-Bench, a benchmark for estimating heterogeneous treatment effects (HTEs) from censored survival data.
It includes implementations of imputation-based meta-learners, double machine learning (DML), survival-adapted meta-learners, and direct survival causal models.

We support experiments on synthetic, semi-synthetic, and real-world datasets.
( Note: MIMIC-related datasets cannot be shared.)

For reproducibility, each dataset has an idx_split.csv file to ensure consistent train/validation/test splits across methods.

## Repository Structure

```
├── benchmark/                   # Main experiment runners
├── models_causal_impute/        # Outcome-imputation method (imputation + meta-learners or DML methods)
├── models_causal_survival/      # Direct survival causal models (e.g., CSF)
├── models_causal_survival_meta/ # Survival meta-learners
├── models_utils/                # Utilities (checkpointing, shared helpers)
├── data/                        # Synthetic, semi-synthetic, and real datasets; generation + preprocessing
├── results/                     # Stored results, organized by dataset and method family
├── scripts/                     # Shell scripts to reproduce experiments
├── notebooks/                   # Analysis and aggregation notebooks
├── environment.yml              # Conda environment specification
└── README.md
```

### Key Modules

- `models_causal_impute/`: Implements outcome-imputation approaches that first impute censored outcomes and then apply standard causal inference methods.
  - `meta_learners.py`: T-Learner, S-Learner, X-Learner, DR-Learner
  - `dml_learners.py`: Double ML, Causal Forest 
  - `survival_eval_impute.py`: Various imputation strategies (IPCW-T, Pseudo-obs, Margin)
  - `regressor_base.py`: Regression base models
  
- `models_causal_survival_meta/`: Implements meta-learners directly adapted for survival analysis
  - `meta_learners_survival.py`: Survival T-Learner, Survival S-Learner, Matching Learner
  - `survival_base.py`: Base class for survival models (RSF, DeepSurv, DeepHit) with hyperparameter tuning
  - `concordance.py`: Survival evaluation utilities

- `models_causal_survival/`: Specialized causal survival models (referred as "Direct-survival CATE models" in the paper)
  - `causal_survival_forest.py`: Implementation of Causal Survival Forests

- `benchmark/`: Python scripts to run experiments:
  - `impute_event_times_precomputations.py`: Precompute event-time imputations
  - `run_meta_learner_impute.py`: Run meta-learners with outcome imputation
  - `run_dml_learner_impute.py`: Run DML/Causal Forest with imputation
  - `run_meta_learner_survival.py`: Run survival-adapted meta-learners
  - `run_causal_survival_forest.py`: Run Causal Survival Forest
  

## Data

### Synthetic Data
Located in `data/synthetic/`:

The synthetic datasets used in this benchmark are generated using `generate_synthetic_data.ipynb`.  
Each `.h5` file corresponds to **one causal configuration** from the paper, and within each file are **five distinct survival scenarios**.  
In total, there are **8 causal configurations × 5 survival scenarios = 40 synthetic datasets**.  

The eight causal configurations include:  
- RCT scenarios with different treatment proportions (`RCT-50.h5` and `RCT-5.h5`)  
- Observational scenario with confounding (`OBS-CPS.h5`)  
- Observational scenario with unobserved confounders (`OBS-UConf.h5`)  
- Observational scenario with positivity violation (`OBS-NoPos.h5`)  
- Informative censoring counterparts of the three observational settings (`OBS-CPS-IC.h5`, `OBS-NoPos-IC.h5`, `OBS-UConf-IC.h5`)  

Each `.h5` file contains data for all **five survival scenarios** under that causal configuration.  

The `data/synthetic/` directory includes:  
- `.h5` files for each causal configuration (each containing five survival scenarios)  
- `idx_split.csv`: predefined train/val/test splits for reproducibility  
- `generate_synthetic_data.ipynb`: notebook to regenerate datasets  


### Semi-Synthetic
Located in `data/semi-synthetic/`:  
- **MIMIC-IV derived semi-synthetic datasets** (not redistributable)  
- **ACTG175 semi-synthetic dataset**  
- Preprocessing notebooks: `prepare_mimic_semi_simulated.ipynb`, `prepare_actg_synthetic.ipynb`


### Real Data
Located in `data/real/`:  
- **ACTG HIV clinical trial data** (`ACTG_175_HIV1/2/3.csv`)  
- **Twins mortality data** (`twin30.csv`, `twin180.csv`)  
- Preprocessing: `prepare_actg_175.py`, `prepare_twin_data.ipynb` 

Each dataset folder includes an `idx_split_*.csv` for reproducible splits.  


## Installation

### Prerequisites

- Python 3.9+
- Conda

### Environment Setup

To set up the required environment:

```bash
# Clone the repository
git clone https://github.com/anonymous/SurvHTE-Benchmark.git
cd SurvHTE-Benchmark

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

The repository includes various scripts to run experiments across different methods and datasets. All the experiments should be ran from the main work directory.

### Precompute imputations
Examples:
```bash
# Run on synthetic datasets
python benchmark/impute_event_times_precomputations.py \
  --dataset_name synthetic \
  --data_dir ./data \
  --train_size 5000 --val_size 2500 --test_size 2500
# Run on mimic semi-synthetic datasets
python benchmark/impute_event_times_precomputations.py \
  --dataset_name mimic_syn \
  --data_dir ./data \
  --train_size 0.5 --val_size 0.25 --test_size 0.25
```

### Experiments with Outcome Imputation-Based Methods
Imputation precomputation is required for outcome imputation-based methods

#### Meta-learners experiments after imputation:
Examples:
```bash
# Run on synthetic datasets
./scripts/synthetic/run_dml_learners_impute_synthetic.sh
# Run on mimic semi-synthetic datasets
./scripts/mimic/run_meta_learners_impute_mimic_syn.sh
```

#### Double ML and Causal Forest Experiments after imputation
Examples:
```bash
# Run on synthetic datasets
./scripts/synthetic/run_dml_learners_impute_synthetic.sh
# Run on mimic semi-synthetic datasets
./scripts/mimic/run_dml_learners_impute_mimic_syn.sh
```

### Survival-adapted meta-learners:

Examples:
```bash
# Run on synthetic datasets
./scripts/synthetic/run_meta_survival_learners_synthetic.sh
# Run on mimic semi-synthetic datasets
./scripts/mimic/run_meta_survival_learners_mimic_syn.sh
```

### Direct-survival CATE models:
```bash
# Run on all supported datasets
./scripts/run_csf_all_datasets.sh
```

## Result Analysis
- All results are stored in `results/` under `{synthetic, semi-synthetic, real}/models_*`.
- The results of experiments are saved as pickle files in the `results/` directory, organized by dataset type (synthetic or real), model category, and specific method. These can be loaded and analyzed using the notebooks in the `notebooks/` directory.

## Acknowledgments

* This code builds on several open-source packages including EconML, scikit-survival, and PyCox
* The ACTG 175 clinical trial data is provided by the AIDS Clinical Trials Group (Data available at [AIDS Clinical Trials Group Study 175](https://archive.ics.uci.edu/dataset/890/aids+clinical+trials+group+study+175))
* The Twin mortality data is derived from the Twin birth registry of NBER (Subset obtained from [GANITE](https://github.com/YorkNishi999/ganite_pytorch/blob/main/data/Twin_data.csv.gz))
