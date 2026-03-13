# SurvHTE-Bench: A Benchmark for Heterogeneous Treatment Effect Estimation in Survival Analysis

This repository provides code for SurvHTE-Bench [[Paper](https://arxiv.org/abs/2603.05483)], a benchmark for estimating heterogeneous treatment effects (HTEs) from censored survival data.
It includes implementations of imputation-based meta-learners, double machine learning (DML), survival-adapted meta-learners, and direct survival causal models (e.g., Causal Survival Forest, SurvITE).

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
├── data_utils/                  # HuggingFace data loaing utilities
├── survhte_base/                # Base interfaces for data generation and learner families
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
  - `survite_model.py`: SurvITE model wrapper for SurvHTE-Benchmark
  - `survite_pytorch.py`: Implementation of SurvITE with PyTorch
  - `survite_trainer.py`: Training utilities for SurvITE

- `benchmark/`: Python scripts to run experiments:
  - `impute_event_times_precomputations.py`: Precompute event-time imputations
  - `run_meta_learner_impute.py`: Run meta-learners with outcome imputation
  - `run_dml_learner_impute.py`: Run DML/Causal Forest with imputation
  - `run_meta_learner_survival.py`: Run survival-adapted meta-learners
  - `run_causal_survival_forest.py`: Run Causal Survival Forest
  - `run_survite.py`: Run SurvITE

### Extensibility: Base Interfaces

To make the benchmark easier to extend, we provide explicit base interfaces in: `survhte_base/`

These include:
* `data_generation_base.py`: `DataGeneration` base class for defining custom data-generating mechanisms (hazards, censoring, treatment policies with/without feedback).
* `outcome_imputation_base.py`: `OutcomeImputationBase` for outcome-imputation methods (with an optional hook to generate imputed outcomes).
* `direct_survival_base.py`: `DirectSurvivalCATEBase` for direct-survival HTE models, including:
  * RMST-based CATE via `predict_cate`
  * Survival-probability CATE via `predict_cate_survprob`
* `survival_meta_learner_base.py`: `SurvivalMetaLearnerBase` for survival meta-learners with analogous interfaces for RMST-based and survival-probability CATEs.

A step-by-step tutorial on how to:
- plug in new data generators,
- add new outcome-imputation methods,
- add new direct-survival CATE models, and
- add new survival meta-learners

is provided in:
`survhte_base/README.md`

See that file for concrete examples of how to subclass these interfaces and register new methods in the benchmark scripts.
  

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


## Dataset on HuggingFace

All shareable datasets (synthetic, semi-synthetic ACTG, and real-world) are available on HuggingFace at:

**[https://huggingface.co/datasets/snoroozi/SurvHTE-Bench](https://huggingface.co/datasets/snoroozi/SurvHTE-Bench)**

The HF repository hosts pre-computed train/val/test splits for 10 repeated experiments, making it easy to evaluate new methods without re-running the data pipeline. Note: MIMIC-related datasets are not included due to data sharing restrictions.

### Loading via `data_utils/hf_load.py`

We provide [`data_utils/hf_load.py`](https://github.com/Shahriarnz14/SurvHTE-Bench/blob/main/data_utils/hf_load.py) with two loading interfaces.

**Install dependencies:**
```bash
pip install datasets pandas numpy
```

#### Interface 1 — `load_data`: Reconstruct full experiment structures

Identical output to the local `load_data()` used throughout the benchmark scripts:

```python
from data_utils.hf_load import load_data

experiment_setups, experiment_repeat_setups = load_data(dataset_name=dataset_name, repo_id="snoroozi/SurvHTE-Bench")
```

`experiment_setups` is a nested dict keyed by `setup_key` → `scenario`:
```python
experiment_setups[setup_key][scenario] = {
    "dataset":  pd.DataFrame,  # covariates + outcome columns
    "summary":  dict,          # summary statistics
    "metadata": dict,          # DGP metadata (synthetic only)
}
```

Supported `dataset_name` values: `"synthetic"`, `"actg_syn"`, `"twin"`, `"actgHC"`, `"actgLC"`.

#### Interface 2 — `load_splits`: Pre-split arrays for the experiment loop

Returns numpy arrays already split into train/val/test for each configuration, scenario, and repeat — ready to pass directly into model training:

```python
from data_utils.hf_load import load_splits

split_dict = load_splits(dataset_name=dataset_name, repo_id="snoroozi/SurvHTE-Bench")
```

The returned structure mirrors the benchmark's experiment loop:
```
split_dict[config_name][scenario_key][rand_idx]["train" | "val" | "test"]
    = (X, W, Y, cate_true)
```

**Example:**
```python
X_train, W_train, Y_train, cate_true_train = split_dict[config_name][scenario_key][rand_idx]["train"]
X_val,   W_val,   Y_val,   cate_true_val   = split_dict[config_name][scenario_key][rand_idx]["val"]
X_test,  W_test,  Y_test,  cate_true_test  = split_dict[config_name][scenario_key][rand_idx]["test"]
```


## Installation

### Prerequisites

- Python 3.9+
- Conda

### Environment Setup

To set up the required environment:

```bash
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


## Citation

If you use SurvHTE-Bench in your research, please cite:

```bibtex
@inproceedings{noroozizadeh2026survhte,
  title={SurvHTE-Bench: A Benchmark for Heterogeneous Treatment Effect Estimation in Survival Analysis},
  author={Noroozizadeh, Shahriar and Shen, Xiaobin and Weiss, Jeremy and Chen, George H.},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2026}
}
```