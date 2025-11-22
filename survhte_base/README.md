# Extending SurvHTE-Bench via Base Interfaces

SurvHTE-Bench is designed to be modular. The `survhte_base/` directory contains
lightweight base interfaces that make it easier to:

1. Define new **data-generating mechanisms** (DGMs) for synthetic experiments.
2. Add new **outcome-imputation methods**.
3. Add new **direct-survival CATE models**.
4. Add new **survival meta-learners**.

These interfaces are used by the benchmark scripts but are intentionally minimal so
you can plug in your own methods without touching the core pipeline.

## Files in `survhte_base/`

- `data_generation_base.py`
  - `DataGeneration`: base class for synthetic data generation.
  - `GeneratedDataset`: simple container for `(X, A, time, event, true_cate, metadata)`.
- `outcome_imputation_base.py`
  - `OutcomeImputationBase`: base class for outcome-imputation methods.
- `direct_survival_base.py`
  - `DirectSurvivalCATEBase`: base class for direct-survival HTE models.
- `survival_meta_learner_base.py`
  - `SurvivalMetaLearnerBase`: base class for survival meta-learners.

Below we show how to extend each of these.

---

## 1. Custom data generation via `DataGeneration`

To explore alternative causal structures, non-linear covariate-treatment relation, different hazards, tail behaviors, or treatment policies
(including feedback policies), subclass `DataGeneration`:

```python
from survhte_base.data_generation_base import DataGeneration, GeneratedDataset
```

### 1.1. Implement a custom generator

Create a new file, e.g. `data/custom_generators.py`:

```python
import numpy as np
from survhte_base.data_generation_base import DataGeneration, GeneratedDataset

class MyDataGenerator(DataGeneration):
    def sample_covariates(self, n: int) -> np.ndarray:
        # Example: 5-dimensional standard normal covariates
        return self.random_state.normal(size=(n, 5))

    def sample_treatment(self, x: np.ndarray) -> np.ndarray:
        # Example: observational setting with logistic propensity
        logits = 0.5 * x[:, 0] - 0.25 * x[:, 1]
        p = 1 / (1 + np.exp(-logits))
        return (self.random_state.uniform(size=x.shape[0]) < p).astype(int)

    def sample_event_time(self, x: np.ndarray, a: np.ndarray) -> np.ndarray:
        # Example: proportional hazards with treatment effect
        base_hazard = np.exp(0.3 * x[:, 0] - 0.2 * x[:, 1])
        hazard = base_hazard * np.where(a == 1, 0.7, 1.0)
        u = self.random_state.uniform(size=x.shape[0])
        return -np.log(u) / hazard

    def sample_censoring_time(self, x: np.ndarray, a: np.ndarray) -> np.ndarray:
        # Example: independent censoring
        u = self.random_state.uniform(size=x.shape[0])
        return -np.log(u) / 0.2
```

You can also override `compute_true_cate` if your DGM admits a closed-form CATE.

### 1.2. Generate data and save to `.h5`

In a script or notebook:

```python
from pathlib import Path

gen = MyDataGenerator(seed=123)
data = gen.generate(n=10000, horizons=None)

# Option 1: use a pandas DataFrame
df = gen.to_dataframe(data)  # columns: x0, ..., x{d-1}, a, time, event

# Option 2: save directly to .h5
out_path = Path("data/synthetic/MY_CONFIG.h5")
gen.to_h5(data, out_path, overwrite=False)
```

You can adapt `to_h5` in `data_generation_base.py` to match the schema expected
by your existing loaders (e.g., scenario groups, multiple horizons, etc.).

Once the file is saved under `data/synthetic/`, you can point the benchmark scripts
to it via the usual `--data_dir` and `--dataset_name` arguments.

---

## 2. Adding new outcome-imputation methods

Outcome-imputation methods live in `models_causal_impute/` and can be based on
`OutcomeImputationBase`:

```python
from survhte_base.outcome_imputation_base import OutcomeImputationBase
```

### 2.1. Minimal implementation

Create a new file, e.g. `models_causal_impute/my_outcome_learner.py`:

```python
import numpy as np
from survhte_base.outcome_imputation_base import OutcomeImputationBase

class MyOutcomeImputationMethod(OutcomeImputationBase):
    def __init__(self, random_state=None, **kwargs):
        super().__init__(
            name="my_outcome_imputation_method",
            random_state=random_state,
            **kwargs,
        )
        # initialize internal regressors / learners here

    def fit(self, x, a, y_imputed, sample_weight=None, **fit_kwargs):
        # fit underlying models using (x, a, y_imputed)
        # ...
        return self

    def predict_cate(self, x, horizon=None, **predict_kwargs):
        # compute tau(x) from the fitted models
        cate = np.zeros(x.shape[0])
        return cate
```

This version assumes that `y_imputed` is precomputed (e.g., via
`benchmark/impute_event_times_precomputations.py` and `survival_eval_impute.py`).

### 2.2. Optional: methods that also generate imputations

If your method has its own imputation logic, implement:

```python
    def predict_imputed_outcome(self, x, a, time=None, event=None, **kwargs):
        # map (X, A, time, event) -> y_imputed
        # ...
        return y_imputed
```

The benchmark can then call `predict_imputed_outcome` instead of relying only on
precomputed imputations, while still using the same `fit`/`predict_cate` interface.

### 2.3. Registering the method in the benchmark

Import your class in the appropriate runner (e.g.,
`benchmark/run_meta_learner_impute.py` or `benchmark/run_dml_learner_impute.py`)
and add it to the dictionary/list of methods to execute, following the pattern used
for existing meta-learners.

---

## 3. Adding new direct-survival CATE models

Direct-survival CATE models live in `models_causal_survival/` and can be based on
`DirectSurvivalCATEBase`:

```python
from survhte_base.direct_survival_base import DirectSurvivalCATEBase
```

### 3.1. Minimal implementation

Create a new file, e.g. `models_causal_survival/my_direct_survival_model.py`:

```python
import numpy as np
from survhte_base.direct_survival_base import DirectSurvivalCATEBase

class MyDirectSurvivalModel(DirectSurvivalCATEBase):
    def __init__(self, time_grid, random_state=None, **kwargs):
        super().__init__(
            name="my_direct_survival_model",
            time_grid=time_grid,
            random_state=random_state,
            **kwargs,
        )
        # initialize internal survival models here (for A=0 and A=1)

    def fit(self, x, a, time, event, **fit_kwargs):
        # fit survival models directly on (time, event)
        # ...
        return self

    def predict_survival(self, x, times, treatment, **predict_kwargs):
        # return S(t | X, A=treatment) as an array of shape (n_samples, n_times)
        surv = np.ones((x.shape[0], len(times)))
        return surv
```

### 3.2. Using different CATE estimands

`DirectSurvivalCATEBase` already provides:

- **RMST-based CATE** via:

  ```python
  cate_rmst = model.predict_cate(x, horizon=h, times=time_grid)
  ```

- **Survival-probability CATE** via:

  ```python
  cate_survprob = model.predict_cate_survprob(x, horizon=h, times=time_grid)
  ```

Both use `predict_survival` under the hood, so most new models only need to
implement `fit` + `predict_survival`.

### 3.3. Registering the method

Import your class in run file similar to `benchmark/run_causal_survival_forest.py` (or a new direct-survival
runner) and add it to the list/dictionary of models that the script iterates over.

---

## 4. Adding new survival meta-learners

Survival meta-learners live in `models_causal_survival_meta/` and can be based on
`SurvivalMetaLearnerBase`:

```python
from survhte_base.survival_meta_learner_base import SurvivalMetaLearnerBase
```

### 4.1. Minimal implementation

Create a new file, e.g. `models_causal_survival_meta/my_survival_meta_learner.py`:

```python
import numpy as np
from survhte_base.survival_meta_learner_base import SurvivalMetaLearnerBase

class MySurvivalMetaLearner(SurvivalMetaLearnerBase):
    def __init__(self, time_grid, random_state=None, **kwargs):
        super().__init__(
            name="my_survival_meta_learner",
            time_grid=time_grid,
            random_state=random_state,
            **kwargs,
        )
        # initialize arm-specific survival models, propensity models, etc.

    def fit(self, x, a, time, event, **fit_kwargs):
        # fit survival models for each treatment arm, possibly with meta-learner logic
        # ...
        return self

    def predict_survival(self, x, times, treatment, **predict_kwargs):
        # return S(t | X, A=treatment) as an array of shape (n_samples, n_times)
        surv = np.ones((x.shape[0], len(times)))
        return surv
```

### 4.2. CATE estimands

As with direct-survival models, the base class provides:

- **RMST-based CATE**:

  ```python
  cate_rmst = meta_learner.predict_cate(x, horizon=h, times=time_grid)
  ```

- **Survival-probability CATE**:

  ```python
  cate_survprob = meta_learner.predict_cate_survprob(x, horizon=h, times=time_grid)
  ```

You only need to implement `fit` and `predict_survival`; the CATE logic is shared.

### 4.3. Registering the meta-learner

Import your class in `benchmark/run_meta_learner_survival.py` and add it to the
dictionary/list of survival meta-learners. The runner will then treat it like any
other method and store results under `results/.../models_causal_survival_meta/`.

---

## 5. Summary

- Use `DataGeneration` to plug in new synthetic DGMs (including non-linear covariate-treatment relations, ignorable-censoring, non-smooth hazards,
  different tails, and feedback-based treatment policies), with a helper `to_h5` for
  saving in a standard format.
- Use `OutcomeImputationBase` for new outcome-imputation methods, optionally
  implementing `predict_imputed_outcome` if the method includes its own imputation.
- Use `DirectSurvivalCATEBase` for direct-survival models, with built-in support for
  RMST-based and survival-probability CATEs.
- Use `SurvivalMetaLearnerBase` for survival meta-learners, again with shared CATE logic
  so you only write the core survival modeling code.

These interfaces are designed so that once you implement a new class and register it
in the benchmark scripts, you automatically get access to the full evaluation pipeline
used in SurvHTE-Bench.
