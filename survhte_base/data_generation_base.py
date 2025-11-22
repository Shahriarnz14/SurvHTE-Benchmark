from __future__ import annotations

import abc
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import h5py
import numpy as np
import pandas as pd


@dataclass
class GeneratedDataset:
    """
    Container for a single generated dataset.

    Attributes
    ----------
    x : np.ndarray
        Covariates, shape (n_samples, n_features).
    a : np.ndarray
        Binary treatment assignment, shape (n_samples,).
    time : np.ndarray
        Observed time (min of event and censoring), shape (n_samples,).
    event : np.ndarray
        Event indicator (1 = event, 0 = censored), shape (n_samples,).
    true_cate : Optional[np.ndarray]
        Optional ground-truth CATE. Can be:
          * (n_samples,) for a single estimand (e.g., RMST-CATE at a fixed horizon),
          * (n_samples, n_horizons) for horizon-specific effects.
    metadata : Dict[str, Any]
        Additional metadata (e.g., horizons, DGM name, parameters).
    """
    x: np.ndarray
    a: np.ndarray
    time: np.ndarray
    event: np.ndarray
    true_cate: Optional[np.ndarray] = None
    metadata: Optional[Dict[str, Any]] = None


class DataGeneration(abc.ABC):
    """
    Base class for synthetic data generation in SurvHTE-Bench.

    Subclass this to introduce new hazard functions, censoring schemes,
    and treatment policies (including feedback / dynamic treatment rules),
    while keeping the downstream evaluation pipeline unchanged.

    Typical usage
    -------------
    >>> gen = MyDataGenerator(seed=123)
    >>> data = gen.generate(n=10000, horizons=np.array([3.0, 5.0]))
    >>> df = gen.to_dataframe(data)
    """

    def __init__(self, seed: Optional[int] = None):
        self.seed = seed
        self.random_state = np.random.RandomState(seed)

    # ---- Core pieces you must implement ---------------------------------

    @abc.abstractmethod
    def sample_covariates(self, n: int) -> np.ndarray:
        """
        Sample baseline covariates X.

        Parameters
        ----------
        n : int
            Number of samples.

        Returns
        -------
        x : np.ndarray of shape (n, d)
        """
        raise NotImplementedError

    @abc.abstractmethod
    def sample_treatment(self, x: np.ndarray) -> np.ndarray:
        """
        Sample treatment assignment A | X.

        This can be:
          * randomized (e.g., Bernoulli with fixed probability),
          * observational with confounding (propensity varies with X),
          * Optionally, dynamic / feedback policies if the generator includes time-varying state.

        Parameters
        ----------
        x : np.ndarray
            Covariates, shape (n, d).

        Returns
        -------
        a : np.ndarray of shape (n,)
        """
        raise NotImplementedError

    @abc.abstractmethod
    def sample_event_time(self, x: np.ndarray, a: np.ndarray) -> np.ndarray:
        """
        Sample event time T(event) | X, A.

        Parameters
        ----------
        x : np.ndarray
        a : np.ndarray

        Returns
        -------
        t_event : np.ndarray of shape (n,)
        """
        raise NotImplementedError

    # ---- Optional pieces you can override if needed ----------------------

    def sample_censoring_time(self, x: np.ndarray, a: np.ndarray) -> np.ndarray:
        """
        Sample censoring time C | X, A.

        Unless set, default is no censoring (C = +inf for all samples) resulting in regression without censoring.
        Override this to introduce informative or non-informative censoring.

        Returns
        -------
        t_cens : np.ndarray of shape (n,)
        """
        return np.full(shape=x.shape[0], fill_value=np.inf, dtype=float)

    def compute_true_cate(
        self,
        x: np.ndarray,
        horizons: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """
        Optionally return ground-truth CATE for evaluation.

        By default, returns None (no ground-truth CATE stored).
        Override this for DGMs where the estimand is analytically known.

        Parameters
        ----------
        x : np.ndarray
            Covariates, shape (n, d).
        horizons : np.ndarray, optional
            Time horizons at which the true CATE is computed.

        Returns
        -------
        true_cate : np.ndarray or None
            Shape (n,) or (n, n_horizons).
        """
        return None

    # ---- High-level helpers ----------------------------------------------

    def generate(
        self,
        n: int,
        horizons: Optional[np.ndarray] = None,
    ) -> GeneratedDataset:
        """
        Generate a full dataset (X, A, time, event[, true_cate]).

        Parameters
        ----------
        n : int
            Number of samples.
        horizons : np.ndarray, optional
            Horizons for which true CATE is computed, if supported.

        Returns
        -------
        GeneratedDataset
        """
        x = self.sample_covariates(n)
        a = self.sample_treatment(x)
        t_event = self.sample_event_time(x, a)
        t_cens = self.sample_censoring_time(x, a)

        time = np.minimum(t_event, t_cens)
        event = (t_event <= t_cens).astype(int)

        true_cate = self.compute_true_cate(x, horizons=horizons)

        return GeneratedDataset(
            x=x,
            a=a,
            time=time,
            event=event,
            true_cate=true_cate,
            metadata={"horizons": horizons},
        )

    def to_dataframe(self, data: GeneratedDataset) -> pd.DataFrame:
        """
        Convert a GeneratedDataset to a long-form DataFrame compatible with
        the existing loading utilities.

        Columns: x0, x1, ..., x{d-1}, a, time, event
        """
        n, d = data.x.shape
        df = pd.DataFrame(data.x, columns=[f"x{i}" for i in range(d)])
        df["a"] = data.a
        df["time"] = data.time
        df["event"] = data.event
        return df
    
    def to_h5(
        self,
        data: GeneratedDataset,
        path: str | Path,
        overwrite: bool = False,
    ) -> None:
        """
        Save a GeneratedDataset to an .h5 file.

        This is a simple default that you can adapt to match your
        existing synthetic-data schema.

        Datasets:
          - X      : (n, d) covariates
          - A      : (n,) treatment
          - time   : (n,) observed time
          - event  : (n,) event indicator
          - true_cate (optional): if provided

        Attributes:
          - any metadata entries are stored as JSON-encoded strings.
        """
        path = Path(path)
        if path.exists() and not overwrite:
            raise FileExistsError(
                f"{path} already exists. Use overwrite=True to overwrite."
            )

        with h5py.File(path, "w") as f:
            f.create_dataset("X", data=data.x)
            f.create_dataset("A", data=data.a)
            f.create_dataset("time", data=data.time)
            f.create_dataset("event", data=data.event)

            if data.true_cate is not None:
                f.create_dataset("true_cate", data=data.true_cate)

            if data.metadata is not None:
                for k, v in data.metadata.items():
                    try:
                        f.attrs[k] = json.dumps(v)
                    except TypeError:
                        # Fallback to string representation
                        f.attrs[k] = str(v)
