from __future__ import annotations

import abc
from typing import Any, Dict, Optional

import numpy as np


class SurvivalMetaLearnerBase(abc.ABC):
    """
    Base class for survival-adapted meta-learners (e.g., Survival T-/S-/Matching-learners).

    The meta-learner typically wraps underlying survival models for each
    treatment arm and produces treatment-specific survival curves and CATEs.

    Provides:
      * `predict_cate` (RMST-based)
      * `predict_cate_survprob` (survival-probability difference at horizon)
    """

    def __init__(
        self,
        name: str,
        time_grid: Optional[np.ndarray] = None,
        random_state: Optional[int] = None,
        **kwargs: Any,
    ):
        self.name = name
        self.time_grid = time_grid
        self.random_state = random_state
        self._init_kwargs = kwargs

    # -------- Required methods --------------------------------------------

    @abc.abstractmethod
    def fit(
        self,
        x: np.ndarray,
        a: np.ndarray,
        time: np.ndarray,
        event: np.ndarray,
        **fit_kwargs: Any,
    ) -> "SurvivalMetaLearnerBase":
        """
        Fit the survival meta-learner on (X, A, time, event).
        """
        raise NotImplementedError

    @abc.abstractmethod
    def predict_survival(
        self,
        x: np.ndarray,
        times: np.ndarray,
        treatment: int,
        **predict_kwargs: Any,
    ) -> np.ndarray:
        """
        Predict survival curves S(t | X, A=treatment) produced by the meta-learner.

        Returns
        -------
        surv : np.ndarray, shape (n_samples, n_times)
        """
        raise NotImplementedError

    # -------- Default CATE implementation (RMST-CATE) ----------------------

    def predict_cate(
        self,
        x: np.ndarray,
        horizon: Optional[float] = None,
        times: Optional[np.ndarray] = None,
        **predict_kwargs: Any,
    ) -> np.ndarray:
        """
        Default CATE estimate using RMST differences, analogously to
        DirectSurvivalCATEBase.
        """
        if times is None:
            if self.time_grid is None:
                raise ValueError("Either `times` or `self.time_grid` must be provided.")
            times = self.time_grid

        times = np.asarray(times)
        if horizon is None:
            horizon = float(times.max())

        mask = times <= horizon
        times_trunc = times[mask]

        surv1 = self.predict_survival(x, times_trunc, treatment=1, **predict_kwargs)
        surv0 = self.predict_survival(x, times_trunc, treatment=0, **predict_kwargs)

        rmst1 = np.trapz(surv1, times_trunc, axis=1)
        rmst0 = np.trapz(surv0, times_trunc, axis=1)
        cate = rmst1 - rmst0
        return cate
    
    # -------- Survival-probability CATE -----------------------------------

    def predict_cate_survprob(
        self,
        x: np.ndarray,
        horizon: float,
        times: Optional[np.ndarray] = None,
        **predict_kwargs: Any,
    ) -> np.ndarray:
        """
        CATE based on survival probability difference at a given horizon:

          tau_surv(x, h) = S_1(h | x) - S_0(h | x)
        """
        if times is None:
            if self.time_grid is None:
                raise ValueError("Either `times` or `self.time_grid` must be provided.")
            times = self.time_grid

        times = np.asarray(times, dtype=float)
        if times.ndim != 1:
            raise ValueError("`times` must be a 1D array of time points.")

        surv1 = self.predict_survival(x, times, treatment=1, **predict_kwargs)
        surv0 = self.predict_survival(x, times, treatment=0, **predict_kwargs)

        n, _ = surv1.shape
        s1_h = np.array(
            [np.interp(horizon, times, surv1[i, :]) for i in range(n)],
            dtype=float,
        )
        s0_h = np.array(
            [np.interp(horizon, times, surv0[i, :]) for i in range(n)],
            dtype=float,
        )
        return s1_h - s0_h

    # -------- Optional helpers --------------------------------------------

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        params = dict(
            name=self.name,
            time_grid=self.time_grid,
            random_state=self.random_state,
        )
        params.update(self._init_kwargs)
        return params

    def set_params(self, **params: Any) -> "SurvivalMetaLearnerBase":
        for k, v in params.items():
            if k in ("name", "time_grid", "random_state"):
                setattr(self, k, v)
            else:
                self._init_kwargs[k] = v
        return self
