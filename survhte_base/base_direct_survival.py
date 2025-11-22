from __future__ import annotations

import abc
from typing import Any, Dict, Optional

import numpy as np


class DirectSurvivalCATEBase(abc.ABC):
    """
    Base class for direct-survival CATE models (e.g., Causal Survival Forest, SurvITE).

    The key assumptions are:
      * the model is trained directly on (time, event) rather than imputed outcomes,
      * we can obtain treatment-specific survival curves S(t | X, A=a),
      * from which CATEs (e.g., RMST-CATE, survival-probability differences)
        can be derived.

    New methods should at least implement `fit` and `predict_survival`.
    The default `predict_cate` uses RMST differences computed from the
    survival curves.

    Provides:
      * `predict_cate` (RMST-based by default)
      * `predict_cate_survprob` (survival-probability difference at a horizon)
    """

    def __init__(
        self,
        name: str,
        time_grid: Optional[np.ndarray] = None,
        random_state: Optional[int] = None,
        **kwargs: Any,
    ):
        self.name = name
        self.time_grid = time_grid  # 1D array of evaluation times (optional)
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
    ) -> "DirectSurvivalCATEBase":
        """
        Fit the model on survival data.

        Parameters
        ----------
        x : np.ndarray, shape (n_samples, n_features)
        a : np.ndarray, shape (n_samples,)
            Treatment assignments.
        time : np.ndarray, shape (n_samples,)
            Observed times (event or censoring).
        event : np.ndarray, shape (n_samples,)
            Event indicators (1 = event, 0 = censored).

        Returns
        -------
        self
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
        Predict survival curves S(t | X, A=treatment).

        Parameters
        ----------
        x : np.ndarray, shape (n_samples, n_features)
        times : np.ndarray, shape (n_times,)
        treatment : int
            Treatment level (0 or 1).

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
        Default CATE estimate based on Restricted Mean Survival Time (RMST).

        RMST_A(h) = ∫_0^h S_A(t | X) dt
        CATE(x, h) = RMST_1(h) - RMST_0(h)

        Parameters
        ----------
        x : np.ndarray
        horizon : float, optional
            Upper limit of integration. If None, uses max(times).
        times : np.ndarray, optional
            Time grid. If None, uses self.time_grid (must be set).

        Returns
        -------
        cate : np.ndarray, shape (n_samples,)
        """
        if times is None:
            if self.time_grid is None:
                raise ValueError("Either `times` or `self.time_grid` must be provided.")
            times = self.time_grid

        if horizon is None:
            horizon = float(times.max())

        times = np.asarray(times)
        mask = times <= horizon
        times_trunc = times[mask]

        surv1 = self.predict_survival(x, times_trunc, treatment=1, **predict_kwargs)
        surv0 = self.predict_survival(x, times_trunc, treatment=0, **predict_kwargs)

        # Trapezoidal integration over time axis (axis=1)
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

        We obtain S_A(h | x) by interpolating the predicted survival curve
        in time (linear interpolation over the provided grid).
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

        # surv1/surv0: (n_samples, n_times)
        n, _ = surv1.shape

        # Interpolate S_A(h | x) for each sample
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

    def set_params(self, **params: Any) -> "DirectSurvivalCATEBase":
        for k, v in params.items():
            if k in ("name", "time_grid", "random_state"):
                setattr(self, k, v)
            else:
                self._init_kwargs[k] = v
        return self
