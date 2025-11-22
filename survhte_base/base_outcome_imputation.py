from __future__ import annotations

import abc
from typing import Any, Dict, Optional

import numpy as np


class OutcomeImputationBase(abc.ABC):
    """
    Base class for outcome-imputation-based HTE estimators in SurvHTE-Bench.

    This class assumes that:
      * event times have already been imputed externally (e.g., IPCW-T, Margin, or pseudo-observations),
      * the method operates on an imputed outcome y_imputed and treatment A.
        OR
      * the model itself performs an imputation step via `predict_imputed_outcome`.

    New methods should implement `fit` and `predict_cate`. The benchmark
    assumes a scikit-learn-like interface.
    """

    def __init__(self, name: str, random_state: Optional[int] = None, **kwargs: Any):
        self.name = name
        self.random_state = random_state
        self._init_kwargs = kwargs

    # -------- Required methods --------------------------------------------

    @abc.abstractmethod
    def fit(
        self,
        x: np.ndarray,
        a: np.ndarray,
        y_imputed: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
        **fit_kwargs: Any,
    ) -> "OutcomeImputationBase":
        """
        Fit the model on imputed outcomes.

        Parameters
        ----------
        x : np.ndarray, shape (n_samples, n_features)
        a : np.ndarray, shape (n_samples,)
            Treatment assignments.
        y_imputed : np.ndarray, shape (n_samples,)
            Imputed outcome used for HTE estimation.
        sample_weight : np.ndarray, optional
            Optional weights.

        Returns
        -------
        self
        """
        raise NotImplementedError

    @abc.abstractmethod
    def predict_cate(
        self,
        x: np.ndarray,
        horizon: Optional[float] = None,
        **predict_kwargs: Any,
    ) -> np.ndarray:
        """
        Predict conditional average treatment effects tau(x, horizon).

        For methods that do not depend on a specific horizon, the
        `horizon` argument can be ignored.

        Returns
        -------
        cate : np.ndarray, shape (n_samples,)
        """
        raise NotImplementedError
    
    # -------- Optional imputation hook ------------------------------------

    def predict_imputed_outcome(
        self,
        x: np.ndarray,
        a: np.ndarray,
        time: Optional[np.ndarray] = None,
        event: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Optional hook for methods that also perform the outcome imputation step.

        If your method directly maps (X, A, time, event) -> y_imputed, you can
        implement this and let the benchmark call it instead of relying on
        precomputed imputations.

        Default behavior: raise NotImplementedError, indicating that this
        estimator expects y_imputed to be provided to `fit`.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement `predict_imputed_outcome`. "
            "Either implement this method or provide `y_imputed` externally."
        )

    # -------- Optional helpers --------------------------------------------

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """
        Parameter API compatible with scikit-learn.
        """
        params = dict(name=self.name, random_state=self.random_state)
        params.update(self._init_kwargs)
        return params

    def set_params(self, **params: Any) -> "OutcomeImputationBase":
        for k, v in params.items():
            if k in ("name", "random_state"):
                setattr(self, k, v)
            else:
                self._init_kwargs[k] = v
        return self
