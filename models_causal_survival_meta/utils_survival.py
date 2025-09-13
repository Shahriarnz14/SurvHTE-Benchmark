import numpy as np
import pandas as pd
from .concordance import concordance_td
from pycox.utils import idx_at_times
from sksurv.metrics import integrated_brier_score

from sksurv.util import check_y_survival
from sksurv.metrics import _check_estimate_2d
from sksurv.nonparametric import SurvivalFunctionEstimator
from sksurv.nonparametric import kaplan_meier_estimator



def get_concordance_score(Y_test, surv, times):
    return concordance_td(
        Y_test[:, 0],
        Y_test[:, 1],
        surv,
        idx_at_times(times, Y_test[:, 0], "post"),
        "antolini",
    )


def get_integrated_brier_score(survival_train, survival_test, surv, times):
    if survival_train is None:
        print("[Warning]: Survival train data not available. Cannot compute integrated Brier score (IBS)")
        return None
    
    _, train_time = check_y_survival(survival_train)
    _, test_time = check_y_survival(survival_test)

    min_time = max(test_time.min(), train_time.min())
    max_time = min(test_time.max(), train_time.max())

    # Cap evaluation times to avoid exceeding training support
    times_exclusive = times[(times >= min_time) & (times < max_time)]
    filtered_surv = surv[(times >= min_time) & (times < max_time)]

    try:
        ibs = integrated_brier_score(survival_train, survival_test, filtered_surv.T, times_exclusive)
        return ibs
    except ValueError:
        # Cap survival_test times to max_eval_time to avoid out-of-support errors
        survival_test_adj = np.array([
            (event, max(min(time, min(times.max(), train_time.max())), max(times.min(), train_time.min()))) for event, time in survival_test
        ], dtype=[("event", "bool"), ("time", "float64")])
        # survival_test_adj = np.array([
        #     (event, min(time, min(times.max(), train_time.max()))) for event, time in survival_test
        # ], dtype=[("event", "bool"), ("time", "float64")])
        # times_exclusive_adj = times[(times >= train_time.min()) & (times < train_time.max())]
        # filtered_surv_adj = surv[(times >= train_time.min()) & (times < train_time.max())]
        return integrated_brier_score(survival_train, survival_test_adj, filtered_surv.T, times_exclusive)


def get_cumulative_dynamic_auc(survival_train, survival_test, surv, times):
    if survival_train is None:
        print("Survival train data not available. Cannot compute Time-Dependent AUC (td-AUC)")
        return None
    
    # Need to pass the cumulative hazard function to compute the Time-Dependent AUC
    # https://k-d-w.org/blog/2021/03/scikit-survival-0.15-released/
    cumulative_hazard = compute_cumulative_hazard_vectorized(surv)
    
    _, test_time = check_y_survival(survival_test)
    times_exclusive = times[(times >= test_time.min()) & (times < test_time.max())]
    filtered_cumulative_hazard = cumulative_hazard[(times >= test_time.min()) & (times < test_time.max())]

    return cumulative_dynamic_auc(survival_train, survival_test, filtered_cumulative_hazard.T, times_exclusive)


def compute_cumulative_hazard_vectorized(surv):
    # H[l|X] = \sum_{m=1}^{l} h[m|X] = \sum_{m=1}^{l} \frac{S[m-1|X]-S[m|X]}{S[m-1|X]}

    # Add S[0,j] = 1 as the first row for all columns
    surv_with_initial = np.vstack((np.ones(surv.shape[1]), surv.values))

    # Create placeholders for the numerator and denominator
    diff = surv_with_initial[:-1] - surv_with_initial[1:]
    denominator = surv_with_initial[:-1]

    # Avoid division by zero in the denominator
    incremental_hazard = np.divide(diff, denominator, out=np.zeros_like(diff), where=denominator != 0)

    # Compute the cumulative sum row-wise
    cumulative_hazard = np.cumsum(incremental_hazard, axis=0)

    cumulative_hazard_df = pd.DataFrame(cumulative_hazard, columns=surv.columns, index=surv.index)
    
    return cumulative_hazard_df


def cumulative_dynamic_auc(survival_train, survival_test, estimate, times, tied_tol=1e-8):
    test_event, test_time = check_y_survival(survival_test)
    estimate, times = _check_estimate_2d(estimate, test_time, times, estimator="cumulative_dynamic_auc")

    n_samples = estimate.shape[0]
    n_times = times.shape[0]
    if estimate.ndim == 1:
        estimate = np.broadcast_to(estimate[:, np.newaxis], (n_samples, n_times))

    # fit and transform IPCW
    cens = CensoringDistributionEstimator()
    cens.fit(survival_train)
    ipcw = cens.predict_ipcw(survival_test)

    # expand arrays to (n_samples, n_times) shape
    test_time = np.broadcast_to(test_time[:, np.newaxis], (n_samples, n_times))
    test_event = np.broadcast_to(test_event[:, np.newaxis], (n_samples, n_times))
    times_2d = np.broadcast_to(times, (n_samples, n_times))
    ipcw = np.broadcast_to(ipcw[:, np.newaxis], (n_samples, n_times))

    # sort each time point (columns) by risk score (descending)
    o = np.argsort(-estimate, axis=0)
    test_time = np.take_along_axis(test_time, o, axis=0)
    test_event = np.take_along_axis(test_event, o, axis=0)
    estimate = np.take_along_axis(estimate, o, axis=0)
    ipcw = np.take_along_axis(ipcw, o, axis=0)

    is_case = (test_time <= times_2d) & test_event
    is_control = test_time > times_2d
    n_controls = is_control.sum(axis=0)

    # prepend row of infinity values
    estimate_diff = np.concatenate((np.broadcast_to(np.inf, (1, n_times)), estimate))
    is_tied = np.absolute(np.diff(estimate_diff, axis=0)) <= tied_tol

    cumsum_tp = np.cumsum(is_case * ipcw, axis=0)
    cumsum_fp = np.cumsum(is_control, axis=0)
    true_pos = cumsum_tp / cumsum_tp[-1]
    false_pos = cumsum_fp / n_controls

    scores = np.empty(n_times, dtype=float)
    it = np.nditer((true_pos, false_pos, is_tied), order="F", flags=["external_loop"])
    with it:
        for i, (tp, fp, mask) in enumerate(it):
            idx = np.flatnonzero(mask) - 1
            # only keep the last estimate for tied risk scores
            tp_no_ties = np.delete(tp, idx)
            fp_no_ties = np.delete(fp, idx)
            # Add an extra threshold position
            # to make sure that the curve starts at (0, 0)
            tp_no_ties = np.r_[0, tp_no_ties]
            fp_no_ties = np.r_[0, fp_no_ties]
            scores[i] = np.trapz(tp_no_ties, fp_no_ties)

    if n_times == 1:
        mean_auc = scores[0]
    else:
        surv = SurvivalFunctionEstimator()
        surv.fit(survival_test)
        s_times = surv.predict_proba(times)
        # compute integral of AUC over survival function
        d = -np.diff(np.r_[1.0, s_times])
        integral = (scores * d).sum()
        mean_auc = integral / (1.0 - s_times[-1])

    return scores, mean_auc


class CensoringDistributionEstimator(SurvivalFunctionEstimator):
    """Kaplan–Meier estimator for the censoring distribution."""

    def fit(self, y):
        """Estimate censoring distribution from training data.

        Parameters
        ----------
        y : structured array, shape = (n_samples,)
            A structured array containing the binary event indicator
            as first field, and time of event or time of censoring as
            second field.

        Returns
        -------
        self
        """
        event, time = check_y_survival(y)
        if event.all():
            self.unique_time_ = np.unique(time)
            self.prob_ = np.ones(self.unique_time_.shape[0])
        else:
            unique_time, prob = kaplan_meier_estimator(event, time, reverse=True)
            self.unique_time_ = np.r_[-np.infty, unique_time]
            self.prob_ = np.r_[1.0, prob]

        return self

    def predict_ipcw(self, y):
        """Return inverse probability of censoring weights at given time points.

        :math:`\\omega_i = \\delta_i / \\hat{G}(y_i)`

        Parameters
        ----------
        y : structured array, shape = (n_samples,)
            A structured array containing the binary event indicator
            as first field, and time of event or time of censoring as
            second field.

        Returns
        -------
        ipcw : array, shape = (n_samples,)
            Inverse probability of censoring weights.
        """
        event, time = check_y_survival(y)
        Ghat = self.predict_proba(time[event])

        Ghat[Ghat == 0.0] += 1e-6

        if (Ghat == 0.0).any():
            print("[Warning]: censoring survival function is zero at one or more time points")

        weights = np.zeros(time.shape[0])
        weights[event] = 1.0 / Ghat

        return weights