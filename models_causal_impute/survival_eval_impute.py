from sklearn.model_selection import KFold
import numpy as np
from tqdm import trange
from dataclasses import InitVar, dataclass, field
import warnings

# Methods adapted to our settings from survival_evaluation package and paper: https://arxiv.org/pdf/2306.01196

class SurvivalEvalImputer:
    def __init__(self, imputation_method="Pseudo_obs", verbose=True):
        self.imputation_method = imputation_method
        self.verbose = verbose


    def fit_transform(self, Y_train, Y_test, impute_train=True):
        """
        Note in our setup, Y_test imputation is not important as we calculate CATE from X_test and W_test
        where our CATE estimator is trained using X_train, W_train and Y_train.
        Nevertheless, we still impute Y_test for the sake of consistency.
        """

        if self.imputation_method == "Pseudo_obs":
            return self._pseudo_obs_imputation(Y_train, Y_test, impute_train=impute_train)
        elif self.imputation_method == "Margin":
            return self._margin_imputation(Y_train, Y_test, impute_train=impute_train)
        elif self.imputation_method == "IPCW-T":
            return self._ipcw_t_imputation(Y_train, Y_test, impute_train=impute_train)
        else:
            raise ValueError(f"Unknown imputation method: {self.imputation_method}")


    def _km_mean(self, times: np.ndarray, survival_probabilities: np.ndarray) -> float:
        """
        Calculate the mean of the Kaplan-Meier curve.

        Parameters
        ----------
        times: np.ndarray, shape = (n_samples, )
            Survival times for KM curve of the testing samples
        survival_probabilities: np.ndarray, shape = (n_samples, )
            Survival probabilities for KM curve of the testing samples

        Returns
        -------
        The mean of the Kaplan-Meier curve.
        """
        # calculate the area under the curve for each interval
        area_probabilities = np.append(1, survival_probabilities)
        area_times = np.append(0, times)
        km_linear_zero = -1 / ((area_probabilities[-1] - 1) / area_times[-1])
        if survival_probabilities[-1] != 0:
            area_times = np.append(area_times, km_linear_zero)
            area_probabilities = np.append(area_probabilities, 0)
        area_diff = np.diff(area_times, 1)
        # we are using trap rule
        average_probabilities = (area_probabilities[0:-1] + area_probabilities[1:]) / 2
        area = np.flip(np.flip(area_diff * average_probabilities).cumsum())
        area = np.append(area, 0)
        # or the step function rule (deprecated for now)
        # area_subs = area_diff * area_probabilities[0:-1]
        # area_subs[-1] = area_subs[-1] / 2
        # area = np.flip(np.flip(area_subs).cumsum())

        # calculate the mean
        probability_index = np.digitize(0, times)
        surv_prob = np.append(1, survival_probabilities)[probability_index]

        return area[0] / surv_prob
    

    def _pseudo_obs_imputation_train(self, Y_train):
        """
        Pseudo-observation imputation method.
        Calculate the best guess time (surrogate time) by the contribution of the censored subjects to KM curve
        """
        event_times = Y_train[:, 0]
        event_indicators = (Y_train[:, 1]).astype(bool)
        max_horizon_time = max(event_times)

        best_guesses = event_times.copy().astype(float)

        for i in trange(len(event_times), desc="Calculating surrogate times for Pseudo-observation", disable=not self.verbose):
            if event_indicators[i] == 1:
                continue

            # train_event_times would be all the event times except the current one
            train_event_times = np.delete(event_times, i)
            train_event_indicators = np.delete(event_indicators, i)
            test_event_time = event_times[i]
            test_event_indicator = event_indicators[i]

            n_train = train_event_times.size

            km_model = KaplanMeierArea(train_event_times, train_event_indicators)

            # Survival eval extrapolates the KM curve to the right until survival probability reaches 0
            # km_linear_zero = km_model.km_linear_zero
            # We instead use the max time in the training set
            km_linear_zero = max_horizon_time

            events, population_counts = km_model.events.copy(), km_model.population_count.copy()
            times = km_model.survival_times.copy()
            probs = km_model.survival_probabilities.copy()

            # get the discrete time points where the event happens, then calculate the area under those discrete time only
            # this doesn't make any difference for step function, but it does for trapezoid rule.
            unique_idx = np.where(events != 0)[0]
            if unique_idx[-1] != len(events) - 1:
                unique_idx = np.append(unique_idx, len(events) - 1)
            times = times[unique_idx]
            population_counts = population_counts[unique_idx]
            events = events[unique_idx]
            probs = probs[unique_idx]
            sub_expect_time = self._km_mean(times.copy(), probs.copy())

            # use the idea of dynamic programming to calculate the multiplier of the KM estimator in advance.
            # if we add a new time point to the KM curve, the multiplier before the new time point will be
            # 1 - event_counts / (population_counts + 1), and the multiplier after the new time point will be
            # the same as before.
            multiplier = 1 - events / population_counts
            multiplier_total = 1 - events / (population_counts + 1)

            total_multiplier = multiplier.copy()
            insert_index = np.searchsorted(times, test_event_time, side='right')
            total_multiplier[:insert_index] = multiplier_total[:insert_index]
            survival_probabilities = np.cumprod(total_multiplier)
            if insert_index == len(times):
                times_addition = np.append(times, test_event_time)
                survival_probabilities_addition = np.append(survival_probabilities, survival_probabilities[-1])
                total_expect_time = self._km_mean(times_addition, survival_probabilities_addition)
            else:
                total_expect_time = self._km_mean(times, survival_probabilities)
            best_guesses[i] = (n_train + 1) * total_expect_time - n_train * sub_expect_time
            if best_guesses[i] < test_event_time:
                best_guesses[i] = test_event_time
                warnings.warn(f"[Train Imputes] Best guess for training sample {i} is less than the observed time. Setting it to the observed time.")

        assert np.all(best_guesses >= 0), "Best guesses should be non-negative"
        assert np.all(best_guesses[Y_train[:, 1] == 0] >= event_times[Y_train[:, 1] == 0]), "Best guesses should be greater than or equal to censor times"
        assert np.all(best_guesses[Y_train[:, 1] == 1] == event_times[Y_train[:, 1] == 1]), "Best guesses should be less than or equal to event times"

        return best_guesses


    def _pseudo_obs_imputation(self, Y_train, Y_test, impute_train=True):
        """
        Pseudo-observation imputation method.
        Calculate the best guess time (surrogate time) by the contribution of the censored subjects to KM curve
        
        Note: We do not need to impute Y_test in our setup, but we still do it for consistency.

        :param Y_train: np.ndarray, shape = (n_samples, 2)
            The training set with observed time and event indicator
        :param Y_test: np.ndarray, shape = (n_samples, 2)
            The test set with observed time and event indicator
        :param impute_train: bool
            Whether to impute the training set

        :return best_guesses_train: np.ndarray, shape = (n_samples, )
            The imputed time for traing set.
            
            (if impute_train is False, the observed time for the training set is returned)
        :return best_guesses: np.ndarray, shape = (n_samples, )
            The imputed time for test set.
        """
        train_event_times = Y_train[:, 0]
        train_event_indicators = (Y_train[:, 1]).astype(bool)
        test_event_times = Y_test[:, 0]
        test_event_indicators = (Y_test[:, 1]).astype(bool)

        n_train = train_event_times.size
        n_test = test_event_times.size

        km_model = KaplanMeierArea(train_event_times, train_event_indicators)

        # Survival eval extrapolates the KM curve to the right until survival probability reaches 0
        # km_linear_zero = km_model.km_linear_zero
        # We instead use the max time in the training set
        km_linear_zero = max(km_model.survival_times)

        test_censor_times = test_event_times[~test_event_indicators]

        events, population_counts = km_model.events.copy(), km_model.population_count.copy()
        times = km_model.survival_times.copy()
        probs = km_model.survival_probabilities.copy()

        # get the discrete time points where the event happens, then calculate the area under those discrete time only
        # this doesn't make any difference for step function, but it does for trapezoid rule.
        unique_idx = np.where(events != 0)[0]
        if unique_idx[-1] != len(events) - 1:
            unique_idx = np.append(unique_idx, len(events) - 1)
        times = times[unique_idx]
        population_counts = population_counts[unique_idx]
        events = events[unique_idx]
        probs = probs[unique_idx]
        sub_expect_time = self._km_mean(times.copy(), probs.copy())


        # use the idea of dynamic programming to calculate the multiplier of the KM estimator in advance.
        # if we add a new time point to the KM curve, the multiplier before the new time point will be
        # 1 - event_counts / (population_counts + 1), and the multiplier after the new time point will be
        # the same as before.
        multiplier = 1 - events / population_counts
        multiplier_total = 1 - events / (population_counts + 1)
        best_guesses = test_event_times.copy().astype(float)

        for i in trange(n_test, desc="Calculating surrogate times for Pseudo-observation", disable=not self.verbose):
            if test_event_indicators[i] != 1:
                total_multiplier = multiplier.copy()
                insert_index = np.searchsorted(times, test_event_times[i], side='right')
                total_multiplier[:insert_index] = multiplier_total[:insert_index]
                survival_probabilities = np.cumprod(total_multiplier)
                if insert_index == len(times):
                    times_addition = np.append(times, test_event_times[i])
                    survival_probabilities_addition = np.append(survival_probabilities, survival_probabilities[-1])
                    total_expect_time = self._km_mean(times_addition, survival_probabilities_addition)
                else:
                    total_expect_time = self._km_mean(times, survival_probabilities)
                best_guesses[i] = (n_train + 1) * total_expect_time - n_train * sub_expect_time
                if best_guesses[i] < test_event_times[i]:
                    best_guesses[i] = test_event_times[i]
                    warnings.warn(f"[Testing Imputes] Best guess for test sample {i} is less than the observed time. Setting it to the observed time.")

        assert np.all(best_guesses >= 0), "Best guesses should be non-negative"
        assert np.all(best_guesses[Y_test[:, 1] == 0] >= test_censor_times), "Best guesses should be greater than or equal to censor times"
        assert np.all(best_guesses[Y_test[:, 1] == 1] == test_event_times[test_event_indicators]), "Best guesses should be less than or equal to event times"


        if impute_train:
            best_guesses_train = self._pseudo_obs_imputation_train(Y_train)
        else:
            best_guesses_train = Y_train[:, 0].copy()

        return best_guesses_train, best_guesses
    

    def _margin_imputation_train(self, Y_train, num_folds=5):
        """
        Margin imputation method.
        Calculate the best guess time (surrogate time) by the contribution of the censored subjects to KM curve
        The L1-margin method proposed by https://www.jmlr.org/papers/v21/18-772.html
        Calculate the best guess survival time given the KM curve and censoring time of that patient.
        :param Y_train: np.ndarray, shape = (n_samples, 2)
            The training set with observed time and event indicator
        :param num_folds: int
            The number of folds for cross-validation
        :return best_guesses: np.ndarray, shape = (n_samples, )
            The imputed time for traing set.
        """
        event_times = Y_train[:, 0]
        event_indicators = Y_train[:, 1].astype(bool)
        max_horizon_time = max(event_times)
        
        n = len(Y_train)
        best_guesses = event_times.copy().astype(float)

        # Only impute censored points
        censored_indices = np.where(~event_indicators)[0]

        if len(censored_indices) == 0:
            return best_guesses
        elif len(censored_indices) == 1:
            km_train_idx = np.setdiff1d(np.arange(n), censored_indices)
            km_model = KaplanMeierArea(event_times[km_train_idx], event_indicators[km_train_idx])

            km_linear_zero = max_horizon_time
            val_censor_times = event_times[censored_indices]

            imputed_val = km_model.best_guess(val_censor_times)
            imputed_val[val_censor_times > km_linear_zero] = val_censor_times[val_censor_times > km_linear_zero]
            
            best_guesses[censored_indices] = imputed_val
            return best_guesses


        if len(censored_indices) < num_folds:
            num_folds = len(censored_indices)
        kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

        # Split only censored indices
        for train_index, val_index in kf.split(censored_indices):
            censored_train_idx = censored_indices[train_index]
            censored_val_idx = censored_indices[val_index]
            
            # Build KM on full training data excluding the censored validation fold
            km_train_idx = np.setdiff1d(np.arange(n), censored_val_idx)
            km_model = KaplanMeierArea(event_times[km_train_idx], event_indicators[km_train_idx])
            
            km_linear_zero = max_horizon_time
            val_censor_times = event_times[censored_val_idx]
            
            imputed_val = km_model.best_guess(val_censor_times)
            imputed_val[val_censor_times > km_linear_zero] = val_censor_times[val_censor_times > km_linear_zero]
            
            best_guesses[censored_val_idx] = imputed_val

        assert np.all(best_guesses >= 0), "Best guesses must be non-negative"
        assert np.all(best_guesses[~event_indicators] >= event_times[~event_indicators]), "Imputed must be ≥ censoring time"
        assert np.all(best_guesses[event_indicators] == event_times[event_indicators]), "Uncensored should match original"

        return best_guesses

    
    def _margin_imputation(self, Y_train, Y_test, impute_train=True):
        """
        Margin imputation method.
        Calculate the best guess time (surrogate time) by the contribution of the censored subjects to KM curve
        
        The L1-margin method proposed by https://www.jmlr.org/papers/v21/18-772.html
        
        Calculate the best guess survival time given the KM curve and censoring time of that patient.

        :param Y_train: np.ndarray, shape = (n_samples, 2)
            The training set with observed time and event indicator
        :param Y_test: np.ndarray, shape = (n_samples, 2)
            The test set with observed time and event indicator
        :param impute_train: bool
            Whether to impute the training set
        :return best_guesses_train: np.ndarray, shape = (n_samples, )
            The imputed time for traing set.
            (if impute_train is False, the observed time for the training set is returned)
        :return best_guesses: np.ndarray, shape = (n_samples, )
            The imputed time for test set.
        """
        train_event_times = Y_train[:, 0]
        train_event_indicators = (Y_train[:, 1]).astype(bool)
        test_event_times = Y_test[:, 0]
        test_event_indicators = (Y_test[:, 1]).astype(bool)

        n_train = train_event_times.size
        n_test = test_event_times.size

        km_model = KaplanMeierArea(train_event_times, train_event_indicators)

        # Survival eval extrapolates the KM curve to the right until survival probability reaches 0
        # km_linear_zero = km_model.km_linear_zero
        # We instead use the max time in the training set
        km_linear_zero = max(km_model.survival_times)

        test_censor_times = test_event_times[~test_event_indicators]

        # The L1-margin method proposed by https://www.jmlr.org/papers/v21/18-772.html
        # Calculate the best guess survival time given the KM curve and censoring time of that patient
        best_guesses_censored_data = km_model.best_guess(test_censor_times)
        best_guesses_censored_data[test_censor_times > km_linear_zero] = test_censor_times[test_censor_times > km_linear_zero]

        best_guesses = test_event_times.copy().astype(float)
        best_guesses[~test_event_indicators] = best_guesses_censored_data
        assert np.all(best_guesses >= 0), "Best guesses should be non-negative"
        assert np.all(best_guesses[Y_test[:, 1] == 0] >= test_censor_times), "Best guesses should be greater than or equal to censor times"
        assert np.all(best_guesses[Y_test[:, 1] == 1] == test_event_times[test_event_indicators]), "Best guesses should be less than or equal to event times"

        if impute_train:
            best_guesses_train = self._margin_imputation_train(Y_train)
        else:
            best_guesses_train = Y_train[:, 0].copy()
        
        return best_guesses_train, best_guesses


    def _ipcw_t_imputation_train(self, Y_train):
        """
        IPCW-T imputation method.
        Calculate the best guess time (surrogate time) based on the subsequent uncensored subjects
        The IPCW-T method proposed by https://arxiv.org/pdf/2306.01196.pdf
        Calculate the best guess survival time given the KM curve and censoring time of that patient.
        :param Y_train: np.ndarray, shape = (n_samples, 2)
            The training set with observed time and event indicator
        :return best_guesses: np.ndarray, shape = (n_samples, )
            The imputed time for traing set.
        """
        train_event_times = Y_train[:, 0]
        train_event_indicators = (Y_train[:, 1]).astype(bool)

        n_train = train_event_times.size

        # Survival eval extrapolates the KM curve to the right until survival probability reaches 0
        # km_linear_zero = km_model.km_linear_zero
        # We instead use the max time in the training set
        km_model = KaplanMeierArea(train_event_times, train_event_indicators)
        km_linear_zero = max(km_model.survival_times)


        # This is the IPCW-T method from https://arxiv.org/pdf/2306.01196.pdf
        # Calculate the best guess time (surrogate time) based on the subsequent uncensored subjects
        best_guesses = np.empty(shape=n_train)
        for i in range(n_train):
            if train_event_indicators[i] == 1:
                best_guesses[i] = train_event_times[i]
            else:
                # Numpy will throw a warning if afterward_event_times are all false. TODO: consider change the code.
                afterward_event_idx = train_event_times[train_event_indicators == 1] > train_event_times[i]
                if afterward_event_idx.sum() == 0:
                    best_guesses[i] = km_linear_zero
                else:
                    best_guesses[i] = np.mean(train_event_times[train_event_indicators == 1][afterward_event_idx])

                if best_guesses[i] < train_event_times[i]:
                    best_guesses[i] = train_event_times[i]
                    warnings.warn(f"[Train Imputes] Best guess for training sample {i} is less than the observed time. Setting it to the observed time.")
                                    
        # NaN values are generated because there are no events after the censor times
        nan_idx = np.argwhere(np.isnan(best_guesses))
        best_guesses = np.delete(best_guesses, nan_idx)

        assert np.all(best_guesses >= 0), "Best guesses should be non-negative"
        assert np.all(best_guesses[Y_train[:, 1] == 0] >= train_event_times[Y_train[:, 1] == 0]), "Best guesses should be greater than or equal to censor times"
        assert np.all(best_guesses[Y_train[:, 1] == 1] == train_event_times[Y_train[:, 1] == 1]), "Best guesses should be less than or equal to event times"

        return best_guesses
    

    def _ipcw_t_imputation(self, Y_train, Y_test, impute_train=True):
        """
        IPCW-T imputation method.
        Calculate the best guess time (surrogate time) based on the subsequent uncensored subjects
        The IPCW-T method proposed by https://arxiv.org/pdf/2306.01196.pdf
        Calculate the best guess survival time given the KM curve and censoring time of that patient.
        :param Y_train: np.ndarray, shape = (n_samples, 2)
            The training set with observed time and event indicator
        :param Y_test: np.ndarray, shape = (n_samples, 2)
            The test set with observed time and event indicator
        :param impute_train: bool
            Whether to impute the training set
        :return best_guesses_train: np.ndarray, shape = (n_samples, )
            The imputed time for traing set.
            (if impute_train is False, the observed time for the training set is returned)
        :return best_guesses: np.ndarray, shape = (n_samples, )
            The imputed time for test set.
        """
        train_event_times = Y_train[:, 0]
        train_event_indicators = (Y_train[:, 1]).astype(bool)
        test_event_times = Y_test[:, 0]
        test_event_indicators = (Y_test[:, 1]).astype(bool)

        n_train = train_event_times.size
        n_test = test_event_times.size

        km_model = KaplanMeierArea(train_event_times, train_event_indicators)

        # Survival eval extrapolates the KM curve to the right until survival probability reaches 0
        # km_linear_zero = km_model.km_linear_zero
        # We instead use the max time in the training set
        km_linear_zero = max(km_model.survival_times)

        test_censor_times = test_event_times[~test_event_indicators]

        # This is the IPCW-T method from https://arxiv.org/pdf/2306.01196.pdf
        # Calculate the best guess time (surrogate time) based on the subsequent uncensored subjects
        best_guesses = np.empty(shape=n_test)
        for i in range(n_test):
            if test_event_indicators[i] == 1:
                best_guesses[i] = test_event_times[i]
            else:
                # Numpy will throw a warning if afterward_event_times are all false. TODO: consider change the code.
                afterward_event_idx = train_event_times[train_event_indicators == 1] > test_event_times[i]
                if afterward_event_idx.sum() == 0:
                    best_guesses[i] = km_linear_zero
                else:
                    best_guesses[i] = np.mean(train_event_times[train_event_indicators == 1][afterward_event_idx])

                if best_guesses[i] < test_event_times[i]:
                    best_guesses[i] = test_event_times[i]
                    warnings.warn(f"[Testing Imputes] Best guess for test sample {i} is less than the observed time. Setting it to the observed time.")
                
        # NaN values are generated because there are no events after the censor times
        nan_idx = np.argwhere(np.isnan(best_guesses))
        best_guesses = np.delete(best_guesses, nan_idx)

        assert np.all(best_guesses >= 0), "Best guesses should be non-negative"
        assert np.all(best_guesses[Y_test[:, 1] == 0] >= test_censor_times), "Best guesses should be greater than or equal to censor times"
        assert np.all(best_guesses[Y_test[:, 1] == 1] == test_event_times[test_event_indicators]), "Best guesses should be less than or equal to event times"

        if impute_train:
            best_guesses_train = self._ipcw_t_imputation_train(Y_train)
        else:
            best_guesses_train = Y_train[:, 0].copy()

        return best_guesses_train, best_guesses

@dataclass
class KaplanMeier:
    """
    This class is borrowed from survival_evaluation package.
    """
    event_times: InitVar[np.array]
    event_indicators: InitVar[np.array]
    survival_times: np.array = field(init=False)
    population_count: np.array = field(init=False)
    events: np.array = field(init=False)
    survival_probabilities: np.array = field(init=False)
    cumulative_dens: np.array = field(init=False)
    probability_dens: np.array = field(init=False)

    def __post_init__(self, event_times, event_indicators):
        index = np.lexsort((event_indicators, event_times))
        unique_times = np.unique(event_times[index], return_counts=True)
        self.survival_times = unique_times[0]
        self.population_count = np.flip(np.flip(unique_times[1]).cumsum())

        event_counter = np.append(0, unique_times[1].cumsum()[:-1])
        event_ind = list()
        for i in range(np.size(event_counter[:-1])):
            event_ind.append(event_counter[i])
            event_ind.append(event_counter[i + 1])
        event_ind.append(event_counter[-1])
        event_ind.append(len(event_indicators))
        self.events = np.add.reduceat(np.append(event_indicators[index], 0), event_ind)[::2]

        event_ratios = 1 - self.events / self.population_count
        self.survival_probabilities = np.cumprod(event_ratios)
        self.cumulative_dens = 1 - self.survival_probabilities
        self.probability_dens = np.diff(np.append(self.cumulative_dens, 1))

    def predict(self, prediction_times: np.array):
        probability_index = np.digitize(prediction_times, self.survival_times)
        probability_index = np.where(
            probability_index == self.survival_times.size + 1,
            probability_index - 1,
            probability_index,
        )
        probabilities = np.append(1, self.survival_probabilities)[probability_index]

        return probabilities


@dataclass
class KaplanMeierArea(KaplanMeier):
    area_times: np.array = field(init=False)
    area_probabilities: np.array = field(init=False)
    area: np.array = field(init=False)
    km_linear_zero: float = field(init=False)

    def __post_init__(self, event_times, event_indicators):
        super().__post_init__(event_times, event_indicators)
        area_probabilities = np.append(1, self.survival_probabilities)
        area_times = np.append(0, self.survival_times)
        self.km_linear_zero = area_times[-1] / (1 - area_probabilities[-1])
        if self.survival_probabilities[-1] != 0:
            area_times = np.append(area_times, self.km_linear_zero)
            area_probabilities = np.append(area_probabilities, 0)

        # we are facing the choice of using the trapzoidal rule or directly using the area under the step function
        # we choose to use trapz because it is more accurate
        area_diff = np.diff(area_times, 1)
        average_probabilities = (area_probabilities[0:-1] + area_probabilities[1:]) / 2
        area = np.flip(np.flip(area_diff * average_probabilities).cumsum())
        # area = np.flip(np.flip(area_diff * area_probabilities[0:-1]).cumsum())

        self.area_times = np.append(area_times, np.inf)
        self.area_probabilities = area_probabilities
        self.area = np.append(area, 0)

    @property
    def mean(self):
        return self.best_guess(np.array([0])).item()

    def best_guess(self, censor_times: np.array):
        # calculate the slope using the [0, 1] - [max_time, S(t|x)]
        slope = (1 - min(self.survival_probabilities)) / (0 - max(self.survival_times))
        # if after the last time point, then the best guess is the linear function
        before_last_idx = censor_times <= max(self.survival_times)
        after_last_idx = censor_times > max(self.survival_times)
        surv_prob = np.empty_like(censor_times).astype(float)
        surv_prob[after_last_idx] = 1 + censor_times[after_last_idx] * slope
        surv_prob[before_last_idx] = self.predict(censor_times[before_last_idx])
        # do not use np.clip(a_min=0) here because we will use surv_prob as the denominator,
        # if surv_prob is below 0 (or 1e-10 after clip), the nominator will be 0 anyway.
        surv_prob = np.clip(surv_prob, a_min=1e-10, a_max=None)

        censor_indexes = np.digitize(censor_times, self.area_times)
        censor_indexes = np.where(
            censor_indexes == self.area_times.size + 1,
            censor_indexes - 1,
            censor_indexes,
        )

        # for those beyond the end point, censor_area = 0
        beyond_idx = censor_indexes > len(self.area_times) - 2
        censor_area = np.zeros_like(censor_times).astype(float)
        # trapzoidal rule:  (x1 - x0) * (f(x0) + f(x1)) * 0.5
        censor_area[~beyond_idx] = ((self.area_times[censor_indexes[~beyond_idx]] - censor_times[~beyond_idx]) *
                                    (self.area_probabilities[censor_indexes[~beyond_idx]] + surv_prob[~beyond_idx])
                                    * 0.5)
        censor_area[~beyond_idx] += self.area[censor_indexes[~beyond_idx]]
        return censor_times + censor_area / surv_prob

    def _km_linear_predict(self, times):
        slope = (1 - min(self.survival_probabilities)) / (0 - max(self.survival_times))

        predict_prob = np.empty_like(times)
        before_last_time_idx = times <= max(self.survival_times)
        after_last_time_idx = times > max(self.survival_times)
        predict_prob[before_last_time_idx] = self.predict(times[before_last_time_idx])
        predict_prob[after_last_time_idx] = np.clip(1 + times[after_last_time_idx] * slope, a_min=0, a_max=None)
        # if time <= max(self.survival_times):
        #     predict_prob = self.predict(time)
        # else:
        #     predict_prob = max(1 + time * slope, 0)
        return predict_prob

    def _compute_best_guess(self, time: float, restricted: bool = False):
        """
        Given a censor time, compute the decensor event time based on the residual mean survival time on KM curves.
        :param time:
        :return:
        """
        # Using integrate.quad from Scipy should be more accurate, but also making the program unbearably slow.
        # The compromised method uses numpy.trapz to approximate the integral using composite trapezoidal rule.
        warnings.warn("This method is deprecated. Use best_guess instead.", DeprecationWarning)
        if restricted:
            last_time = max(self.survival_times)
        else:
            last_time = self.km_linear_zero
        time_range = np.linspace(time, last_time, 2000)
        if self.predict(time) == 0:
            best_guess = time
        else:
            best_guess = time + np.trapezoid(self._km_linear_predict(time_range), time_range) / self.predict(time)

        return best_guess