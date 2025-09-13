import rpy2.robjects as ro
from rpy2.robjects import r, numpy2ri, FloatVector
from rpy2.robjects.packages import importr, PackageNotInstalledError
import numpy as np
from sklearn.metrics import mean_squared_error

class CausalSurvivalForestGRF:
    def __init__(self, failure_times_grid_size=100, horizon=None, target="RMST", min_node_size=5, seed=None):
        """
        Causal Survival Forest using the grf package in R.
        :param failure_times_grid_size: Number of points in the grid for failure times.
        :param horizon: The maximum time horizon for predictions. If None, it will be set to the maximum event time in the training data.
        :param target: The target for the causal survival forest. Default is "RMST" (Restricted Mean Survival Time).
        """
        self.failure_times_grid_size = failure_times_grid_size
        self.model = None
        self.horizon = horizon # if not provided, will be set to the maximum event time in the training data
        self.target = target
        self.min_node_size = min_node_size
        self.seed = seed

        # Activate numpy <-> R automatic conversion
        numpy2ri.activate()

        # Import the required R packages
        try:
            self.grf = importr('grf')
            self.stats = importr('stats')
        except PackageNotInstalledError as e:
            raise ImportError("The 'grf' R package is not installed. Please run `install.packages('grf')` in R.") from e

    def _to_r_matrix(self, np_matrix):
        return r.matrix(np_matrix, nrow=np_matrix.shape[0], ncol=np_matrix.shape[1])

    def _to_r_float_vector(self, np_array):
        return FloatVector(np_array.tolist())

    def _make_failure_time_grid(self, Y_time):
        return np.linspace(Y_time.min(), Y_time.max(), num=self.failure_times_grid_size)

    def fit(self, X_train, W_train, Y_train, failure_times_grid=None):
        """
        Y_train: np.array of shape (n, 2), where
            Y_train[:, 0] = event times
            Y_train[:, 1] = event indicators (1=event, 0=censored)
        """
        X_train_r = self._to_r_matrix(X_train)
        Y_time = Y_train[:, 0]
        Y_event = Y_train[:, 1]

        Y_time_r = self._to_r_float_vector(Y_time)
        Y_event_r = self._to_r_float_vector(Y_event)
        W_train_r = self._to_r_float_vector(W_train)

        if self.horizon is None:
            # Set horizon to the maximum event time in the training data
            self.horizon = float(Y_time.max())

        # Create a grid of failure times
        if failure_times_grid is None:
            failure_times_grid = self._make_failure_time_grid(Y_time)
        failure_times_r = self._to_r_float_vector(failure_times_grid)

        if self.seed is None:
            self.model = self.grf.causal_survival_forest(
                X_train_r,
                Y_time_r,
                W_train_r,
                Y_event_r,
                target=self.target,
                failure_times=failure_times_r,
                horizon=self.horizon,
                min_node_size=self.min_node_size
            )
        else:
            self.model = self.grf.causal_survival_forest(
                X_train_r,
                Y_time_r,
                W_train_r,
                Y_event_r,
                target=self.target,
                failure_times=failure_times_r,
                horizon=self.horizon,
                min_node_size=self.min_node_size,
                seed=self.seed
            )

    def predict_cate(self, X, W=None):
        """
        Predicts the CATE using the fitted model.
        :param X: np.array of shape (n, p)
        :param W: np.array of shape (n, 1) with the treatment assignment (Not used in CSF)
        :return: np.array of shape (n, 1) with the predicted CATE
        """
        if self.model is None:
            raise RuntimeError("You must call `fit` before `predict`.")
        X_test_r = self._to_r_matrix(X)
        prediction_r = self.stats.predict(self.model, X_test_r)
        return np.array(prediction_r.rx2("predictions"))

    def evaluate(self, X, cate_true, W=None):
        """
        Evaluate CATE predictions using mean squared error.
        
        Parameters:
        - X (np.ndarray): Test features.
        - cate_true (np.ndarray): Ground-truth CATE values (from simulation).
        - W (np.ndarray): Treatment assignment (not used in this method).
        
        Returns:
        - mse (float): Mean squared error.
        - cate_pred (np.ndarray): Predicted CATE values.
        """
        cate_pred = self.predict_cate(X)
        ate_pred = np.mean(cate_pred)
        mse = mean_squared_error(cate_true, cate_pred)
        return mse, cate_pred, ate_pred