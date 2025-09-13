from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
from scipy.spatial.distance import cdist
from .survival_base import SurvivalModelBase

class BaseMetaLearnerSurvival(ABC):
    """
    Abstract base class for causal meta-learners with survival outcomes.
    
    Subclasses must implement:
    - fit(X_train, W_train, Y_train)
    - predict_cate(X)
    """
    def __init__(self, base_model_name='DeepSurv', base_model_params=None, 
                 base_model_grid=None, metric='mean', max_time=np.inf):
        """
        Initialize the survival meta-learner.
        
        Parameters:
        - base_model_name (str): Type of survival model (CoxPH, RandomSurvivalForest, etc.)
        - base_model_params (dict): Parameters for the base model
        - base_model_grid (dict): Grid search parameters for model selection
        - metric (str): Survival metric to use ('median' or 'mean')
        - max_time (float): Maximum time for restricted mean survival time
        """
        self.base_model_name = base_model_name
        self.base_model_params = base_model_params if base_model_params else {}
        self.base_model_grid = base_model_grid if base_model_grid else {}
        self.metric = metric
        self.max_time = max_time
        self.models = {}
        self.evaluation_test_dict = {}

    @abstractmethod
    def fit(self, X_train, W_train, Y_train):
        """
        Fit the meta-learner on training data.
        """
        pass
    
    @abstractmethod
    def evaluate_test(self, X_test, Y_test, W_test):
        """
        Evaluate base models on test data.
        
        Parameters:
        - X_test (np.ndarray): Test features
        - Y_test (np.ndarray): Test targets
        
        Returns:
        - dict: Evaluation metrics for each base model
        """
        pass

    @abstractmethod
    def predict_cate(self, X):
        """
        Predict Conditional Average Treatment Effect (CATE) on test data.
        """
        pass

    def evaluate(self, X, cate_true, W=None):
        """
        Evaluate CATE predictions using mean squared error.
        
        Parameters:
        - X (np.ndarray): Test features
        - cate_true (np.ndarray): Ground-truth CATE values (from simulation)
        - W (np.ndarray, optional): Treatment assignment
        
        Returns:
        - mse (float): Mean squared error
        - cate_pred (np.ndarray): Predicted CATE values
        - ate_pred (float): Average Treatment Effect
        """
        cate_pred = self.predict_cate(X, W)
        ate_pred = np.mean(cate_pred)
        mse = mean_squared_error(cate_true, cate_pred)
        return mse, cate_pred, ate_pred


class TLearnerSurvival(BaseMetaLearnerSurvival):
    """
    T-Learner for survival data: Trains separate survival models for treated and control groups.
    """
    def fit(self, X_train, W_train, Y_train):
        """
        Fit separate models for treatment and control groups.
        
        Parameters:
        - X_train (np.ndarray): Training features
        - W_train (np.ndarray): Treatment indicators (0/1)
        - Y_train (np.ndarray): Training targets (time, event)
        """
        model_treated = SurvivalModelBase(
            model_type=self.base_model_name,
            hyperparams=self.base_model_params,
            hyperparams_grid=self.base_model_grid
        )
        model_control = SurvivalModelBase(
            model_type=self.base_model_name,
            hyperparams=self.base_model_params,
            hyperparams_grid=self.base_model_grid
        )
        
        # Fit models on treatment and control groups
        model_treated.fit(X_train[W_train == 1], Y_train[W_train == 1])
        model_control.fit(X_train[W_train == 0], Y_train[W_train == 0])
        
        self.models['treated'] = model_treated
        self.models['control'] = model_control
    
    def evaluate_test(self, X_test, Y_test, W_test):
        """
        Evaluate base models on test data.
        
        Parameters:
        - X_test (np.ndarray): Test features
        - Y_test (np.ndarray): Test targets
        - W_test (np.ndarray): Treatment indicators
        
        Returns:
        - dict: Evaluation metrics for each base model
        """
        self.evaluation_test_dict = {}
        
        # Evaluate treated model
        treated_eval = self.models['treated'].evaluate(X_test[W_test == 1], Y_test[W_test == 1])
        self.evaluation_test_dict['treated'] = treated_eval
        
        # Evaluate control model
        control_eval = self.models['control'].evaluate(X_test[W_test == 0], Y_test[W_test == 0])
        self.evaluation_test_dict['control'] = control_eval
        
        return self.evaluation_test_dict

    def predict_cate(self, X, W=None):
        """
        Predict CATE as difference between treatment and control predictions.
        
        Parameters:
        - X (np.ndarray): Test features
        - W (np.ndarray, optional): Not used in T-Learner
        
        Returns:
        - np.ndarray: Predicted CATE values
        """
        # Predict using specified metric (median or mean survival time)
        mu1 = self.models['treated'].predict_metric(X, metric=self.metric, max_time=self.max_time)
        mu0 = self.models['control'].predict_metric(X, metric=self.metric, max_time=self.max_time)
        return mu1 - mu0


class SLearnerSurvival(BaseMetaLearnerSurvival):
    """
    S-Learner for survival data: Trains a single model using treatment as a feature.
    """
    def fit(self, X_train, W_train, Y_train):
        """
        Fit a single model with treatment as feature.
        
        Parameters:
        - X_train (np.ndarray): Training features
        - W_train (np.ndarray): Treatment indicators (0/1)
        - Y_train (np.ndarray): Training targets (time, event)
        """
        # Augment features with treatment indicator
        X_aug = np.column_stack((X_train, W_train))
        
        model = SurvivalModelBase(
            model_type=self.base_model_name,
            hyperparams=self.base_model_params,
            hyperparams_grid=self.base_model_grid
        )
        model.fit(X_aug, Y_train)
        self.models['s'] = model
    
    def evaluate_test(self, X_test, Y_test, W_test):
        """
        Evaluate base model on test data.
        
        Parameters:
        - X_test (np.ndarray): Test features
        - Y_test (np.ndarray): Test targets
        - W_test (np.ndarray): Treatment indicators
        
        Returns:
        - dict: Evaluation metrics for the base model
        """
        self.evaluation_test_dict = {}
        
        # Augment features with treatment indicator
        X_aug = np.column_stack((X_test, W_test))
        
        # Evaluate model
        model_eval = self.models['s'].evaluate(X_aug, Y_test)
        self.evaluation_test_dict['s'] = model_eval
        
        return self.evaluation_test_dict

    def predict_cate(self, X, W=None):
        """
        Predict CATE by contrasting predictions with W=1 and W=0.
        
        Parameters:
        - X (np.ndarray): Test features
        - W (np.ndarray, optional): Not used in S-Learner
        
        Returns:
        - np.ndarray: Predicted CATE values
        """
        # Create counterfactual inputs
        X0 = np.column_stack((X, np.zeros(len(X))))
        X1 = np.column_stack((X, np.ones(len(X))))
        
        # Predict using specified metric
        mu0 = self.models['s'].predict_metric(X0, metric=self.metric, max_time=self.max_time)
        mu1 = self.models['s'].predict_metric(X1, metric=self.metric, max_time=self.max_time)
        return mu1 - mu0


class MatchingLearnerSurvival(BaseMetaLearnerSurvival):
    """
    Matching-Learner for survival data: Uses nearest neighbor matching to estimate treatment effects.
    """
    def __init__(self, base_model_name='DeepSurv', base_model_params=None, base_model_grid=None,
                 metric='mean', max_time=np.inf, num_matches=5, distance_metric='euclidean'):
        """
        Initialize the matching learner.
        
        Parameters:
        - base_model_name (str): Type of survival model
        - base_model_params (dict): Parameters for the base model
        - base_model_grid (dict): Grid search parameters for model selection
        - metric (str): Survival metric to use ('median' or 'mean')
        - max_time (float): Maximum time for restricted mean survival time
        - num_matches (int): Number of nearest neighbors to match
        - distance_metric (str): Distance metric for matching
        """
        super().__init__(base_model_name, base_model_params, base_model_grid, metric, max_time)
        self.num_matches = num_matches
        self.distance_metric = distance_metric
        
    def fit(self, X_train, W_train, Y_train):
        """
        Fit the model for matching-based estimation.
        
        Parameters:
        - X_train (np.ndarray): Training features
        - W_train (np.ndarray): Treatment indicators (0/1)
        - Y_train (np.ndarray): Training targets (time, event)
        """
        # Store training data for matching
        self.X_train = X_train
        self.W_train = W_train
        self.Y_train = Y_train
        
        # Train a survival model for predictions
        X_aug = np.column_stack((X_train, W_train))
        model = SurvivalModelBase(
            model_type=self.base_model_name,
            hyperparams=self.base_model_params,
            hyperparams_grid=self.base_model_grid
        )
        model.fit(X_aug, Y_train)
        self.models['model'] = model
    
    def evaluate_test(self, X_test, Y_test, W_test):
        """
        Evaluate base model on test data.
        
        Parameters:
        - X_test (np.ndarray): Test features
        - Y_test (np.ndarray): Test targets
        - W_test (np.ndarray): Treatment indicators
        
        Returns:
        - dict: Evaluation metrics for the base model
        """
        self.evaluation_test_dict = {}
        
        # Augment features with treatment indicator
        X_aug = np.column_stack((X_test, W_test))
        
        # Evaluate model
        model_eval = self.models['model'].evaluate(X_aug, Y_test)
        self.evaluation_test_dict['model'] = model_eval
        
        return self.evaluation_test_dict

    def predict_cate(self, X, W):
        """
        Predict CATE using nearest-neighbor matching.
        
        Parameters:
        - X (np.ndarray): Test features
        - W (np.ndarray): Treatment indicators for test data
        
        Returns:
        - np.ndarray: Predicted CATE values
        """
        # Use the true treatment for predictions, then get counterfactuals via matching
        X_aug = np.column_stack((X, W))
        
        # Get the predicted outcomes for the actual treatment
        true_outcomes = self.models['model'].predict_metric(
            X_aug, metric=self.metric, max_time=self.max_time
        )
        
        # Get matched neighbors with opposite treatment
        opposite_outcomes = self._get_opposite_treatment_outcomes(X, W)
        
        # CATE is (Y_1 - Y_0) * (2*W - 1) to account for treatment direction
        cate = (true_outcomes - opposite_outcomes) * (2*W - 1)
        
        return cate
    
    def _get_opposite_treatment_outcomes(self, X, W):
        """
        Get outcomes for opposite treatment using nearest neighbor matching.
        
        Parameters:
        - X (np.ndarray): Test features
        - W (np.ndarray): Treatment indicators
        
        Returns:
        - np.ndarray: Predicted outcomes for opposite treatment
        """
        # Calculate distances between test points and training data
        distances = cdist(X, self.X_train, metric=self.distance_metric)
        
        opposite_outcomes = np.zeros(X.shape[0])
        
        for i in range(X.shape[0]):
            # Determine which group to match with (opposite of actual treatment)
            opposite_treatment = 1 - W[i]
            match_indices = np.where(self.W_train == opposite_treatment)[0]
            
            # Get distances to opposite treatment group
            match_distances = distances[i, match_indices]
            
            # Find nearest neighbors
            neighbors = match_indices[np.argsort(match_distances)[:self.num_matches]]
            
            # Create features with opposite treatment
            X_aug_matches = np.column_stack((
                self.X_train[neighbors], 
                np.full(min(self.num_matches, len(neighbors)), opposite_treatment)
            ))
            
            # Get predicted outcomes
            match_outcomes = self.models['model'].predict_metric(
                X_aug_matches, metric=self.metric, max_time=self.max_time
            )
            
            # Average the outcomes
            opposite_outcomes[i] = np.mean(match_outcomes)
                
        return opposite_outcomes