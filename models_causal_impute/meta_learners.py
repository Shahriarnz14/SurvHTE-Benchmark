from abc import ABC, abstractmethod
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, mean_absolute_error, r2_score
import numpy as np
from econml.dr import DRLearner
from econml.metalearners import TLearner, SLearner, XLearner
from econml.inference import BootstrapInference
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from xgboost import XGBRegressor
from .regressor_base import RegressorBaseLearner

class BaseMetaLearner(ABC):
    """
    Abstract base class for causal meta-learners.
    
    Subclasses must implement:
    - fit(X_train, W_train, Y_train)
    - predict_cate(X)
    """
    def __init__(self, base_model_name='ridge', num_bootstrap_samples=100):
        """
        Initialize the meta-learner with a specified base model type.
        
        Parameters:
        - base_model_name (str): Model type to use (ridge, lasso, rf, gbr, xgb).
        """
        self.base_model_name = base_model_name
        self.model = None
        self.num_bootstrap_samples = num_bootstrap_samples

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
        - X (np.ndarray): Test features.
        - cate_true (np.ndarray): Ground-truth CATE values (from simulation).
        - W (np.ndarray): Treatment assignment (not used in this method).
        
        Returns:
        - mse (float): Mean squared error.
        - cate_pred (np.ndarray): Predicted CATE values.
        - ate_pred (object): Average Treatment Effect (ATE) prediction.
        """
        cate_pred = self.predict_cate(X)
        # ate_pred = np.mean(cate_pred)
        ate_pred = self.model.ate_inference(X)
        mse = mean_squared_error(cate_true, cate_pred)
        return mse, cate_pred, ate_pred
    
    
class T_Learner(BaseMetaLearner):
    """
    T-Learner: Trains separate models for treated and control groups.
    """
    def fit(self, X_train, W_train, Y_train):
        # define the model_final and model_regression based on the base_model_name
        if self.base_model_name == 'ridge':
            underlying_model = Ridge()
        elif self.base_model_name == 'lasso':
            underlying_model = Lasso()
        elif self.base_model_name == 'rf':
            underlying_model = RandomForestRegressor(random_state=42, n_jobs=-1)
        elif self.base_model_name == 'gbr':
            underlying_model = GradientBoostingRegressor(random_state=42)
        elif self.base_model_name == 'xgb':
            underlying_model = XGBRegressor(random_state=42, verbosity=0, n_jobs=-1)
        else:
            raise ValueError(f"Unsupported model name: {self.base_model_name}")
        
        bootstap = BootstrapInference(n_bootstrap_samples=self.num_bootstrap_samples, n_jobs=1)

        self.model = TLearner(
            models=underlying_model,
        )
        self.model.fit(Y_train, W_train, X=X_train, inference=bootstap)

    def predict_cate(self, X):
        return self.model.effect(X)

    def evaluate_test(self, X_test, Y_test, W_test):

        self.evaluation_test_dict = {}
        
        treated_eval = {'mae': mean_absolute_error(self.model.models[1].predict(X_test[W_test == 1]), Y_test[W_test == 1]),
                        'r2':  r2_score(self.model.models[1].predict(X_test[W_test == 1]), Y_test[W_test == 1])}
        self.evaluation_test_dict['treated'] = treated_eval

        control_eval = {'mae': mean_absolute_error(self.model.models[0].predict(X_test[W_test == 0]), Y_test[W_test == 0]),
                        'r2':  r2_score(self.model.models[0].predict(X_test[W_test == 0]), Y_test[W_test == 0])}
        self.evaluation_test_dict['control'] = control_eval

        return self.evaluation_test_dict

    
class S_Learner(BaseMetaLearner):
    """
    S-Learner: Trains a single model using treatment as a feature.
    """
    def fit(self, X_train, W_train, Y_train):
        # define the model_final and model_regression based on the base_model_name
        if self.base_model_name == 'ridge':
            underlying_model = Ridge()
        elif self.base_model_name == 'lasso':
            underlying_model = Lasso()
        elif self.base_model_name == 'rf':
            underlying_model = RandomForestRegressor(random_state=42, n_jobs=-1)
        elif self.base_model_name == 'gbr':
            underlying_model = GradientBoostingRegressor(random_state=42)
        elif self.base_model_name == 'xgb':
            underlying_model = XGBRegressor(random_state=42, verbosity=0, n_jobs=-1)
        else:
            raise ValueError(f"Unsupported model name: {self.base_model_name}")
        
        bootstap = BootstrapInference(n_bootstrap_samples=self.num_bootstrap_samples, n_jobs=1)

        self.model = SLearner(
            overall_model=underlying_model,
        )
        self.model.fit(Y_train, W_train, X=X_train, inference=bootstap)

    def predict_cate(self, X):
        return self.model.effect(X)

    def evaluate_test(self, X_test, Y_test, W_test):

        self.evaluation_test_dict = {}
        
        # unsure exactly how to augment the treatment variable (selected based on training performance)
        X_aug = np.column_stack((X_test, 1-W_test, W_test)) 
        self.evaluation_test_dict['s'] = {
            'mae': mean_absolute_error(self.model.overall_model.predict(X_aug), Y_test),
            'r2': r2_score(self.model.overall_model.predict(X_aug), Y_test)
        }

        return self.evaluation_test_dict


class X_Learner(BaseMetaLearner):
    """
    X-Learner: Uses imputed treatment effects from T-Learner to train tau models,
    then combines with a propensity model.
    """
    def fit(self, X_train, W_train, Y_train):
        # define the model_final and model_regression based on the base_model_name
        if self.base_model_name == 'ridge':
            underlying_model = Ridge()
        elif self.base_model_name == 'lasso':
            underlying_model = Lasso()
        elif self.base_model_name == 'rf':
            underlying_model = RandomForestRegressor(random_state=42, n_jobs=-1)
        elif self.base_model_name == 'gbr':
            underlying_model = GradientBoostingRegressor(random_state=42)
        elif self.base_model_name == 'xgb':
            underlying_model = XGBRegressor(random_state=42, verbosity=0, n_jobs=-1)
        else:
            raise ValueError(f"Unsupported model name: {self.base_model_name}")
        
        bootstap = BootstrapInference(n_bootstrap_samples=self.num_bootstrap_samples, n_jobs=1)

        self.model = XLearner(
            models=underlying_model,
            # propensity_model=LogisticRegression(),
            cate_models=underlying_model,
        )
        self.model.fit(Y_train, W_train, X=X_train, inference=bootstap)

    def predict_cate(self, X):
        return self.model.effect(X)

    def evaluate_test(self, X_test, Y_test, W_test):

        self.evaluation_test_dict = {}
        
        treated_eval = {'mae': mean_absolute_error(self.model.models[1].predict(X_test[W_test == 1]), Y_test[W_test == 1]),
                        'r2':  r2_score(self.model.models[1].predict(X_test[W_test == 1]), Y_test[W_test == 1])}
        self.evaluation_test_dict['mu1'] = treated_eval

        control_eval = {'mae': mean_absolute_error(self.model.models[0].predict(X_test[W_test == 0]), Y_test[W_test == 0]),
                        'r2':  r2_score(self.model.models[0].predict(X_test[W_test == 0]), Y_test[W_test == 0])}
        self.evaluation_test_dict['mu0'] = control_eval

        # self.evaluation_test_dict['propensity'] = None # EconML does not expose the trained propensity model

        # self.evaluation_test_dict['tau0'] = None # EconML does not expose the trained tau0 model
        # self.evaluation_test_dict['tau1'] = None # EconML does not expose the trained tau1 model

        return self.evaluation_test_dict


class DR_Learner(BaseMetaLearner):
    """
    DR-Learner: Doubly robust learner using a single model_final.
    """
    def fit(self, X_train, W_train, Y_train):
        # base_model_final = RegressorBaseLearner(model_name=self.base_model_name)
        # model_final = base_model_final.grid_search  # assume this returns a scikit-learn-style estimator

        # base_model_regression = RegressorBaseLearner(model_name=self.base_model_name)
        # model_regression = base_model_regression.grid_search  # assume this returns a scikit-learn-style estimator

        # define the model_final and model_regression based on the base_model_name
        if self.base_model_name == 'ridge':
            model_final = Ridge()
            model_regression = Ridge()
        elif self.base_model_name == 'lasso':
            model_final = Lasso()
            model_regression = Lasso()
        elif self.base_model_name == 'rf':
            model_final = RandomForestRegressor(random_state=42)
            model_regression = RandomForestRegressor(random_state=42)
        elif self.base_model_name == 'gbr':
            model_final = GradientBoostingRegressor(random_state=42)
            model_regression = GradientBoostingRegressor(random_state=42)
        elif self.base_model_name == 'xgb':
            model_final = XGBRegressor(random_state=42, verbosity=0, n_jobs=-1)
            model_regression = XGBRegressor(random_state=42, verbosity=0, n_jobs=-1)
        else:
            raise ValueError(f"Unsupported model name: {self.base_model_name}")

        self.model = DRLearner(
            model_regression=model_regression,
            # model_propensity=LogisticRegression(),
            model_final=model_final,
            random_state=42,
        )
        bootstap = BootstrapInference(n_bootstrap_samples=self.num_bootstrap_samples, n_jobs=1)
        self.model.fit(Y_train, W_train, X=X_train, inference=bootstap)

    def predict_cate(self, X):
        return self.model.effect(X)

    def evaluate_test(self, X_test, Y_test, W_test):
        # DRLearner does not expose base models the way T-/S-/X-Learners do
        # We can only evaluate some parts of the model
        model_regression_eval = { 
                                    'mae': np.mean([mean_absolute_error(fold_model.predict(np.column_stack((X_test, W_test))), Y_test) 
                                                    for fold_model in self.model.models_regression[0]]),
                                    'r2': np.mean([r2_score(fold_model.predict(np.column_stack((X_test, W_test))), Y_test)
                                                   for fold_model in self.model.models_regression[0]])
                                }

        model_propensity_eval = { 
                                    'auc': np.mean([roc_auc_score(W_test, fold_model.predict_proba(X_test)[:, 1])
                                                    for fold_model in self.model.models_propensity[0]]),
                                    'f1': np.mean([f1_score(W_test, (fold_model.predict_proba(X_test)[:, 1] >= 0.5).astype(int))
                                                   for fold_model in self.model.models_propensity[0]])
                                }
        
        # Store evaluations in a dictionary
        self.evaluation_test_dict = {'model_regression': model_regression_eval,
                                     'propensity': model_propensity_eval}
        
        return self.evaluation_test_dict


