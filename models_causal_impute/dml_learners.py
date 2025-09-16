from abc import ABC, abstractmethod
from sklearn.metrics import mean_squared_error
import numpy as np
from econml.dml import DML, CausalForestDML
from econml.sklearn_extensions.linear_model import StatsModelsLinearRegression
from econml.inference import BootstrapInference
import os
import pickle
from models_utils.checkpoint_utils import ensure_dir

class BaseDirectLearner(ABC):
    """
    Abstract base class for double machine learning causal effect estimators like DoubleML and Causal Forest.
    """
    def __init__(self, num_bootstrap_samples=100):
        self.model = None
        self.num_bootstrap_samples = num_bootstrap_samples

    @abstractmethod
    def fit(self, X_train, W_train, Y_train):
        pass

    @abstractmethod
    def predict_cate(self, X):
        pass

    def evaluate(self, X, cate_true, W=None):
        """
        Evaluate predicted CATE against ground-truth CATE.
        """
        cate_pred = self.predict_cate(X)
        mse = mean_squared_error(cate_true, cate_pred)
        # ate_pred = np.mean(cate_pred)
        ate_pred = self.model.ate_inference(X)
        return mse, cate_pred, ate_pred
    
    def save_model(self, filepath):
        """Save the DML learner model."""
        ensure_dir(os.path.dirname(filepath))
        model_data = {
            'model_type': self.__class__.__name__,
            'model': self.model,
            'num_bootstrap_samples': self.num_bootstrap_samples
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """Load a saved DML learner model."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.num_bootstrap_samples = model_data['num_bootstrap_samples']
        print(f"Model loaded from {filepath}")
        return self

    @classmethod
    def load(cls, filepath):
        """Class method to load a model."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        instance = cls()
        instance.model = model_data['model']
        instance.num_bootstrap_samples = model_data['num_bootstrap_samples']
        return instance
    

class DoubleML(BaseDirectLearner):
    """
    Double Machine Learning (Partially Linear) using EconML DML.
    """
    def __init__(self):
        super().__init__()
        self.model = DML(
            model_final=StatsModelsLinearRegression(fit_intercept=False),
            model_y='auto',
            model_t='auto',
            discrete_treatment=True,
            random_state=42
        )

    def fit(self, X_train, W_train, Y_train):
        bootstap = BootstrapInference(n_bootstrap_samples=self.num_bootstrap_samples, n_jobs=1)
        self.model.fit(Y_train, W_train, X=X_train, inference=bootstap)

    def predict_cate(self, X):
        return self.model.effect(X)
    

class CausalForest(BaseDirectLearner):
    """
    Causal Forest using EconML's CausalForestDML.
    """
    def __init__(self):
        super().__init__()
        self.model = CausalForestDML(
            model_y='auto',
            model_t='auto',
            discrete_treatment=True,
            random_state=42
        )

    def fit(self, X_train, W_train, Y_train):
        bootstap = BootstrapInference(n_bootstrap_samples=self.num_bootstrap_samples, n_jobs=1)
        self.model.fit(Y_train, W_train, X=X_train, inference=bootstap)

    def predict_cate(self, X):
        return self.model.effect(X)
