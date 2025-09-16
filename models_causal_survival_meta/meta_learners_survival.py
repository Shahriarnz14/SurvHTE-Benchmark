from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
from scipy.spatial.distance import cdist
from .survival_base import SurvivalModelBase
import os
import pickle
import torch
import importlib
from models_utils.checkpoint_utils import ensure_dir


def _class_path(obj):
    """Return 'module.ClassName' for an object's class."""
    return obj.__class__.__module__ + "." + obj.__class__.__name__


def _import_from_path(path):
    """Import class from 'module.ClassName' string."""
    module_name, class_name = path.rsplit(".", 1)
    return getattr(importlib.import_module(module_name), class_name)


class BaseMetaLearnerSurvival(ABC):
    """
    Abstract base class for causal meta-learners with survival outcomes.
    """

    def __init__(self, base_model_name='DeepSurv', base_model_params=None,
                 base_model_grid=None, metric='mean', max_time=np.inf):
        """
        Initialize the survival meta-learner.
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
        pass

    @abstractmethod
    def evaluate_test(self, X_test, Y_test, W_test):
        pass

    @abstractmethod
    def predict_cate(self, X):
        pass

    def evaluate(self, X, cate_true, W=None):
        """Evaluate CATE predictions using mean squared error."""
        cate_pred = self.predict_cate(X, W)
        ate_pred = np.mean(cate_pred)
        mse = mean_squared_error(cate_true, cate_pred)
        return mse, cate_pred, ate_pred

    # ---------------------- FIXED SAVE / LOAD ---------------------- #
    def save_model(self, filepath):
        """Save the survival meta-learner model safely (no deep wrappers pickled)."""
        ensure_dir(os.path.dirname(filepath))

        model_data = {
            'model_type': self.__class__.__name__,
            'base_model_name': self.base_model_name,
            'base_model_params': self.base_model_params,
            'base_model_grid': self.base_model_grid,
            'metric': self.metric,
            'max_time': self.max_time,
            'models': {},
            'evaluation_test_dict': self.evaluation_test_dict
        }

        for key, model_wrapper in self.models.items():
            model_type = getattr(model_wrapper, "model_type", None)

            if model_type in ['DeepSurv', 'DeepHit']:
                # Store only a lightweight spec; don't pickle the wrapper itself.
                wrapper_spec = {
                    'model_type': model_type,
                    'wrapper_class': _class_path(model_wrapper),
                    # These two are optional but recommended to add to your wrapper __init__
                    'init_args': getattr(model_wrapper, 'init_args', ()),
                    'init_kwargs': getattr(model_wrapper, 'init_kwargs', {}),
                    # If available, capture model class & hparams for reconstruction
                    'model_class': (_class_path(model_wrapper.model)
                                    if getattr(model_wrapper, 'model', None) is not None else None),
                    'model_hparams': getattr(getattr(model_wrapper, 'model', None), 'hparams', None),
                }
                model_data['models'][key] = wrapper_spec

            else:
                # Non-deep wrappers are usually pickleable; if not, convert to a similar spec.
                try:
                    pickle.dumps(model_wrapper, protocol=pickle.HIGHEST_PROTOCOL)
                    model_data['models'][key] = model_wrapper
                except Exception:
                    model_data['models'][key] = {
                        'model_type': model_type,
                        'wrapper_class': _class_path(model_wrapper),
                        'init_args': getattr(model_wrapper, 'init_args', ()),
                        'init_kwargs': getattr(model_wrapper, 'init_kwargs', {}),
                        '_non_pickleable': True
                    }

        # Save the lightweight pickle
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Save deep model state_dicts separately
        for key, model_wrapper in self.models.items():
            model_type = getattr(model_wrapper, "model_type", None)
            if model_type in ['DeepSurv', 'DeepHit'] and getattr(model_wrapper, 'model', None):
                if hasattr(model_wrapper.model, 'net'):
                    torch_path = filepath.replace('.pkl', f'_{key}_state.pt')
                    torch.save(model_wrapper.model.net.state_dict(), torch_path)

        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """Load a saved survival meta-learner model."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.base_model_name = model_data['base_model_name']
        self.base_model_params = model_data['base_model_params']
        self.base_model_grid = model_data['base_model_grid']
        self.metric = model_data['metric']
        self.max_time = model_data['max_time']
        self.evaluation_test_dict = model_data.get('evaluation_test_dict', {})

        # Recreate models
        self.models = {}
        for key, stored in model_data['models'].items():
            if isinstance(stored, dict) and stored.get('model_type') in ['DeepSurv', 'DeepHit']:
                # Rebuild wrapper
                WrapperCls = _import_from_path(stored['wrapper_class'])
                init_args = tuple(stored.get('init_args', ()))
                init_kwargs = dict(stored.get('init_kwargs', {}))
                wrapper = WrapperCls(*init_args, **init_kwargs)

                # If the wrapper already creates .model with .net, load weights
                torch_path = filepath.replace('.pkl', f'_{key}_state.pt')
                if os.path.exists(torch_path):
                    try:
                        if getattr(wrapper, 'model', None) and getattr(wrapper.model, 'net', None):
                            state_dict = torch.load(torch_path, map_location='cpu')
                            wrapper.model.net.load_state_dict(state_dict)
                        # If wrapper has no model yet but we have a class/hparams, try to construct
                        elif stored.get('model_class'):
                            ModelCls = _import_from_path(stored['model_class'])
                            model_hparams = stored.get('model_hparams') or {}
                            wrapper.model = ModelCls(**model_hparams)
                            if getattr(wrapper.model, 'net', None):
                                state_dict = torch.load(torch_path, map_location='cpu')
                                wrapper.model.net.load_state_dict(state_dict)
                    except Exception as e:
                        # Best-effort: keep a usable wrapper even if weights can't be loaded
                        print(f"[load_model] Warning: failed to load state for '{key}': {e}")
                self.models[key] = wrapper

            elif isinstance(stored, dict) and stored.get('_non_pickleable'):
                # Rebuild previously non-pickleable classical wrapper
                WrapperCls = _import_from_path(stored['wrapper_class'])
                init_args = tuple(stored.get('init_args', ()))
                init_kwargs = dict(stored.get('init_kwargs', {}))
                self.models[key] = WrapperCls(*init_args, **init_kwargs)

            else:
                # Plain pickled classical model
                self.models[key] = stored

        print(f"Model loaded from {filepath}")
        return self
    # -------------------- END FIXED SAVE / LOAD -------------------- #


class TLearnerSurvival(BaseMetaLearnerSurvival):
    """
    T-Learner for survival data: Trains separate survival models for treated and control groups.
    """
    def fit(self, X_train, W_train, Y_train):
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

        model_treated.fit(X_train[W_train == 1], Y_train[W_train == 1])
        model_control.fit(X_train[W_train == 0], Y_train[W_train == 0])

        self.models['treated'] = model_treated
        self.models['control'] = model_control

    def evaluate_test(self, X_test, Y_test, W_test):
        self.evaluation_test_dict = {}
        # Guard against empty slices
        if np.any(W_test == 1):
            treated_eval = self.models['treated'].evaluate(X_test[W_test == 1], Y_test[W_test == 1])
            self.evaluation_test_dict['treated'] = treated_eval
        if np.any(W_test == 0):
            control_eval = self.models['control'].evaluate(X_test[W_test == 0], Y_test[W_test == 0])
            self.evaluation_test_dict['control'] = control_eval
        return self.evaluation_test_dict

    def predict_cate(self, X, W=None):
        mu1 = self.models['treated'].predict_metric(X, metric=self.metric, max_time=self.max_time)
        mu0 = self.models['control'].predict_metric(X, metric=self.metric, max_time=self.max_time)
        return mu1 - mu0


class SLearnerSurvival(BaseMetaLearnerSurvival):
    """
    S-Learner for survival data: Trains a single model using treatment as a feature.
    """
    def fit(self, X_train, W_train, Y_train):
        X_aug = np.column_stack((X_train, W_train))
        model = SurvivalModelBase(
            model_type=self.base_model_name,
            hyperparams=self.base_model_params,
            hyperparams_grid=self.base_model_grid
        )
        model.fit(X_aug, Y_train)
        self.models['s'] = model

    def evaluate_test(self, X_test, Y_test, W_test):
        self.evaluation_test_dict = {}
        X_aug = np.column_stack((X_test, W_test))
        model_eval = self.models['s'].evaluate(X_aug, Y_test)
        self.evaluation_test_dict['s'] = model_eval
        return self.evaluation_test_dict

    def predict_cate(self, X, W=None):
        X0 = np.column_stack((X, np.zeros(len(X))))
        X1 = np.column_stack((X, np.ones(len(X))))
        mu0 = self.models['s'].predict_metric(X0, metric=self.metric, max_time=self.max_time)
        mu1 = self.models['s'].predict_metric(X1, metric=self.metric, max_time=self.max_time)
        return mu1 - mu0


class MatchingLearnerSurvival(BaseMetaLearnerSurvival):
    """
    Matching-Learner for survival data: Uses nearest neighbor matching to estimate treatment effects.
    """
    def __init__(self, base_model_name='DeepSurv', base_model_params=None, base_model_grid=None,
                 metric='mean', max_time=np.inf, num_matches=5, distance_metric='euclidean'):
        super().__init__(base_model_name, base_model_params, base_model_grid, metric, max_time)
        self.num_matches = num_matches
        self.distance_metric = distance_metric

    def fit(self, X_train, W_train, Y_train):
        self.X_train = X_train
        self.W_train = W_train
        self.Y_train = Y_train

        X_aug = np.column_stack((X_train, W_train))
        model = SurvivalModelBase(
            model_type=self.base_model_name,
            hyperparams=self.base_model_params,
            hyperparams_grid=self.base_model_grid
        )
        model.fit(X_aug, Y_train)
        self.models['model'] = model

    def evaluate_test(self, X_test, Y_test, W_test):
        self.evaluation_test_dict = {}
        X_aug = np.column_stack((X_test, W_test))
        model_eval = self.models['model'].evaluate(X_aug, Y_test)
        self.evaluation_test_dict['model'] = model_eval
        return self.evaluation_test_dict

    def predict_cate(self, X, W):
        X_aug = np.column_stack((X, W))
        true_outcomes = self.models['model'].predict_metric(
            X_aug, metric=self.metric, max_time=self.max_time
        )
        opposite_outcomes = self._get_opposite_treatment_outcomes(X, W)
        cate = (true_outcomes - opposite_outcomes) * (2 * W - 1)
        return cate

    def _get_opposite_treatment_outcomes(self, X, W):
        distances = cdist(X, self.X_train, metric=self.distance_metric)
        opposite_outcomes = np.zeros(X.shape[0])

        for i in range(X.shape[0]):
            opposite_treatment = 1 - W[i]
            match_indices = np.where(self.W_train == opposite_treatment)[0]

            if match_indices.size == 0:
                # Fallback: if no opposite-treatment samples exist, use model prediction directly
                X_aug_fallback = np.column_stack((X[i:i+1], np.full(1, opposite_treatment)))
                opposite_outcomes[i] = self.models['model'].predict_metric(
                    X_aug_fallback, metric=self.metric, max_time=self.max_time
                )[0]
                continue

            match_distances = distances[i, match_indices]
            k = min(self.num_matches, match_indices.size)
            neighbors = match_indices[np.argsort(match_distances)[:k]]

            X_aug_matches = np.column_stack((
                self.X_train[neighbors],
                np.full(k, opposite_treatment)
            ))

            match_outcomes = self.models['model'].predict_metric(
                X_aug_matches, metric=self.metric, max_time=self.max_time
            )
            opposite_outcomes[i] = np.mean(match_outcomes)

        return opposite_outcomes
