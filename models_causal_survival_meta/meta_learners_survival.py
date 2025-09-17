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
        """Simplified saver:
        - RSF/etc: pickle the wrapper
        - DeepSurv/DeepHit (PyCox): torch-save the net + small sidecar files
        """
        ensure_dir(os.path.dirname(filepath))

        model_data = {
            'model_type': self.__class__.__name__,
            'base_model_name': self.base_model_name,
            'base_model_params': self.base_model_params,
            'base_model_grid': self.base_model_grid,
            'metric': self.metric,
            'max_time': self.max_time,
            'models': {},  # metadata map; heavy tensors go to sidecar files
            'evaluation_test_dict': self.evaluation_test_dict,
        }

        for key, wrap in self.models.items():
            mtype = getattr(wrap, "model_type", None)

            # ---------- PyCox models ----------
            if mtype in ('DeepSurv', 'DeepHit'):
                if not getattr(wrap, 'model', None) or not getattr(wrap.model, 'net', None):
                    raise RuntimeError(f"Model '{key}' missing .model.net; cannot save.")

                # save full nn.Module (architecture + weights)
                net_path = filepath.replace('.pkl', f'_{key}_net.pth')
                torch.save(wrap.model.net, net_path)

                entry = {
                    'kind': 'pycox',
                    'model_type': mtype,
                    'pycox_model_class': wrap.model.__class__.__module__ + '.' + wrap.model.__class__.__name__,
                    'net_path': os.path.basename(net_path),
                }

                # DeepSurv (CoxPH): persist hazards
                if mtype == 'DeepSurv':
                    bh = getattr(wrap.model, 'baseline_hazards_', None)
                    if bh is not None:
                        bh_path = filepath.replace('.pkl', f'_{key}_basehaz.pkl')
                        with open(bh_path, 'wb') as bf:
                            pickle.dump(bh, bf, protocol=pickle.HIGHEST_PROTOCOL)
                        entry['basehaz_path'] = os.path.basename(bh_path)

                    bch = getattr(wrap.model, 'baseline_cumulative_hazards_', None)
                    if bch is not None:
                        bch_path = filepath.replace('.pkl', f'_{key}_basecumhaz.pkl')
                        with open(bch_path, 'wb') as bf:
                            pickle.dump(bch, bf, protocol=pickle.HIGHEST_PROTOCOL)
                        entry['basecumhaz_path'] = os.path.basename(bch_path)

                    di = getattr(wrap.model, 'duration_index', None)
                    if di is not None:
                        di_path = filepath.replace('.pkl', f'_{key}_duration_index.pkl')
                        with open(di_path, 'wb') as df:
                            pickle.dump(di, df, protocol=pickle.HIGHEST_PROTOCOL)
                        entry['duration_index_path'] = os.path.basename(di_path)

                # DeepHit: persist duration_index
                if mtype == 'DeepHit':
                    di = getattr(wrap.model, 'duration_index', None)
                    if di is not None:
                        di_path = filepath.replace('.pkl', f'_{key}_duration_index.pkl')
                        with open(di_path, 'wb') as df:
                            pickle.dump(di, df, protocol=pickle.HIGHEST_PROTOCOL)
                        entry['duration_index_path'] = os.path.basename(di_path)

                model_data['models'][key] = entry

            # ---------- Classical (e.g., RandomSurvivalForest) ----------
            else:
                model_data['models'][key] = wrap  # original simple path

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """Robust loader: rebuild models and repair expected keys from sidecars if needed."""
        import importlib
        import pandas as pd  # for cumsum fallback; ok if unused

        def _import(path):
            mod, cls = path.rsplit('.', 1)
            return getattr(importlib.import_module(mod), cls)

        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        self.base_model_name = data['base_model_name']
        self.base_model_params = data['base_model_params']
        self.base_model_grid = data['base_model_grid']
        self.metric = data['metric']
        self.max_time = data['max_time']
        self.evaluation_test_dict = data.get('evaluation_test_dict', {})

        base_dir = os.path.dirname(filepath)
        self.models = {}

        # ---------- First pass: rebuild anything present in the pickle ----------
        for key, stored in data['models'].items():
            # PyCox path (DeepSurv / DeepHit)
            if isinstance(stored, dict) and stored.get('kind') == 'pycox':
                mtype = stored['model_type']
                net_path = os.path.join(base_dir, stored['net_path'])
                net = torch.load(net_path, map_location='cpu')

                from .survival_base import SurvivalModelBase  # local import to avoid cycles
                wrap = SurvivalModelBase(
                    model_type=mtype,
                    hyperparams=self.base_model_params,
                    hyperparams_grid=self.base_model_grid
                )

                PyCoxModelCls = _import(stored['pycox_model_class'])
                try:
                    wrap.model = PyCoxModelCls(net=net)
                except TypeError:
                    wrap.model = PyCoxModelCls(net)

                # Optional eval mode
                try:
                    wrap.model.net.eval()
                except Exception:
                    pass

                # Restore DeepSurv hazards + duration index
                if mtype == 'DeepSurv':
                    bh_path = stored.get('basehaz_path')
                    bch_path = stored.get('basecumhaz_path')
                    di_path = stored.get('duration_index_path')

                    if di_path:
                        with open(os.path.join(base_dir, di_path), 'rb') as df:
                            wrap.model.duration_index = pickle.load(df)
                    if bh_path:
                        with open(os.path.join(base_dir, bh_path), 'rb') as bf:
                            wrap.model.baseline_hazards_ = pickle.load(bf)
                    if bch_path:
                        with open(os.path.join(base_dir, bch_path), 'rb') as bf:
                            wrap.model.baseline_cumulative_hazards_ = pickle.load(bf)
                    # Fallback: derive cumulative from hazards if missing
                    if getattr(wrap.model, 'baseline_hazards_', None) is not None and \
                    getattr(wrap.model, 'baseline_cumulative_hazards_', None) is None:
                        try:
                            wrap.model.baseline_cumulative_hazards_ = wrap.model.baseline_hazards_.cumsum(axis=0)
                        except Exception:
                            pass

                # Restore DeepHit duration index
                if mtype == 'DeepHit':
                    di_path = stored.get('duration_index_path')
                    if di_path:
                        with open(os.path.join(base_dir, di_path), 'rb') as df:
                            wrap.model.duration_index = pickle.load(df)

                self.models[key] = wrap
                continue

            # Classical (e.g., RSF) -> already pickleable
            self.models[key] = stored

        # ---------- Second pass: repair expected keys from sidecars if missing ----------
        learner_kind = data.get('model_type', '')
        base_model = self.base_model_name

        def build_from_sidecars(key_needed, pycox_type):
            """Create a SurvivalModelBase for `key_needed` using saved sidecars."""
            net_path = filepath.replace('.pkl', f'_{key_needed}_net.pth')
            if not os.path.exists(net_path):
                return False  # nothing to rebuild
            net = torch.load(net_path, map_location='cpu')

            from .survival_base import SurvivalModelBase
            wrap = SurvivalModelBase(
                model_type=pycox_type,
                hyperparams=self.base_model_params,
                hyperparams_grid=self.base_model_grid
            )

            # Choose the PyCox class from the type
            if pycox_type == 'DeepSurv':
                # pycox.models.cox.CoxPH is the class used by DeepSurv wrappers
                try:
                    PyCoxModelCls = _import('pycox.models.cox.CoxPH')
                except Exception:
                    PyCoxModelCls = _import('pycox.models.CoxPH')
            elif pycox_type == 'DeepHit':
                # DeepHit class path (common in pycox)
                try:
                    PyCoxModelCls = _import('pycox.models.deephit.DeepHit')
                except Exception:
                    PyCoxModelCls = _import('pycox.models.DeepHit')

            try:
                wrap.model = PyCoxModelCls(net=net)
            except TypeError:
                wrap.model = PyCoxModelCls(net)

            # Restore extras
            if pycox_type == 'DeepSurv':
                bhp = filepath.replace('.pkl', f'_{key_needed}_basehaz.pkl')
                bchp = filepath.replace('.pkl', f'_{key_needed}_basecumhaz.pkl')
                dip = filepath.replace('.pkl', f'_{key_needed}_duration_index.pkl')
                if os.path.exists(dip):
                    with open(dip, 'rb') as fdi:
                        wrap.model.duration_index = pickle.load(fdi)
                if os.path.exists(bhp):
                    with open(bhp, 'rb') as fbh:
                        wrap.model.baseline_hazards_ = pickle.load(fbh)
                if os.path.exists(bchp):
                    with open(bchp, 'rb') as fbch:
                        wrap.model.baseline_cumulative_hazards_ = pickle.load(fbch)
                # Fallback derive cumulative
                if getattr(wrap.model, 'baseline_hazards_', None) is not None and \
                getattr(wrap.model, 'baseline_cumulative_hazards_', None) is None:
                    try:
                        wrap.model.baseline_cumulative_hazards_ = wrap.model.baseline_hazards_.cumsum(axis=0)
                    except Exception:
                        pass
            elif pycox_type == 'DeepHit':
                dip = filepath.replace('.pkl', f'_{key_needed}_duration_index.pkl')
                if os.path.exists(dip):
                    with open(dip, 'rb') as fdi:
                        wrap.model.duration_index = pickle.load(fdi)

            self.models[key_needed] = wrap
            return True

        if learner_kind == 'TLearnerSurvival':
            # Ensure both keys exist
            for need in ('treated', 'control'):
                if need not in self.models:
                    if base_model in ('DeepSurv', 'DeepHit'):
                        ok = build_from_sidecars(need, base_model)
                        if not ok:
                            print(f"[load_model] Warning: missing '{need}' model and no sidecars found.")
                    else:
                        print(f"[load_model] Warning: missing '{need}' model for RSF; was it saved?")
        elif learner_kind == 'SLearnerSurvival':
            if 's' not in self.models:
                if base_model in ('DeepSurv', 'DeepHit'):
                    build_from_sidecars('s', base_model)
                else:
                    print("[load_model] Warning: missing 's' model for RSF; was it saved?")
        elif learner_kind == 'MatchingLearnerSurvival':
            if 'model' not in self.models:
                if base_model in ('DeepSurv', 'DeepHit'):
                    build_from_sidecars('model', base_model)
                else:
                    print("[load_model] Warning: missing 'model' for RSF; was it saved?")

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
