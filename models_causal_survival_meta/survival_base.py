from sklearn.model_selection import train_test_split, GridSearchCV
import numpy as np
import pandas as pd
import torch
from lifelines import KaplanMeierFitter, CoxPHFitter
from sksurv.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxPHSurvivalAnalysis
from pycox.models import DeepHitSingle, CoxPH
import torchtuples as tt
import inspect

from .utils_survival import get_concordance_score, get_integrated_brier_score, get_cumulative_dynamic_auc

class SurvivalModelBase:
    """
    Base class for survival models with consistent interface for meta-learners.
    
    Supported models:
    - 'CoxPH': Cox Proportional Hazards model using scikit-survival
    - 'RandomSurvivalForest': Random Survival Forest using scikit-survival
    - 'DeepSurv': Deep Cox Proportional Hazards using pycox
    - 'DeepHit': DeepHit model using pycox
    """
    def __init__(self, model_type='DeepSurv', hyperparams=None, hyperparams_grid=None, 
                 extrapolate_median=False, random_state=42, cv=5):
        """
        Initialize the survival model.
        
        Parameters:
        - model_type (str): Type of survival model to use
        - hyperparams (dict): Default model hyperparameters
        - hyperparams_grid (dict): Grid of hyperparameters for model selection
        - extrapolate_median (bool): Whether to extrapolate median survival time
        - random_state (int): Random seed for reproducibility
        - cv (int): Number of cross-validation folds
        """
        self.model_type = model_type
        self.hyperparams = hyperparams if hyperparams else {}
        self.hyperparams_grid = hyperparams_grid if hyperparams_grid else {}
        self.random_seed = random_state
        self.extrapolate_median = extrapolate_median
        self.cv = cv
        self.model = self._initialize_model()
        self.time_grid = None
        self.survival_train = None
        self.best_params = None

    def _initialize_model(self):
        """Initialize the specific survival model instance."""
        if self.model_type == "CoxPH":
            return CoxPHSurvivalAnalysis()
        elif self.model_type == "RandomSurvivalForest":
            filtered_hyperparams = {k: v for k, v in self.hyperparams.items() 
                                  if k in inspect.signature(RandomSurvivalForest).parameters}
            return RandomSurvivalForest(**filtered_hyperparams)
        elif self.model_type == "DeepSurv":
            return None  # Initialized in `fit` for PyCox
        elif self.model_type == "DeepHit":
            return None  # Initialized in `fit` for PyCox
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def fit(self, X_train, Y_train, covariate_names=None):
        """
        Fit the survival model on training data with model selection.
        
        Parameters:
        - X_train (np.ndarray): Training features
        - Y_train (np.ndarray): Training targets (time, event)
        - covariate_names (list): Optional list of feature names
        """
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        torch.cuda.manual_seed(self.random_seed)

        try:
            # Split into train and validation sets for model selection
            X_train_fit, X_val, Y_train_fit, Y_val = train_test_split(
                X_train, Y_train, test_size=0.2, random_state=self.random_seed,
                stratify=Y_train[:, 1]
            )
        except ValueError:
            # If stratified split fails, we only have one event type (event/censored) in Y_train
            X_train_fit = X_train
            Y_train_fit = Y_train
            X_val = X_train
            Y_val = Y_train
            print("[Warning]: Stratified split for validation failed. Using full training data for fitting.")

        if len(Y_val) == 1:
            # If validation set has only one sample, we use training data for validation
            X_val = X_train_fit
            Y_val = Y_train_fit
            print("[Warning]: Validation set had only one sample. Using training data for validation.")

        # Checking that validation set has at least one event
        if not np.any(Y_val[:, 1] == 1):
            # Find index of a class-1 sample in the training set
            idx_in_train = np.where(Y_train_fit[:, 1] == 1)[0]

            if idx_in_train.size > 0:
                # Take the first such index (or random if preferred)
                idx_to_duplicate = idx_in_train[0]

                # Get the example
                x_dup = X_train_fit[idx_to_duplicate]
                y_dup = Y_train_fit[idx_to_duplicate]

                # Duplicate it into validation set
                X_val = np.concatenate([X_val, x_dup[None, :]], axis=0)
                Y_val = np.concatenate([Y_val, y_dup[None, :]], axis=0)
                # Shuffle the validation set
                perm = np.random.permutation(X_val.shape[0])
                X_val = X_val[perm]
                Y_val = Y_val[perm]
                print("[Warning]: All validation was censored. Duplicating single event example into validation set.")
            else:
                print("[Warning]: All train is censored. No event example available to duplicate into validation set.")

        self.survival_train = self._prepare_sksurv_data(Y_train)

        if self.model_type == "CoxPH":
            self._fit_coxph(X_train_fit, Y_train_fit, X_val, Y_val)
        elif self.model_type == "RandomSurvivalForest":
            self._fit_rsf(X_train_fit, Y_train_fit, X_val, Y_val)
        elif self.model_type == "DeepSurv":
            self._fit_deepsurv(X_train_fit, Y_train_fit, X_val, Y_val)
        elif self.model_type == "DeepHit":
            self._fit_deephit(X_train_fit, Y_train_fit, X_val, Y_val)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def _prepare_cox_data(self, X, Y, covariate_names=None):
        """Prepare data in the format required by CoxPH."""
        if covariate_names is None:
            columns = [f"X{i}" for i in range(X.shape[1])]
        else:
            columns = covariate_names
        df = pd.DataFrame(X, columns=columns)
        df["time"] = Y[:, 0]
        df["event"] = Y[:, 1]
        return df
    
    def _prepare_sksurv_data(self, Y):
        """Prepare data in the format required by scikit-survival."""
        return np.array([(bool(event), time) for time, event in Y], 
                       dtype=[("event", "bool"), ("time", "float64")])
    
    def _fit_coxph(self, X_train, Y_train, X_val, Y_val):
        """Fit Cox Proportional Hazards model using scikit-survival with model selection."""
        y_train_struct = self._prepare_sksurv_data(Y_train)
        y_val_struct = self._prepare_sksurv_data(Y_val)
        
        if self.hyperparams_grid:
            # Get parameter grid
            param_grid = self.hyperparams_grid.get('CoxPH', {'alpha': [0.001, 0.01, 0.1, 1.0]})
            
            # Manual parameter search using validation set
            best_score = -float('inf')
            best_params = None
            best_model = None
            
            # Generate all parameter combinations
            param_keys = list(param_grid.keys())
            param_values = list(param_grid.values())
            
            from itertools import product
            for params in product(*param_values):
                param_dict = {k: v for k, v in zip(param_keys, params)}
                
                # Create and fit model with current parameters
                model = CoxPHSurvivalAnalysis(**param_dict)
                model.fit(X_train, y_train_struct)
                
                # Evaluate on validation set
                # Use concordance index as the score
                score = model.score(X_val, y_val_struct)
                if score > best_score:
                    best_score = score
                    best_params = param_dict
                    best_model = model
            
            # Set the best model and parameters
            self.model = best_model
            self.best_params = best_params
        else:
            # No grid search, just fit with default parameters
            self.model = CoxPHSurvivalAnalysis()
            self.model.fit(X_train, y_train_struct)

    def _fit_rsf(self, X_train, Y_train, X_val, Y_val):
        """Fit Random Survival Forest model with validation set for model selection."""
        y_train_struct = self._prepare_sksurv_data(Y_train)
        y_val_struct = self._prepare_sksurv_data(Y_val)
        
        if self.hyperparams_grid:
            # Get parameter grid
            param_grid = self.hyperparams_grid.get('RandomSurvivalForest', {
                'n_estimators': [50, 100],
                'min_samples_split': [5, 10],
                'min_samples_leaf': [3, 5]
            })
            
            # Manual parameter search using validation set
            best_score = -float('inf')
            best_params = None
            best_model = None
            
            # Generate all parameter combinations
            param_keys = list(param_grid.keys())
            param_values = list(param_grid.values())
            
            from itertools import product
            for params in product(*param_values):
                param_dict = {k: v for k, v in zip(param_keys, params)}
                param_dict['random_state'] = self.random_seed
                param_dict['n_jobs'] = -1  # Use all cores
                
                # Create and fit model with current parameters
                model = RandomSurvivalForest(**param_dict)
                model.fit(X_train, y_train_struct)
                
                # Evaluate on validation set
                # Use concordance index as the score
                score = model.score(X_val, y_val_struct)
                if score > best_score:
                    best_score = score
                    best_params = param_dict
                    best_model = model
            
            # Set the best model and parameters
            self.model = best_model
            self.best_params = best_params
        else:
            # No grid search, just fit with default parameters
            self.model = RandomSurvivalForest(
                n_estimators=self.hyperparams.get("n_estimators", 100),
                min_samples_split=self.hyperparams.get("min_samples_split", 10),
                min_samples_leaf=self.hyperparams.get("min_samples_leaf", 5),
                n_jobs=-1,
                random_state=self.random_seed,
            )
            self.model.fit(X_train, y_train_struct)

    def _fit_deepsurv(self, X_train, Y_train, X_val, Y_val):
        """Fit DeepSurv (Deep Cox) model with model selection."""
        # Model selection for DeepSurv
        best_model = None
        best_concordance = -float('inf')  # Higher concordance is better
        
        # Default grid if none specified
        default_grid = {
            'num_nodes': [64, 128],
            'dropout': [0.1, 0.2],
            'lr': [0.01, 0.001],
            'epochs': [100, 500]
        }
        
        param_grid = self.hyperparams_grid.get('DeepSurv', default_grid)
        
        # Try different hyperparameter combinations
        for num_nodes in param_grid.get('num_nodes', [128]):
            for dropout in param_grid.get('dropout', [0.1]):
                for lr in param_grid.get('lr', [0.01]):
                    for epochs in param_grid.get('epochs', [200]):
                        # Create the network
                        net = tt.practical.MLPVanilla(
                            X_train.shape[1],
                            num_nodes,
                            out_features=1,
                            batch_norm=True,  # Always True
                            dropout=dropout,
                            output_bias=False
                        )
                        model = CoxPH(net, tt.optim.Adam)
                        model.optimizer.set_lr(lr)
                        
                        # Train with early stopping
                        train_data = (X_train.astype(np.float32), 
                                    (Y_train[:, 0].astype(np.float32), Y_train[:, 1].astype(int)))
                        val_data = (X_val.astype(np.float32), 
                                (Y_val[:, 0].astype(np.float32), Y_val[:, 1].astype(int)))
                        
                        batch_size = 64
                        if X_train.shape[0] % batch_size == 1:
                            if X_train.shape[0] % (batch_size + 1) != 1:
                                batch_size += 1
                            elif X_train.shape[0] % (batch_size - 1) != 1:
                                batch_size -= 1
                            else:
                                batch_size = batch_size // 2
                        
                        model.fit(
                            *train_data,
                            batch_size,
                            epochs,
                            callbacks=[tt.callbacks.EarlyStopping()],
                            verbose=False,
                            val_data=val_data,
                            val_batch_size=512
                        )
                        
                        # Compute baseline hazards required for prediction
                        model.compute_baseline_hazards()
                        
                        # Get survival curves for validation set
                        surv_curves = model.predict_surv_df(val_data[0])
                        times = surv_curves.index.to_numpy()
                        
                        # Calculate concordance index
                        try:
                            concordance_td = get_concordance_score(Y_val, surv_curves, times)
                        # catch zero division error
                        except ZeroDivisionError as e: # Could Happen if no comparable pairs for small training size
                            print(f"[Warning]: No comparable pairs for validation set (training continues): {str(e)}")
                            concordance_td = 0
                        
                        if concordance_td > best_concordance:
                            best_concordance = concordance_td
                            best_model = model
                            self.best_params = {
                                'num_nodes': num_nodes,
                                'dropout': dropout,
                                'lr': lr,
                                'epochs': epochs
                            }

        self.model = best_model
        self.model.compute_baseline_hazards()

    def _fit_deephit(self, X_train, Y_train, X_val, Y_val):
        """Fit DeepHit model with model selection."""
        # Default grid if none specified
        default_grid = {
            'num_nodes': [64, 128],
            'dropout': [0.1, 0.2],
            'lr': [0.01, 0.001],
            'epochs': [100, 500]
        }
        
        param_grid = self.hyperparams_grid.get('DeepHit', default_grid)
        best_model = None
        best_concordance = -float('inf')  # Higher concordance is better
        
        # Create label transform
        num_durations = min(np.unique(Y_train[:, 0]).shape[0], 100)  # Limit durations for efficiency
        labtrans = DeepHitSingle.label_transform(num_durations)
        
        # Transform the data
        Y_train_fit = labtrans.fit_transform(Y_train[:, 0].astype(np.float32), Y_train[:, 1].astype(int))
        try:
            Y_val_fit = labtrans.transform(Y_val[:, 0].astype(np.float32), Y_val[:, 1].astype(int))
        except ValueError:
            Y_val_fit = labtrans.transform(np.vstack((Y_val, Y_train))[:, 0].astype(np.float32), np.vstack((Y_val, Y_train))[:, 1].astype(int))
            Y_val_fit = (Y_val_fit[0][:Y_val.shape[0]], Y_val_fit[1][:Y_val.shape[0]])
        
        # Grid search
        for num_nodes in param_grid.get('num_nodes', [128]):
            for dropout in param_grid.get('dropout', [0.1]):
                for lr in param_grid.get('lr', [0.01]):
                    for epochs in param_grid.get('epochs', [200]):
                        # Create network
                        net = tt.practical.MLPVanilla(
                            X_train.shape[1],
                            num_nodes,
                            labtrans.out_features,
                            batch_norm=True,  # Always True
                            dropout=dropout
                        )
                        model = DeepHitSingle(
                            net, tt.optim.Adam, alpha=0.2, sigma=0.1, duration_index=labtrans.cuts
                        )
                        model.optimizer.set_lr(lr)

                        batch_size = 64
                        if X_train.shape[0] % batch_size == 1:
                            if X_train.shape[0] % (batch_size + 1) != 1:
                                batch_size += 1
                            elif X_train.shape[0] % (batch_size - 1) != 1:
                                batch_size -= 1
                            else:
                                batch_size = batch_size // 2
                        
                        # Train model
                        model.fit(
                            X_train.astype(np.float32),
                            Y_train_fit,
                            batch_size=batch_size,
                            epochs=epochs,
                            callbacks=[tt.callbacks.EarlyStopping()],
                            verbose=False,
                            val_data=(X_val.astype(np.float32), Y_val_fit),
                            val_batch_size=512
                        )
                        
                        # Get survival curves for validation set
                        # Use interpolate for smoother curves
                        surv_curves = model.interpolate(10).predict_surv_df(X_val.astype(np.float32))
                        times = surv_curves.index.to_numpy()
                        
                        # Calculate concordance index
                        try:
                            concordance_td = get_concordance_score(Y_val, surv_curves, times)
                        # catch zero division error
                        except ZeroDivisionError as e: # Could Happen if no comparable pairs for small training size
                            print(f"[Warning]: No comparable pairs for validation set (training continues): {str(e)}")
                            concordance_td = 0
                        
                        if concordance_td > best_concordance:
                            best_concordance = concordance_td
                            best_model = model
                            self.best_params = {
                                'num_nodes': num_nodes,
                                'dropout': dropout,
                                'lr': lr,
                                'epochs': epochs
                            }
        
        # Set the best model and parameters
        self.model = best_model
        self.time_grid = labtrans.cuts

    def evaluate(self, X_test, Y_test):
        """
        Evaluate model performance on test data with exception handling.
        
        Parameters:
        - X_test (np.ndarray): Test features
        - Y_test (np.ndarray): Test targets (time, event)
        
        Returns:
        - dict or float: Performance metrics or concordance index
        """
        if self.model is None:
            raise ValueError("Model has not been fitted yet.")
            
        surv = self.predict_survival_curve(X_test)
        times = surv.index.to_numpy()
        self.survival_test = self._prepare_sksurv_data(Y_test)
        
        # Calculate concordance index with exception handling
        try:
            concordance_td = get_concordance_score(Y_test, surv, times)
        except Exception as e:
            print(f"[Warning]: Error calculating concordance_td for the current random-seed. (Base-learner result of this repeat will be excluded): {str(e)}")
            concordance_td = np.nan
            
        # Calculate integrated brier score with exception handling
        try:
            ibs = get_integrated_brier_score(self.survival_train, self.survival_test, surv, times)
        except Exception as e:
            # print(f"[Warning]: Error calculating integrated brier score for the current random-seed. (Base-learner result of this repeat will be excluded): {str(e)}")
            ibs = np.nan
            
        # # Calculate time-dependent AUC with exception handling
        # try:
        #     td_auc = get_cumulative_dynamic_auc(self.survival_train, self.survival_test, surv, times)
        # except Exception as e:
        #     print(f"Error calculating cumulative dynamic AUC: {str(e)}")
        #     td_auc = np.nan

        results = {
            "concordance_td": concordance_td,
            "integrated_brier_score": ibs,
            # "td_auc": td_auc,
            # "times": times
        }
        return results
    
    def predict_survival_curve(self, X):
        """
        Predict survival curves for new data.
        
        Parameters:
        - X (np.ndarray): Features
        
        Returns:
        - pd.DataFrame: Survival curves with times as index
        """
        if self.model is None:
            # Return dummy survival curve if model doesn't exist
            dummy_times = np.linspace(0, 10, 100)
            dummy_curves = np.ones((len(X), len(dummy_times)))
            return pd.DataFrame(dummy_curves.T, index=dummy_times)
            
        if self.model_type == "CoxPH":
            return pd.DataFrame(
                self.model.predict_survival_function(X, return_array=True).T, 
                index=self.model.unique_times_
            )
        elif self.model_type == "RandomSurvivalForest":
            return pd.DataFrame(
                self.model.predict_survival_function(X, return_array=True).T, 
                index=self.model.unique_times_
            )
        elif self.model_type == "DeepSurv":
            return self.model.predict_surv_df(X.astype(np.float32))
        elif self.model_type == "DeepHit":
            return self.model.interpolate(10).predict_surv_df(X.astype(np.float32))
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
    def predict_metric(self, X, metric="median", max_time=np.inf):
        """
        Predict a summary metric from the survival curve.
        
        Parameters:
        - X (np.ndarray): Features
        - metric (str): 'median' or 'mean'
        - max_time (float): Maximum time for restricted mean survival time
        
        Returns:
        - np.ndarray: Predicted metric values
        """
        survival_curves = self.predict_survival_curve(X)
        times = survival_curves.index.to_numpy()
        
        if metric == "median":
            return self._compute_median_survival_times(survival_curves, times)
        elif metric == "mean":
            return self._compute_restricted_mean_survival_times(survival_curves, times, max_time=max_time)
        else:
            raise ValueError(f"Unsupported metric: {metric}. Supported metrics: ['median', 'mean']")

    def _compute_restricted_mean_survival_times(self, survival_curves, times, max_time=np.inf):
        """Calculate restricted mean survival time."""
        rmt = []
        for curve in np.array(survival_curves).T:
            rmt.append(np.trapz(
                y=curve[:np.searchsorted(times, max_time, side='right')], 
                x=times[:np.searchsorted(times, max_time, side='right')]
            ))
        return np.array(rmt)
    
    def _compute_median_survival_times(self, survival_curves, times):
        """Calculate median survival time."""
        median_survival_times = []
        for curve in np.array(survival_curves).T:
            if curve[-1] > .5:
                # Extrapolate last time point to be the median survival time
                if self.extrapolate_median: 
                    median_survival_times.append(times[-1])
                else:
                    median_survival_times.append(np.inf)
            else:
                median_survival_times.append(
                    times[np.searchsorted(-curve, [-.5])[0]])
        return np.array(median_survival_times)