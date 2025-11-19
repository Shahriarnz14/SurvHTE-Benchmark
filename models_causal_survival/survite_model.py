"""
SurvITE model wrapper for SurvHTE-Benchmark
Follows the same interface as causal_survival_forest.py
"""

import numpy as np
import torch
import warnings
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
import os
import sys

# Add parent directory to path to import survite modules
from .survite_pytorch import SurvITE
from .survite_trainer import SurvITETrainer


class SurvITEModel:
    """
    SurvITE wrapper for the benchmark
    
    Parameters:
    -----------
    horizon : int or None
        Maximum time horizon for predictions. If None, will be set based on training data.
    target : str
        Target for estimation. Default is "RMST" (Restricted Mean Survival Time).
    x_dim : int or None
        Input dimension. Will be inferred from training data if None.
    z_dim : int
        Latent representation dimension (default: 100)
    h_dim1 : int
        Hidden dimension for representation network (default: 100)
    h_dim2 : int
        Hidden dimension for hypothesis network (default: 100)
    num_layers1 : int
        Number of layers in representation network (default: 3)
    num_layers2 : int
        Number of layers in hypothesis network (default: 2)
    activation : str
        Activation function (default: 'relu')
    dropout : float
        Dropout rate (default: 0.3)
    ipm_type : str
        IPM type for domain adaptation (default: 'wasserstein')
    beta : float
        IPM regularization weight (default: 1e-3)
    gamma : float
        Smoothing regularization weight (default: 0)
    lr : float
        Learning rate (default: 1e-3)
    batch_size : int
        Batch size for training (default: 512)
    epochs : int
        Maximum training epochs (default: 20000)
    patience : int
        Early stopping patience (default: 20)
    seed : int or None
        Random seed for reproducibility
    device : str or None
        Device to use ('cuda', 'cpu', or None for auto-detect)
    verbose : bool
        Whether to print training progress
    """
    
    def __init__(self, 
                 horizon=None, 
                 target="RMST",
                 x_dim=None,
                 z_dim=100,
                 h_dim1=100,
                 h_dim2=100,
                 num_layers1=3,
                 num_layers2=2,
                 activation='relu',
                 dropout=0.3,
                 ipm_type='wasserstein',
                 beta=1e-3,
                 gamma=0,
                 lr=1e-3,
                 batch_size=512,
                 epochs=20000,
                 patience=20,
                 seed=None,
                 device=None,
                 verbose=False):
        
        self.horizon = horizon
        self.target = target
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.h_dim1 = h_dim1
        self.h_dim2 = h_dim2
        self.num_layers1 = num_layers1
        self.num_layers2 = num_layers2
        self.activation = activation
        self.dropout = dropout
        self.ipm_type = ipm_type
        self.beta = beta
        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.seed = seed
        self.verbose = verbose
        
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        self.model = None
        self.trainer = None
        self.time_grid = None  # Will store unique times from training data
        
        # Set random seeds
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
    
    def _create_time_grid_and_discretize(self, Y_time, horizon=None):
        """
        Create time grid from unique event times and convert continuous times to discrete indices.
        
        This is the key discretization step:
        - Extract all unique event times from training data
        - Sort them to create a time grid
        - Map each continuous time to its index in the grid
        
        Args:
            Y_time: continuous event times
            horizon: optional max time to include in grid (in original units)
        
        Returns:
            time_grid: sorted array of unique times (continuous values)
            Y_time_discrete: discrete indices corresponding to Y_time
            t_max: maximum discrete time index
        """
        # Get unique times and sort them
        unique_times = np.unique(Y_time)
        
        # If horizon specified, filter times
        if horizon is not None:
            unique_times = unique_times[unique_times <= horizon]
            # Ensure horizon is included if it's not already
            if horizon not in unique_times:
                unique_times = np.append(unique_times, horizon)
                unique_times.sort()
        
        # Add time 0 if not present (for baseline)
        if unique_times[0] > 0:
            unique_times = np.concatenate([[0], unique_times])
        
        time_grid = unique_times
        t_max = len(time_grid) - 1
        
        # Convert continuous times to discrete indices
        # For each time in Y_time, find its index in time_grid
        Y_time_discrete = np.searchsorted(time_grid, Y_time, side='right') - 1
        Y_time_discrete = np.clip(Y_time_discrete, 0, t_max)
        
        return time_grid, Y_time_discrete, t_max
    
    def fit(self, X_train, W_train, Y_train, failure_times_grid=None):
        """
        Fit the SurvITE model
        
        Parameters:
        -----------
        X_train : np.array
            Training features of shape (n, p)
        W_train : np.array
            Treatment assignments of shape (n,)
        Y_train : np.array
            Outcomes of shape (n, 2), where:
            - Y_train[:, 0] = event times (continuous)
            - Y_train[:, 1] = event indicators (1=event, 0=censored)
        failure_times_grid : np.array or None
            Grid of failure times (not used in SurvITE, kept for compatibility)
        """
        
        # Extract event times (continuous) and indicators
        Y_time_continuous = Y_train[:, 0]
        Y_event = Y_train[:, 1]
        
        # Set horizon if not specified (in original time units)
        if self.horizon is None:
            self.horizon = float(Y_time_continuous.max())
        
        # Create time grid and discretize times
        # This maps continuous times to discrete indices efficiently
        self.time_grid, Y_time_discrete, t_max_actual = \
            self._create_time_grid_and_discretize(Y_time_continuous, self.horizon)
        
        # Update t_max to actual number of discrete time points
        self.t_max_discrete = t_max_actual
        
        if self.verbose:
            print(f"Created time grid with {len(self.time_grid)} unique time points")
            print(f"Time range: [{self.time_grid[0]:.2f}, {self.time_grid[-1]:.2f}]")
            print(f"Horizon: {self.horizon:.2f} (original units)")
        
        # Set input dimension
        if self.x_dim is None:
            self.x_dim = X_train.shape[1]
        
        # Initialize model with discrete t_max
        self.model = SurvITE(
            x_dim=self.x_dim,
            z_dim=self.z_dim,
            h_dim1=self.h_dim1,
            h_dim2=self.h_dim2,
            num_layers1=self.num_layers1,
            num_layers2=self.num_layers2,
            t_max=self.t_max_discrete,
            activation=self.activation,
            dropout=self.dropout,
            ipm_type=self.ipm_type,
            beta=self.beta,
            gamma=self.gamma,
            device=self.device,
            time_grid=self.time_grid  # Pass time grid to model
        )
        
        # Initialize trainer
        self.trainer = SurvITETrainer(self.model, lr=self.lr)
        
        # Split into train/validation if needed
        X_tr, X_val, Y_time_discrete_tr, Y_time_discrete_val, Y_event_tr, Y_event_val, W_tr, W_val = \
            train_test_split(X_train, Y_time_discrete, Y_event, W_train, 
                           test_size=0.2, random_state=self.seed)
        
        # Train the model (uses discrete times)
        history = self.trainer.fit(
            x_train=X_tr,
            y_train=Y_event_tr,
            t_train=Y_time_discrete_tr,  # Now using discrete indices
            a_train=W_tr,
            x_val=X_val,
            y_val=Y_event_val,
            t_val=Y_time_discrete_val,  # Now using discrete indices
            a_val=W_val,
            epochs=self.epochs,
            batch_size=self.batch_size,
            check_step=100,
            patience=self.patience,
            verbose=self.verbose
        )
        
        return history
    
    def predict_cate(self, X, W=None):
        """
        Predict CATE (Conditional Average Treatment Effect) using RMST
        
        Parameters:
        -----------
        X : np.array
            Features of shape (n, p)
        W : np.array or None
            Treatment assignments (not used in prediction)
        
        Returns:
        --------
        np.array
            Predicted CATE values of shape (n,)
        """
        if self.model is None:
            raise RuntimeError("You must call `fit` before `predict`.")
        
        # Predict RMST-based ITE
        # horizon is in original time units, not indices
        if self.target == "RMST":
            cate = self.model.predict_rmst_ite(X, horizon=self.horizon)
        else:
            # Could implement other targets here
            warnings.warn(f"Target '{self.target}' not fully implemented, using RMST")
            cate = self.model.predict_rmst_ite(X, horizon=self.horizon)
        
        return cate
    
    def predict_survival(self, X, W, target_times=None):
        """
        Predict survival curves
        
        Parameters:
        -----------
        X : np.array
            Features of shape (n, p)
        W : np.array
            Treatment assignments of shape (n,)
        target_times : np.array or None
            Optional array of continuous times for predictions.
            If None, returns survival at discrete time grid points.
        
        Returns:
        --------
        np.array
            Predicted survival curves of shape (n, n_times)
        """
        if self.model is None:
            raise RuntimeError("You must call `fit` before `predict`.")
        
        surv1 = self.model.predict_survival_A1(X, target_times=target_times)
        surv0 = self.model.predict_survival_A0(X, target_times=target_times)
        
        # Select based on actual treatment
        survival = np.zeros_like(surv1)
        survival[W == 0] = surv0[W == 0]
        survival[W == 1] = surv1[W == 1]
        
        return survival
    
    def evaluate(self, X, cate_true, W=None):
        """
        Evaluate CATE predictions using mean squared error
        
        Parameters:
        -----------
        X : np.ndarray
            Test features
        cate_true : np.ndarray
            Ground-truth CATE values
        W : np.ndarray
            Treatment assignment (not used in this method)
        
        Returns:
        --------
        mse : float
            Mean squared error
        cate_pred : np.ndarray
            Predicted CATE values
        ate_pred : float
            Predicted ATE (average treatment effect)
        """
        cate_pred = self.predict_cate(X, W)
        ate_pred = np.mean(cate_pred)
        rmse = root_mean_squared_error(cate_true, cate_pred)
        return rmse, cate_pred, ate_pred
    
    def save_model(self, path):
        """Save model checkpoint"""
        if self.trainer is not None:
            self.trainer.save_model(path)
            # Also save hyperparameters and time_grid
            import pickle
            hyperparams = {
                'horizon': self.horizon,
                'target': self.target,
                'x_dim': self.x_dim,
                'z_dim': self.z_dim,
                'h_dim1': self.h_dim1,
                'h_dim2': self.h_dim2,
                'num_layers1': self.num_layers1,
                'num_layers2': self.num_layers2,
                'activation': self.activation,
                'dropout': self.dropout,
                'ipm_type': self.ipm_type,
                'beta': self.beta,
                'gamma': self.gamma,
                'lr': self.lr,
                'batch_size': self.batch_size,
                'time_grid': self.time_grid,  # Save time grid
                't_max_discrete': self.t_max_discrete  # Save discrete t_max
            }
            with open(path + '.params', 'wb') as f:
                pickle.dump(hyperparams, f)
    
    def load_model(self, path):
        """Load model checkpoint"""
        import pickle
        
        # Load hyperparameters
        with open(path + '.params', 'rb') as f:
            hyperparams = pickle.load(f)
        
        # Update model parameters
        for key, value in hyperparams.items():
            setattr(self, key, value)
        
        # Initialize model with loaded parameters
        self.model = SurvITE(
            x_dim=self.x_dim,
            z_dim=self.z_dim,
            h_dim1=self.h_dim1,
            h_dim2=self.h_dim2,
            num_layers1=self.num_layers1,
            num_layers2=self.num_layers2,
            t_max=self.t_max_discrete,  # Use discrete t_max
            activation=self.activation,
            dropout=self.dropout,
            ipm_type=self.ipm_type,
            beta=self.beta,
            gamma=self.gamma,
            device=self.device,
            time_grid=self.time_grid  # Restore time grid
        )
        
        # Load model weights
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Initialize trainer
        self.trainer = SurvITETrainer(self.model, lr=self.lr)