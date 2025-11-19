"""
SurvITE: Survival Individualized Treatment Effect Estimator
PyTorch implementation based on https://github.com/chl8856/survITE
Paper: https://arxiv.org/pdf/2110.14001
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple
import warnings


class RepresentationNetwork(nn.Module):
    """Phi network that maps inputs to shared representation"""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, 
                 num_layers: int, activation: str = 'elu', dropout: float = 0.3):
        super().__init__()
        
        self.layers = nn.ModuleList()
        
        # Build hidden layers
        for i in range(num_layers):
            if i == 0:
                self.layers.append(nn.Linear(input_dim, hidden_dim))
            else:
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            
            # Add batch norm
            self.layers.append(nn.BatchNorm1d(hidden_dim))
            
            # Add activation
            if activation == 'elu':
                self.layers.append(nn.ELU())
            elif activation == 'relu':
                self.layers.append(nn.ReLU())
            else:
                self.layers.append(nn.Tanh())
            
            # Add dropout
            self.layers.append(nn.Dropout(dropout))
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.output_layer(x)


class HypothesisNetwork(nn.Module):
    """Hypothesis network for hazard prediction"""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_layers: int, activation: str = 'relu', dropout: float = 0.3):
        super().__init__()
        
        self.layers = nn.ModuleList()
        
        # Build hidden layers
        for i in range(num_layers):
            if i == 0:
                self.layers.append(nn.Linear(input_dim, hidden_dim))
            else:
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            
            # Add batch norm
            self.layers.append(nn.BatchNorm1d(hidden_dim))
            
            # Add activation
            if activation == 'elu':
                self.layers.append(nn.ELU())
            elif activation == 'relu':
                self.layers.append(nn.ReLU())
            else:
                self.layers.append(nn.Tanh())
            
            # Add dropout
            self.layers.append(nn.Dropout(dropout))
        
        # Output layer with sigmoid for hazard probabilities
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.sigmoid(self.output_layer(x))


class IPMUtils:
    """Integral Probability Metric utilities for domain adaptation"""
    
    @staticmethod
    def pdist2sq(X, Y):
        """Computes squared Euclidean distance between all pairs"""
        # X: [n1, d], Y: [n2, d]
        # Returns: [n1, n2] distance matrix
        n1 = X.size(0)
        n2 = Y.size(0)
        
        # Expand dimensions for broadcasting
        X_exp = X.unsqueeze(1)  # [n1, 1, d]
        Y_exp = Y.unsqueeze(0)  # [1, n2, d]
        
        # Compute squared distances
        dist = torch.sum((X_exp - Y_exp) ** 2, dim=2)
        return dist
    
    @staticmethod
    def mmd2_lin(X1, X2, W1=None, W2=None, p=0.5):
        """Linear Maximum Mean Discrepancy"""
        if W1 is None:
            W1 = torch.ones(X1.size(0), device=X1.device)
        if W2 is None:
            W2 = torch.ones(X2.size(0), device=X2.device)
        
        # Normalize weights
        W1 = W1 / W1.sum()
        W2 = W2 / W2.sum()
        
        # Weighted means
        mean1 = torch.sum(W1.unsqueeze(1) * X1, dim=0)
        mean2 = torch.sum(W2.unsqueeze(1) * X2, dim=0)
        
        # MMD
        mmd = torch.sum((2.0 * p * mean1 - 2.0 * (1.0 - p) * mean2) ** 2)
        return mmd
    
    @staticmethod
    def wasserstein(X1, X2, W1=None, W2=None, p=0.5, lam=10, its=10):
        """Wasserstein distance using Sinkhorn iterations"""
        n1 = X1.size(0)
        n2 = X2.size(0)
        
        # Compute distance matrix
        M = IPMUtils.pdist2sq(X1, X2)
        
        if W1 is None:
            W1 = torch.ones(n1, device=X1.device)
        if W2 is None:
            W2 = torch.ones(n2, device=X2.device)
        
        # Normalize weights
        W1 = W1 / W1.sum()
        W2 = W2 / W2.sum()
        
        # Create weight mask
        W_mask = W1.unsqueeze(1) * W2.unsqueeze(0)
        
        # Estimate lambda and delta
        M_mean = torch.sum(M * W_mask)
        delta = M.max().detach()
        eff_lam = (lam / M_mean).detach()
        
        # Add dummy points
        Mt = torch.cat([M, delta * torch.ones(1, n2, device=M.device)], dim=0)
        col = torch.cat([delta * torch.ones(n1 + 1, 1, device=M.device)], dim=1)
        col[-1, -1] = 0
        Mt = torch.cat([Mt, col], dim=1)
        
        # Marginal distributions
        a = torch.cat([p * W1, (1 - p) * torch.ones(1, device=X1.device)])
        b = torch.cat([(1 - p) * W2, p * torch.ones(1, device=X2.device)])
        
        # Sinkhorn iterations
        K = torch.exp(-eff_lam * Mt) + 1e-6
        u = torch.ones_like(a)
        
        for _ in range(its):
            v = b / (K.T @ u + 1e-8)
            u = a / (K @ v + 1e-8)
        
        # Transport plan
        T = u.unsqueeze(1) * K * v.unsqueeze(0)
        
        # Wasserstein distance
        D = 2 * torch.sum(T * Mt)
        return D


class SurvITE(nn.Module):
    """
    SurvITE: Survival Individualized Treatment Effect Estimator
    
    Hyperparameters:
        - x_dim: Input dimension
        - z_dim: Latent representation dimension (default: 100)
        - h_dim1: Hidden dimension for representation network (default: 100)
        - h_dim2: Hidden dimension for hypothesis network (default: 100)
        - num_layers1: Number of layers in representation network (default: 3)
        - num_layers2: Number of layers in hypothesis network (default: 2)
        - t_max: Maximum time horizon (default: 30)
        - activation: Activation function ('elu', 'relu', 'tanh') (default: 'relu')
        - dropout: Dropout rate (default: 0.3)
        - ipm_type: IPM type ('wasserstein', 'mmd', 'no_ipm') (default: 'wasserstein')
        - beta: IPM regularization weight (default: 1e-3)
        - gamma: Smoothing regularization weight (default: 0)
        - lr: Learning rate (default: 1e-3)
        - batch_size: Batch size (default: 512)
    """
    
    def __init__(self, 
                 x_dim: int,
                 z_dim: int = 100,
                 h_dim1: int = 100,
                 h_dim2: int = 100,
                 num_layers1: int = 3,
                 num_layers2: int = 2,
                 t_max: int = 30,
                 activation: str = 'elu',
                 dropout: float = 0.3,
                 ipm_type: str = 'wasserstein',
                 beta: float = 1e-3,
                 gamma: float = 0,
                 device: str = None,
                 time_grid: np.ndarray = None):
        super().__init__()
        
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.t_max = t_max
        self.ipm_type = ipm_type
        self.beta = beta
        self.gamma = gamma
        
        # Store time grid for continuous time mapping
        # time_grid[i] = continuous time corresponding to discrete index i
        if time_grid is not None:
            self.time_grid = time_grid
        else:
            # Default: assume discrete time indices
            self.time_grid = np.arange(t_max + 1, dtype=np.float32)
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Build networks
        self.phi = RepresentationNetwork(
            x_dim, h_dim1, z_dim, num_layers1, activation, dropout
        )
        
        # Two hypothesis networks for treatment and control
        self.h1 = HypothesisNetwork(
            z_dim, h_dim2, t_max + 1, num_layers2, activation, dropout
        )
        self.h0 = HypothesisNetwork(
            z_dim, h_dim2, t_max + 1, num_layers2, activation, dropout
        )
        
        self.to(self.device)
    
    def forward(self, x, a):
        """Forward pass"""
        # Get representation
        z = self.phi(x)
        
        # Get hazards for both treatments
        h1 = self.h1(z)  # Hazards for treatment=1
        h0 = self.h0(z)  # Hazards for treatment=0
        
        # Select based on actual treatment
        hazards = torch.where(
            a.unsqueeze(1).expand(-1, self.t_max + 1),
            h1, h0
        )
        
        return hazards, h1, h0, z
    
    def compute_survival(self, hazards):
        """Convert hazards to survival probabilities"""
        # S(t) = prod_{j=1}^{t} (1 - h_j)
        survival = torch.cumprod(1 - hazards, dim=1)
        return survival
    
    def compute_loss(self, x, y, t, a, weights=None):
        """Compute the total loss"""
        batch_size = x.size(0)
        
        # Forward pass
        hazards, h1, h0, z = self.forward(x, a)
        
        # Create time mask (for observed times)
        time_indices = t.long()
        mask = torch.zeros(batch_size, self.t_max + 1, device=self.device)
        
        for i in range(batch_size):
            if time_indices[i] <= self.t_max:
                mask[i, :time_indices[i] + 1] = 1
        
        # Compute negative log-likelihood
        eps = 1e-8
        
        # For events (y=1): -log(h(t)) - sum_{j<t} log(1-h(j))
        # For censoring (y=0): - sum_{j<=t} log(1-h(j))
        log_hazards = torch.log(hazards + eps)
        log_survival = torch.log(1 - hazards + eps)
        
        # Event contribution
        event_loss = torch.zeros(batch_size, device=self.device)
        for i in range(batch_size):
            ti = min(time_indices[i].item(), self.t_max)
            if y[i] == 1:  # Event occurred
                event_loss[i] = -log_hazards[i, ti]
                if ti > 0:
                    event_loss[i] -= torch.sum(log_survival[i, :ti])
            else:  # Censored
                event_loss[i] = -torch.sum(log_survival[i, :ti+1])
        
        # Apply weights if provided
        if weights is not None:
            event_loss = event_loss * weights
        
        nll_loss = event_loss.mean()
        
        # IPM loss for domain adaptation
        ipm_loss = torch.tensor(0.0, device=self.device)
        if self.beta > 0 and self.ipm_type != 'no_ipm':
            # Separate treated and control representations
            z1 = z[a == 1]
            z0 = z[a == 0]
            
            if len(z1) > 0 and len(z0) > 0:
                if weights is not None:
                    w1 = weights[a == 1]
                    w0 = weights[a == 0]
                else:
                    w1 = None
                    w0 = None
                
                if self.ipm_type == 'wasserstein':
                    ipm_loss = IPMUtils.wasserstein(z1, z0, w1, w0)
                elif self.ipm_type == 'mmd':
                    ipm_loss = IPMUtils.mmd2_lin(z1, z0, w1, w0)
        
        # Smoothing regularization
        smooth_loss = torch.tensor(0.0, device=self.device)
        if self.gamma > 0:
            # Penalize large differences between consecutive time points
            smooth_loss = torch.mean((hazards[:, 1:] - hazards[:, :-1]) ** 2)
        
        # Total loss
        total_loss = nll_loss + self.beta * ipm_loss + self.gamma * smooth_loss
        
        return {
            'total_loss': total_loss,
            'nll_loss': nll_loss,
            'ipm_loss': ipm_loss,
            'smooth_loss': smooth_loss
        }
    
    def predict_hazard_A1(self, x):
        """Predict hazards for treatment=1"""
        self.eval()
        with torch.no_grad():
            x_tensor = torch.FloatTensor(x).to(self.device)
            z = self.phi(x_tensor)
            h1 = self.h1(z)
            return h1.cpu().numpy()
    
    def _interpolate_survival(self, survival_discrete, target_times):
        """
        Interpolate survival curves from discrete time grid to target continuous times.
        
        Args:
            survival_discrete: [n_samples, t_max+1] survival at discrete time points
            target_times: array of continuous times to interpolate to
        
        Returns:
            survival_interpolated: [n_samples, len(target_times)] interpolated survival
        """
        n_samples = survival_discrete.shape[0]
        n_target = len(target_times)
        survival_interp = np.zeros((n_samples, n_target))
        
        for i in range(n_samples):
            # Use numpy interp: handles extrapolation by extending edge values
            survival_interp[i] = np.interp(
                target_times,
                self.time_grid,
                survival_discrete[i],
                left=1.0,  # Before first time point, survival = 1
                right=survival_discrete[i, -1]  # After last time, use last value
            )
        
        return survival_interp
    
    def predict_hazard_A0(self, x):
        """Predict hazards for treatment=0"""
        self.eval()
        with torch.no_grad():
            x_tensor = torch.FloatTensor(x).to(self.device)
            z = self.phi(x_tensor)
            h0 = self.h0(z)
            return h0.cpu().numpy()
    
    def predict_survival_A1(self, x, target_times=None):
        """
        Predict survival curves for treatment=1
        
        Args:
            x: features
            target_times: optional array of continuous times for interpolation
                         if None, returns survival at discrete time grid points
        
        Returns:
            survival: [n_samples, n_times] survival probabilities
        """
        h1 = self.predict_hazard_A1(x)
        survival_discrete = np.cumprod(1 - h1, axis=1)
        
        if target_times is not None:
            return self._interpolate_survival(survival_discrete, target_times)
        return survival_discrete
    
    def predict_survival_A0(self, x, target_times=None):
        """
        Predict survival curves for treatment=0
        
        Args:
            x: features
            target_times: optional array of continuous times for interpolation
                         if None, returns survival at discrete time grid points
        
        Returns:
            survival: [n_samples, n_times] survival probabilities
        """
        h0 = self.predict_hazard_A0(x)
        survival_discrete = np.cumprod(1 - h0, axis=1)
        
        if target_times is not None:
            return self._interpolate_survival(survival_discrete, target_times)
        return survival_discrete
    
    def predict_rmst_ite(self, x, horizon=None):
        """
        Predict Restricted Mean Survival Time (RMST) ITE
        
        RMST = integral of survival function from 0 to horizon
        ITE = RMST(treatment=1) - RMST(treatment=0)
        
        Args:
            x: features
            horizon: time horizon in original continuous time units (not index)
                    if None, uses max time in time_grid
        
        Returns:
            ite: RMST-based treatment effect
        """
        if horizon is None:
            horizon = self.time_grid[-1]
        
        # Create time points for integration up to horizon
        # Use time_grid points up to horizon
        eval_times = self.time_grid[self.time_grid <= horizon]
        
        # If horizon is beyond last time point, add it
        if horizon > self.time_grid[-1] and np.isfinite(horizon):
            eval_times = np.append(eval_times, horizon)
        
        # Get survival at discrete grid points
        surv1 = self.predict_survival_A1(x)  # [n, t_max+1]
        surv0 = self.predict_survival_A0(x)  # [n, t_max+1]
        
        # Interpolate to evaluation times
        surv1_eval = self._interpolate_survival(surv1, eval_times)  # [n, len(eval_times)]
        surv0_eval = self._interpolate_survival(surv0, eval_times)  # [n, len(eval_times)]
        
        # Compute RMST using trapezoidal rule
        # RMST = int_0^horizon S(t) dt ≈ sum of trapezoids
        rmst1 = np.trapz(surv1_eval, eval_times, axis=1)
        rmst0 = np.trapz(surv0_eval, eval_times, axis=1)
        
        # ITE
        ite = rmst1 - rmst0
        return ite