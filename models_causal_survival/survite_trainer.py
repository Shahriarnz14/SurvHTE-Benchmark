"""
Training utilities for SurvITE model
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Optional, Tuple
import time
from sklearn.model_selection import train_test_split


class SurvITETrainer:
    """Trainer class for SurvITE model"""
    
    def __init__(self, model, lr=1e-3, weight_decay=0):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.device = model.device
        
    def get_minibatch(self, x, y, t, a, weights=None, batch_size=512):
        """Sample a minibatch from the data"""
        n = x.shape[0]
        if batch_size >= n:
            idx = np.arange(n)
        else:
            idx = np.random.choice(n, batch_size, replace=False)
        
        x_batch = torch.FloatTensor(x[idx]).to(self.device)
        y_batch = torch.FloatTensor(y[idx]).to(self.device)
        t_batch = torch.FloatTensor(t[idx]).to(self.device)
        # a_batch = torch.FloatTensor(a[idx]).to(self.device)
        a_batch = torch.as_tensor(a[idx], device=self.device, dtype=torch.bool)
        
        if weights is not None:
            w_batch = torch.FloatTensor(weights[idx]).to(self.device)
        else:
            w_batch = None
        
        return x_batch, y_batch, t_batch, a_batch, w_batch
    
    def train_step(self, x, y, t, a, weights=None, batch_size=512):
        """Single training step"""
        self.model.train()
        
        # Get minibatch
        x_batch, y_batch, t_batch, a_batch, w_batch = self.get_minibatch(
            x, y, t, a, weights, batch_size
        )
        
        # Forward pass
        loss_dict = self.model.compute_loss(x_batch, y_batch, t_batch, a_batch, w_batch)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss_dict['total_loss'].backward()
        self.optimizer.step()
        
        return {k: v.item() for k, v in loss_dict.items()}
    
    def evaluate(self, x, y, t, a, weights=None, batch_size=512):
        """Evaluate on a dataset"""
        self.model.eval()
        
        total_loss = 0
        nll_loss = 0
        ipm_loss = 0
        smooth_loss = 0
        n_batches = 0
        
        with torch.no_grad():
            # Process in batches to handle large datasets
            n = x.shape[0]
            for i in range(0, n, batch_size):
                end_idx = min(i + batch_size, n)
                
                x_batch = torch.FloatTensor(x[i:end_idx]).to(self.device)
                y_batch = torch.FloatTensor(y[i:end_idx]).to(self.device)
                t_batch = torch.FloatTensor(t[i:end_idx]).to(self.device)
                # a_batch = torch.FloatTensor(a[i:end_idx]).to(self.device)
                a_batch = torch.as_tensor(a[i:end_idx], device=self.device, dtype=torch.bool)
                
                if weights is not None:
                    w_batch = torch.FloatTensor(weights[i:end_idx]).to(self.device)
                else:
                    w_batch = None
                
                loss_dict = self.model.compute_loss(x_batch, y_batch, t_batch, a_batch, w_batch)
                
                total_loss += loss_dict['total_loss'].item()
                nll_loss += loss_dict['nll_loss'].item()
                ipm_loss += loss_dict['ipm_loss'].item()
                smooth_loss += loss_dict['smooth_loss'].item()
                n_batches += 1
        
        return {
            'total_loss': total_loss / n_batches,
            'nll_loss': nll_loss / n_batches,
            'ipm_loss': ipm_loss / n_batches,
            'smooth_loss': smooth_loss / n_batches
        }
    
    def fit(self, x_train, y_train, t_train, a_train, 
            x_val=None, y_val=None, t_val=None, a_val=None,
            weights_train=None, weights_val=None,
            epochs=20000, batch_size=512, 
            check_step=100, patience=20, verbose=True):
        """
        Train the model with early stopping
        
        Args:
            x_train: Training features
            y_train: Event indicators (1=event, 0=censored)
            t_train: Observed times
            a_train: Treatment assignments
            x_val: Validation features (optional)
            y_val: Validation event indicators (optional)
            t_val: Validation observed times (optional)
            a_val: Validation treatment assignments (optional)
            weights_train: Training sample weights (optional)
            weights_val: Validation sample weights (optional)
            epochs: Maximum number of epochs
            batch_size: Batch size
            check_step: Steps between validation checks
            patience: Early stopping patience
            verbose: Print progress
        
        Returns:
            Dictionary with training history
        """
        
        # If no validation set provided, create one
        if x_val is None:
            x_train, x_val, y_train, y_val, t_train, t_val, a_train, a_val = \
                train_test_split(x_train, y_train, t_train, a_train, 
                               test_size=0.2, random_state=42)
            
            if weights_train is not None:
                weights_train, weights_val = train_test_split(
                    weights_train, test_size=0.2, random_state=42
                )
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_nll': [],
            'val_nll': [],
            'train_ipm': [],
            'val_ipm': []
        }
        
        best_val_loss = float('inf')
        best_state = None
        patience_counter = 0
        
        # Training loop
        for itr in range(epochs):
            # Training step
            train_losses = self.train_step(
                x_train, y_train, t_train, a_train, weights_train, batch_size
            )
            
            # Validation and logging
            if (itr + 1) % check_step == 0:
                # Evaluate on validation set
                val_losses = self.evaluate(
                    x_val, y_val, t_val, a_val, weights_val, batch_size
                )
                
                # Store history
                history['train_loss'].append(train_losses['total_loss'])
                history['val_loss'].append(val_losses['total_loss'])
                history['train_nll'].append(train_losses['nll_loss'])
                history['val_nll'].append(val_losses['nll_loss'])
                history['train_ipm'].append(train_losses['ipm_loss'])
                history['val_ipm'].append(val_losses['ipm_loss'])
                
                if verbose:
                    print(f"Iter {itr+1:5d} | "
                          f"Train Loss: {train_losses['total_loss']:.4f} "
                          f"(NLL: {train_losses['nll_loss']:.4f}, "
                          f"IPM: {train_losses['ipm_loss']:.4f}) | "
                          f"Val Loss: {val_losses['total_loss']:.4f} "
                          f"(NLL: {val_losses['nll_loss']:.4f}, "
                          f"IPM: {val_losses['ipm_loss']:.4f})")
                
                # Early stopping
                if val_losses['total_loss'] < best_val_loss:
                    best_val_loss = val_losses['total_loss']
                    best_state = self.model.state_dict().copy()
                    patience_counter = 0
                    if verbose:
                        print(f"  -> New best validation loss: {best_val_loss:.4f}")
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        if verbose:
                            print(f"Early stopping at iteration {itr+1}")
                        break
        
        # Load best model
        if best_state is not None:
            self.model.load_state_dict(best_state)
        
        return history
    
    def save_model(self, path):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'x_dim': self.model.x_dim,
                'z_dim': self.model.z_dim,
                't_max': self.model.t_max,
                'ipm_type': self.model.ipm_type,
                'beta': self.model.beta,
                'gamma': self.model.gamma
            }
        }, path)
    
    def load_model(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        return checkpoint['model_config']