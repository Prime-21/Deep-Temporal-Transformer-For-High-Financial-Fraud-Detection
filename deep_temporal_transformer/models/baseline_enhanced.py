"""
Enhanced Baseline Models for Fraud Detection Comparison

This module provides baseline implementations:
- Random Forest 
- Logistic Regression
- LSTM for sequence modeling
- Temporal CNN

Author: Prasad Kharat
"""

import logging
from typing import Dict, Any, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

try:
    from ..utils.utils import setup_logging
    logger = setup_logging()
except:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


class LSTMBaseline(nn.Module):
    """
    LSTM baseline for sequential fraud detection.
    
    Args:
        input_dim: Number of input features
        hidden_dim: LSTM hidden dimension
        num_layers: Number of LSTM layers
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        
        lstm_output_dim = hidden_dim * 2
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        pooled = lstm_out.mean(dim=1)
        logits = self.classifier(pooled).squeeze(-1)
        return logits


class TemporalCNN(nn.Module):
    """
    Temporal CNN baseline using 1D convolutions.
    
    Args:
        input_dim: Number of input features
        channels: Channel dimensions for conv layers
        kernel_size: Convolution kernel size
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        input_dim: int,
        channels: list = None,
        kernel_size: int = 3,
        dropout: float = 0.2
    ):
        super().__init__()
        
        if channels is None:
            channels = [128, 256, 512]
        
        conv_blocks = []
        in_channels = input_dim
        
        for out_channels in channels:
            conv_blocks.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_channels = out_channels
        
        self.conv_blocks = nn.Sequential(*conv_blocks)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(channels[-1], 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.conv_blocks(x)
        x = self.global_pool(x).squeeze(-1)
        logits = self.classifier(x).squeeze(-1)
        return logits


class EnhancedBaselineModels:
    """
    Collection of baseline models for comparison.
    
    Args:
        random_state: Random seed
        device: PyTorch device
    """
    
    def __init__(self, random_state: int = 42, device: Optional[torch.device] = None):
        self.random_state = random_state
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {}
        logger.info(f"Enhanced Baseline Models initialized on {self.device}")
    
    def flatten_sequences(self, X_seq: np.ndarray) -> np.ndarray:
        """Flatten sequences to features for traditional ML."""
        features_mean = X_seq.mean(axis=1)
        features_std = X_seq.std(axis=1)
        features_max = X_seq.max(axis=1)
        features_min = X_seq.min(axis=1)
        features_last = X_seq[:, -1, :]
        
        X_flat = np.concatenate([
            features_mean, features_std, features_max, 
            features_min, features_last
        ], axis=1)
        
        return X_flat
    
    def train_random_forest(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        n_estimators: int = 200,
        max_depth: Optional[int] = 20,
        class_weight: str = 'balanced'
    ) -> Dict[str, Any]:
        """Train Random Forest."""
        logger.info("Training Random Forest...")
        
        X_train_flat = self.flatten_sequences(X_train)
        if X_val is not None:
            X_val_flat = self.flatten_sequences(X_val)
        
        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight=class_weight,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        rf.fit(X_train_flat, y_train)
        self.models['random_forest'] = rf
        
        metrics = {}
        if X_val is not None and y_val is not None:
            y_pred = rf.predict(X_val_flat)
            y_prob = rf.predict_proba(X_val_flat)[:, 1]
            
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_val, y_pred, average='binary', zero_division=0
            )
            auc = roc_auc_score(y_val, y_prob)
            
            metrics = {
                'f1': f1,
                'precision': precision,
                'recall': recall,
                'auc': auc
            }
            
            logger.info(f"F1: {f1:.4f} | AUC: {auc:.4f}")
        
        return metrics
    
    def train_logistic_regression(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Train Logistic Regression."""
        logger.info("Training Logistic Regression...")
        
        X_train_flat = self.flatten_sequences(X_train)
        if X_val is not None:
            X_val_flat = self.flatten_sequences(X_val)
        
        lr = LogisticRegression(
            class_weight='balanced',
            random_state=self.random_state,
            max_iter=1000
        )
        
        lr.fit(X_train_flat, y_train)
        self.models['logistic_regression'] = lr
        
        metrics = {}
        if X_val is not None and y_val is not None:
            y_pred = lr.predict(X_val_flat)
            y_prob = lr.predict_proba(X_val_flat)[:, 1]
            
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_val, y_pred, average='binary', zero_division=0
            )
            auc = roc_auc_score(y_val, y_prob)
            
            metrics = {
                'f1': f1,
                'precision': precision,
                'recall': recall,
                'auc': auc
            }
            
            logger.info(f"F1: {f1:.4f} | AUC: {auc:.4f}")
        
        return metrics
    
    def compare_models(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Dict]:
        """Evaluate all trained models."""
        X_test_flat = self.flatten_sequences(X_test)
        results = {}
        
        for name, model in self.models.items():
            if hasattr(model, 'predict_proba'):
                y_pred = model.predict(X_test_flat)
                y_prob = model.predict_proba(X_test_flat)[:, 1]
            else:
                continue
            
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test, y_pred, average='binary', zero_division=0
            )
            auc = roc_auc_score(y_test, y_prob)
            
            results[name] = {
                'f1': f1,
                'precision': precision,
                'recall': recall,
                'auc': auc
            }
        
        return results


__all__ = ['EnhancedBaselineModels', 'LSTMBaseline', 'TemporalCNN']
