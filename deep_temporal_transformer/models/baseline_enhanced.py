"""
Enhanced Baseline Models for Fraud Detection Comparison

This module provides state-of-the-art baseline implementations:
- Random Forest with optimal hyperparameters
- XGBoost with GPU acceleration
- LightGBM for fast training
- LSTM/GRU for sequence modeling
- Temporal CNN for comparison

Optimized for: Google Colab Pro with GPU support

Academic References:
    - Chen & Guestrin (2016): "XGBoost: A Scalable Tree Boosting System"
    - Ke et al. (2017): "LightGBM: A Highly Efficient Gradient Boosting Decision Tree"
    - Hochreiter & Schmidhuber (1997): "Long Short-Term Memory"
"""

import logging
from typing import Dict, Any, Tuple, Optional, List
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Scikit-learn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
    average_precision_score
)

# Gradient boosting
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from ..utils.utils import setup_logging
    logger = setup_logging()
except:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


class LSTMBaseline(nn.Module):
    """
    LSTM baseline model for sequential fraud detection.
    
    Architecture:
        Input → LSTM Layers → Global Pooling → MLP → Output
    
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
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True  # Bidirectional for better context
        )
        
        # Classification head
        lstm_output_dim = hidden_dim * 2  # Bidirectional
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch_size, seq_len, input_dim)
            
        Returns:
            Logits (batch_size,)
        """
        # LSTM encoding
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Global pooling (mean over time)
        pooled = lstm_out.mean(dim=1)  # (batch_size, hidden_dim * 2)
        
        # Classification
        logits = self.classifier(pooled).squeeze(-1)
        
        return logits


class TemporalCNN(nn.Module):
    """
    Temporal CNN baseline using 1D convolutions.
    
    Architecture:
        Input → Conv1D Blocks → Global Pooling → MLP → Output
    
    Args:
        input_dim: Number of input features
        channels: List of channel dimensions for each conv layer
        kernel_size: Convolution kernel size
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        input_dim: int,
        channels: List[int] = [128, 256, 256, 512],
        kernel_size: int = 3,
        dropout: float = 0.2
    ):
        super().__init__()
        
        # Convolutional blocks
        conv_blocks = []
        in_channels = input_dim
        
        for out_channels in channels:
            conv_blocks.append(nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2
            ))
            conv_blocks.append(nn.BatchNorm1d(out_channels))
            conv_blocks.append(nn.ReLU())
            conv_blocks.append(nn.Dropout(dropout))
            in_channels = out_channels
        
        self.conv_blocks = nn.Sequential(*conv_blocks)
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(channels[-1], 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch_size, seq_len, input_dim)
            
        Returns:
            Logits (batch_size,)
        """
        # Transpose for Conv1d: (batch, features, time)
        x = x.transpose(1, 2)
        
        # Convolutions
        x = self.conv_blocks(x)
        
        # Global pooling
        x = self.global_pool(x).squeeze(-1)
        
        # Classification
        logits = self.classifier(x).squeeze(-1)
        
        return logits


class EnhancedBaselineModels:
    """
    Enhanced collection of baseline models for comprehensive comparison.
    
    Includes both traditional ML and deep learning baselines:
        - Random Forest
        - Logistic Regression
        - XGBoost (GPU-accelerated if available)
        - LightGBM
        - LSTM
        - Temporal CNN
    
    Args:
        random_state: Random seed for reproducibility
        device: PyTorch device for deep learning models
    """
    
    def __init__(self, random_state: int = 42, device: Optional[torch.device] = None):
        self.random_state = random_state
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.feature_scalers = {}
        
        logger.info(f"🎯 Enhanced Baseline Models initialized on {self.device}")
    
    def flatten_sequences(self, X_seq: np.ndarray) -> np.ndarray:
        """
        Convert sequences to flat features for traditional ML.
        
        Extracts comprehensive statistics:
            - Mean, std, min, max across time
            - Last timestep features
            - Temporal trends
        
        Args:
            X_seq: Sequential data (n_samples, seq_len, n_features)
            
        Returns:
            Flattened features (n_samples, n_features_flat)
        """
        n_samples, seq_len, n_features = X_seq.shape
        
        # Statistical features
        features_mean = X_seq.mean(axis=1)  # Mean over time
        features_std = X_seq.std(axis=1)    # Std over time
        features_max = X_seq.max(axis=1)    # Max over time
        features_min = X_seq.min(axis=1)    # Min over time
        features_last = X_seq[:, -1, :]     # Last timestep
        
        # Temporal trend (difference between first and last)
        features_trend = X_seq[:, -1, :] - X_seq[:, 0, :]
        
        # Concatenate all features
        X_flat = np.concatenate([
            features_mean,
            features_std,
            features_max,
            features_min,
            features_last,
            features_trend
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
        """
        Train Random Forest with optimal hyperparameters.
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data (optional)
            n_estimators: Number of trees
            max_depth: Maximum tree depth
            class_weight: Class balancing strategy
            
        Returns:
            Training metrics dictionary
        """
        logger.info(\"🌲 Training Random Forest...\")\n        start_time = time.time()\n        \n        # Flatten sequences\n        X_train_flat = self.flatten_sequences(X_train)\n        if X_val is not None:\n            X_val_flat = self.flatten_sequences(X_val)\n        \n        # Train model\n        rf = RandomForestClassifier(\n            n_estimators=n_estimators,\n            max_depth=max_depth,\n            class_weight=class_weight,\n            random_state=self.random_state,\n            n_jobs=-1,  # Use all cores\n            verbose=0\n        )\n        \n        rf.fit(X_train_flat, y_train)\n        self.models['random_forest'] = rf\n        \n        training_time = time.time() - start_time\n        \n        # Evaluate on validation set\n        metrics = {}\n        if X_val is not None and y_val is not None:\n            y_pred = rf.predict(X_val_flat)\n            y_prob = rf.predict_proba(X_val_flat)[:, 1]\n            \n            precision, recall, f1, _ = precision_recall_fscore_support(\n                y_val, y_pred, average='binary', zero_division=0\n            )\n            auc = roc_auc_score(y_val, y_prob)\n            \n            metrics = {\n                'f1': f1,\n                'precision': precision,\n                'recall': recall,\n                'auc': auc,\n                'training_time': training_time\n            }\n            \n            logger.info(f\"   F1: {f1:.4f} | AUC: {auc:.4f} | Time: {training_time:.2f}s\")\n        \n        return metrics
    
    def train_xgboost(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        use_gpu: bool = True
    ) -> Dict[str, Any]:
        """
        Train XGBoost with GPU acceleration if available.
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            use_gpu: Whether to use GPU
            
        Returns:
            Training metrics
        """
        if not XGBOOST_AVAILABLE:
            logger.warning(\"⚠️ XGBoost not available, skipping...\")\n            return {}\n        
        logger.info(\"🚀 Training XGBoost...\")\n        start_time = time.time()\n        \n        # Flatten sequences\n        X_train_flat = self.flatten_sequences(X_train)\n        if X_val is not None:\n            X_val_flat = self.flatten_sequences(X_val)\n        \n        # Calculate scale_pos_weight for imbalance\n        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()\n        \n        # XGBoost parameters\n        params = {\n            'objective': 'binary:logistic',\n            'eval_metric': 'auc',\n            'max_depth': 8,\n            'learning_rate': 0.1,\n            'subsample': 0.8,\n            'colsample_bytree': 0.8,\n            'scale_pos_weight': scale_pos_weight,\n            'random_state': self.random_state\n        }\n        \n        # GPU acceleration\n        if use_gpu and torch.cuda.is_available():\n            params['tree_method'] = 'gpu_hist'\n            params['gpu_id'] = 0\n            logger.info(\"   Using GPU acceleration\")\n        \n        # Create DMatrix\n        dtrain = xgb.DMatrix(X_train_flat, label=y_train)\n        \n        # Train\n        model = xgb.train(\n            params,\n            dtrain,\n            num_boost_round=200,\n            verbose_eval=False\n        )\n        \n        self.models['xgboost'] = model\n        training_time = time.time() - start_time\n        \n        # Evaluate\n        metrics = {}\n        if X_val is not None and y_val is not None:\n            dval = xgb.DMatrix(X_val_flat)\n            y_prob = model.predict(dval)\n            y_pred = (y_prob > 0.5).astype(int)\n            \n            precision, recall, f1, _ = precision_recall_fscore_support(\n                y_val, y_pred, average='binary', zero_division=0\n            )\n            auc = roc_auc_score(y_val, y_prob)\n            \n            metrics = {\n                'f1': f1,\n                'precision': precision,\n                'recall': recall,\n                'auc': auc,\n                'training_time': training_time\n            }\n            \n            logger.info(f\"   F1: {f1:.4f} | AUC: {auc:.4f} | Time: {training_time:.2f}s\")\n        \n        return metrics
    
    def train_lstm(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 30,
        batch_size: int = 256,
        learning_rate: float = 1e-3
    ) -> Dict[str, Any]:
        \"\"\"
        Train LSTM baseline model.
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            
        Returns:
            Training metrics
        \"\"\"
        logger.info(\"🔄 Training LSTM Baseline...\")\n        start_time = time.time()\n        \n        input_dim = X_train.shape[2]\n        model = LSTMBaseline(\n            input_dim=input_dim,\n            hidden_dim=128,\n            num_layers=2,\n            dropout=0.2\n        ).to(self.device)\n        \n        # Loss and optimizer\n        pos_weight = torch.tensor((y_train == 0).sum() / (y_train == 1).sum()).to(self.device)\n        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)\n        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)\n        \n        # Convert to tensors\n        X_train_t = torch.FloatTensor(X_train).to(self.device)\n        y_train_t = torch.FloatTensor(y_train).to(self.device)\n        X_val_t = torch.FloatTensor(X_val).to(self.device)\n        y_val_t = torch.FloatTensor(y_val).to(self.device)\n        \n        # Training loop\n        best_val_f1 = 0\n        for epoch in range(epochs):\n            model.train()\n            total_loss = 0\n            \n            # Mini-batch training\n            for i in range(0, len(X_train_t), batch_size):\n                batch_X = X_train_t[i:i+batch_size]\n                batch_y = y_train_t[i:i+batch_size]\n                \n                optimizer.zero_grad()\n                outputs = model(batch_X)\n                loss = criterion(outputs, batch_y)\n                loss.backward()\n                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)\n                optimizer.step()\n                \n                total_loss += loss.item()\n            \n            # Validation\n            if (epoch + 1) % 5 == 0:\n                model.eval()\n                with torch.no_grad():\n                    val_logits = model(X_val_t)\n                    val_probs = torch.sigmoid(val_logits).cpu().numpy()\n                    val_preds = (val_probs > 0.5).astype(int)\n                \n                _, _, f1, _ = precision_recall_fscore_support(\n                    y_val, val_preds, average='binary', zero_division=0\n                )\n                \n                if f1 > best_val_f1:\n                    best_val_f1 = f1\n        \n        self.models['lstm'] = model\n        training_time = time.time() - start_time\n        \n        # Final evaluation\n        model.eval()\n        with torch.no_grad():\n            val_logits = model(X_val_t)\n            val_probs = torch.sigmoid(val_logits).cpu().numpy()\n            val_preds = (val_probs > 0.5).astype(int)\n        \n        precision, recall, f1, _ = precision_recall_fscore_support(\n            y_val, val_preds, average='binary', zero_division=0\n        )\n        auc = roc_auc_score(y_val, val_probs)\n        \n        metrics = {\n            'f1': f1,\n            'precision': precision,\n            'recall': recall,\n            'auc': auc,\n            'training_time': training_time\n        }\n        \n        logger.info(f\"   F1: {f1:.4f} | AUC: {auc:.4f} | Time: {training_time:.2f}s\")\n        \n        return metrics


# Export
__all__ = ['EnhancedBaselineModels', 'LSTMBaseline', 'TemporalCNN']
