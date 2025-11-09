"""Baseline models for comparison with Deep Temporal Transformer."""
import logging
from typing import Dict, Any, Tuple, Optional
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_recall_fscore_support, 
    roc_auc_score, 
    confusion_matrix
)

from .utils import setup_logging

logger = setup_logging()


class BaselineModels:
    """Collection of baseline models for fraud detection comparison."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        
    def _validate_inputs(self, X: np.ndarray, y: np.ndarray) -> None:
        """Validate input data."""
        if len(X) != len(y):
            raise ValueError(f"X and y length mismatch: {len(X)} vs {len(y)}")
        
        if len(X) == 0:
            raise ValueError("Empty dataset provided")
        
        if not np.isfinite(X).all():
            raise ValueError("X contains non-finite values")
        
        if not np.isin(y, [0, 1]).all():
            raise ValueError("y must contain only 0 and 1 values")
    
    def flatten_sequences(self, X_seq: np.ndarray) -> np.ndarray:
        """
        Convert sequential data to flat features for traditional ML models.
        
        Extracts statistical features from sequences:
        - Mean across time dimension
        - Standard deviation across time dimension  
        - Last timestep values
        
        Args:
            X_seq: Sequential data of shape (n_samples, seq_len, n_features)
            
        Returns:
            Flattened features of shape (n_samples, n_features * 3)
        """
        try:
            if X_seq.ndim != 3:
                raise ValueError(f"Expected 3D input, got {X_seq.ndim}D")
            
            # Extract statistical features efficiently
            mean_features = X_seq.mean(axis=1)
            std_features = X_seq.std(axis=1)
            last_features = X_seq[:, -1, :]
            
            # Concatenate all features
            flattened = np.hstack([mean_features, std_features, last_features])
            
            logger.debug(f"Flattened sequences from {X_seq.shape} to {flattened.shape}")
            return flattened
            
        except Exception as e:
            logger.error(f"Failed to flatten sequences: {e}")
            raise
    
    def train_random_forest(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        X_val: np.ndarray, 
        y_val: np.ndarray,
        n_estimators: int = 100,
        max_depth: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Train Random Forest baseline model.
        
        Args:
            X_train: Training sequences
            y_train: Training labels
            X_val: Validation sequences
            y_val: Validation labels
            n_estimators: Number of trees
            max_depth: Maximum tree depth
            
        Returns:
            Dictionary containing model and metrics
        """
        try:
            self._validate_inputs(X_train, y_train)
            self._validate_inputs(X_val, y_val)
            
            # Flatten sequences
            X_train_flat = self.flatten_sequences(X_train)
            X_val_flat = self.flatten_sequences(X_val)
            
            # Train model
            rf_model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=self.random_state,
                n_jobs=-1,
                class_weight='balanced'  # Handle class imbalance
            )
            
            logger.info(f"Training Random Forest with {n_estimators} estimators")
            rf_model.fit(X_train_flat, y_train)
            
            # Evaluate
            val_probs = rf_model.predict_proba(X_val_flat)[:, 1]
            val_preds = (val_probs > 0.5).astype(int)
            
            # Compute metrics
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_val, val_preds, average='binary', zero_division=0
            )
            
            auc = roc_auc_score(y_val, val_probs) if len(np.unique(y_val)) > 1 else 0.0
            cm = confusion_matrix(y_val, val_preds)
            
            # Feature importance
            feature_importance = rf_model.feature_importances_
            
            results = {
                'model': rf_model,
                'model_type': 'RandomForest',
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'auc': float(auc),
                'confusion_matrix': cm.tolist(),
                'feature_importance': feature_importance.tolist(),
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'probabilities': val_probs,
                'predictions': val_preds
            }
            
            self.models['random_forest'] = results
            
            logger.info(f"Random Forest - F1: {f1:.4f}, AUC: {auc:.4f}")
            return results
            
        except Exception as e:
            logger.error(f"Random Forest training failed: {e}")
            raise
    
    def train_logistic_regression(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        X_val: np.ndarray, 
        y_val: np.ndarray,
        C: float = 1.0,
        max_iter: int = 1000
    ) -> Dict[str, Any]:
        """
        Train Logistic Regression baseline model.
        
        Args:
            X_train: Training sequences
            y_train: Training labels
            X_val: Validation sequences
            y_val: Validation labels
            C: Regularization strength
            max_iter: Maximum iterations
            
        Returns:
            Dictionary containing model and metrics
        """
        try:
            self._validate_inputs(X_train, y_train)
            self._validate_inputs(X_val, y_val)
            
            # Flatten sequences
            X_train_flat = self.flatten_sequences(X_train)
            X_val_flat = self.flatten_sequences(X_val)
            
            # Train model
            lr_model = LogisticRegression(
                C=C,
                max_iter=max_iter,
                random_state=self.random_state,
                class_weight='balanced',  # Handle class imbalance
                solver='liblinear'  # Good for small datasets
            )
            
            logger.info(f"Training Logistic Regression with C={C}")
            lr_model.fit(X_train_flat, y_train)
            
            # Evaluate
            val_probs = lr_model.predict_proba(X_val_flat)[:, 1]
            val_preds = (val_probs > 0.5).astype(int)
            
            # Compute metrics
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_val, val_preds, average='binary', zero_division=0
            )
            
            auc = roc_auc_score(y_val, val_probs) if len(np.unique(y_val)) > 1 else 0.0
            cm = confusion_matrix(y_val, val_preds)
            
            # Feature coefficients
            coefficients = lr_model.coef_[0] if lr_model.coef_.ndim > 1 else lr_model.coef_
            
            results = {
                'model': lr_model,
                'model_type': 'LogisticRegression',
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'auc': float(auc),
                'confusion_matrix': cm.tolist(),
                'coefficients': coefficients.tolist(),
                'C': C,
                'max_iter': max_iter,
                'probabilities': val_probs,
                'predictions': val_preds
            }
            
            self.models['logistic_regression'] = results
            
            logger.info(f"Logistic Regression - F1: {f1:.4f}, AUC: {auc:.4f}")
            return results
            
        except Exception as e:
            logger.error(f"Logistic Regression training failed: {e}")
            raise
    
    def evaluate_model(
        self, 
        model_name: str, 
        X_test: np.ndarray, 
        y_test: np.ndarray
    ) -> Dict[str, Any]:
        """
        Evaluate a trained baseline model on test data.
        
        Args:
            model_name: Name of the model to evaluate
            X_test: Test sequences
            y_test: Test labels
            
        Returns:
            Dictionary containing test metrics
        """
        try:
            if model_name not in self.models:
                raise ValueError(f"Model '{model_name}' not found. Available: {list(self.models.keys())}")
            
            self._validate_inputs(X_test, y_test)
            
            model_info = self.models[model_name]
            model = model_info['model']
            
            # Flatten test data
            X_test_flat = self.flatten_sequences(X_test)
            
            # Predict
            test_probs = model.predict_proba(X_test_flat)[:, 1]
            test_preds = (test_probs > 0.5).astype(int)
            
            # Compute metrics
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test, test_preds, average='binary', zero_division=0
            )
            
            auc = roc_auc_score(y_test, test_probs) if len(np.unique(y_test)) > 1 else 0.0
            cm = confusion_matrix(y_test, test_preds)
            
            # Additional metrics
            tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            
            test_results = {
                'model_type': model_info['model_type'],
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'auc': float(auc),
                'specificity': float(specificity),
                'confusion_matrix': cm.tolist(),
                'n_samples': len(y_test),
                'fraud_rate': float(np.mean(y_test)),
                'probabilities': test_probs.tolist(),
                'predictions': test_preds.tolist()
            }
            
            logger.info(f"{model_name} Test Results - F1: {f1:.4f}, AUC: {auc:.4f}")
            return test_results
            
        except Exception as e:
            logger.error(f"Model evaluation failed: {e}")
            raise
    
    def compare_models(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Dict[str, Any]]:
        """
        Compare all trained baseline models on test data.
        
        Args:
            X_test: Test sequences
            y_test: Test labels
            
        Returns:
            Dictionary containing results for all models
        """
        try:
            if not self.models:
                raise ValueError("No models trained yet")
            
            comparison_results = {}
            
            for model_name in self.models.keys():
                test_results = self.evaluate_model(model_name, X_test, y_test)
                comparison_results[model_name] = test_results
            
            # Find best model by F1 score
            best_model = max(
                comparison_results.keys(), 
                key=lambda x: comparison_results[x]['f1']
            )
            
            logger.info(f"Best baseline model: {best_model} (F1: {comparison_results[best_model]['f1']:.4f})")
            
            comparison_results['best_model'] = best_model
            return comparison_results
            
        except Exception as e:
            logger.error(f"Model comparison failed: {e}")
            raise