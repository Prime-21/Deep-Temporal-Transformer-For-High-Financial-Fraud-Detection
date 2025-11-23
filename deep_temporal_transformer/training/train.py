"""Training module for Deep Temporal Transformer."""
import os
import time
import logging
from typing import Dict, Any, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn

# Set PyTorch seeds for reproducibility
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
from sklearn.metrics import (
    precision_recall_fscore_support, 
    roc_auc_score, 
    confusion_matrix,
    classification_report
)

try:
    from ..models.model_enhanced import DeepTemporalTransformerEnhanced, FocalLossEnhanced
    from ..configs.config import Config
    from ..utils.utils import setup_logging, ensure_dir, save_json, EarlyStopping
    logger = setup_logging()
except:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


class ModelTrainer:
    """Secure and robust trainer for Deep Temporal Transformer."""
    
    def __init__(self, config: Config, device: torch.device):
        self.config = config
        self.device = device
        self.model: Optional[DeepTemporalTransformer] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.criterion: Optional[nn.Module] = None
        self.early_stopping: Optional[EarlyStopping] = None
        
    def setup_model(self, input_dim: int) -> None:
        """Initialize model with proper configuration."""
        try:
            self.model = DeepTemporalTransformerEnhanced(
                input_dim=input_dim,
                seq_len=self.config.model.seq_len,
                d_model=self.config.model.d_model,
                nhead=self.config.model.nhead,
                num_layers=self.config.model.num_layers,
                dim_feedforward=self.config.model.dim_feedforward,
                memory_slots=self.config.model.memory_slots,
                dropout=self.config.model.dropout,
                emb_dims=self.config.model.emb_dims
            ).to(self.device)
            
            # Setup optimizer
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay
            )
            
            # Setup loss function
            self.criterion = FocalLossEnhanced(
                alpha=self.config.training.focal_alpha,
                gamma=self.config.training.focal_gamma
            )
            
            # Setup early stopping
            self.early_stopping = EarlyStopping(
                patience=self.config.training.patience
            )
            
            logger.info(f"Model setup complete. Parameters: {self._count_parameters():,}")
            
        except Exception as e:
            logger.error(f"Failed to setup model: {e}")
            raise
    
    def _count_parameters(self) -> int:
        """Count trainable parameters."""
        if self.model is None:
            return 0
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def _validate_inputs(self, X: np.ndarray, y: np.ndarray) -> None:
        """Validate input data."""
        if len(X) != len(y):
            raise ValueError(f"X and y length mismatch: {len(X)} vs {len(y)}")
        
        if len(X) == 0:
            raise ValueError("Empty dataset provided")
        
        if X.ndim != 3:
            raise ValueError(f"X must be 3D array, got {X.ndim}D")
        
        if not np.isfinite(X).all():
            raise ValueError("X contains non-finite values")
        
        if not np.isin(y, [0, 1]).all():
            raise ValueError("y must contain only 0 and 1 values")
    
    def evaluate_model(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        batch_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Evaluate model performance with comprehensive metrics.
        
        Args:
            X: Input sequences (n_samples, seq_len, n_features)
            y: Target labels (n_samples,)
            batch_size: Batch size for evaluation
            
        Returns:
            Dictionary containing evaluation metrics
        """
        try:
            self._validate_inputs(X, y)
            
            if self.model is None:
                raise RuntimeError("Model not initialized")
            
            batch_size = batch_size or self.config.training.batch_size
            self.model.eval()
            
            all_probs = []
            all_preds = []
            inference_times = []
            
            with torch.no_grad():
                n_samples = len(X)
                for i in range(0, n_samples, batch_size):
                    batch_X = X[i:i + batch_size]
                    
                    # Convert to tensor
                    batch_tensor = torch.tensor(
                        batch_X, 
                        dtype=torch.float32, 
                        device=self.device
                    )
                    
                    # Measure inference time
                    start_time = time.time()
                    logits, _ = self.model(batch_tensor)
                    inference_time = (time.time() - start_time) / len(batch_X)
                    inference_times.append(inference_time)
                    
                    # Get predictions efficiently
                    with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                        probs = torch.sigmoid(logits).cpu().numpy()
                    preds = (probs > 0.5).astype(np.int8)
                    
                    all_probs.extend(probs)
                    all_preds.extend(preds)
            
            # Convert to numpy arrays
            all_probs = np.array(all_probs)
            all_preds = np.array(all_preds)
            
            # Compute metrics
            precision, recall, f1, _ = precision_recall_fscore_support(
                y, all_preds, average='binary', zero_division=0
            )
            
            auc = roc_auc_score(y, all_probs) if len(np.unique(y)) > 1 else 0.0
            cm = confusion_matrix(y, all_preds)
            
            # Additional metrics
            tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            
            metrics = {
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'auc': float(auc),
                'specificity': float(specificity),
                'confusion_matrix': cm.tolist(),
                'avg_inference_time': float(np.mean(inference_times)),
                'n_samples': len(y),
                'fraud_rate': float(np.mean(y))
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise
    
    def train_epoch(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray, 
        batch_size: int
    ) -> float:
        """Train for one epoch."""
        try:
            if self.model is None or self.optimizer is None or self.criterion is None:
                raise RuntimeError("Model components not initialized")
            
            self.model.train()
            epoch_losses = []
            
            # Efficient data shuffling
            indices = np.random.permutation(len(X_train))
            
            for i in range(0, len(X_train), batch_size):
                end_idx = min(i + batch_size, len(X_train))
                batch_indices = indices[i:end_idx]
                batch_X = X_train[batch_indices]
                batch_y = y_train[batch_indices]
                
                # Convert to tensors
                batch_X_tensor = torch.tensor(
                    batch_X, 
                    dtype=torch.float32, 
                    device=self.device
                )
                batch_y_tensor = torch.tensor(
                    batch_y, 
                    dtype=torch.float32, 
                    device=self.device
                )
                
                # Forward pass
                self.optimizer.zero_grad()
                logits, _ = self.model(batch_X_tensor)
                loss = self.criterion(logits, batch_y_tensor)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                epoch_losses.append(loss.item())
            
            return float(np.mean(epoch_losses))
            
        except Exception as e:
            logger.error(f"Training epoch failed: {e}")
            raise
    
    def train(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        X_val: np.ndarray, 
        y_val: np.ndarray
    ) -> Dict[str, Any]:
        """
        Train the model with early stopping and comprehensive logging.
        
        Args:
            X_train: Training sequences
            y_train: Training labels
            X_val: Validation sequences
            y_val: Validation labels
            
        Returns:
            Training history and final metrics
        """
        try:
            # Validate inputs
            self._validate_inputs(X_train, y_train)
            self._validate_inputs(X_val, y_val)
            
            if self.model is None:
                raise RuntimeError("Model not initialized")
            
            # Setup output directory
            ensure_dir(self.config.output_dir)
            
            # Training history
            history = {
                'train_loss': [],
                'val_metrics': [],
                'best_epoch': 0,
                'best_f1': 0.0
            }
            
            best_model_state = None
            
            logger.info(f"Starting training for {self.config.training.epochs} epochs")
            
            for epoch in range(1, self.config.training.epochs + 1):
                # Train epoch
                train_loss = self.train_epoch(
                    X_train, y_train, self.config.training.batch_size
                )
                
                # Validate
                val_metrics = self.evaluate_model(X_val, y_val)
                
                # Update history
                history['train_loss'].append(train_loss)
                history['val_metrics'].append(val_metrics)
                
                # Log progress
                logger.info(
                    f"Epoch {epoch:3d}: "
                    f"train_loss={train_loss:.4f}, "
                    f"val_f1={val_metrics['f1']:.4f}, "
                    f"val_auc={val_metrics['auc']:.4f}"
                )
                
                # Save best model
                if val_metrics['f1'] > history['best_f1']:
                    history['best_f1'] = val_metrics['f1']
                    history['best_epoch'] = epoch
                    best_model_state = self.model.state_dict().copy()
                    
                    # Save checkpoint
                    from .security_fixes import validate_path
                    checkpoint_path = validate_path(os.path.join(self.config.output_dir, 'best_model.pt'), ['.pt', '.pth'])
                    torch.save(best_model_state, checkpoint_path)
                
                # Early stopping check
                if self.early_stopping(val_metrics['f1']):
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
            
            # Load best model
            if best_model_state is not None:
                self.model.load_state_dict(best_model_state)
                logger.info(f"Loaded best model from epoch {history['best_epoch']}")
            
            # Save training history
            from .security_fixes import validate_path
            history_path = validate_path(os.path.join(self.config.output_dir, 'training_history.json'), ['.json'])
            save_json(history, history_path)
            
            return history
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def save_model(self, path: str) -> None:
        """Save model state safely."""
        try:
            if self.model is None:
                raise RuntimeError("No model to save")
            
            # Validate path
            from .security_fixes import validate_path
            normalized_path = validate_path(path, ['.pt', '.pth'])
            
            ensure_dir(os.path.dirname(normalized_path))
            
            # Save model state
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'config': self.config,
                'model_class': 'DeepTemporalTransformer'
            }, normalized_path)
            
            logger.info(f"Model saved to {normalized_path}")
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise
    
    def load_model(self, path: str, input_dim: int) -> None:
        """Load model state safely."""
        try:
            # Validate path
            from .security_fixes import validate_path
            normalized_path = validate_path(path, ['.pt', '.pth'])
            if not os.path.exists(normalized_path):
                raise FileNotFoundError(f"Model file not found: {normalized_path}")
            
            # Load checkpoint
            checkpoint = torch.load(normalized_path, map_location=self.device)
            
            # Setup model if not already done
            if self.model is None:
                self.setup_model(input_dim)
            
            # Load state
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            logger.info(f"Model loaded from {normalized_path}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise