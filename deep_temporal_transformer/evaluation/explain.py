"""Model interpretability and explanation module."""
import os
import logging
from typing import Optional, Dict, Any, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch

# Set PyTorch seeds for reproducibility
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

try:
    from ..utils.utils import setup_logging, ensure_dir
    logger = setup_logging()
except:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


class ModelExplainer:
    """Model interpretability and visualization tools."""
    
    def __init__(self, model, device: torch.device):
        self.model = model
        self.device = device
        
    def plot_attention_weights(
        self, 
        attention_weights: np.ndarray, 
        output_path: Optional[str] = None,
        title: str = "Memory Attention Weights"
    ) -> None:
        """
        Plot memory attention weights for interpretability.
        
        Args:
            attention_weights: Attention weights array (batch_size, memory_slots)
            output_path: Path to save the plot
            title: Plot title
        """
        try:
            # Average across batch dimension
            avg_attention = np.mean(attention_weights, axis=0)
            
            # Create figure
            plt.figure(figsize=(12, 6))
            
            # Plot attention weights
            plt.subplot(1, 2, 1)
            plt.bar(range(len(avg_attention)), avg_attention)
            plt.title(f'{title} (Average)')
            plt.xlabel('Memory Slot')
            plt.ylabel('Attention Weight')
            plt.grid(True, alpha=0.3)
            
            # Plot attention heatmap for top samples
            plt.subplot(1, 2, 2)
            n_samples = min(20, attention_weights.shape[0])
            sns.heatmap(
                attention_weights[:n_samples], 
                cmap='viridis',
                cbar=True,
                xticklabels=False,
                yticklabels=range(n_samples)
            )
            plt.title(f'{title} (Sample Heatmap)')
            plt.xlabel('Memory Slot')
            plt.ylabel('Sample')
            
            plt.tight_layout()
            
            if output_path:
                from ..utils.security_fixes import validate_path
                normalized_path = validate_path(output_path, ['.png', '.jpg', '.pdf'])
                ensure_dir(os.path.dirname(normalized_path))
                plt.savefig(normalized_path, dpi=300, bbox_inches='tight')
                logger.info(f"Attention plot saved to {normalized_path}")
                plt.close()
            else:
                plt.show()
                
        except Exception as e:
            logger.error(f"Failed to plot attention weights: {e}")
            raise
    
    def plot_feature_importance(
        self, 
        feature_names: list, 
        importance_scores: np.ndarray,
        output_path: Optional[str] = None,
        title: str = "Feature Importance",
        top_k: int = 20
    ) -> None:
        """
        Plot feature importance scores.
        
        Args:
            feature_names: List of feature names
            importance_scores: Importance scores array
            output_path: Path to save the plot
            title: Plot title
            top_k: Number of top features to display
        """
        try:
            if len(feature_names) != len(importance_scores):
                raise ValueError("Feature names and scores length mismatch")
            
            # Sort by importance
            sorted_indices = np.argsort(importance_scores)[::-1]
            top_indices = sorted_indices[:top_k]
            
            top_features = [feature_names[i] for i in top_indices]
            top_scores = importance_scores[top_indices]
            
            # Create plot
            plt.figure(figsize=(10, 8))
            bars = plt.barh(range(len(top_features)), top_scores)
            plt.yticks(range(len(top_features)), top_features)
            plt.xlabel('Importance Score')
            plt.title(title)
            plt.gca().invert_yaxis()
            
            # Add value labels on bars
            for i, (bar, score) in enumerate(zip(bars, top_scores)):
                plt.text(score + 0.001, i, f'{score:.3f}', 
                        va='center', ha='left', fontsize=9)
            
            plt.grid(True, alpha=0.3, axis='x')
            plt.tight_layout()
            
            if output_path:
                from ..utils.security_fixes import validate_path
                normalized_path = validate_path(output_path, ['.png', '.jpg', '.pdf'])
                ensure_dir(os.path.dirname(normalized_path))
                plt.savefig(normalized_path, dpi=300, bbox_inches='tight')
                logger.info(f"Feature importance plot saved to {normalized_path}")
                plt.close()
            else:
                plt.show()
                
        except Exception as e:
            logger.error(f"Failed to plot feature importance: {e}")
            raise
    
    def plot_confusion_matrix(
        self, 
        X_or_cm: np.ndarray,
        y: Optional[np.ndarray] = None,
        class_names: list = ['Normal', 'Fraud'],
        output_path: Optional[str] = None,
        title: str = "Confusion Matrix"
    ) -> None:
        """
        Plot confusion matrix with proper formatting.
        
        Args:
            X_or_cm: Either a 2x2 confusion matrix OR input data X (will compute CM)
            y: True labels (required if X_or_cm is input data)
            class_names: List of class names
            output_path: Path to save the plot
            title: Plot title
        """
        try:
            # Determine if we got a confusion matrix or need to compute it
            if X_or_cm.ndim == 2 and X_or_cm.shape[0] == 2 and X_or_cm.shape[1] == 2:
                # Already a confusion matrix
                confusion_matrix = X_or_cm
            elif y is not None:
                # Compute confusion matrix from X and y
                from sklearn.metrics import confusion_matrix as compute_cm
                
                # Get predictions
                self.model.eval()
                with torch.no_grad():
                    X_tensor = torch.tensor(X_or_cm, dtype=torch.float32, device=self.device)
                    logits, _, _ = self.model(X_tensor)
                    probs = torch.sigmoid(logits).cpu().numpy()
                    y_pred = (probs > 0.5).astype(int)
                
                confusion_matrix = compute_cm(y, y_pred)
            else:
                raise ValueError("Either provide a 2x2 confusion matrix or both X and y")
            
            plt.figure(figsize=(8, 6))
            
            # Create heatmap
            sns.heatmap(
                confusion_matrix, 
                annot=True, 
                fmt='d',
                cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                cbar=True
            )
            
            plt.title(title)
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            
            # Add percentage annotations
            total = np.sum(confusion_matrix)
            for i in range(len(class_names)):
                for j in range(len(class_names)):
                    percentage = confusion_matrix[i, j] / total * 100
                    plt.text(j + 0.5, i + 0.7, f'({percentage:.1f}%)', 
                            ha='center', va='center', fontsize=10, color='gray')
            
            plt.tight_layout()
            
            if output_path:
                from ..utils.security_fixes import validate_path
                normalized_path = validate_path(output_path, ['.png', '.jpg', '.pdf'])
                ensure_dir(os.path.dirname(normalized_path))
                plt.savefig(normalized_path, dpi=300, bbox_inches='tight')
                logger.info(f"Confusion matrix plot saved to {normalized_path}")
                plt.close()
            else:
                plt.show()
                
        except Exception as e:
            logger.error(f"Failed to plot confusion matrix: {e}")
            raise
    
    def plot_training_history(
        self, 
        history: Dict[str, Any],
        output_path: Optional[str] = None
    ) -> None:
        """
        Plot training history metrics.
        
        Args:
            history: Training history dictionary
            output_path: Path to save the plot
        """
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            epochs = range(1, len(history['train_loss']) + 1)
            
            # Training loss
            axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Training Loss')
            axes[0, 0].set_title('Training Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].legend()
            
            # Validation metrics
            val_f1 = [m['f1'] for m in history['val_metrics']]
            val_auc = [m['auc'] for m in history['val_metrics']]
            val_precision = [m['precision'] for m in history['val_metrics']]
            val_recall = [m['recall'] for m in history['val_metrics']]
            
            # F1 and AUC
            axes[0, 1].plot(epochs, val_f1, 'g-', label='F1 Score')
            axes[0, 1].plot(epochs, val_auc, 'r-', label='AUC')
            axes[0, 1].set_title('Validation F1 & AUC')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Score')
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].legend()
            
            # Precision and Recall
            axes[1, 0].plot(epochs, val_precision, 'b-', label='Precision')
            axes[1, 0].plot(epochs, val_recall, 'orange', label='Recall')
            axes[1, 0].set_title('Validation Precision & Recall')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Score')
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].legend()
            
            # Best epoch indicator
            best_epoch = history.get('best_epoch', 0)
            best_f1 = history.get('best_f1', 0)
            
            axes[1, 1].text(0.1, 0.8, f'Best Epoch: {best_epoch}', 
                           transform=axes[1, 1].transAxes, fontsize=14)
            axes[1, 1].text(0.1, 0.6, f'Best F1: {best_f1:.4f}', 
                           transform=axes[1, 1].transAxes, fontsize=14)
            axes[1, 1].text(0.1, 0.4, f'Total Epochs: {len(epochs)}', 
                           transform=axes[1, 1].transAxes, fontsize=14)
            axes[1, 1].set_title('Training Summary')
            axes[1, 1].axis('off')
            
            plt.tight_layout()
            
            if output_path:
                from ..utils.security_fixes import validate_path
                normalized_path = validate_path(output_path, ['.png', '.jpg', '.pdf'])
                ensure_dir(os.path.dirname(normalized_path))
                plt.savefig(normalized_path, dpi=300, bbox_inches='tight')
                logger.info(f"Training history plot saved to {normalized_path}")
                plt.close()
            else:
                plt.show()
                
        except Exception as e:
            logger.error(f"Failed to plot training history: {e}")
            raise
    
    def get_sample_explanations(
        self, 
        X_sample: np.ndarray, 
        y_sample: np.ndarray,
        n_samples: int = 5
    ) -> Dict[str, Any]:
        """
        Get explanations for sample predictions.
        
        Args:
            X_sample: Sample input sequences
            y_sample: Sample true labels
            n_samples: Number of samples to explain
            
        Returns:
            Dictionary containing explanations
        """
        try:
            if self.model is None:
                raise RuntimeError("Model not provided")
            
            torch.manual_seed(42)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(42)
            self.model.eval()
            
            # Select random samples
            indices = np.random.choice(len(X_sample), size=min(n_samples, len(X_sample)), replace=False)
            
            explanations = []
            
            with torch.no_grad():
                for idx in indices:
                    sample_x = X_sample[idx:idx+1]  # Keep batch dimension
                    sample_y = y_sample[idx]
                    
                    # Convert to tensor
                    sample_tensor = torch.tensor(sample_x, dtype=torch.float32, device=self.device)
                    
                    # Get prediction and attention
                    logits, attention = self.model(sample_tensor)
                    prob = torch.sigmoid(logits).cpu().numpy()[0]
                    pred = int(prob > 0.5)
                    
                    explanation = {
                        'sample_idx': int(idx),
                        'true_label': int(sample_y),
                        'predicted_label': pred,
                        'prediction_probability': float(prob),
                        'attention_weights': attention.cpu().numpy()[0].tolist(),
                        'input_sequence': sample_x[0].tolist()
                    }
                    
                    explanations.append(explanation)
            
            return {
                'explanations': explanations,
                'n_samples': len(explanations)
            }
            
        except Exception as e:
            logger.error(f"Failed to generate sample explanations: {e}")
            raise