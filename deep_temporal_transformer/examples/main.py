"""Main execution script for Deep Temporal Transformer fraud detection."""
import os
import sys
import argparse
import logging
from typing import Optional

import torch
import numpy as np

# Set PyTorch seeds for reproducibility
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

from .config import get_default_config, Config
from ..utils.utils import set_random_seeds, get_device, setup_logging, save_json
from .data import DataProcessor
from .model import DeepTemporalTransformer
from .train import ModelTrainer
from .baseline import BaselineModels
from .explain import ModelExplainer

logger = setup_logging()


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Deep Temporal Transformer for Financial Fraud Detection')
    
    parser.add_argument('--data-path', type=str, default=None,
                       help='Path to credit card dataset CSV file')
    parser.add_argument('--output-dir', type=str, default='outputs',
                       help='Output directory for results')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=256,
                       help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--seq-len', type=int, default=8,
                       help='Sequence length for temporal modeling')
    parser.add_argument('--d-model', type=int, default=256,
                       help='Model dimension')
    parser.add_argument('--num-layers', type=int, default=6,
                       help='Number of transformer layers')
    parser.add_argument('--random-seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--run-baselines', action='store_true',
                       help='Run baseline models for comparison')
    parser.add_argument('--generate-plots', action='store_true',
                       help='Generate visualization plots')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda', 'mps'],
                       help='Device to use for training')
    
    return parser.parse_args()


def setup_device(device_arg: str) -> torch.device:
    """Setup computation device."""
    if device_arg == 'auto':
        device = get_device()
    else:
        device = torch.device(device_arg)
    
    logger.info(f"Using device: {device}")
    return device


def run_baseline_comparison(
    X_train: np.ndarray, 
    y_train: np.ndarray,
    X_val: np.ndarray, 
    y_val: np.ndarray,
    X_test: np.ndarray, 
    y_test: np.ndarray,
    output_dir: str
) -> dict:
    """Run baseline model comparison."""
    try:
        logger.info("Running baseline model comparison...")
        
        baseline_models = BaselineModels(random_state=42)
        
        # Train Random Forest
        rf_results = baseline_models.train_random_forest(
            X_train, y_train, X_val, y_val,
            n_estimators=100, max_depth=20
        )
        
        # Train Logistic Regression
        lr_results = baseline_models.train_logistic_regression(
            X_train, y_train, X_val, y_val,
            C=1.0, max_iter=1000
        )
        
        # Compare on test set
        comparison_results = baseline_models.compare_models(X_test, y_test)
        
        # Save results
        from ..utils.security_fixes import validate_path
        baseline_results_path = validate_path(os.path.join(output_dir, 'baseline_results.json'), ['.json'])
        save_json(comparison_results, baseline_results_path)
        
        logger.info("Baseline comparison completed")
        return comparison_results
        
    except Exception as e:
        logger.error(f"Baseline comparison failed: {e}")
        raise


def run_deep_temporal_transformer(
    config: Config,
    X_train: np.ndarray, 
    y_train: np.ndarray,
    X_val: np.ndarray, 
    y_val: np.ndarray,
    X_test: np.ndarray, 
    y_test: np.ndarray,
    device: torch.device
) -> dict:
    """Train and evaluate Deep Temporal Transformer."""
    try:
        logger.info("Training Deep Temporal Transformer...")
        
        # Initialize trainer
        trainer = ModelTrainer(config, device)
        
        # Setup model
        input_dim = X_train.shape[-1]
        trainer.setup_model(input_dim)
        
        # Train model
        training_history = trainer.train(X_train, y_train, X_val, y_val)
        
        # Evaluate on test set
        test_results = trainer.evaluate_model(X_test, y_test)
        
        # Save model
        from ..utils.security_fixes import validate_path
        model_path = validate_path(os.path.join(config.output_dir, 'deep_temporal_transformer.pt'), ['.pt', '.pth'])
        trainer.save_model(model_path)
        
        # Combine results
        results = {
            'training_history': training_history,
            'test_results': test_results,
            'model_path': model_path
        }
        
        # Save results
        results_path = validate_path(os.path.join(config.output_dir, 'transformer_results.json'), ['.json'])
        save_json(results, results_path)
        
        logger.info(f"Deep Temporal Transformer - Test F1: {test_results['f1']:.4f}, AUC: {test_results['auc']:.4f}")
        return results
        
    except Exception as e:
        logger.error(f"Deep Temporal Transformer training failed: {e}")
        raise


def generate_visualizations(
    transformer_results: dict,
    baseline_results: dict,
    config: Config,
    device: torch.device,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> None:
    """Generate visualization plots."""
    try:
        logger.info("Generating visualization plots...")
        
        # Load trained model for explanations
        model_path = transformer_results['model_path']
        model = DeepTemporalTransformerEnhanced(
            input_dim=X_test.shape[-1],
            seq_len=config.model.seq_len,
            d_model=config.model.d_model,
            nhead=config.model.nhead,
            num_layers=config.model.num_layers,
            dim_feedforward=config.model.dim_feedforward,
            memory_slots=config.model.memory_slots,
            dropout=config.model.dropout,
            emb_dims=config.model.emb_dims
        ).to(device)
        
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        explainer = ModelExplainer(model, device)
        
        # Plot training history
        from ..utils.security_fixes import validate_path
        training_history = transformer_results['training_history']
        history_plot_path = validate_path(os.path.join(config.output_dir, 'training_history.png'), ['.png'])
        explainer.plot_training_history(training_history, history_plot_path)
        
        # Plot confusion matrix for transformer
        transformer_cm = np.array(transformer_results['test_results']['confusion_matrix'])
        cm_plot_path = validate_path(os.path.join(config.output_dir, 'transformer_confusion_matrix.png'), ['.png'])
        explainer.plot_confusion_matrix(transformer_cm, output_path=cm_plot_path, 
                                      title='Deep Temporal Transformer - Confusion Matrix')
        
        # Plot attention weights for sample predictions
        sample_indices = np.random.choice(len(X_test), size=100, replace=False)
        X_sample = X_test[sample_indices]
        
        with torch.no_grad():
            model.eval()
            sample_tensor = torch.tensor(X_sample, dtype=torch.float32, device=device)
            _, attention_weights = model(sample_tensor)
            attention_np = attention_weights.cpu().numpy()
        
        attention_plot_path = validate_path(os.path.join(config.output_dir, 'attention_weights.png'), ['.png'])
        explainer.plot_attention_weights(attention_np, attention_plot_path)
        
        # Compare model performance
        comparison_data = {
            'Deep Temporal Transformer': transformer_results['test_results'],
            'Random Forest': baseline_results.get('random_forest', {}),
            'Logistic Regression': baseline_results.get('logistic_regression', {})
        }
        
        # Create performance comparison plot
        import matplotlib.pyplot as plt
        
        models = list(comparison_data.keys())
        f1_scores = [comparison_data[model].get('f1', 0) for model in models]
        auc_scores = [comparison_data[model].get('auc', 0) for model in models]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # F1 scores
        bars1 = ax1.bar(models, f1_scores, color=['blue', 'green', 'orange'])
        ax1.set_title('F1 Score Comparison')
        ax1.set_ylabel('F1 Score')
        ax1.set_ylim(0, 1)
        for bar, score in zip(bars1, f1_scores):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{score:.3f}', ha='center', va='bottom')
        
        # AUC scores
        bars2 = ax2.bar(models, auc_scores, color=['blue', 'green', 'orange'])
        ax2.set_title('AUC Score Comparison')
        ax2.set_ylabel('AUC Score')
        ax2.set_ylim(0, 1)
        for bar, score in zip(bars2, auc_scores):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{score:.3f}', ha='center', va='bottom')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        comparison_plot_path = validate_path(os.path.join(config.output_dir, 'model_comparison.png'), ['.png'])
        plt.savefig(comparison_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Visualization plots generated successfully")
        
    except Exception as e:
        logger.error(f"Visualization generation failed: {e}")
        raise


def main():
    """Main execution function."""
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Setup logging
        logger.info("Starting Deep Temporal Transformer for Financial Fraud Detection")
        
        # Set random seeds for reproducibility
        set_random_seeds(args.random_seed)
        
        # Setup device
        device = setup_device(args.device)
        
        # Create configuration
        config = get_default_config()
        
        # Update config with command line arguments
        config.training.epochs = args.epochs
        config.training.batch_size = args.batch_size
        config.training.learning_rate = args.learning_rate
        config.model.seq_len = args.seq_len
        config.model.d_model = args.d_model
        config.model.num_layers = args.num_layers
        config.data.seq_len = args.seq_len
        config.output_dir = args.output_dir
        config.random_seed = args.random_seed
        config.device = str(device)
        
        # Process data
        logger.info("Processing data...")
        data_processor = DataProcessor(seq_len=config.data.seq_len, random_state=config.random_seed)
        X_train, y_train, X_val, y_val, X_test, y_test = data_processor.process_data(args.data_path)
        
        logger.info(f"Data shapes - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        logger.info(f"Fraud rates - Train: {np.mean(y_train):.4f}, Val: {np.mean(y_val):.4f}, Test: {np.mean(y_test):.4f}")
        
        # Update model input dimension
        config.model.input_dim = X_train.shape[-1]
        
        # Run baseline models if requested
        baseline_results = {}
        if args.run_baselines:
            baseline_results = run_baseline_comparison(
                X_train, y_train, X_val, y_val, X_test, y_test, config.output_dir
            )
        
        # Train Deep Temporal Transformer
        transformer_results = run_deep_temporal_transformer(
            config, X_train, y_train, X_val, y_val, X_test, y_test, device
        )
        
        # Generate visualizations if requested
        if args.generate_plots:
            generate_visualizations(
                transformer_results, baseline_results, config, device, X_test, y_test
            )
        
        # Save final configuration
        from ..utils.security_fixes import validate_path
        config_path = validate_path(os.path.join(config.output_dir, 'config.json'), ['.json'])
        save_json(config.__dict__, config_path)
        
        logger.info("Execution completed successfully!")
        
        # Print summary
        print("\n" + "="*60)
        print("FRAUD DETECTION RESULTS SUMMARY")
        print("="*60)
        
        if baseline_results:
            print("\nBaseline Models:")
            for model_name, results in baseline_results.items():
                if model_name != 'best_model' and isinstance(results, dict):
                    print(f"  {model_name}: F1={results.get('f1', 0):.4f}, AUC={results.get('auc', 0):.4f}")
        
        print(f"\nDeep Temporal Transformer:")
        test_results = transformer_results['test_results']
        print(f"  F1={test_results['f1']:.4f}, AUC={test_results['auc']:.4f}")
        print(f"  Precision={test_results['precision']:.4f}, Recall={test_results['recall']:.4f}")
        print(f"  Avg Inference Time: {test_results['avg_inference_time']:.6f}s per transaction")
        
        print(f"\nResults saved to: {config.output_dir}")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()