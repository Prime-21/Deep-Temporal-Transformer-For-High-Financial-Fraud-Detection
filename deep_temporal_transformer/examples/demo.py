"""Demo script for Deep Temporal Transformer fraud detection."""
import logging
from deep_temporal_transformer import (
    get_default_config,
    DataProcessor,
    ModelTrainer,
    EnhancedBaselineModels,
    set_random_seeds,
    get_device
)
from deep_temporal_transformer.evaluation.explain import ModelExplainer
from deep_temporal_transformer.utils.utils import setup_logging

def run_demo():
    """Run a complete demo of the fraud detection system."""
    
    # Setup logging
    logger = setup_logging()
    logger.info("Starting Deep Temporal Transformer Demo")
    
    try:
        # Set random seeds for reproducibility
        set_random_seeds(42)
        
        # Get configuration
        config = get_default_config()
        config.training.epochs = 5  # Quick demo
        config.training.batch_size = 128
        
        # Setup device
        device = get_device()
        logger.info(f"Using device: {device}")
        
        # Process data (will generate synthetic data)
        logger.info("Processing data...")
        data_processor = DataProcessor(seq_len=8, random_state=42)
        X_train, y_train, X_val, y_val, X_test, y_test = data_processor.process_data()
        
        logger.info(f"Data shapes - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        
        # Train baseline models
        logger.info("Training baseline models...")
        baseline_models = EnhancedBaselineModels(random_state=42)
        
        rf_results = baseline_models.train_random_forest(
            X_train, y_train, X_val, y_val, n_estimators=50
        )
        
        lr_results = baseline_models.train_logistic_regression(
            X_train, y_train, X_val, y_val
        )
        
        # Train Deep Temporal Transformer
        logger.info("Training Deep Temporal Transformer...")
        trainer = ModelTrainer(config, device)
        trainer.setup_model(input_dim=X_train.shape[-1])
        
        # Train model
        history = trainer.train(X_train, y_train, X_val, y_val)
        
        # Evaluate all models on test set
        logger.info("Evaluating models...")
        
        # Transformer results
        transformer_results = trainer.evaluate_model(X_test, y_test)
        
        # Baseline results
        baseline_comparison = baseline_models.compare_models(X_test, y_test)
        
        # Print results
        print("\n" + "="*60)
        print("FRAUD DETECTION DEMO RESULTS")
        print("="*60)
        
        print(f"\nRandom Forest:")
        rf_test = baseline_comparison['random_forest']
        print(f"  F1: {rf_test['f1']:.4f}, AUC: {rf_test['auc']:.4f}")
        print(f"  Precision: {rf_test['precision']:.4f}, Recall: {rf_test['recall']:.4f}")
        
        print(f"\nLogistic Regression:")
        lr_test = baseline_comparison['logistic_regression']
        print(f"  F1: {lr_test['f1']:.4f}, AUC: {lr_test['auc']:.4f}")
        print(f"  Precision: {lr_test['precision']:.4f}, Recall: {lr_test['recall']:.4f}")
        
        print(f"\nDeep Temporal Transformer:")
        print(f"  F1: {transformer_results['f1']:.4f}, AUC: {transformer_results['auc']:.4f}")
        print(f"  Precision: {transformer_results['precision']:.4f}, Recall: {transformer_results['recall']:.4f}")
        print(f"  Avg Inference Time: {transformer_results['avg_inference_time']:.6f}s per transaction")
        
        # Model interpretability
        logger.info("Generating model explanations...")
        explainer = ModelExplainer(trainer.model, device)
        
        # Get sample explanations
        explanations = explainer.get_sample_explanations(X_test[:100], y_test[:100], n_samples=3)
        
        print(f"\nSample Predictions (first 3):")
        for i, exp in enumerate(explanations['explanations']):
            print(f"  Sample {i+1}: True={exp['true_label']}, Pred={exp['predicted_label']}, "
                  f"Prob={exp['prediction_probability']:.4f}")
        
        print("\n" + "="*60)
        print("Demo completed successfully!")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise

if __name__ == "__main__":
    run_demo()