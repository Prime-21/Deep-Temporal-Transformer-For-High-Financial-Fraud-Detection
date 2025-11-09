"""Colab-optimized execution script."""
import torch
import numpy as np
from deep_temporal_transformer import (
    get_default_config, DataProcessor, ModelTrainer, 
    set_random_seeds, get_device
)

def run_colab_demo():
    """Run optimized demo for Colab Pro environment."""
    try:
        # Set seeds for reproducibility
        set_random_seeds(42)
        
        # Colab-optimized config
        config = get_default_config()
        config.training.epochs = 5  # Fast demo
        config.training.batch_size = 64 if torch.cuda.is_available() else 32
        config.data.n_samples = 20000  # Smaller dataset for Colab
        config.model.d_model = 128  # Smaller model
        config.model.num_layers = 3
        
        # Setup device
        device = get_device()
        print(f"🚀 Using device: {device}")
        
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Process data
        print("📊 Processing synthetic fraud data...")
        processor = DataProcessor(seq_len=4, random_state=42)  # Shorter sequences
        X_train, y_train, X_val, y_val, X_test, y_test = processor.process_data()
        
        print(f"Data shapes - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        print(f"Fraud rates - Train: {np.mean(y_train):.1%}, Val: {np.mean(y_val):.1%}, Test: {np.mean(y_test):.1%}")
        
        # Train model
        print("🤖 Training Deep Temporal Transformer...")
        trainer = ModelTrainer(config, device)
        trainer.setup_model(input_dim=X_train.shape[-1])
        
        # Quick training
        history = trainer.train(X_train, y_train, X_val, y_val)
        
        # Evaluate
        print("📈 Evaluating model...")
        results = trainer.evaluate_model(X_test, y_test)
        
        # Display results
        print("\n" + "="*50)
        print("🎯 COLAB PRO DEMO RESULTS")
        print("="*50)
        print(f"F1 Score: {results['f1']:.4f}")
        print(f"AUC Score: {results['auc']:.4f}")
        print(f"Precision: {results['precision']:.4f}")
        print(f"Recall: {results['recall']:.4f}")
        print(f"Avg Inference Time: {results['avg_inference_time']:.6f}s per transaction")
        print("✅ Demo completed successfully!")
        
        return results
        
    except Exception as e:
        print(f"❌ Error in Colab demo: {e}")
        # Fallback to simple demo
        return {"f1": 0.0, "auc": 0.0, "error": str(e)}

if __name__ == "__main__":
    run_colab_demo()