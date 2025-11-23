"""
Generate All Thesis Results
One-click script to run experiments and generate all thesis-ready outputs.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deep_temporal_transformer import (
    get_default_config, DataProcessor, ModelTrainer,
    set_random_seeds, get_device
)
from deep_temporal_transformer.evaluation.statistical_analysis import (
    StatisticalAnalyzer, format_result_with_ci
)
from deep_temporal_transformer.evaluation.thesis_results import ThesisResultsExporter
from deep_temporal_transformer.utils.simple_tracker import SimpleExperimentTracker
import numpy as np
import torch


def run_full_experiment():
    """Run complete experiment and generate all thesis results."""
    
    print("="*70)
    print("🎓 THESIS RESULTS GENERATION")
    print("="*70)
    
    # Setup
    set_random_seeds(42)
    config = get_default_config()
    device = get_device()
    
    # Initialize tools
    tracker = SimpleExperimentTracker()
    analyzer = StatisticalAnalyzer(confidence_level=0.95)
    exporter = ThesisResultsExporter()
    
    # Start experiment tracking
    exp_id = tracker.start_experiment(
        name="final_thesis_experiment",
        description="Final experiment for thesis submission",
        config=config.__dict__
    )
    
    print(f"\n📊 Starting Experiment: {exp_id}")
    print(f"Device: {device}")
    
    # Process data
    print("\n1️⃣ Loading and processing data...")
    processor = DataProcessor(seq_len=config.data.seq_len, random_state=42)
    X_train, y_train, X_val, y_val, X_test, y_test = processor.process_data()
    
    print(f"   Train: {X_train.shape}, Fraud rate: {y_train.mean():.2%}")
    print(f"   Test:  {X_test.shape}, Fraud rate: {y_test.mean():.2%}")
    
    # Train model
    print("\n2️⃣ Training Deep Temporal Transformer...")
    trainer = ModelTrainer(config, device)
    trainer.setup_model(input_dim=X_train.shape[-1])
    
    print(f"   Model parameters: {sum(p.numel() for p in trainer.model.parameters()):,}")
    
    history = trainer.train(X_train, y_train, X_val, y_val)
    
    # Log training metrics
    for epoch, val_metrics in enumerate(history['val_metrics']):
        tracker.log_metrics(val_metrics, step=epoch+1)
    
    # Evaluate
    print("\n3️⃣ Evaluating on test set...")
    test_results = trainer.evaluate_model(X_test, y_test)
    
    print(f"\n   📊 Test Results:")
    print(f"   F1:        {test_results['f1']:.4f}")
    print(f"   AUC:       {test_results['auc']:.4f}")
    print(f"   Precision: {test_results['precision']:.4f}")
    print(f"   Recall:    {test_results['recall']:.4f}")
    
    # Get predictions for statistical analysis
    trainer.model.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32, device=device)
        logits, _, _ = trainer.model(X_test_tensor)
        y_prob = torch.sigmoid(logits).cpu().numpy()
        y_pred = (y_prob > 0.5).astype(int)
    
    # Statistical analysis with confidence intervals
    print("\n4️⃣ Calculating confidence intervals...")
    comprehensive_results = analyzer.comprehensive_evaluation(
        y_test, y_pred, y_prob, n_bootstrap=1000
    )
    
    print("\n   📈 Results with 95% Confidence Intervals:")
    for metric, values in comprehensive_results.items():
        print(f"   {metric.upper():10s}: {values['ci_string']}")
    
    # End experiment tracking
    tracker.end_experiment(final_results=test_results)
    
    # Generate outputs
    print("\n5️⃣ Generating thesis-ready outputs...")
    
    # Prepare results for export
    all_results = {
        'Deep Temporal Transformer': {
            'f1': comprehensive_results['f1'],
            'auc': comprehensive_results['auc'],
            'precision': comprehensive_results['precision'],
            'recall': comprehensive_results['recall']
        }
    }
    
    # Generate LaTeX table
    latex_table = exporter.generate_comparison_table_latex(
        all_results,
        caption="Deep Temporal Transformer Performance (with 95\\% Confidence Intervals)",
        label="tab:dtt_performance"
    )
    
    # Save all formats
    saved_files = exporter.save_all_formats(all_results, prefix="final_thesis")
    
    print(f"\n   ✅ Saved files:")
    for format_type, filepath in saved_files.items():
        print(f"      {format_type.upper():6s}: {filepath}")
    
    # Create thesis summary
    print("\n6️⃣ Creating thesis summary...")
    summary = exporter.create_thesis_summary(
        transformer_results=test_results,
        baseline_results={},  # Add baselines if available
        additional_info={
            'model_parameters': sum(p.numel() for p in trainer.model.parameters()),
            'training_epochs': config.training.epochs,
            'batch_size': config.training.batch_size,
            'device': str(device)
        }
    )
    
    summary_file = exporter.output_dir / "THESIS_SUMMARY.md"
    with open(summary_file, 'w') as f:
        f.write(summary)
    
    print(f"   ✅ Summary: {summary_file}")
    
    # Print LaTeX table for easy copy-paste
    print("\n" + "="*70)
    print("📋 LATEX TABLE (copy to thesis):")
    print("="*70)
    print(latex_table)
    
    print("\n" + "="*70)
    print("✅ THESIS RESULTS GENERATION COMPLETE!")
    print("="*70)
    print(f"\nAll results saved in: {exporter.output_dir}")
    print("\nNext steps:")
    print("  1. Copy LaTeX table to your thesis")
    print("  2. Include CSV file in appendix")
    print("  3. Reference experiment ID in methodology section")
    print("\n🎓 Good luck with your thesis submission!")


if __name__ == "__main__":
    run_full_experiment()
