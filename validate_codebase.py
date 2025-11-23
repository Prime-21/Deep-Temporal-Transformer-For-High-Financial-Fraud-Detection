#!/usr/bin/env python
"""
Codebase Validation Script
Tests all imports and common patterns to ensure no errors
"""

import sys
import importlib

def test_imports():
    """Test all main imports"""
    print("="*60)
    print("🔍 TESTING IMPORTS")
    print("="*60)
    
    errors = []
    
    # Test main package imports
    try:
        from deep_temporal_transformer import (
            get_default_config, DataProcessor, ModelTrainer,
            set_random_seeds, get_device
        )
        print("✅ Main package imports")
    except Exception as e:
        errors.append(f"❌ Main package: {e}")
        print(f"❌ Main package imports: {e}")
    
    # Test model imports
    try:
        from deep_temporal_transformer.models.model_enhanced import (
            DeepTemporalTransformerEnhanced, FocalLossEnhanced
        )
        print("✅ Model imports")
    except Exception as e:
        errors.append(f"❌ Models: {e}")
        print(f"❌ Model imports: {e}")
    
    # Test advanced transformer
    try:
        from deep_temporal_transformer.models.advanced_transformer import (
            DeepTemporalTransformerAdvanced
        )
        print("✅ Advanced transformer imports")
    except Exception as e:
        errors.append(f"❌ Advanced transformer: {e}")
        print(f"❌ Advanced transformer imports: {e}")
    
    # Test baseline models
    try:
        from deep_temporal_transformer.models.baseline_enhanced import (
            EnhancedBaselineModels, LSTMBaseline, TemporalCNN
        )
        print("✅ Baseline model imports")
    except Exception as e:
        errors.append(f"❌ Baseline models: {e}")
        print(f"❌ Baseline model imports: {e}")
    
    # Test evaluation
    try:
        from deep_temporal_transformer.evaluation.explain import ModelExplainer
        print("✅ Evaluation imports")
    except Exception as e:
        errors.append(f"❌ Evaluation: {e}")
        print(f"❌ Evaluation imports: {e}")
    
    # Test configs
    try:
        from deep_temporal_transformer.configs.config import Config, get_default_config
        print("✅ Config imports")
    except Exception as e:
        errors.append(f"❌ Configs: {e}")
        print(f"❌ Config imports: {e}")
    
    # Test utils
    try:
        from deep_temporal_transformer.utils.utils import (
            setup_logging, ensure_dir, set_random_seeds, get_device
        )
        print("✅ Utils imports")
    except Exception as e:
        errors.append(f"❌ Utils: {e}")
        print(f"❌ Utils imports: {e}")
    
    return errors


def test_initialization():
    """Test basic initialization"""
    print("\n" + "="*60)
    print("🔍 TESTING INITIALIZATION")
    print("="*60)
    
    errors = []
    
    try:
        from deep_temporal_transformer import get_default_config, get_device
        
        config = get_default_config()
        print(f"✅ Config created: d_model={config.model.d_model}")
        
        device = get_device()
        print(f"✅ Device detected: {device}")
        
    except Exception as e:
        errors.append(f"❌ Initialization: {e}")
        print(f"❌ Initialization failed: {e}")
    
    return errors


def test_model_forward():
    """Test model forward pass"""
    print("\n" + "="*60)
    print("🔍 TESTING MODEL FORWARD PASS")
    print("="*60)
    
    errors = []
    
    try:
        import torch
        from deep_temporal_transformer.models.model_enhanced import DeepTemporalTransformerEnhanced
        from deep_temporal_transformer import get_default_config, get_device
        
        config = get_default_config()
        device = get_device()
        
        model = DeepTemporalTransformerEnhanced(
            input_dim=14,
            seq_len=8,
            d_model=config.model.d_model,
            nhead=config.model.nhead,
            num_layers=config.model.num_layers,
            dim_feedforward=config.model.dim_feedforward,
            memory_slots=config.model.memory_slots,
            dropout=config.model.dropout,
            emb_dims=config.model.emb_dims
        ).to(device)
        
        # Test forward pass
        batch_size = 4
        seq_len = 8
        input_dim = 14
        
        x = torch.randn(batch_size, seq_len, input_dim).to(device)
        
        # Should return 3 values
        logits, attention_weights, intermediates = model(x)
        
        print(f"✅ Forward pass successful:")
        print(f"   Logits shape: {logits.shape}")
        print(f"   Attention weights shape: {attention_weights.shape}")
        print(f"   Intermediates: {intermediates is not None}")
        
        # Check return value unpacking
        logits2, _, _ = model(x)
        print(f"✅ Unpacking works correctly")
        
    except Exception as e:
        errors.append(f"❌ Model forward: {e}")
        print(f"❌ Model forward pass failed: {e}")
        import traceback
        traceback.print_exc()
    
    return errors


def main():
    """Run all validation tests"""
    print("\n")
    print("🚀 DEEP TEMPORAL TRANSFORMER - CODEBASE VALIDATION")
    print("="*60)
    
    all_errors = []
    
    # Run tests
    all_errors.extend(test_imports())
    all_errors.extend(test_initialization())
    all_errors.extend(test_model_forward())
    
    # Summary
    print("\n" + "="*60)
    print("📊 VALIDATION SUMMARY")
    print("="*60)
    
    if not all_errors:
        print("✅ ALL TESTS PASSED!")
        print("🎉 Your codebase is ready for Colab!")
        return 0
    else:
        print(f"❌ FOUND {len(all_errors)} ERRORS:")
        for i, error in enumerate(all_errors, 1):
            print(f"{i}. {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
