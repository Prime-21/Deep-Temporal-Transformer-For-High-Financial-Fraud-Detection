"""
Quick test script to verify the code is working correctly.
Run this after installing dependencies.
"""

print("Testing Deep Temporal Transformer imports...")

try:
    from deep_temporal_transformer import (
        get_default_config, DataProcessor, ModelTrainer,
        set_random_seeds, get_device
    )
    print("✅ Basic imports successful!")
    
    # Test configuration
    config = get_default_config()
    print(f"✅ Config loaded: d_model={config.model.d_model}, layers={config.model.num_layers}")
    
    # Test random seeds
    set_random_seeds(42)
    print("✅ Random seeds set")
    
    # Test device detection
    device = get_device()
    print(f"✅ Device detected: {device}")
    
    print("\n" + "="*60)
    print("🎉 ALL TESTS PASSED!")
    print("="*60)
    print("\nYou can now:")
    print("1. Run the demo: python deep_temporal_transformer/examples/demo.py")
    print("2. Run the full pipeline: python deep_temporal_transformer/examples/main.py")
    print("3. Use the Colab notebook: run_on_colab.ipynb")
    
except ImportError as e:
    print(f"\n❌ Import Error: {e}")
    print("\nPlease install dependencies first:")
    print("pip install torch scikit-learn pandas matplotlib seaborn")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
