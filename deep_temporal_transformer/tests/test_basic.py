"""Basic tests for the package."""
import numpy as np
from deep_temporal_transformer import get_default_config, DataProcessor


def test_config():
    """Test configuration creation."""
    config = get_default_config()
    assert config.model.d_model > 0
    assert config.training.batch_size > 0


def test_data_processor():
    """Test data processing."""
    processor = DataProcessor(seq_len=4, random_state=42)
    X_train, y_train, X_val, y_val, X_test, y_test = processor.process_data()
    
    assert X_train.shape[0] > 0
    assert len(X_train) == len(y_train)
    assert X_train.shape[1] == 4  # seq_len
    assert np.all(np.isin(y_train, [0, 1]))


if __name__ == "__main__":
    test_config()
    test_data_processor()
    print("All tests passed!")