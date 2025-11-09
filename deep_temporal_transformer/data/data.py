"""Data processing module for financial fraud detection."""
import os
import logging
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional, Any
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

from .utils import setup_logging

logger = setup_logging()


class DataProcessor:
    """Secure data processing class for financial fraud detection."""
    
    def __init__(self, seq_len: int = 8, random_state: int = 42):
        self.seq_len = seq_len
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.encoders: Dict[str, LabelEncoder] = {}
        self.feature_cols: List[str] = []
    
    def synthesize_creditcard_data(
        self, 
        n_samples: int = 100000, 
        fraud_ratio: float = 0.001
    ) -> pd.DataFrame:
        """Generate synthetic credit card transaction data."""
        try:
            rng = np.random.RandomState(self.random_state)
            n_fraud = max(1, int(n_samples * fraud_ratio))
            n_legit = n_samples - n_fraud
            
            # Generate temporal features
            time_stamps = rng.randint(0, n_samples * 10, size=n_samples)
            
            # Generate transaction amounts (log-normal distribution)
            legit_amounts = rng.lognormal(mean=2.0, sigma=1.5, size=n_legit)
            fraud_amounts = rng.lognormal(mean=4.0, sigma=2.0, size=n_fraud)
            amounts = np.concatenate([legit_amounts, fraud_amounts])
            
            # Generate PCA-like features (V1-V10)
            features = rng.randn(n_samples, 10) * 2.0
            # Fraud transactions have different patterns
            features[-n_fraud:, :3] += rng.normal(loc=3.0, scale=1.0, size=(n_fraud, 3))
            
            # Generate categorical features
            user_ids = rng.randint(0, 20000, size=n_samples)
            device_ids = rng.randint(0, 5000, size=n_samples)
            merchant_cats = rng.randint(0, 50, size=n_samples)
            
            # Create labels
            labels = np.array([0] * n_legit + [1] * n_fraud)
            
            # Create DataFrame
            df = pd.DataFrame(
                features, 
                columns=[f"V{i+1}" for i in range(features.shape[1])]
            )
            df["Time"] = time_stamps
            df["Amount"] = amounts
            df["user_id"] = user_ids
            df["device_id"] = device_ids
            df["merchant_cat"] = merchant_cats
            df["Class"] = labels
            
            # Shuffle and sort by time
            df = df.sample(frac=1.0, random_state=self.random_state).reset_index(drop=True)
            df = df.sort_values("Time").reset_index(drop=True)
            
            logger.info(f"Generated {len(df)} synthetic transactions with {n_fraud} fraud cases")
            return df
            
        except Exception as e:
            logger.error(f"Failed to generate synthetic data: {e}")
            raise
    
    def load_creditcard_data(self, path: Optional[str] = None) -> pd.DataFrame:
        """Load credit card data with fallback to synthetic generation."""
        try:
            if path and os.path.exists(path):
                # Validate path security
                from .security_fixes import validate_path
                normalized_path = validate_path(path, ['.csv'])
                
                df = pd.read_csv(normalized_path)
                
                # Validate required columns
                required_cols = ['Time', 'Amount', 'Class']
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    raise ValueError(f"Missing required columns: {missing_cols}")
                
                # Add missing categorical columns if needed
                if 'user_id' not in df.columns:
                    df['user_id'] = df.index % 10000
                if 'device_id' not in df.columns:
                    df['device_id'] = df.index % 3000
                if 'merchant_cat' not in df.columns:
                    df['merchant_cat'] = df.index % 50
                
                logger.info(f"Loaded {len(df)} transactions from {path}")
                return df
            else:
                logger.warning("Data file not found, generating synthetic data")
                return self.synthesize_creditcard_data()
                
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise
    
    def temporal_split(
        self, 
        df: pd.DataFrame, 
        train_frac: float = 0.7, 
        val_frac: float = 0.15
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data temporally to avoid data leakage."""
        try:
            df_sorted = df.sort_values('Time').reset_index(drop=True)
            n = len(df_sorted)
            
            train_end = int(n * train_frac)
            val_end = int(n * (train_frac + val_frac))
            
            train_df = df_sorted.iloc[:train_end].reset_index(drop=True)
            val_df = df_sorted.iloc[train_end:val_end].reset_index(drop=True)
            test_df = df_sorted.iloc[val_end:].reset_index(drop=True)
            
            logger.info(f"Split data: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
            return train_df, val_df, test_df
            
        except Exception as e:
            logger.error(f"Failed to split data: {e}")
            raise
    
    def encode_categorical_features(
        self, 
        train_df: pd.DataFrame, 
        val_df: pd.DataFrame, 
        test_df: pd.DataFrame,
        id_cols: Tuple[str, ...] = ('user_id', 'device_id', 'merchant_cat')
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Encode categorical features safely."""
        try:
            train_df = train_df.copy()
            val_df = val_df.copy()
            test_df = test_df.copy()
            
            for col in id_cols:
                if col not in train_df.columns:
                    continue
                
                # Fit encoder on all data to handle unseen categories
                all_values = pd.concat([
                    train_df[col], val_df[col], test_df[col]
                ]).astype(str)
                
                encoder = LabelEncoder()
                encoder.fit(all_values)
                
                # Transform each split
                train_df[f"{col}_enc"] = encoder.transform(train_df[col].astype(str))
                val_df[f"{col}_enc"] = encoder.transform(val_df[col].astype(str))
                test_df[f"{col}_enc"] = encoder.transform(test_df[col].astype(str))
                
                self.encoders[col] = encoder
            
            return train_df, val_df, test_df
            
        except Exception as e:
            logger.error(f"Failed to encode categorical features: {e}")
            raise
    
    def build_sequences(
        self, 
        df: pd.DataFrame, 
        feature_cols: List[str],
        id_col: str = 'user_id_enc'
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build temporal sequences for each user."""
        try:
            sequences = []
            labels = []
            timestamps = []
            
            # Group by user and process sequences
            for user_id, group in df.groupby(id_col, sort=False):
                group = group.sort_values('Time').reset_index(drop=True)
                
                features = group[feature_cols].values.astype(np.float32)
                targets = group['Class'].values.astype(np.int32)
                times = group['Time'].values.astype(np.int64)
                
                # Create sliding windows
                for i in range(len(group)):
                    start_idx = max(0, i - self.seq_len + 1)
                    window = features[start_idx:i + 1]
                    
                    # Pad if necessary
                    if len(window) < self.seq_len:
                        padding = np.zeros((self.seq_len - len(window), len(feature_cols)), dtype=np.float32)
                        window = np.vstack([padding, window])
                    
                    sequences.append(window)
                    labels.append(targets[i])
                    timestamps.append(times[i])
            
            X = np.stack(sequences) if sequences else np.zeros((0, self.seq_len, len(feature_cols)))
            y = np.array(labels, dtype=np.int32)
            t = np.array(timestamps, dtype=np.int64)
            
            logger.info(f"Built {len(X)} sequences with shape {X.shape}")
            return X, y, t
            
        except Exception as e:
            logger.error(f"Failed to build sequences: {e}")
            raise
    
    def normalize_features(
        self, 
        X_train: np.ndarray, 
        X_val: np.ndarray, 
        X_test: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Normalize features using training data statistics."""
        try:
            # Efficient reshaping and scaling
            original_shapes = [X_train.shape, X_val.shape, X_test.shape]
            n_features = X_train.shape[-1]
            
            # Fit scaler on training data
            X_train_flat = X_train.reshape(-1, n_features)
            self.scaler.fit(X_train_flat)
            
            # Transform efficiently
            X_train_norm = self.scaler.transform(X_train_flat).reshape(original_shapes[0])
            X_val_norm = self.scaler.transform(X_val.reshape(-1, n_features)).reshape(original_shapes[1])
            X_test_norm = self.scaler.transform(X_test.reshape(-1, n_features)).reshape(original_shapes[2])
            
            return X_train_norm, X_val_norm, X_test_norm
            
        except Exception as e:
            logger.error(f"Failed to normalize features: {e}")
            raise
    
    def process_data(self, data_path: Optional[str] = None) -> Tuple[np.ndarray, ...]:
        """Complete data processing pipeline."""
        try:
            # Load data
            df = self.load_creditcard_data(data_path)
            
            # Split temporally
            train_df, val_df, test_df = self.temporal_split(df)
            
            # Encode categorical features
            train_df, val_df, test_df = self.encode_categorical_features(train_df, val_df, test_df)
            
            # Define feature columns
            self.feature_cols = [col for col in train_df.columns if col.startswith('V')] + \
                               ['Amount', 'user_id_enc', 'device_id_enc', 'merchant_cat_enc']
            
            # Build sequences
            X_train, y_train, t_train = self.build_sequences(train_df, self.feature_cols)
            X_val, y_val, t_val = self.build_sequences(val_df, self.feature_cols)
            X_test, y_test, t_test = self.build_sequences(test_df, self.feature_cols)
            
            # Normalize features
            X_train, X_val, X_test = self.normalize_features(X_train, X_val, X_test)
            
            logger.info("Data processing completed successfully")
            return X_train, y_train, X_val, y_val, X_test, y_test
            
        except Exception as e:
            logger.error(f"Data processing failed: {e}")
            raise