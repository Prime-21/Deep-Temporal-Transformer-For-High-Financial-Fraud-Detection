"""Setup script for Deep Temporal Transformer package."""
from setuptools import setup, find_packages

setup(
    name="deep-temporal-transformer",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.21.0", 
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0"
    ],
    python_requires=">=3.8",
    author="Prasad Kharat",
    description="Deep Temporal Transformer for High Frequency Financial Fraud Detection",
    long_description="A state-of-the-art transformer-based model for detecting fraudulent transactions in high-frequency financial data.",
    package_dir={'': '.'},
    entry_points={
        'console_scripts': [
            'deep-temporal-transformer=deep_temporal_transformer.examples.main:main',
        ],
    },
)