import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
from typing import Union, Tuple, Optional, Dict





# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    X = np.random.randn(1000, 10)  # 1000 samples, 10 features

    # Initialize and fit the upsampler
    upsampler = DataUpsampler(
        latent_dim=50,
        generator_layers=[128, 256, 512],
        discriminator_layers=[512, 256, 128],
        scaler_type='standard',
        random_state=42
    )

    # Fit the model
    history = upsampler.fit(X, epochs=2000, batch_size=32, verbose=True)

    # Generate new samples
    synthetic_samples = upsampler.generate_samples(n_samples=500)

    # Upsample to specific size
    upsampled_data = upsampler.upsample_to_size(X, target_size=1500)

    print(f"Original data shape: {X.shape}")
    print(f"Synthetic samples shape: {synthetic_samples.shape}")
    print(f"Upsampled data shape: {upsampled_data.shape}")