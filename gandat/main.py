# Example usage
if __name__ == "__main__":
    # Create sample data with multiple types
    np.random.seed(42)

    sample_data = pd.DataFrame({
        'continuous': np.random.normal(0, 1, 1000),
        'discrete': np.random.randint(0, 100, 1000),
        'categorical': np.random.choice(['A', 'B', 'C'], 1000),
        'boolean': np.random.choice([True, False], 1000),
        'datetime': pd.date_range('2023-01-01', periods=1000, freq='D')
    })

    # Initialize and fit the upsampler
    upsampler = UniversalDataUpsampler(
        latent_dim=50,
        generator_layers=[128, 256, 512],
        discriminator_layers=[512, 256, 128],
        random_state=42
    )

    # Fit the model
    history = upsampler.fit(sample_data, epochs=2000, batch_size=32, verbose=True)

    # Generate new samples
    synthetic_samples = upsampler.generate_samples(n_samples=500)

    # Upsample to specific size
    upsampled_data = upsampler.upsample_to_size(sample_data, target_size=1500)

    print("\nOriginal Data Types:")
    print(sample_data.dtypes)
    print("\nSynthetic Data Types:")
    print(synthetic_samples.dtypes)