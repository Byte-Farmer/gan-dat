class DataUpsampler:
    """
    A general-purpose data upsampling tool using GANs.
    Supports various data types and provides flexible configuration options.
    """

    def __init__(self,
                 latent_dim: int = 100,
                 generator_layers: Optional[list] = None,
                 discriminator_layers: Optional[list] = None,
                 scaler_type: str = 'standard',
                 learning_rate: float = 0.0002,
                 beta_1: float = 0.5,
                 random_state: int = 42):
        """
        Initialize the DataUpsampler.

        Parameters:
        -----------
        latent_dim : int
            Dimension of the latent space for the generator
        generator_layers : list, optional
            List of layer sizes for the generator
        discriminator_layers : list, optional
            List of layer sizes for the discriminator
        scaler_type : str
            Type of scaling to apply ('standard' or 'minmax')
        learning_rate : float
            Learning rate for Adam optimizer
        beta_1 : float
            Beta1 parameter for Adam optimizer
        random_state : int
            Random seed for reproducibility
        """
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.random_state = random_state

        # Set random seeds
        tf.random.set_seed(random_state)
        np.random.seed(random_state)

        # Initialize scaler
        self.scaler_type = scaler_type
        self.scaler = StandardScaler() if scaler_type == 'standard' else MinMaxScaler()

        # Set default architecture if none provided
        self.generator_layers = generator_layers or [256, 512, 1024]
        self.discriminator_layers = discriminator_layers or [1024, 512, 256]

        self.is_fitted = False
        self.input_dim = None

    def _build_generator(self) -> models.Sequential:
        """Build the generator network."""
        model = models.Sequential()

        # First layer
        model.add(layers.Dense(self.generator_layers[0], input_dim=self.latent_dim))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.BatchNormalization())

        # Hidden layers
        for layer_size in self.generator_layers[1:]:
            model.add(layers.Dense(layer_size))
            model.add(layers.LeakyReLU(alpha=0.2))
            model.add(layers.BatchNormalization())

        # Output layer
        model.add(layers.Dense(self.input_dim, activation='tanh'))

        return model

    def _build_discriminator(self) -> models.Sequential:
        """Build the discriminator network."""
        model = models.Sequential()

        # First layer
        model.add(layers.Dense(self.discriminator_layers[0], input_dim=self.input_dim))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.Dropout(0.3))

        # Hidden layers
        for layer_size in self.discriminator_layers[1:]:
            model.add(layers.Dense(layer_size))
            model.add(layers.LeakyReLU(alpha=0.2))
            model.add(layers.Dropout(0.3))

        # Output layer
        model.add(layers.Dense(1, activation='sigmoid'))

        return model

    def _initialize_gan(self):
        """Initialize the GAN architecture."""
        # Build and compile discriminator
        self.discriminator = self._build_discriminator()
        self.discriminator.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate, beta_1=self.beta_1),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        # Build generator
        self.generator = self._build_generator()

        # Build combined model
        self.discriminator.trainable = False
        self.combined = models.Sequential([self.generator, self.discriminator])
        self.combined.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate, beta_1=self.beta_1),
            loss='binary_crossentropy'
        )

    def fit(self,
            data: Union[np.ndarray, pd.DataFrame],
            epochs: int = 2000,
            batch_size: int = 32,
            verbose: bool = True) -> Dict:
        """
        Fit the DataUpsampler to the training data.

        Parameters:
        -----------
        data : array-like or DataFrame
            Training data to fit the model
        epochs : int
            Number of training epochs
        batch_size : int
            Size of training batches
        verbose : bool
            Whether to print training progress

        Returns:
        --------
        dict : Training history
        """
        # Convert data to numpy array if needed
        if isinstance(data, pd.DataFrame):
            data = data.values

        self.input_dim = data.shape[1]

        # Scale the data
        X_scaled = self.scaler.fit_transform(data)

        # Initialize GAN
        self._initialize_gan()

        history = {
            'discriminator_loss': [],
            'generator_loss': []
        }

        # Training loop
        for epoch in range(epochs):
            # Train discriminator
            idx = np.random.randint(0, X_scaled.shape[0], batch_size)
            real_samples = X_scaled[idx]

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            fake_samples = self.generator.predict(noise, verbose=0)

            d_loss_real = self.discriminator.train_on_batch(real_samples, np.ones((batch_size, 1)))
            d_loss_fake = self.discriminator.train_on_batch(fake_samples, np.zeros((batch_size, 1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # Train generator
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            g_loss = self.combined.train_on_batch(noise, np.ones((batch_size, 1)))

            history['discriminator_loss'].append(d_loss[0])
            history['generator_loss'].append(g_loss)

            if verbose and epoch % 200 == 0:
                print(f"Epoch {epoch}, D Loss: {d_loss[0]:.4f}, G Loss: {g_loss:.4f}")

        self.is_fitted = True
        return history

    def generate_samples(self, n_samples: int) -> np.ndarray:
        """
        Generate new synthetic samples.

        Parameters:
        -----------
        n_samples : int
            Number of samples to generate

        Returns:
        --------
        np.ndarray : Generated samples (inverse transformed to original scale)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before generating samples")

        noise = np.random.normal(0, 1, (n_samples, self.latent_dim))
        synthetic_samples = self.generator.predict(noise, verbose=0)

        # Inverse transform the samples
        return self.scaler.inverse_transform(synthetic_samples)

    def upsample_to_size(self,
                         data: Union[np.ndarray, pd.DataFrame],
                         target_size: int) -> np.ndarray:
        """
        Upsample data to reach a target size.

        Parameters:
        -----------
        data : array-like or DataFrame
            Original data to be upsampled
        target_size : int
            Desired size of the dataset after upsampling

        Returns:
        --------
        np.ndarray : Upsampled dataset
        """
        if isinstance(data, pd.DataFrame):
            data = data.values

        current_size = len(data)
        if target_size <= current_size:
            raise ValueError("Target size must be greater than current size")

        # Calculate number of synthetic samples needed
        n_synthetic = target_size - current_size

        # Generate synthetic samples
        synthetic_samples = self.generate_samples(n_synthetic)

        # Combine original and synthetic samples
        return np.vstack([data, synthetic_samples])