import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from datetime import datetime
import warnings
from typing import Union, Dict, List, Optional, Tuple


class UniversalDataUpsampler:
    """
    A universal data upsampling tool that handles multiple data types:
    - Continuous (float)
    - Discrete (int)
    - Categorical
    - Boolean
    - DateTime
    """

    def __init__(self,
                 latent_dim: int = 100,
                 generator_layers: Optional[list] = None,
                 discriminator_layers: Optional[list] = None,
                 learning_rate: float = 0.0002,
                 beta_1: float = 0.5,
                 random_state: int = 42):
        """
        Initialize the UniversalDataUpsampler.

        Parameters:
        -----------
        latent_dim : int
            Dimension of the latent space for the generator
        generator_layers : list, optional
            List of layer sizes for the generator
        discriminator_layers : list, optional
            List of layer sizes for the discriminator
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

        # Set default architecture if none provided
        self.generator_layers = generator_layers or [256, 512, 1024]
        self.discriminator_layers = discriminator_layers or [1024, 512, 256]

        # Initialize preprocessing components
        self.continuous_scaler = StandardScaler()
        self.discrete_scaler = MinMaxScaler()
        self.label_encoders = {}
        self.column_types = {}
        self.categorical_dims = {}
        self.date_min = {}
        self.date_max = {}

        # Set random seeds
        tf.random.set_seed(random_state)
        np.random.seed(random_state)

        self.is_fitted = False
        self.input_dim = None

    def _detect_column_types(self, df: pd.DataFrame) -> Dict:
        """
        Detect the type of each column in the dataframe.

        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataframe

        Returns:
        --------
        dict : Dictionary mapping column names to their detected types
        """
        column_types = {}

        for column in df.columns:
            # Get unique values
            unique_values = df[column].nunique()
            dtype = df[column].dtype

            # Check if datetime
            if pd.api.types.is_datetime64_any_dtype(dtype):
                column_types[column] = 'datetime'

            # Check if boolean
            elif pd.api.types.is_bool_dtype(dtype):
                column_types[column] = 'boolean'

            # Check if categorical
            elif pd.api.types.is_categorical_dtype(dtype) or \
                    (pd.api.types.is_object_dtype(dtype) and unique_values <= 100):
                column_types[column] = 'categorical'

            # Check if discrete (integer)
            elif pd.api.types.is_integer_dtype(dtype):
                column_types[column] = 'discrete'

            # Assume continuous for remaining numeric types
            elif pd.api.types.is_numeric_dtype(dtype):
                column_types[column] = 'continuous'

            else:
                warnings.warn(f"Column {column} has unhandled dtype {dtype}. Treating as categorical.")
                column_types[column] = 'categorical'

        return column_types

    def _preprocess_column(self,
                           series: pd.Series,
                           column: str,
                           column_type: str,
                           fit: bool = False) -> np.ndarray:
        """
        Preprocess a single column based on its type.

        Parameters:
        -----------
        series : pandas.Series
            Column data to preprocess
        column : str
            Column name
        column_type : str
            Type of the column
        fit : bool
            Whether to fit the preprocessors

        Returns:
        --------
        numpy.ndarray : Preprocessed column data
        """
        if column_type == 'continuous':
            if fit:
                return self.continuous_scaler.fit_transform(series.values.reshape(-1, 1))
            return self.continuous_scaler.transform(series.values.reshape(-1, 1))

        elif column_type == 'discrete':
            if fit:
                return self.discrete_scaler.fit_transform(series.values.reshape(-1, 1))
            return self.discrete_scaler.transform(series.values.reshape(-1, 1))

        elif column_type == 'categorical':
            if fit:
                self.label_encoders[column] = LabelEncoder()
                encoded = self.label_encoders[column].fit_transform(series)
                self.categorical_dims[column] = len(self.label_encoders[column].classes_)
            else:
                encoded = self.label_encoders[column].transform(series)
            return np.eye(self.categorical_dims[column])[encoded]

        elif column_type == 'boolean':
            return series.astype(int).values.reshape(-1, 1)

        elif column_type == 'datetime':
            if fit:
                self.date_min[column] = series.min()
                self.date_max[column] = series.max()

            timestamps = series.map(datetime.timestamp)
            normalized = (timestamps - self.date_min[column].timestamp()) / \
                         (self.date_max[column].timestamp() - self.date_min[column].timestamp())
            return normalized.values.reshape(-1, 1)

    def _postprocess_column(self,
                            data: np.ndarray,
                            column: str,
                            column_type: str) -> pd.Series:
        """
        Convert preprocessed data back to original format.

        Parameters:
        -----------
        data : numpy.ndarray
            Preprocessed data
        column : str
            Column name
        column_type : str
            Type of the column

        Returns:
        --------
        pandas.Series : Postprocessed column data
        """
        if column_type == 'continuous':
            return pd.Series(
                self.continuous_scaler.inverse_transform(data).flatten(),
                name=column
            )

        elif column_type == 'discrete':
            values = self.discrete_scaler.inverse_transform(data).flatten()
            return pd.Series(np.round(values).astype(int), name=column)

        elif column_type == 'categorical':
            encoded_idx = np.argmax(data, axis=1)
            return pd.Series(
                self.label_encoders[column].inverse_transform(encoded_idx),
                name=column
            )

        elif column_type == 'boolean':
            return pd.Series(data.flatten() > 0.5, name=column)

        elif column_type == 'datetime':
            timestamps = data.flatten() * \
                         (self.date_max[column].timestamp() - self.date_min[column].timestamp()) + \
                         self.date_min[column].timestamp()
            return pd.Series(
                pd.to_datetime(timestamps, unit='s'),
                name=column
            )

    def _preprocess_data(self,
                         df: pd.DataFrame,
                         fit: bool = False) -> np.ndarray:
        """
        Preprocess all columns in the dataframe.

        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataframe
        fit : bool
            Whether to fit the preprocessors

        Returns:
        --------
        numpy.ndarray : Preprocessed data
        """
        processed_columns = []

        if fit:
            self.column_types = self._detect_column_types(df)

        for column, column_type in self.column_types.items():
            processed = self._preprocess_column(
                df[column],
                column,
                column_type,
                fit
            )
            processed_columns.append(processed)

        return np.hstack(processed_columns)

    def _postprocess_data(self, data: np.ndarray) -> pd.DataFrame:
        """
        Convert preprocessed data back to original format.

        Parameters:
        -----------
        data : numpy.ndarray
            Preprocessed data

        Returns:
        --------
        pandas.DataFrame : Postprocessed data
        """
        start_idx = 0
        processed_columns = []

        for column, column_type in self.column_types.items():
            if column_type == 'categorical':
                width = self.categorical_dims[column]
            else:
                width = 1

            column_data = data[:, start_idx:start_idx + width]
            processed = self._postprocess_column(column_data, column, column_type)
            processed_columns.append(processed)

            start_idx += width

        return pd.concat(processed_columns, axis=1)

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

    def fit(self,
            data: Union[pd.DataFrame, np.ndarray],
            epochs: int = 2000,
            batch_size: int = 32,
            verbose: bool = True) -> Dict:
        """
        Fit the model to the data.

        Parameters:
        -----------
        data : pandas.DataFrame or numpy.ndarray
            Training data
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
        # Convert to DataFrame if necessary
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data)

        # Preprocess data
        X_processed = self._preprocess_data(data, fit=True)
        self.input_dim = X_processed.shape[1]

        # Initialize GAN
        self._initialize_gan()

        history = {
            'discriminator_loss': [],
            'generator_loss': []
        }

        # Training loop
        for epoch in range(epochs):
            # Train discriminator
            idx = np.random.randint(0, X_processed.shape[0], batch_size)
            real_samples = X_processed[idx]

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            fake_samples = self.generator.predict(noise, verbose=0)

            d_loss_real = self.discriminator.train_on_batch(
                real_samples,
                np.ones((batch_size, 1))
            )
            d_loss_fake = self.discriminator.train_on_batch(
                fake_samples,
                np.zeros((batch_size, 1))
            )
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # Train generator
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            g_loss = self.combined.train_on_batch(
                noise,
                np.ones((batch_size, 1))
            )

            history['discriminator_loss'].append(d_loss[0])
            history['generator_loss'].append(g_loss)

            if verbose and epoch % 200 == 0:
                print(f"Epoch {epoch}, D Loss: {d_loss[0]:.4f}, G Loss: {g_loss:.4f}")

        self.is_fitted = True
        return history

    def generate_samples(self, n_samples: int) -> pd.DataFrame:
        """
        Generate new synthetic samples.

        Parameters:
        -----------
        n_samples : int
            Number of samples to generate

        Returns:
        --------
        pandas.DataFrame : Generated samples
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before generating samples")

        noise = np.random.normal(0, 1, (n_samples, self.latent_dim))
        synthetic_data = self.generator.predict(noise, verbose=0)

        return self._postprocess_data(synthetic_data)

    def upsample_to_size(self,
                         data: Union[pd.DataFrame, np.ndarray],
                         target_size: int) -> pd.DataFrame:
        """Upsample data to reach a target size."""
        # Convert to DataFrame if necessary
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data)

        current_size = len(data)
        if target_size <= current_size:
            raise ValueError("Target size must be greater than current size")

        # Generate synthetic samples
        n_synthetic = target_size - current_size
        synthetic_samples = self.generate_samples(n_synthetic)

        # Combine original and synthetic samples
        return pd.concat([data, synthetic_samples], ignore_index=True)