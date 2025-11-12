import os
import numpy as np
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

class BaseLSTMModel:
    """Base class for LSTM models with common functionality."""
    
    def __init__(self, config):
        """Initialize the base LSTM model.
        
        Args:
            config: Configuration object containing model parameters
        """
        self.config = config
        self.model = None
        self.preprocessor = None
    
    def prepare_sequences(self, texts):
        """Convert texts to padded sequences.
        
        Args:
            texts: List of text strings
            
        Returns:
            array: Padded sequences
        """
        if not hasattr(self, 'preprocessor'):
            raise ValueError("Preprocessor not set. Call set_preprocessor() first.")
        return self.preprocessor.prepare_sequences(
            texts, 
            max_len=self.config.MAX_LEN,
            vocab_size=self.config.VOCAB_SIZE
        )
    
    def set_preprocessor(self, preprocessor):
        """Set the text preprocessor.
        
        Args:
            preprocessor: TextPreprocessor instance
        """
        self.preprocessor = preprocessor
    
    def train(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        """Train the model.
        
        Args:
            X_train: Training text data
            y_train: Training labels
            X_val: Validation text data (optional)
            y_val: Validation labels (optional)
            **kwargs: Additional arguments for model.fit()
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
            
        # Prepare sequences
        X_train_seq = self.prepare_sequences(X_train)
        
        # Prepare validation data if provided
        validation_data = None
        if X_val is not None and y_val is not None:
            X_val_seq = self.prepare_sequences(X_val)
            validation_data = (X_val_seq, y_val)
        
        # Define callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss' if validation_data else 'loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss' if validation_data else 'loss',
                factor=0.5,
                patience=2,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        # Train the model
        history = self.model.fit(
            X_train_seq, y_train,
            validation_data=validation_data,
            batch_size=self.config.BATCH_SIZE,
            epochs=self.config.EPOCHS,
            callbacks=callbacks,
            **kwargs
        )
        
        return history
    
    def predict(self, texts):
        """Make predictions.
        
        Args:
            texts: List of text strings
            
        Returns:
            array: Predicted class indices
        """
        sequences = self.prepare_sequences(texts)
        return np.argmax(self.model.predict(sequences), axis=1)
    
    def save_model(self, model_path):
        """Save the model to disk.
        
        Args:
            model_path: Path to save the model
        """
        # Ensure the directory exists
        model_path = str(model_path)
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Ensure .keras extension is used
        if not model_path.endswith('.keras'):
            model_path = model_path + '.keras'
        
        # Save the model
        self.model.save(model_path)
        print(f"Model saved to {model_path}")
        
        # Save preprocessor if available
        if hasattr(self, 'preprocessor') and self.preprocessor is not None:
            preprocessor_path = os.path.join(os.path.dirname(model_path), 'preprocessor.joblib')
            import joblib
            joblib.dump(self.preprocessor, preprocessor_path)
    
    @classmethod
    def load_model(cls, model_path, config=None):
        """Load a saved model.
        
        Args:
            model_path: Path to the saved model
            config: Configuration object (optional)
            
        Returns:
            Instance of the model
        """
        # Add .keras extension if not present
        model_path = str(model_path)
        if not model_path.endswith('.keras'):
            model_path = model_path + '.keras'
            
        if not Path(model_path).exists():
            # Try without extension for backward compatibility
            if Path(model_path.replace('.keras', '')).exists():
                model_path = model_path.replace('.keras', '')
            else:
                raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Create instance and load the model
        instance = cls(config)
        instance.model = tf.keras.models.load_model(model_path)
        
        # Try to load preprocessor
        preprocessor_path = os.path.join(os.path.dirname(model_path), 'preprocessor.joblib')
        if os.path.exists(preprocessor_path):
            import joblib
            instance.preprocessor = joblib.load(preprocessor_path)
            
        return instance
        
        return instance


class LSTM_Pretrained(BaseLSTMModel):
    """LSTM model with pre-trained Word2Vec embeddings."""
    
    def __init__(self, config):
        """Initialize the LSTM model with pre-trained embeddings.
        
        Args:
            config: Configuration object containing model parameters
        """
        super().__init__(config)
        self.embedding_matrix = None
    
    def build_model(self, num_classes, word_index=None, embedding_matrix=None):
        """Build the LSTM model with pre-trained embeddings.
        
        Args:
            num_classes: Number of output classes
            word_index: Word to index mapping
            embedding_matrix: Pre-trained embedding matrix
        """
        if embedding_matrix is not None:
            self.embedding_matrix = embedding_matrix
            vocab_size, embedding_dim = embedding_matrix.shape
            
            # Input layer
            input_layer = Input(shape=(self.config.MAX_LEN,))
            
            # Embedding layer (frozen)
            embedding_layer = Embedding(
                input_dim=vocab_size,
                output_dim=embedding_dim,
                weights=[embedding_matrix],
                input_length=self.config.MAX_LEN,
                trainable=False,
                mask_zero=True
            )(input_layer)
            
            # LSTM layers
            x = Bidirectional(LSTM(
                128,
                return_sequences=True,
                kernel_regularizer=l2(0.01),
                recurrent_dropout=0.2
            ))(embedding_layer)
            x = Dropout(0.5)(x)
            
            x = LSTM(
                64,
                kernel_regularizer=l2(0.01),
                recurrent_dropout=0.2
            )(x)
            x = Dropout(0.5)(x)
            
            # Output layer
            output_layer = Dense(
                num_classes,
                activation='softmax',
                kernel_regularizer=l2(0.01)
            )(x)
            
            # Create model
            self.model = Model(inputs=input_layer, outputs=output_layer)
            
            # Compile model
            self.model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
        else:
            raise ValueError("embedding_matrix must be provided for LSTM_Pretrained model")


class LSTM_Scratch(BaseLSTMModel):
    """LSTM model with trainable embeddings from scratch."""
    
    def build_model(self, num_classes, word_index=None):
        """Build the LSTM model with trainable embeddings.
        
        Args:
            num_classes: Number of output classes
            word_index: Word to index mapping (unused in this implementation)
        """
        # Input layer
        input_layer = Input(shape=(self.config.MAX_LEN,))
        
        # Embedding layer (trainable)
        x = Embedding(
            input_dim=self.config.VOCAB_SIZE,
            output_dim=self.config.EMBEDDING_DIM,
            input_length=self.config.MAX_LEN,
            mask_zero=True,
            embeddings_regularizer=l2(0.0001)
        )(input_layer)
        
        # LSTM layers
        x = Bidirectional(LSTM(
            128,
            return_sequences=True,
            kernel_regularizer=l2(0.01),
            recurrent_dropout=0.2
        ))(x)
        x = Dropout(0.5)(x)
        
        x = LSTM(
            64,
            kernel_regularizer=l2(0.01),
            recurrent_dropout=0.2
        )(x)
        x = Dropout(0.5)(x)
        
        # Output layer
        output_layer = Dense(
            num_classes,
            activation='softmax',
            kernel_regularizer=l2(0.01)
        )(x)
        
        # Create model
        self.model = Model(inputs=input_layer, outputs=output_layer)
        
        # Compile model
        self.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
