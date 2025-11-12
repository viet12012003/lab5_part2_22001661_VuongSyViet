import os
import numpy as np
from gensim.models import Word2Vec
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class W2V_Dense:
    """
    Word2Vec + Dense Neural Network model for text classification.
    This model combines Word2Vec word embeddings with a Dense neural network.
    """
    
    def __init__(self, config=None):
        """Initialize the Word2Vec + Dense model.
        
        Args:
            config: Configuration object containing model parameters
        """
        self.config = config
        self.model = None
        self.w2v_model = None
        self.preprocessor = None
        self.vector_size = config.EMBEDDING_DIM if config else 100
        
    def set_preprocessor(self, preprocessor):
        """Set the text preprocessor.
        
        Args:
            preprocessor: Preprocessor object with text processing methods
        """
        self.preprocessor = preprocessor
    
    def train_word2vec(self, sentences, vector_size=100, window=5, min_count=1, workers=4, epochs=10):
        """Train Word2Vec model on the given sentences.
        
        Args:
            sentences: List of tokenized sentences or list of strings
            vector_size: Dimensionality of the word vectors
            window: Maximum distance between the current and predicted word
            min_count: Ignores words with total frequency lower than this
            workers: Number of worker threads
            epochs: Number of iterations over the corpus
            
        Returns:
            Word2Vec: Trained Word2Vec model
        """
        print("Training Word2Vec model...")
        
        # Ensure input is list of tokenized sentences
        tokenized_sentences = [s.split() if isinstance(s, str) else s for s in sentences]
        
        # Train Word2Vec model
        self.w2v_model = Word2Vec(
            sentences=tokenized_sentences,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=workers,
            epochs=epochs,
            sg=1  # Skip-gram
        )
        self.vector_size = vector_size
        print(f"Word2Vec model trained with {len(self.w2v_model.wv)} words in vocabulary")
        return self.w2v_model
    
    def document_vector(self, doc):
        """Convert a document to its average word vector representation.
        
        Args:
            doc: Input document (string or list of tokens)
            
        Returns:
            numpy.ndarray: Document vector
        """
        if self.w2v_model is None:
            raise ValueError("Word2Vec model not trained. Call train_word2vec() first.")
            
        # Convert input to list of words
        if isinstance(doc, str):
            words = doc.split()
        else:
            words = list(doc)
            
        # Filter words that are in the vocabulary
        words = [w for w in words if w in self.w2v_model.wv]
        
        if not words:
            # Return zero vector if no words in vocabulary
            return np.zeros(self.vector_size)
            
        # Calculate average of word vectors
        word_vectors = [self.w2v_model.wv[word] for word in words]
        return np.mean(word_vectors, axis=0)
    
    def build_model(self, num_classes, learning_rate=0.001):
        """Build the Dense neural network model.
        
        Args:
            num_classes: Number of output classes
            learning_rate: Learning rate for the optimizer
        """
        self.model = Sequential([
            Dense(128, activation='relu', input_shape=(self.vector_size,)),
            Dropout(0.5),
            Dense(num_classes, activation='softmax')
        ])
        
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def train(self, X_train, y_train, X_val=None, y_val=None, batch_size=32, epochs=10):
        """Train the Dense model on pre-trained Word2Vec embeddings.
        
        Args:
            X_train: Training text data
            y_train: Training labels
            X_val: Validation text data (optional)
            y_val: Validation labels (optional)
            batch_size: Batch size for training
            epochs: Number of training epochs
            
        Returns:
            History: Training history
        """
        if self.w2v_model is None:
            raise ValueError("Word2Vec model not trained. Call train_word2vec() first.")
        
        # Preprocess text if preprocessor is available
        if self.preprocessor is not None:
            print("Preprocessing training data...")
            X_train = [self.preprocessor.preprocess_text(doc) for doc in X_train]
            if X_val is not None:
                X_val = [self.preprocessor.preprocess_text(doc) for doc in X_val]
            
        # Convert text to document vectors
        print("Converting text to document vectors...")
        X_train_vec = np.array([self.document_vector(doc) for doc in X_train])
        
        # Build the model if not already built
        if self.model is None:
            num_classes = len(np.unique(y_train))
            self.build_model(num_classes)
            
        # Prepare validation data
        validation_data = None
        if X_val is not None and y_val is not None:
            X_val_vec = np.array([self.document_vector(doc) for doc in X_val])
            validation_data = (X_val_vec, y_val)
        
        # Define callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-5)
        ]
        
        # Train the model
        print("Training Dense model...")
        history = self.model.fit(
            X_train_vec, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def predict(self, texts, return_proba=False):
        """Make predictions on new text data.
        
        Args:
            texts: List of text strings or tokenized sentences
            return_proba: Whether to return class probabilities
            
        Returns:
            If return_proba is True, returns a tuple of (predicted_class_indices, class_probabilities)
            Otherwise, returns only the predicted_class_indices
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
            
        # Preprocess text if preprocessor is available
        if self.preprocessor is not None:
            texts = [self.preprocessor.preprocess_text(doc) for doc in texts]
            
        # Convert text to document vectors
        X_vec = np.array([self.document_vector(text) for text in texts])
        
        # Get prediction probabilities
        proba = self.model.predict(X_vec, verbose=0)
        
        if return_proba:
            return np.argmax(proba, axis=1), proba
        return np.argmax(proba, axis=1)
    
    def evaluate(self, X_test, y_test, output_dir=None):
        """Evaluate the model on test data and generate reports.
        
        Args:
            X_test: Test text data
            y_test: Test labels
            output_dir: Directory to save evaluation results (optional)
            
        Returns:
            dict: Dictionary containing evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        print("\nEvaluating model on test data...")
        
        # Preprocess text if preprocessor is available
        if self.preprocessor is not None:
            X_test = [self.preprocessor.preprocess_text(doc) for doc in X_test]
            
        # Convert text to document vectors
        X_test_vec = np.array([self.document_vector(doc) for doc in X_test])
        
        # Make predictions
        y_pred = np.argmax(self.model.predict(X_test_vec, verbose=0), axis=1)
        
        # Calculate metrics
        report = classification_report(y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # Calculate test loss
        test_loss = self.model.evaluate(X_test_vec, y_test, verbose=0)
        if isinstance(test_loss, list):
            test_loss = test_loss[0]  # Get the loss value if multiple metrics are returned
        
        # Print classification report and test loss
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        print(f"Test Loss: {test_loss:.4f}")
        
        # Add test loss to the report
        report['test_loss'] = test_loss
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            # Save confusion matrix
            plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
            
            # Save classification report
            with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
                f.write(classification_report(y_test, y_pred))
            
            print(f"\nEvaluation results saved to {output_dir}")
        
        plt.show()
        
        return {
            'classification_report': report,
            'confusion_matrix': conf_matrix.tolist(),
            'test_loss': test_loss
        }
    
    def save_model(self, filepath):
        """Save the model to disk.
        
        Args:
            filepath: Path to save the model
        """
        # Create directory if it doesn't exist
        os.makedirs(filepath, exist_ok=True)
        
        # Save Word2Vec model
        if self.w2v_model is not None:
            w2v_path = os.path.join(filepath, 'w2v.model')
            self.w2v_model.save(w2v_path)
        
        # Save Keras model
        if self.model is not None:
            model_path = os.path.join(filepath, 'dense_model.keras')
            self.model.save(model_path)
            
        # Save preprocessor if available
        if self.preprocessor is not None:
            import joblib
            preprocessor_path = os.path.join(filepath, 'preprocessor.joblib')
            joblib.dump(self.preprocessor, preprocessor_path)
        
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath, config=None):
        """Load a saved model from disk.
        
        Args:
            filepath: Path to the saved model
            config: Configuration object (optional)
            
        Returns:
            W2V_Dense: Loaded model instance
        """
        import tensorflow as tf
        import joblib
        
        # Initialize model
        model = cls(config)
        
        # Load Word2Vec model
        w2v_path = os.path.join(filepath, 'w2v.model')
        if os.path.exists(w2v_path):
            model.w2v_model = Word2Vec.load(w2v_path)
            model.vector_size = model.w2v_model.vector_size
        else:
            raise FileNotFoundError(f"Word2Vec model not found at {w2v_path}")
        
        # Load Keras model
        model_path = os.path.join(filepath, 'dense_model.keras')
        if os.path.exists(model_path):
            model.model = tf.keras.models.load_model(model_path)
        else:
            print(f"Warning: Keras model not found at {model_path}")
            
        # Load preprocessor if available
        preprocessor_path = os.path.join(filepath, 'preprocessor.joblib')
        if os.path.exists(preprocessor_path):
            model.preprocessor = joblib.load(preprocessor_path)
            
        return model