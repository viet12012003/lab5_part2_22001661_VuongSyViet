import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

class TFIDF_LogisticRegression:
    def __init__(self, config=None):
        """Initialize the TF-IDF + Logistic Regression model.
        
        Args:
            config: Configuration object containing model parameters
                  (can be None when loading a saved model)
        """
        self.config = config
        self.model = None
        if config is not None:  # Only initialize model if config is provided
            self.model = Pipeline([
                ('tfidf', TfidfVectorizer(
                    max_features=config.VOCAB_SIZE,
                    ngram_range=(1, 2),
                    stop_words='english'
                )),
                ('clf', LogisticRegression(
                    max_iter=1000,
                    random_state=42,
                    class_weight='balanced'
                ))
            ])
    
    def train(self, X_train, y_train):
        """Train the model.
        
        Args:
            X_train: Training text data
            y_train: Training labels
        """
        self.model.fit(X_train, y_train)
    
    def predict(self, X):
        """Make predictions.
        
        Args:
            X: Input text data
            
        Returns:
            array: Predicted labels
        """
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Predict class probabilities.
        
        Args:
            X: Input text data
            
        Returns:
            array: Class probabilities
        """
        return self.model.predict_proba(X)
    
    def save_model(self, filepath):
        """Save the model to disk.
        
        Args:
            filepath: Path to save the model
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.model, filepath)
    
    @classmethod
    def load_model(cls, filepath, config=None):
        """Load a saved model from disk.
        
        Args:
            filepath: Path to the saved model
            config: Optional config object (needed for compatibility with other models)
            
        Returns:
            TFIDF_LogisticRegression: Loaded model instance
        """
        model = cls(config)  # Create instance with config
        model.model = joblib.load(filepath)
        return model
