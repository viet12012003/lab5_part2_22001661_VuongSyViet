import pandas as pd
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
from .preprocessor import TextPreprocessor

class DataLoader:
    def __init__(self, config):
        """Initialize the DataLoader with configuration.
        
        Args:
            config: Configuration object containing paths and parameters
        """
        self.config = config
        self.preprocessor = TextPreprocessor()
        self.label_encoder = LabelEncoder()
        
    def load_datasets(self):
        """Load and preprocess train, validation, and test datasets.
        
        Returns:
            dict: Dictionary containing preprocessed datasets and labels
        """
        print("Loading and preprocessing data...")
        
        try:
            # Load datasets
            train_df = pd.read_csv(self.config.TRAIN_FILE)
            val_df = pd.read_csv(self.config.VAL_FILE)
            test_df = pd.read_csv(self.config.TEST_FILE)
            
            # Check required columns
            for df, name in [(train_df, 'train'), (val_df, 'validation'), (test_df, 'test')]:
                if 'text' not in df.columns or 'category' not in df.columns:
                    raise ValueError(f"{name.capitalize()} CSV must contain 'text' and 'category' columns")
            
            # Preprocess text data
            print("Preprocessing text data...")
            X_train = train_df['text'].apply(self.preprocessor.preprocess_text).values
            X_val = val_df['text'].apply(self.preprocessor.preprocess_text).values
            X_test = test_df['text'].apply(self.preprocessor.preprocess_text).values
            
            # Encode labels
            self.label_encoder.fit(train_df['category'].values)
            y_train = self.label_encoder.transform(train_df['category'].values)
            y_val = self.label_encoder.transform(val_df['category'].values)
            y_test = self.label_encoder.transform(test_df['category'].values)
            
            return {
                'X_train': X_train, 'y_train': y_train,
                'X_val': X_val, 'y_val': y_val,
                'X_test': X_test, 'y_test': y_test
            }
            
        except FileNotFoundError as e:
            print(f"Error: Required data files not found. Please check the file paths in config.py")
            print(f"Expected files:")
            print(f"- {self.config.TRAIN_FILE}")
            print(f"- {self.config.VAL_FILE}")
            print(f"- {self.config.TEST_FILE}")
            raise
        except Exception as e:
            print(f"Error loading datasets: {str(e)}")
            raise
    
    def get_label_encoder(self):
        """Get the label encoder used for encoding intents.
        
        Returns:
            LabelEncoder: Fitted label encoder
        """
        return self.label_encoder
    
    def get_preprocessor(self):
        """Get the text preprocessor.
        
        Returns:
            TextPreprocessor: Text preprocessor instance
        """
        return self.preprocessor
