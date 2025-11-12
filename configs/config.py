import os
from pathlib import Path

class Config:
    def __init__(self):
        # Base directories
        self.BASE_DIR = Path(__file__).parent.parent
        self.DATA_DIR = self.BASE_DIR / 'data'
        self.OUTPUTS_DIR = self.BASE_DIR / 'outputs'
        
        # Data files
        self.RAW_DATA_DIR = self.DATA_DIR / 'hwu'
        self.TRAIN_FILE = self.RAW_DATA_DIR / 'train.csv'
        self.VAL_FILE = self.RAW_DATA_DIR / 'val.csv'
        self.TEST_FILE = self.RAW_DATA_DIR / 'test.csv'
        
        # Output directories
        self.MODEL_DIR = self.OUTPUTS_DIR / 'models'
        self.RESULTS_DIR = self.OUTPUTS_DIR / 'results'
        
        # Create directories if they don't exist
        for directory in [self.MODEL_DIR, self.RESULTS_DIR, self.RAW_DATA_DIR]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Model parameters
        self.EMBEDDING_DIM = 100
        self.MAX_LEN = 100
        self.VOCAB_SIZE = 10000
        self.BATCH_SIZE = 32
        self.EPOCHS = 30
        self.LEARNING_RATE = 0.001
        
        # Model files
        self.TFIDF_MODEL = self.MODEL_DIR / "tfidf_lr_model.joblib"
        self.W2V_MODEL = self.MODEL_DIR / "w2v_dense_model"
        self.LSTM_PRETRAINED = self.MODEL_DIR / "lstm_pretrained.keras"
        self.LSTM_SCRATCH = self.MODEL_DIR / "lstm_scratch.keras"
