import os
import sys
import time
import numpy as np
from pathlib import Path
from datetime import datetime
from gensim.models import Word2Vec

from configs.config import Config
from src.data.data_loader import DataLoader
from src.models.tfidf_lr import TFIDF_LogisticRegression
from src.models.w2v_dense import W2V_Dense
from src.models.lstm_models import LSTM_Pretrained, LSTM_Scratch

def train_tfidf_lr(data, config):
    """Train TF-IDF + Logistic Regression model."""
    print("\n" + "="*50)
    print("TRAINING TF-IDF + LOGISTIC REGRESSION")
    print("="*50)
    
    start_time = time.time()
    try:
        tfidf_lr = TFIDF_LogisticRegression(config)
        tfidf_lr.train(data['X_train'], data['y_train'])
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(str(config.TFIDF_MODEL)), exist_ok=True)
        tfidf_lr.save_model(str(config.TFIDF_MODEL))
        
        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time:.2f} seconds")
        print(f"Model saved to {config.TFIDF_MODEL}")
        return True
        
    except Exception as e:
        print(f"\nError training TF-IDF + Logistic Regression: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def train_w2v_dense(data, config):
    """Train Word2Vec + Dense model."""
    print("\n" + "="*50)
    print("TRAINING WORD2VEC + DENSE")
    print("="*50)
    
    start_time = time.time()
    try:
        # Initialize model
        w2v_dense = W2V_Dense(config)
        
        # Get and set preprocessor
        data_loader = DataLoader(config)
        data_loader.load_datasets()
        preprocessor = data_loader.get_preprocessor()
        w2v_dense.set_preprocessor(preprocessor)
        
        # Train Word2Vec
        print("\nTraining Word2Vec model...")
        w2v_dense.train_word2vec(
            data['X_train'],
            vector_size=config.EMBEDDING_DIM,
            window=5,
            min_count=1,
            workers=4,
            epochs=30
        )
        
        # Train Dense model
        print("\nTraining Dense model...")
        history = w2v_dense.train(
            X_train=data['X_train'],
            y_train=data['y_train'],
            X_val=data['X_val'],
            y_val=data['y_val'],
            batch_size=config.BATCH_SIZE,
            epochs=config.EPOCHS
        )
        
        # Save model
        model_dir = str(config.W2V_MODEL)
        os.makedirs(model_dir, exist_ok=True)
        w2v_dense.save_model(model_dir)
        
        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time:.2f} seconds")
        print(f"Model saved to {model_dir}")
        
        # Evaluate on test set
        print("\nEvaluating on test set...")
        w2v_dense.evaluate(
            data['X_test'],
            data['y_test'],
            output_dir=str(config.RESULTS_DIR / 'w2v_dense')
        )
        
        return True
        
    except Exception as e:
        print(f"\nError training Word2Vec + Dense: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def train_lstm_pretrained(data, config):
    """Train LSTM with pretrained Word2Vec embeddings."""
    print("\n" + "="*50)
    print("TRAINING LSTM WITH PRETRAINED EMBEDDINGS")
    print("="*50)
    
    start_time = time.time()
    try:
        # Initialize model
        lstm_pretrained = LSTM_Pretrained(config)
        
        # Get preprocessor and prepare sequences
        data_loader = DataLoader(config)
        data_loader.load_datasets()
        preprocessor = data_loader.get_preprocessor()
        lstm_pretrained.set_preprocessor(preprocessor)
        
        # Prepare sequences and get word index
        X_train_seq = preprocessor.prepare_sequences(
            data['X_train'], 
            max_len=config.MAX_LEN,
            vocab_size=config.VOCAB_SIZE,
            fit_tokenizer=True
        )
        word_index = preprocessor.tokenizer.word_index
        
        # Load Word2Vec model for embeddings
        w2v_model_path = os.path.join(str(config.W2V_MODEL), 'w2v.model')
        if not os.path.exists(w2v_model_path):
            raise FileNotFoundError(f"Word2Vec model not found at {w2v_model_path}")
            
        w2v_model = Word2Vec.load(w2v_model_path)
        
        # Create embedding matrix
        embedding_matrix = np.zeros((config.VOCAB_SIZE, config.EMBEDDING_DIM))
        for word, i in word_index.items():
            if i < config.VOCAB_SIZE and word in w2v_model.wv:
                embedding_matrix[i] = w2v_model.wv[word]
        
        # Build and train model
        num_classes = len(np.unique(data['y_train']))
        lstm_pretrained.build_model(num_classes=num_classes, embedding_matrix=embedding_matrix)
        
        lstm_pretrained.train(
            data['X_train'], data['y_train'],
            data['X_val'], data['y_val']
        )
        
        # Save model
        os.makedirs(os.path.dirname(str(config.LSTM_PRETRAINED)), exist_ok=True)
        lstm_pretrained.save_model(str(config.LSTM_PRETRAINED))
        
        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time:.2f} seconds")
        print(f"Model saved to {config.LSTM_PRETRAINED}")
        return True
        
    except Exception as e:
        print(f"\nError training LSTM with pretrained embeddings: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def train_lstm_scratch(data, config):
    """Train LSTM from scratch."""
    print("\n" + "="*50)
    print("TRAINING LSTM FROM SCRATCH")
    print("="*50)
    
    start_time = time.time()
    try:
        # Initialize model
        lstm_scratch = LSTM_Scratch(config)
        
        # Get preprocessor and prepare sequences
        data_loader = DataLoader(config)
        data_loader.load_datasets()
        preprocessor = data_loader.get_preprocessor()
        lstm_scratch.set_preprocessor(preprocessor)
        
        # Prepare sequences and get word index
        X_train_seq = preprocessor.prepare_sequences(
            data['X_train'], 
            max_len=config.MAX_LEN,
            vocab_size=config.VOCAB_SIZE,
            fit_tokenizer=True
        )
        word_index = preprocessor.tokenizer.word_index
        
        # Build and train model
        num_classes = len(np.unique(data['y_train']))
        lstm_scratch.build_model(num_classes=num_classes, word_index=word_index)
        
        lstm_scratch.train(
            data['X_train'], data['y_train'],
            data['X_val'], data['y_val']
        )
        
        # Save model
        os.makedirs(os.path.dirname(str(config.LSTM_SCRATCH)), exist_ok=True)
        lstm_scratch.save_model(str(config.LSTM_SCRATCH))
        
        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time:.2f} seconds")
        print(f"Model saved to {config.LSTM_SCRATCH}")
        return True
        
    except Exception as e:
        print(f"\nError training LSTM from scratch: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    # Start time
    start_time = time.time()
    print("\n" + "="*80)
    print(f"STARTING MODEL TRAINING - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    try:
        # Initialize configuration
        config = Config()
        
        # Create necessary directories
        os.makedirs(config.MODEL_DIR, exist_ok=True)
        os.makedirs(config.RESULTS_DIR, exist_ok=True)
        
        # Load and preprocess data
        print("\nLoading and preprocessing data...")
        data_loader = DataLoader(config)
        data = data_loader.load_datasets()
        
        # Dictionary of training functions
        training_functions = {
            "TF-IDF + Logistic Regression": train_tfidf_lr,
            "Word2Vec + Dense": train_w2v_dense,
            "LSTM with Pretrained Embeddings": train_lstm_pretrained,
            "LSTM from Scratch": train_lstm_scratch
        }
        
        # Train each model
        results = {}
        for name, train_func in training_functions.items():
            print("\n" + "="*80)
            print(f"TRAINING: {name.upper()}")
            print("="*80)
            
            success = train_func(data, config)
            results[name] = "SUCCESS" if success else "FAILED"
        
        # Print summary
        total_time = time.time() - start_time
        print("\n" + "="*80)
        print("TRAINING SUMMARY")
        print("="*80)
        
        for name, status in results.items():
            print(f"{name}: {status}")
        
        print(f"\nTotal training time: {total_time:.2f} seconds")
        print("="*80)
        
        if all(status == "SUCCESS" for status in results.values()):
            print("\nAll models trained successfully!")
            return 0
        else:
            print("\nSome models failed to train. Check the logs above for details.")
            return 1
            
    except Exception as e:
        print(f"\nFatal error during training: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
