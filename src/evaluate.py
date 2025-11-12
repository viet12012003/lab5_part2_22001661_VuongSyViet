import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
from sklearn.metrics import classification_report, f1_score

from configs.config import Config
from src.data.data_loader import DataLoader
from src.models.tfidf_lr import TFIDF_LogisticRegression
from src.models.w2v_dense import W2V_Dense
from src.models.lstm_models import LSTM_Pretrained, LSTM_Scratch

class ModelEvaluator:
    def __init__(self, config):
        """Initialize the ModelEvaluator with configuration.
        
        Args:
            config: Configuration object containing paths and parameters
        """
        self.config = config
        self.data_loader = DataLoader(config)
        self.results = []
        
    def load_data(self):
        """Load and preprocess the test data."""
        print("Loading data...")
        data = self.data_loader.load_datasets()
        self.X_test = data['X_test']
        self.y_test = data['y_test']
        self.label_encoder = self.data_loader.get_label_encoder()
        
    def load_models(self):
        """Load all trained models with error handling."""
        print("\nLoading models...")
        # Create results directory if it doesn't exist
        os.makedirs(self.config.RESULTS_DIR, exist_ok=True)
        
        # Initialize all models with config
        self.models = {}
        
        def try_load_model(model_name, loader, path, config=None):
            """Helper function to safely load a model with error handling."""
            try:
                if not os.path.exists(str(path)):
                    print(f"Model file not found: {path}")
                    return None
                print(f"Loading {model_name}...")
                if config is not None:
                    return loader(str(path), config=config)
                return loader(str(path))
            except Exception as e:
                print(f"Error loading {model_name}: {str(e)}")
                import traceback
                traceback.print_exc()
                return None
        
        # Try loading each model with error handling
        models_to_load = [
            ('TF-IDF + Logistic Regression', TFIDF_LogisticRegression.load_model, self.config.TFIDF_MODEL, self.config),
            ('Word2Vec + Dense', W2V_Dense.load_model, self.config.W2V_MODEL, self.config),
            ('LSTM (Pre-trained)', LSTM_Pretrained.load_model, self.config.LSTM_PRETRAINED, self.config),
            ('LSTM (Scratch)', LSTM_Scratch.load_model, self.config.LSTM_SCRATCH, self.config)
        ]
        
        for name, loader, path, config in models_to_load:
            model = try_load_model(name, loader, path, config)
            if model is not None:
                self.models[name] = model
        
        if not self.models:
            raise RuntimeError("\nFailed to load any models. Please check the following:\n"
                            "1. Model files exist in the expected locations\n"
                            "2. You have the correct versions of all dependencies\n"
                            "3. The models were saved correctly during training\n"
                            "You may need to retrain the models if the files are corrupted.")
                            
        print(f"\nSuccessfully loaded {len(self.models)} out of {len(models_to_load)} models.")
        return self.models
        
    def evaluate_all_models(self):
        """Evaluate all models on the test set with enhanced error handling."""
        print("\nEvaluating models...\n")
        
        # Track which models have been evaluated to prevent duplicates
        evaluated_models = set()
        
        for model_name, model in list(self.models.items()):
            try:
                print(f"\n=== Evaluating {model_name} ===")
                
                # Skip if model is None (failed to load)
                if model is None:
                    print(f"[WARNING] {model_name} - model is None, skipping...")
                    self.models.pop(model_name, None)
                    continue
                    
                # Additional check for Word2Vec + Dense model
                if 'Word2Vec' in model_name and not hasattr(model, 'w2v_model'):
                    print(f"[WARNING] {model_name} - Word2Vec model not found, skipping...")
                    self.models.pop(model_name, None)
                    continue
                    
                # Skip if this model has already been evaluated
                if model_name in evaluated_models:
                    print(f"[INFO] {model_name} - already evaluated, skipping...")
                    continue
                
                # Make predictions on test set with better error handling
                try:
                    if 'TF-IDF' in model_name:
                        # For TF-IDF models
                        print("Making predictions with TF-IDF model...")
                        y_pred = model.predict(self.X_test)
                        y_proba = model.predict_proba(self.X_test) if hasattr(model, 'predict_proba') else None
                        
                    else:
                        # For neural models (Word2Vec + Dense, LSTM, etc.)
                        print("Preparing data for neural model...")
                        
                        # Check if model is properly initialized
                        if not hasattr(model, 'model') or model.model is None:
                            print(f"[WARNING] {model_name} - Model not properly initialized")
                            self.models.pop(model_name, None)
                            continue
                            
                        # Special handling for Word2Vec + Dense
                        if 'Word2Vec' in model_name and hasattr(model, 'evaluate'):
                            print(f"Word2Vec vocabulary size: {len(model.w2v_model.wv) if hasattr(model.w2v_model, 'wv') else 'N/A'}")
                            
                            # Check if Word2Vec model is properly trained
                            if not hasattr(model.w2v_model, 'wv') or len(model.w2v_model.wv) == 0:
                                print("[WARNING] Word2Vec model has no word vectors, skipping...")
                                self.models.pop(model_name, None)
                                continue
                            
                            # Evaluate the model
                            print("Evaluating Word2Vec + Dense model...")
                            try:
                                eval_results = model.evaluate(
                                    self.X_test, 
                                    self.y_test, 
                                    output_dir=str(self.config.RESULTS_DIR / 'w2v_dense')
                                )
                                
                                # Add results to the summary
                                result = {
                                    'Model': model_name,
                                    'F1-Score (Macro)': eval_results['classification_report']['macro avg']['f1-score'],
                                    'Test Loss': eval_results['test_loss']
                                }
                                self.results.append(result)
                                evaluated_models.add(model_name)
                                
                                print(f"Evaluation complete for {model_name}")
                                print(f"F1-Score (Macro): {result['F1-Score (Macro)']:.4f}")
                                print(f"Test Loss: {result['Test Loss']:.4f}")
                                
                                # Skip the rest of the prediction logic
                                continue
                                
                            except Exception as e:
                                print(f"[ERROR] Error evaluating {model_name}: {str(e)}")
                                import traceback
                                traceback.print_exc()
                                self.models.pop(model_name, None)
                                continue
                            
                        # Special handling for Word2Vec + Dense
                        if 'Word2Vec' in model_name and hasattr(model, 'w2v_model'):
                            print(f"Word2Vec vocabulary size: {len(model.w2v_model.wv) if hasattr(model.w2v_model, 'wv') else 'N/A'}")
                            
                            # Check if Word2Vec model is properly trained
                            if not hasattr(model.w2v_model, 'wv') or len(model.w2v_model.wv) == 0:
                                print("[WARNING] Word2Vec model has no word vectors, skipping...")
                                self.models.pop(model_name, None)
                                continue
                            
                            # Convert text to document vectors
                            print("Converting text to document vectors...")
                            try:
                                X_test_vec = np.array([model.document_vector(doc) for doc in self.X_test])
                                print(f"Document vectors shape: {X_test_vec.shape}")
                                
                                # Make predictions
                                print("Making predictions...")
                                y_pred_proba = model.model.predict(X_test_vec, verbose=0)
                                y_pred = np.argmax(y_pred_proba, axis=1)
                                y_proba = y_pred_proba
                                
                                # Skip the rest of the prediction logic
                                continue
                                
                            except Exception as e:
                                print(f"[ERROR] Error in document vector prediction: {str(e)}")
                                raise
                        
                        # Prepare data for prediction
                        try:
                            if hasattr(model, 'prepare_sequences'):
                                print("Preparing sequences...")
                                X_test_processed = model.prepare_sequences(self.X_test)
                            else:
                                X_test_processed = self.X_test
                                
                            # Make predictions
                            print("Making predictions...")
                            if hasattr(model, 'predict'):
                                y_pred = model.predict(self.X_test)
                                y_proba = model.model.predict(X_test_processed, verbose=0) if hasattr(model, 'model') else None
                            else:
                                y_pred = model.model.predict(X_test_processed, verbose=0)
                                y_proba = y_pred  # For models that output probabilities directly
                                
                            # Convert probabilities to class predictions if needed
                            if y_proba is not None and len(y_proba.shape) > 1 and y_proba.shape[1] > 1:
                                y_pred = np.argmax(y_proba, axis=1)
                                
                        except Exception as e:
                            print(f"[ERROR] Error during prediction: {str(e)}")
                            print("Attempting fallback prediction method...")
                            
                            # Fallback method for Word2Vec + Dense
                            if 'Word2Vec' in model_name and hasattr(model, 'document_vector'):
                                try:
                                    print("Using fallback document vector method...")
                                    X_test_vec = np.array([model.document_vector(doc) for doc in self.X_test])
                                    y_pred = model.model.predict(X_test_vec, verbose=0)
                                    y_proba = y_pred
                                    y_pred = (y_pred > 0.5).astype(int).flatten()
                                except Exception as e2:
                                    print(f"[ERROR] Fallback method failed: {str(e2)}")
                                    raise e
                            else:
                                raise e
                                
                except Exception as e:
                    if 'Word2Vec' in model_name and hasattr(model, 'document_vector'):
                        print("Trying alternative prediction method...")
                        try:
                            # Alternative prediction using document vectors
                            X_test_vec = []
                            for doc in self.X_test:
                                vec = model.document_vector(doc)
                                if vec is not None:
                                    X_test_vec.append(vec)
                                else:
                                    # Handle case where document_vector returns None
                                    X_test_vec.append(np.zeros(model.vector_size))
                            
                            X_test_vec = np.array(X_test_vec)
                            print(f"Document vectors shape: {X_test_vec.shape}")
                            
                            # Make predictions
                            y_pred_proba = model.model.predict(X_test_vec, verbose=0)
                            y_pred = np.argmax(y_pred_proba, axis=1)
                            y_proba = y_pred_proba
                            
                            # If we got here, continue with evaluation
                            print("Successfully made predictions with alternative method")
                            
                        except Exception as e2:
                            print(f"[ERROR] Alternative prediction method failed: {str(e2)}")
                            print("Skipping this model...")
                            self.models.pop(model_name, None)
                            continue
                    else:
                        print(f"[ERROR] Failed to make predictions with {model_name}: {str(e)}")
                        import traceback
                        traceback.print_exc()
                        print("Skipping this model...")
                        self.models.pop(model_name, None)
                        continue
                
                try:
                    # Calculate evaluation metrics
                    f1_macro = f1_score(self.y_test, y_pred, average='macro')
                    
                    # Store results
                    result = {'Model': model_name, 'F1-Score (Macro)': f1_macro}
                    
                    # Try to get test loss for non-TF-IDF models
                    if 'TF-IDF' not in model_name and hasattr(model, 'model'):
                        try:
                            if hasattr(model, 'prepare_sequences'):
                                X_test_processed = model.prepare_sequences(self.X_test)
                                test_loss = model.model.evaluate(X_test_processed, self.y_test, verbose=0)
                                result['Test Loss'] = test_loss[0] if isinstance(test_loss, list) else test_loss
                            else:
                                test_loss = model.model.evaluate(self.X_test, self.y_test, verbose=0)
                                result['Test Loss'] = test_loss[0] if isinstance(test_loss, list) else test_loss
                        except:
                            result['Test Loss'] = 'N/A'
                    else:
                        result['Test Loss'] = 'N/A'
                    
                    self.results.append(result)
                    
                    # Print detailed classification report
                    print(f"\n{model_name} Classification Report:")
                    print(classification_report(
                        self.y_test, 
                        y_pred, 
                        target_names=self.label_encoder.classes_
                    ))
                    
                    # Save prediction results if we have probabilities
                    if y_proba is not None:
                        self._save_predictions(model_name, y_pred, y_proba)
                    
                except Exception as e:
                    print(f"Error evaluating {model_name}: {str(e)}")
                    continue
                
            except Exception as e:
                print(f"Unexpected error evaluating {model_name}: {str(e)}")
                print("Skipping this model...")
                self.models.pop(model_name, None)
                continue
        
        if not self.models:
            print("\nNo models were successfully evaluated.")
        else:
            print(f"\nSuccessfully evaluated {len(self.models)} models.")
    
    def _save_predictions(self, model_name, y_pred, y_proba):
        """Save prediction results to CSV."""
        results_df = pd.DataFrame({
            'text': self.X_test,
            'true_label': self.label_encoder.inverse_transform(self.y_test),
            'predicted_label': self.label_encoder.inverse_transform(y_pred),
            'confidence': np.max(y_proba, axis=1) if y_proba is not None else [1.0] * len(y_pred)
        })
        
        # Save results to CSV
        output_file = self.config.RESULTS_DIR / f"{model_name.lower().replace(' ', '_')}_predictions.csv"
        results_df.to_csv(output_file, index=False)
        print(f"Saved predictions to {output_file}")
    
    def analyze_hard_cases(self):
        """Analyze challenging test cases and save results."""
        print("\nAnalyzing hard cases...")
        
        # Test cases with true intents
        test_cases = [
            ("can you remind me to not call my mom", "reminder_create"),
            ("is it going to be sunny or rainy tomorrow", "weather_query"),
            ("find a flight from new york to london but not through paris", "flight_search")
        ]
        
        results = []
        results_dir = self.config.RESULTS_DIR
        results_dir.mkdir(parents=True, exist_ok=True)
        
        with open(results_dir / 'hard_cases_analysis.txt', 'w', encoding='utf-8') as f:
            for text, true_intent in test_cases:
                f.write(f"\nText: {text}\n")
                f.write(f"True Intent: {true_intent}\n" + "-"*50 + "\n")
                
                case = {'text': text, 'true_intent': true_intent}
                
                for name, model in self.models.items():
                    try:
                        # Get prediction
                        if 'TF-IDF' in name:
                            pred = model.predict([text])[0]
                            proba = model.predict_proba([text])[0] if hasattr(model, 'predict_proba') else None
                        elif 'Word2Vec' in name and hasattr(model, 'predict'):
                            try:
                                # Get both predictions and probabilities
                                pred, pred_proba = model.predict([text], return_proba=True)
                                pred = pred[0]  # Get single prediction
                                proba = pred_proba[0]  # Get probabilities for the single prediction
                                
                            except Exception as e:
                                raise ValueError(f"Error processing Word2Vec prediction: {str(e)}")
                        else:  # LSTM models
                            if hasattr(model, 'prepare_sequences'):
                                seq = model.prepare_sequences([text])
                                proba = model.model.predict(seq, verbose=0)[0]
                            else:
                                proba = model.model.predict([text], verbose=0)[0]
                            pred = np.argmax(proba)
                        
                        # Get prediction details
                        pred_label = self.label_encoder.inverse_transform([pred])[0]
                        
                        # Ensure confidence is a scalar between 0 and 1
                        if proba is not None and len(proba) > 0:
                            confidence = float(proba[pred])
                            # Ensure confidence is within [0,1] range
                            confidence = max(0.0, min(1.0, confidence))
                        else:
                            confidence = 0.0
                            
                        is_correct = (pred_label == true_intent)
                        
                        # Log result
                        f.write(f"{name}: {pred_label} (Confidence: {confidence:.2f})\n")
                        case[name] = {
                            'prediction': pred_label,
                            'confidence': confidence,
                            'is_correct': is_correct
                        }
                        
                    except Exception as e:
                        f.write(f"{name}: Error - {str(e)}\n")
                        case[name] = {
                            'prediction': f"Error: {str(e)}",
                            'confidence': 0.0,
                            'is_correct': False
                        }
                
                results.append(case)
                f.write("\n" + "="*50 + "\n")
        
        # Save detailed results
        pd.DataFrame(results).to_csv(results_dir / 'hard_cases_detailed.csv', index=False)
        print(f"Results saved to {results_dir / 'hard_cases_analysis.txt'}")
        
        # Print summary
        self._print_summary(results)
        return results
        
    def _print_summary(self, results):
        """Print summary of model performance on test cases."""
        if not results:
            return
            
        print("\n" + "="*50)
        print("HARD CASES ANALYSIS SUMMARY")
        print("="*50)
        
        # Get model names (excluding 'text' and 'true_intent')
        model_names = [k for k in results[0].keys() if k not in ['text', 'true_intent']]
        
        # Calculate stats
        stats = {}
        for name in model_names:
            correct = sum(1 for r in results if r.get(name, {}).get('is_correct', False))
            total = len([r for r in results if name in r])
            confidences = [r[name].get('confidence', 0) for r in results if name in r]
            
            stats[name] = {
                'accuracy': correct / total if total > 0 else 0,
                'avg_confidence': np.mean(confidences) if confidences else 0,
                'correct': correct,
                'total': total
            }
        
        # Print table
        print(f"\n{'Model':<30} | {'Correct':<8} | {'Total':<5} | {'Accuracy':<10} | {'Avg Confidence'}")
        print("-" * 70)
        for name, s in stats.items():
            print(f"{name:<30} | {s['correct']:>7} | {s['total']:>4} | {s['accuracy']:>8.2%}   | {s['avg_confidence']:.4f}")
        
        print("\n" + "="*50 + "\n")
    
    def generate_summary(self):
        """Generate a summary table of model comparison results."""
        # Create DataFrame from results
        df = pd.DataFrame(self.results)
        
        # Save to file
        output_file = self.config.RESULTS_DIR / 'model_comparison.csv'
        df.to_csv(output_file, index=False)
        
        # Print results table
        print("\n" + "="*60)
        print("MODEL COMPARISON SUMMARY")
        print("="*60)
        print(tabulate(df, headers='keys', tablefmt='grid', showindex=False, floatfmt=".4f"))
        print(f"\nDetailed results saved to: {output_file}")
        
        # Generate comparison plot
        self._plot_comparison(df)
    
    def _plot_comparison(self, df):
        """Generate comparison plots for model evaluation metrics."""
        plt.figure(figsize=(12, 6))
        
        # F1-Score plot
        plt.subplot(1, 2, 1)
        sns.barplot(x='Model', y='F1-Score (Macro)', data=df)
        plt.title('F1-Score (Macro) Comparison')
        plt.xticks(rotation=45, ha='right')
        
        # Loss plot (if available)
        if 'Test Loss' in df.columns and not all(df['Test Loss'] == 'N/A'):
            loss_df = df[df['Test Loss'] != 'N/A']
            if not loss_df.empty:
                plt.subplot(1, 2, 2)
                sns.barplot(x='Model', y='Test Loss', data=loss_df)
                plt.title('Test Loss Comparison')
                plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        
        # Save the plot
        plot_file = self.config.RESULTS_DIR / 'model_comparison.png'
        plt.savefig(plot_file, bbox_inches='tight')
        plt.close()
        print(f"\nComparison plot saved to: {plot_file}")

def main():
    # Initialize configuration
    config = Config()
    
    # Initialize evaluator
    evaluator = ModelEvaluator(config)
    
    # Load data
    evaluator.load_data()
    
    # Load trained models
    evaluator.load_models()
    
    # Evaluate all models
    print("\nEvaluating models...")
    evaluator.evaluate_all_models()
    
    # Analyze challenging cases
    print("\nAnalyzing hard cases...")
    evaluator.analyze_hard_cases()
    
    # Generate summary report
    print("\nGenerating summary report...")
    evaluator.generate_summary()

if __name__ == "__main__":
    main()
