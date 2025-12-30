"""
Train and Compare module
Trains DecisionTree and RandomForest models, compares them, and saves the best one
"""

import pandas as pd
import numpy as np
import pickle
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model_factory import DecisionTreeModel, RandomForestModel


def load_and_prepare_data(data_path: str):
    """Load and prepare training data"""
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Separate features and target
    X = df.drop('protocol_id', axis=1)
    y = df['protocol_id']
    
    # Encode categorical features
    label_encoders = {}
    for column in X.columns:
        if X[column].dtype == 'object':
            le = LabelEncoder()
            X[column] = le.fit_transform(X[column])
            label_encoders[column] = le
    
    print(f"Data loaded: {len(df)} samples, {len(X.columns)} features")
    print(f"Features: {list(X.columns)}")
    print(f"Target classes: {sorted(y.unique())}")
    
    return X.values, y.values, label_encoders


def train_and_compare(data_path: str, model_output_dir: str):
    """Train both models, compare them, and save the best one"""
    
    # Load and prepare data
    X, y, label_encoders = load_and_prepare_data(data_path)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Initialize models
    models = {
        'DecisionTree': DecisionTreeModel(max_depth=10, random_state=42),
        'RandomForest': RandomForestModel(n_estimators=100, max_depth=10, random_state=42)
    }
    
    results = {}
    
    # Train and evaluate each model
    print("\n" + "="*60)
    print("TRAINING AND EVALUATION")
    print("="*60)
    
    for name, model in models.items():
        print(f"\n{name}:")
        print("-" * 40)
        
        # Train
        print("Training...")
        model.train(X_train, y_train)
        
        # Evaluate
        print("Evaluating...")
        metrics = model.evaluate(X_test, y_test)
        results[name] = {
            'model': model,
            'metrics': metrics
        }
        
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1_score']:.4f}")
    
    # Compare models and select the best one
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    
    best_model_name = None
    best_score = -1
    
    for name, result in results.items():
        # Use F1-score as the primary metric (harmonic mean of precision and recall)
        score = result['metrics']['f1_score']
        print(f"\n{name}:")
        print(f"  Combined Score (F1): {score:.4f}")
        
        if score > best_score:
            best_score = score
            best_model_name = name
    
    # Save the best model
    print("\n" + "="*60)
    print(f"BEST MODEL: {best_model_name}")
    print("="*60)
    print(f"F1-Score: {best_score:.4f}")
    print(f"Precision: {results[best_model_name]['metrics']['precision']:.4f}")
    print(f"Recall: {results[best_model_name]['metrics']['recall']:.4f}")
    
    # Create models directory if it doesn't exist
    os.makedirs(model_output_dir, exist_ok=True)
    
    best_model_path = os.path.join(model_output_dir, 'best_model.pkl')
    results[best_model_name]['model'].save(best_model_path)
    print(f"\nBest model saved to: {best_model_path}")
    
    # Also save label encoders for later use
    encoders_path = os.path.join(model_output_dir, 'label_encoders.pkl')
    with open(encoders_path, 'wb') as f:
        pickle.dump(label_encoders, f)
    print(f"Label encoders saved to: {encoders_path}")
    
    return best_model_name, results[best_model_name]['metrics']


if __name__ == "__main__":
    # Define paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'train.csv')
    model_output_dir = os.path.join(base_dir, 'models')
    
    # Train and compare models
    best_model, metrics = train_and_compare(data_path, model_output_dir)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
