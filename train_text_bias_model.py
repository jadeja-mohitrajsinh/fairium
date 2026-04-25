"""Train ML classifier for text bias detection."""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
from pathlib import Path


def train_bias_classifier():
    """Train a text bias classifier using the labeled dataset."""
    print("Loading training data...")
    
    # Load training data
    df = pd.read_csv('data/text_bias_training_data.csv')
    
    # Prepare features and labels
    X = df['text']
    y = df['label']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create TF-IDF vectorizer
    print("Creating TF-IDF features...")
    vectorizer = TfidfVectorizer(
        max_features=1000,
        ngram_range=(1, 2),
        stop_words='english',
        min_df=1,
        max_df=0.9
    )
    
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Train classifier
    print("Training classifier...")
    classifier = LogisticRegression(
        random_state=42,
        max_iter=1000,
        class_weight='balanced'
    )
    classifier.fit(X_train_tfidf, y_train)
    
    # Evaluate
    y_pred = classifier.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel accuracy: {accuracy:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save model and vectorizer
    model_dir = Path('data/models')
    model_dir.mkdir(exist_ok=True, parents=True)
    
    model_data = {
        'classifier': classifier,
        'vectorizer': vectorizer,
        'bias_types': df['bias_type'].unique().tolist()
    }
    
    joblib.dump(model_data, 'data/models/text_bias_classifier.joblib')
    print("\nModel saved to data/models/text_bias_classifier.joblib")
    
    return model_data


if __name__ == '__main__':
    train_bias_classifier()
