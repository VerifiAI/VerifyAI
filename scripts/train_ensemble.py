#!/usr/bin/env python3
"""
Ensemble Pipeline Training Script
Generates sample data and trains the ensemble pipeline for fake news detection
"""

import logging
import numpy as np
from sklearn.model_selection import train_test_split
from ensemble_pipeline import create_ensemble_pipeline
import joblib
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_sample_data(n_samples=1000):
    """
    Generate sample fake news data for training
    """
    logger.info(f"Generating {n_samples} sample texts...")
    
    # Sample fake news texts
    fake_texts = [
        "BREAKING: Scientists discover aliens living among us in secret underground cities!",
        "SHOCKING: Government hiding cure for cancer to protect pharmaceutical profits!",
        "URGENT: 5G towers causing mass bird deaths and mind control experiments!",
        "EXCLUSIVE: Celebrity caught in massive conspiracy to control world economy!",
        "WARNING: Vaccines contain microchips for government surveillance programs!",
        "REVEALED: Climate change is hoax created by secret world government!",
        "ALERT: Social media platforms secretly reading your private thoughts!",
        "EXPOSED: Fast food chains using human meat in their products!",
        "CRITICAL: Water supply contaminated with mind-altering chemicals!",
        "URGENT: Moon landing was filmed in Hollywood studio basement!"
    ] * (n_samples // 20 + 1)
    
    # Sample real news texts
    real_texts = [
        "Local government announces new infrastructure development project for downtown area.",
        "University researchers publish study on renewable energy efficiency improvements.",
        "Stock market shows steady growth following positive economic indicators this quarter.",
        "New healthcare facility opens to serve rural communities in the region.",
        "Technology company launches innovative software solution for small businesses.",
        "Environmental protection agency reports improvement in air quality standards.",
        "Education department introduces new curriculum for STEM subjects in schools.",
        "Transportation authority completes major highway renovation project ahead of schedule.",
        "Agricultural department releases guidelines for sustainable farming practices.",
        "Public health officials recommend seasonal vaccination for vulnerable populations."
    ] * (n_samples // 20 + 1)
    
    # Combine and create labels
    texts = fake_texts[:n_samples//2] + real_texts[:n_samples//2]
    labels = [1] * (n_samples//2) + [0] * (n_samples//2)  # 1 = fake, 0 = real
    
    # Shuffle the data
    combined = list(zip(texts, labels))
    np.random.shuffle(combined)
    texts, labels = zip(*combined)
    
    return list(texts), list(labels)

def train_ensemble_pipeline():
    """
    Train the ensemble pipeline with sample data
    """
    logger.info("Starting ensemble pipeline training...")
    
    # Generate sample data
    texts, labels = generate_sample_data(1000)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    logger.info(f"Training set: {len(X_train)} samples")
    logger.info(f"Test set: {len(X_test)} samples")
    
    # Create and train pipeline
    pipeline = create_ensemble_pipeline(use_optuna=False, n_trials=10)  # Faster training
    
    # Train the pipeline
    pipeline.fit(X_train, y_train)
    
    # Test the pipeline
    predictions = pipeline.predict(X_test)
    accuracy = np.mean(predictions['ensemble_prediction'] == y_test)
    logger.info(f"Test accuracy: {accuracy:.3f}")
    
    # Save the trained pipeline
    model_path = 'trained_ensemble_pipeline.pkl'
    pipeline.save_pipeline(model_path)
    logger.info(f"Pipeline saved to {model_path}")
    
    return pipeline

if __name__ == "__main__":
    try:
        trained_pipeline = train_ensemble_pipeline()
        logger.info("✓ Ensemble pipeline training completed successfully!")
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise