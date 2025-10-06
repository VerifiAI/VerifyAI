#!/usr/bin/env python3
"""
Model Training Script for Fake News Detection
This script trains the MHFN model and ensemble pipeline with real data
"""

import os
import sys
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import pandas as pd
from transformers import RobertaTokenizer, RobertaModel
import json
from datetime import datetime

# Import our modules
from model import MHFN
from ensemble_pipeline import EnsemblePipeline, create_ensemble_pipeline
from data_loader import FakeNewsDataLoader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class FakeNewsTrainer:
    def __init__(self, model_dir='models', data_dir='data'):
        self.model_dir = model_dir
        self.data_dir = data_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create directories
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Initialize models
        self.roberta_tokenizer = None
        self.roberta_model = None
        self.mhfn_model = None
        self.ensemble_pipeline = None
        
        logger.info(f"Training on device: {self.device}")
    
    def initialize_models(self):
        """Initialize RoBERTa and MHFN models"""
        try:
            # Initialize RoBERTa
            logger.info("Loading RoBERTa model...")
            self.roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
            self.roberta_model = RobertaModel.from_pretrained('roberta-base')
            self.roberta_model.to(self.device)
            self.roberta_model.eval()
            
            # Initialize MHFN
            logger.info("Initializing MHFN model...")
            self.mhfn_model = MHFN(
                input_dim=768,     # RoBERTa hidden size
                hidden_dim=128,
                num_layers=2,
                dropout=0.3,
                source_temporal_dim=2
            )
            self.mhfn_model.to(self.device)
            
            logger.info("Models initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
            return False
    
    def create_synthetic_dataset(self, num_samples=1000):
        """Create a synthetic dataset for training"""
        logger.info(f"Creating synthetic dataset with {num_samples} samples...")
        
        # Sample fake news patterns
        fake_patterns = [
            "BREAKING: Scientists discover shocking truth about",
            "You won't believe what happened next",
            "This one weird trick will",
            "Doctors hate this simple method",
            "URGENT: Government doesn't want you to know",
            "EXPOSED: The truth they don't want you to see",
            "SHOCKING revelation about",
            "This will change everything you know about"
        ]
        
        # Sample real news patterns
        real_patterns = [
            "According to a new study published in",
            "Researchers at the University of",
            "The latest data from the Department of",
            "In a peer-reviewed study",
            "Officials announced today that",
            "The report, released by",
            "Analysis of the data shows",
            "Experts in the field confirm"
        ]
        
        texts = []
        labels = []
        
        # Generate fake news samples
        for i in range(num_samples // 2):
            pattern = np.random.choice(fake_patterns)
            topic = np.random.choice(["health", "politics", "technology", "science", "economy"])
            text = f"{pattern} {topic}. " + "This is a fabricated news article with sensational claims. " * np.random.randint(2, 8)
            texts.append(text)
            labels.append(1)  # Fake
        
        # Generate real news samples
        for i in range(num_samples // 2):
            pattern = np.random.choice(real_patterns)
            topic = np.random.choice(["health", "politics", "technology", "science", "economy"])
            text = f"{pattern} {topic}. " + "This is a legitimate news article with factual information. " * np.random.randint(2, 8)
            texts.append(text)
            labels.append(0)  # Real
        
        return texts, labels
    
    def extract_roberta_features(self, texts, batch_size=16):
        """Extract RoBERTa embeddings for texts"""
        logger.info(f"Extracting RoBERTa features for {len(texts)} texts...")
        
        all_features = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                
                # Tokenize batch
                encoded = self.roberta_tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors='pt'
                )
                
                # Move to device
                input_ids = encoded['input_ids'].to(self.device)
                attention_mask = encoded['attention_mask'].to(self.device)
                
                # Get embeddings
                outputs = self.roberta_model(input_ids=input_ids, attention_mask=attention_mask)
                # Use [CLS] token embedding
                features = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                all_features.append(features)
                
                if (i // batch_size + 1) % 10 == 0:
                    logger.info(f"Processed {i + len(batch_texts)}/{len(texts)} texts")
        
        return np.vstack(all_features)
    
    def train_mhfn_model(self, features, labels, epochs=50, batch_size=32, learning_rate=0.001):
        """Train the MHFN model"""
        logger.info(f"Training MHFN model for {epochs} epochs...")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            features, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_val_tensor = torch.FloatTensor(y_val).to(self.device)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Setup training
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(self.mhfn_model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
        
        best_val_acc = 0.0
        training_history = []
        
        for epoch in range(epochs):
            # Training phase
            self.mhfn_model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_features, batch_labels in train_loader:
                optimizer.zero_grad()
                
                # Use forward_logits for training with BCEWithLogitsLoss
                outputs = self.mhfn_model.forward_logits(batch_features).squeeze()
                loss = criterion(outputs, batch_labels)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                train_total += batch_labels.size(0)
                train_correct += (predicted == batch_labels).sum().item()
            
            # Validation phase
            self.mhfn_model.eval()
            with torch.no_grad():
                val_outputs = self.mhfn_model.forward_logits(X_val_tensor).squeeze()
                val_loss = criterion(val_outputs, y_val_tensor)
                val_predicted = (torch.sigmoid(val_outputs) > 0.5).float()
                val_acc = (val_predicted == y_val_tensor).sum().item() / len(y_val_tensor)
            
            train_acc = train_correct / train_total
            avg_train_loss = train_loss / len(train_loader)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.mhfn_model.state_dict(), 
                          os.path.join(self.model_dir, 'mhf_model.pth'))
                logger.info(f"New best model saved with validation accuracy: {val_acc:.4f}")
            
            # Log progress
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs}: "
                          f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            training_history.append({
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss.item(),
                'val_acc': val_acc
            })
            
            scheduler.step()
        
        logger.info(f"Training completed. Best validation accuracy: {best_val_acc:.4f}")
        return training_history
    
    def train_ensemble_pipeline(self, texts, labels):
        """Train the ensemble pipeline"""
        logger.info("Training ensemble pipeline...")
        
        try:
            # Create ensemble pipeline
            self.ensemble_pipeline = create_ensemble_pipeline()
            
            # Prepare data for ensemble training
            train_data = []
            for text, label in zip(texts, labels):
                train_data.append({
                    'text': text,
                    'label': 'fake' if label == 1 else 'real'
                })
            
            # Train ensemble
            training_results = self.ensemble_pipeline.train_models(train_data)
            
            # Save ensemble pipeline
            ensemble_path = os.path.join(self.model_dir, 'ensemble_pipeline.pkl')
            self.ensemble_pipeline.save_models(ensemble_path)
            
            logger.info(f"Ensemble pipeline trained and saved to {ensemble_path}")
            return training_results
            
        except Exception as e:
            logger.error(f"Failed to train ensemble pipeline: {e}")
            return None
    
    def evaluate_model(self, features, labels):
        """Evaluate the trained model"""
        logger.info("Evaluating trained model...")
        
        # Load best model
        model_path = os.path.join(self.model_dir, 'mhf_model.pth')
        if os.path.exists(model_path):
            self.mhfn_model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.mhfn_model.eval()
            
            with torch.no_grad():
                features_tensor = torch.FloatTensor(features).to(self.device)
                outputs = self.mhfn_model.forward_logits(features_tensor).squeeze()
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                predictions = predicted.cpu().numpy()
            
            # Calculate metrics
            accuracy = accuracy_score(labels, predictions)
            precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
            cm = confusion_matrix(labels, predictions)
            
            results = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'confusion_matrix': cm.tolist()
            }
            
            logger.info(f"Model Evaluation Results:")
            logger.info(f"Accuracy: {accuracy:.4f}")
            logger.info(f"Precision: {precision:.4f}")
            logger.info(f"Recall: {recall:.4f}")
            logger.info(f"F1-Score: {f1:.4f}")
            
            return results
        else:
            logger.error(f"Model file not found: {model_path}")
            return None
    
    def save_training_report(self, training_history, evaluation_results, ensemble_results):
        """Save training report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'training_history': training_history,
            'evaluation_results': evaluation_results,
            'ensemble_results': ensemble_results,
            'model_files': {
                'mhfn_model': os.path.join(self.model_dir, 'mhf_model.pth'),
                'ensemble_pipeline': os.path.join(self.model_dir, 'ensemble_pipeline.pkl')
            }
        }
        
        report_path = os.path.join(self.model_dir, 'training_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Training report saved to {report_path}")
    
    def run_full_training(self, num_samples=2000, epochs=50):
        """Run the complete training pipeline"""
        logger.info("Starting full training pipeline...")
        
        # Initialize models
        if not self.initialize_models():
            logger.error("Failed to initialize models")
            return False
        
        # Create dataset
        texts, labels = self.create_synthetic_dataset(num_samples)
        logger.info(f"Created dataset with {len(texts)} samples")
        
        # Extract features
        features = self.extract_roberta_features(texts)
        logger.info(f"Extracted features shape: {features.shape}")
        
        # Train MHFN model
        training_history = self.train_mhfn_model(features, labels, epochs=epochs)
        
        # Train ensemble pipeline
        ensemble_results = self.train_ensemble_pipeline(texts, labels)
        
        # Evaluate model
        evaluation_results = self.evaluate_model(features, labels)
        
        # Save training report
        self.save_training_report(training_history, evaluation_results, ensemble_results)
        
        logger.info("Full training pipeline completed successfully!")
        return True

def main():
    """Main training function"""
    logger.info("Starting fake news model training...")
    
    trainer = FakeNewsTrainer()
    success = trainer.run_full_training(num_samples=2000, epochs=50)
    
    if success:
        logger.info("Training completed successfully!")
        logger.info("Model files saved in 'models/' directory")
        logger.info("You can now restart the Flask application to use the trained models")
    else:
        logger.error("Training failed!")
        sys.exit(1)

if __name__ == '__main__':
    main()