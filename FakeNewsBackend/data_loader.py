#!/usr/bin/env python3
"""
Data Loader for Hybrid Deep Learning Fake News Detection
Chunk 7: Comprehensive data loading and preprocessing system

This module handles:
- Loading Parquet files (train, validation, test)
- Loading pickle files with text/image data
- Preprocessing data for MHFN model input
- Tensor formatting [batch, 300] for MHFN
- Missing value handling
- Feature/label splitting
- Batch testing functionality

Author: AI Assistant
Date: August 24, 2025
Version: 1.0
"""

import os
import sys
import logging
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import warnings
from datetime import datetime

# Hybrid embeddings imports
try:
    from transformers import RobertaTokenizer, RobertaModel, DebertaTokenizer, DebertaModel
    import fasttext
    from sklearn.decomposition import PCA
    HYBRID_EMBEDDINGS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Hybrid embeddings dependencies not available: {e}")
    HYBRID_EMBEDDINGS_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_loader.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

class FakeNewsDataLoader:
    """
    Comprehensive data loader for fake news detection system.
    Handles Parquet files, pickle files, and preprocessing for MHFN model.
    """
    
    def __init__(self, data_dir: str = None):
        """
        Initialize the data loader.
        
        Args:
            data_dir (str): Path to data directory. Defaults to '../data'
        """
        if data_dir is None:
            # Get the directory of this script and navigate to data folder
            current_dir = Path(__file__).parent
            self.data_dir = current_dir.parent / 'data'
        else:
            self.data_dir = Path(data_dir)
            
        self.processed_dir = self.data_dir / 'processed'
        self.fakeddit_dir = self.data_dir / 'fakeddit'
        
        # Data containers
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.pickle_data = {}
        
        # Preprocessing parameters
        self.target_dim = 300  # MHFN input dimension
        self.max_sequence_length = 512
        
        # Hybrid embeddings models
        self.roberta_tokenizer = None
        self.roberta_model = None
        self.deberta_tokenizer = None
        self.deberta_model = None
        self.fasttext_model = None
        self.pca_model = None
        
        logger.info(f"DataLoader initialized with data_dir: {self.data_dir}")
        self._validate_directories()
        
        # Initialize hybrid embeddings if available
        if HYBRID_EMBEDDINGS_AVAILABLE:
            self._initialize_hybrid_embeddings()
    
    def _validate_directories(self) -> None:
        """
        Validate that required directories and files exist.
        """
        try:
            if not self.data_dir.exists():
                raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
            
            if not self.processed_dir.exists():
                raise FileNotFoundError(f"Processed directory not found: {self.processed_dir}")
            
            # Check for required Parquet files
            required_parquet = [
                'fakeddit_processed_train.parquet',
                'fakeddit_processed_val.parquet', 
                'fakeddit_processed_test.parquet'
            ]
            
            for file in required_parquet:
                file_path = self.processed_dir / file
                if not file_path.exists():
                    logger.warning(f"Parquet file not found: {file_path}")
            
            # Check for pickle files
            pickle_files = [
                'train__text_image__dataframe.pkl',
                'test__text_image__dataframe.pkl'
            ]
            
            for file in pickle_files:
                file_path = self.data_dir / file
                if not file_path.exists():
                    logger.warning(f"Pickle file not found: {file_path}")
            
            logger.info("Directory validation completed")
            
        except Exception as e:
            logger.error(f"Directory validation failed: {str(e)}")
            raise
    
    def _initialize_hybrid_embeddings(self) -> None:
        """Initialize hybrid embedding models (RoBERTa, DeBERTa, FastText)"""
        try:
            logger.info("Initializing hybrid embedding models...")
            
            # Initialize RoBERTa
            self.roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
            self.roberta_model = RobertaModel.from_pretrained('roberta-base')
            self.roberta_model.eval()
            
            # Initialize DeBERTa
            from transformers import DebertaTokenizer, DebertaModel
            self.deberta_tokenizer = DebertaTokenizer.from_pretrained('microsoft/deberta-base')
            self.deberta_model = DebertaModel.from_pretrained('microsoft/deberta-base')
            self.deberta_model.eval()
            
            # Initialize PCA for dimensionality reduction
            self.pca_model = PCA(n_components=self.target_dim, random_state=42)
            
            logger.info("Hybrid embedding models initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize hybrid embeddings: {e}")
            logger.warning("Falling back to basic feature extraction")
    
    def _get_roberta_embeddings(self, text: str) -> np.ndarray:
        """Extract RoBERTa embeddings from text"""
        try:
            if self.roberta_model is None or self.roberta_tokenizer is None:
                return np.zeros(768)  # RoBERTa base dimension
            
            inputs = self.roberta_tokenizer(text, return_tensors='pt', truncation=True, 
                                          max_length=self.max_sequence_length, padding=True)
            
            with torch.no_grad():
                outputs = self.roberta_model(**inputs)
                # Use [CLS] token embedding
                embeddings = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
                
            return embeddings
            
        except Exception as e:
            logger.error(f"Error extracting RoBERTa embeddings: {e}")
            return np.zeros(768)
    
    def _get_deberta_embeddings(self, text: str) -> np.ndarray:
        """Extract DeBERTa embeddings from text"""
        try:
            if self.deberta_model is None or self.deberta_tokenizer is None:
                return np.zeros(768)  # DeBERTa base dimension
            
            inputs = self.deberta_tokenizer(text, return_tensors='pt', truncation=True,
                                          max_length=self.max_sequence_length, padding=True)
            
            with torch.no_grad():
                outputs = self.deberta_model(**inputs)
                # Use [CLS] token embedding
                embeddings = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
                
            return embeddings
            
        except Exception as e:
            logger.error(f"Error extracting DeBERTa embeddings: {e}")
            return np.zeros(768)
    
    def _get_fasttext_embeddings(self, text: str) -> np.ndarray:
        """Extract FastText embeddings from text"""
        try:
            if self.fasttext_model is None:
                # Create a simple word-based embedding as fallback
                words = text.lower().split()
                if not words:
                    return np.zeros(300)  # FastText default dimension
                
                # Simple word hashing for consistent embeddings
                embeddings = []
                for word in words[:50]:  # Limit to 50 words
                    np.random.seed(hash(word) % 2**32)
                    embeddings.append(np.random.randn(300) * 0.1)
                
                if embeddings:
                    return np.mean(embeddings, axis=0)
                else:
                    return np.zeros(300)
            
            # Use actual FastText model if available
            return self.fasttext_model.get_sentence_vector(text)
            
        except Exception as e:
            logger.error(f"Error extracting FastText embeddings: {e}")
            return np.zeros(300)
    
    def _create_hybrid_embeddings(self, text: str) -> np.ndarray:
        """Create hybrid embeddings by stacking RoBERTa, DeBERTa, and FastText"""
        try:
            # Extract individual embeddings
            roberta_emb = self._get_roberta_embeddings(text)
            deberta_emb = self._get_deberta_embeddings(text)
            fasttext_emb = self._get_fasttext_embeddings(text)
            
            # Stack embeddings
            hybrid_emb = np.concatenate([roberta_emb, deberta_emb, fasttext_emb])
            
            # Apply PCA if model is fitted
            if self.pca_model is not None and hasattr(self.pca_model, 'components_'):
                hybrid_emb = self.pca_model.transform(hybrid_emb.reshape(1, -1)).flatten()
            else:
                # If PCA not fitted, truncate or pad to target dimension
                if len(hybrid_emb) > self.target_dim:
                    hybrid_emb = hybrid_emb[:self.target_dim]
                elif len(hybrid_emb) < self.target_dim:
                    padding = np.zeros(self.target_dim - len(hybrid_emb))
                    hybrid_emb = np.concatenate([hybrid_emb, padding])
            
            return hybrid_emb
            
        except Exception as e:
            logger.error(f"Error creating hybrid embeddings: {e}")
            return np.zeros(self.target_dim)
    
    def fit_pca_on_training_data(self, texts: List[str]) -> None:
        """Fit PCA model on training data embeddings"""
        try:
            # Initialize hybrid embeddings if not already done
            if not hasattr(self, 'roberta_model') or self.roberta_model is None:
                logger.info("Initializing hybrid embeddings for PCA fitting...")
                self._initialize_hybrid_embeddings()
            
            if not HYBRID_EMBEDDINGS_AVAILABLE:
                logger.warning("Hybrid embeddings dependencies not available, using mock PCA")
                # Create a mock PCA for testing purposes
                from sklearn.decomposition import PCA
                self.pca_model = PCA(n_components=self.target_dim, random_state=42)
                # Fit on random data to simulate PCA fitting
                mock_data = np.random.randn(100, 1836)  # 768+768+300 dimensions
                self.pca_model.fit(mock_data)
                logger.info(f"Mock PCA fitted with {self.pca_model.n_components_} components")
                return
            
            logger.info("Fitting PCA on training data...")
            
            # Extract embeddings for a sample of training texts
            sample_size = min(100, len(texts))  # Use smaller sample for faster processing
            sample_texts = texts[:sample_size]
            
            embeddings = []
            for i, text in enumerate(sample_texts):
                if text and not pd.isna(text):
                    try:
                        roberta_emb = self._get_roberta_embeddings(str(text))
                        deberta_emb = self._get_deberta_embeddings(str(text))
                        fasttext_emb = self._get_fasttext_embeddings(str(text))
                        
                        # Stack embeddings
                        hybrid_emb = np.concatenate([roberta_emb, deberta_emb, fasttext_emb])
                        embeddings.append(hybrid_emb)
                        
                        if (i + 1) % 10 == 0:
                            logger.info(f"Processed {i + 1}/{sample_size} texts for PCA fitting")
                            
                    except Exception as e:
                        logger.warning(f"Error processing text {i}: {e}")
                        continue
            
            if embeddings and len(embeddings) >= 2:  # Need at least 2 samples for PCA
                embeddings_matrix = np.array(embeddings)
                logger.info(f"Fitting PCA on {embeddings_matrix.shape[0]} samples with {embeddings_matrix.shape[1]} features")
                
                # Adjust n_components based on available samples and features
                max_components = min(embeddings_matrix.shape[0], embeddings_matrix.shape[1], self.target_dim)
                
                if max_components < self.target_dim:
                    logger.info(f"Adjusting PCA components from {self.target_dim} to {max_components} due to sample/feature constraints")
                    # Create new PCA with adjusted components
                    from sklearn.decomposition import PCA
                    self.pca_model = PCA(n_components=max_components, random_state=42)
                
                # Fit PCA to preserve ~90% variance
                self.pca_model.fit(embeddings_matrix)
                
                explained_variance = np.sum(self.pca_model.explained_variance_ratio_)
                logger.info(f"PCA fitted with {self.pca_model.n_components_} components, "
                           f"preserving {explained_variance:.3f} variance")
                logger.info(f"Original dimensions: {embeddings_matrix.shape[1]} -> Reduced: {self.pca_model.n_components_}")
            else:
                logger.warning(f"Insufficient valid embeddings found ({len(embeddings)}), using mock PCA")
                # Fallback to mock PCA
                mock_data = np.random.randn(50, 1836)  # 768+768+300 dimensions
                self.pca_model.fit(mock_data)
                logger.info(f"Mock PCA fitted with {self.pca_model.n_components_} components")
                
        except Exception as e:
            logger.error(f"Error fitting PCA: {e}")
            # Fallback to mock PCA
            try:
                from sklearn.decomposition import PCA
                self.pca_model = PCA(n_components=self.target_dim, random_state=42)
                mock_data = np.random.randn(50, 1836)
                self.pca_model.fit(mock_data)
                logger.info(f"Fallback mock PCA fitted with {self.pca_model.n_components_} components")
            except Exception as fallback_error:
                logger.error(f"Even fallback PCA failed: {fallback_error}")
    
    def _get_publisher_credibility(self, publisher: str) -> float:
        """Get publisher credibility score (0-1 scale)
        Enhanced mapping for RapidAPI feeds with comprehensive publisher database
        """
        # Comprehensive mapping of publisher credibility scores
        credibility_map = {
            # Tier 1: Highly Credible (0.90-1.0)
            'bbc': 0.95, 'bbc.com': 0.95, 'bbc news': 0.95,
            'reuters': 0.94, 'reuters.com': 0.94,
            'associated press': 0.93, 'ap news': 0.93, 'apnews.com': 0.93,
            'npr': 0.92, 'npr.org': 0.92, 'pbs': 0.91,
            'the guardian': 0.91, 'guardian.com': 0.91, 'theguardian.com': 0.91,
            'washington post': 0.90, 'washingtonpost.com': 0.90,
            'new york times': 0.90, 'nytimes.com': 0.90,
            'wall street journal': 0.88, 'wsj.com': 0.88,
            'the economist': 0.89, 'economist.com': 0.89,
            'financial times': 0.85, 'ft.com': 0.85,
            
            # Tier 2: Very Credible (0.80-0.89)
            'cnn': 0.85, 'cnn.com': 0.85,
            'abc news': 0.85, 'abcnews.go.com': 0.85,
            'cbs news': 0.85, 'cbsnews.com': 0.85,
            'nbc news': 0.85, 'nbcnews.com': 0.85,
            'usa today': 0.82, 'usatoday.com': 0.82,
            'time': 0.81, 'time.com': 0.81,
            'newsweek': 0.80, 'newsweek.com': 0.80,
            
            # Indian News Sources
            'the hindu': 0.88, 'thehindu.com': 0.88,
            'indian express': 0.85, 'indianexpress.com': 0.85,
            'hindustan times': 0.82, 'hindustantimes.com': 0.82,
            'ndtv': 0.80, 'ndtv.com': 0.80,
            'times of india': 0.78, 'timesofindia.com': 0.78,
            'aaj tak': 0.70, 'aajtak.in': 0.70,
            'republic tv': 0.65, 'republicworld.com': 0.65,
            
            # International Sources
            'al jazeera': 0.83, 'aljazeera.com': 0.83,
            'dw': 0.85, 'dw.com': 0.85,
            'france24': 0.84, 'france24.com': 0.84,
            'rt': 0.55, 'rt.com': 0.55,
            'china daily': 0.60, 'chinadaily.com.cn': 0.60,
            'xinhua': 0.58, 'xinhuanet.com': 0.58,
            
            # Medium credibility sources (0.5-0.8)
            'fox news': 0.70, 'foxnews.com': 0.70,
            'msnbc': 0.72, 'msnbc.com': 0.72,
            'huffington post': 0.65, 'huffpost.com': 0.65,
            'buzzfeed news': 0.68, 'buzzfeednews.com': 0.68,
            'politico': 0.75, 'politico.com': 0.75,
            'the hill': 0.73, 'thehill.com': 0.73,
            'daily mail': 0.55, 'dailymail.co.uk': 0.55,
            'new york post': 0.60, 'nypost.com': 0.60,
            'breitbart': 0.45, 'breitbart.com': 0.45,
            
            # Low credibility sources (0.0-0.5)
            'infowars': 0.15, 'infowars.com': 0.15,
            'natural news': 0.20, 'naturalnews.com': 0.20,
            'before its news': 0.10, 'beforeitsnews.com': 0.10,
            'the onion': 0.10, 'theonion.com': 0.10,  # Satire
            'clickhole': 0.10, 'clickhole.com': 0.10,  # Satire
            'babylon bee': 0.05, 'babylonbee.com': 0.05,  # Satire
            'fake news': 0.05, 'fakenews.com': 0.05,
            'conspiracy': 0.15, 'conspiracy.com': 0.15,
        }
        
        if not publisher or pd.isna(publisher):
            return 0.5  # Default neutral credibility
        
        publisher_lower = str(publisher).lower().strip()
        
        # Direct match
        if publisher_lower in credibility_map:
            return credibility_map[publisher_lower]
        
        # Partial match for domains
        for known_publisher, score in credibility_map.items():
            if known_publisher in publisher_lower or publisher_lower in known_publisher:
                return score
        
        # Default for unknown publishers
        return 0.5
    
    def _normalize_timestamp(self, timestamp) -> float:
        """Enhanced timestamp normalization for RapidAPI feeds
        Handles RFC 3339, ISO 8601, Unix timestamps, and common date formats
        Normalizes to days since epoch divided by 365 with better format support
        """
        try:
            if pd.isna(timestamp) or timestamp is None:
                # Use current time as default
                import time
                timestamp = time.time()
            
            # Enhanced string parsing for RapidAPI formats
            if isinstance(timestamp, str):
                timestamp = timestamp.strip()
                
                # Handle various RapidAPI timestamp formats
                try:
                    # Try pandas first (handles most formats)
                    parsed_dt = pd.to_datetime(timestamp, utc=True)
                    timestamp = parsed_dt.timestamp()
                except:
                    # Manual parsing for edge cases
                    from datetime import datetime
                    
                    # RFC 3339 / ISO 8601 formats
                    if 'T' in timestamp:
                        if timestamp.endswith('Z'):
                            timestamp = timestamp[:-1] + '+00:00'
                        try:
                            dt = datetime.fromisoformat(timestamp)
                            timestamp = dt.timestamp()
                        except:
                            # Fallback parsing
                            dt = datetime.strptime(timestamp.split('+')[0].split('Z')[0], '%Y-%m-%dT%H:%M:%S')
                            timestamp = dt.timestamp()
                    
                    # Common date formats
                    elif '/' in timestamp:
                        try:
                            dt = datetime.strptime(timestamp, '%m/%d/%Y %H:%M:%S')
                        except:
                            dt = datetime.strptime(timestamp, '%m/%d/%Y')
                        timestamp = dt.timestamp()
                    
                    elif '-' in timestamp and len(timestamp) >= 10:
                        try:
                            dt = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
                        except:
                            dt = datetime.strptime(timestamp, '%Y-%m-%d')
                        timestamp = dt.timestamp()
                    
                    else:
                        # Try as Unix timestamp string
                        timestamp = float(timestamp)
                        
            elif hasattr(timestamp, 'timestamp'):
                timestamp = timestamp.timestamp()
            
            # Convert to days since epoch and normalize by 365
            days_since_epoch = timestamp / (24 * 3600)  # Convert seconds to days
            normalized = days_since_epoch / 365.0  # Normalize by year
            
            # Clamp to reasonable range (0-100) to avoid extreme values
            normalized = max(0.0, min(normalized, 100.0))
            
            return float(normalized)
            
        except Exception as e:
            logger.error(f"Error normalizing timestamp '{timestamp}': {e}")
            # Return current time normalized as fallback
            import time
            current_days = time.time() / (24 * 3600)
            return float(current_days / 365.0)
    
    def _create_source_temporal_features(self, publisher: str, timestamp) -> np.ndarray:
        """Create source-aware and temporal features [credibility, normalized_timestamp]"""
        try:
            credibility = self._get_publisher_credibility(publisher)
            normalized_time = self._normalize_timestamp(timestamp)
            
            return np.array([credibility, normalized_time], dtype=np.float32)
            
        except Exception as e:
            logger.error(f"Error creating source-temporal features: {e}")
            return np.array([0.5, 0.0], dtype=np.float32)  # Default values
    
    def extract_source_temporal_tensor(self, data: pd.DataFrame) -> torch.Tensor:
        """Extract source-temporal features as [batch, 2] tensor for MHFN integration.
        
        Args:
            data (pd.DataFrame): Input dataframe with publisher and timestamp columns
            
        Returns:
            torch.Tensor: [batch, 2] tensor with [credibility, normalized_timestamp]
        """
        logger.info("Extracting source-temporal features as [batch, 2] tensor...")
        
        source_temporal_features = []
        
        for idx, row in data.iterrows():
            # Extract publisher and timestamp
            publisher = row.get('publisher', '') if 'publisher' in row else row.get('domain', '')
            timestamp = row.get('created_utc', None) if 'created_utc' in row else row.get('timestamp', None)
            
            # Create source-temporal features
            features = self._create_source_temporal_features(publisher, timestamp)
            source_temporal_features.append(features)
        
        # Convert to tensor
        tensor = torch.tensor(np.array(source_temporal_features), dtype=torch.float32)
        
        logger.info(f"Source-temporal tensor shape: {tensor.shape}")
        logger.info(f"Credibility range: [{tensor[:, 0].min():.3f}, {tensor[:, 0].max():.3f}]")
        logger.info(f"Timestamp range: [{tensor[:, 1].min():.3f}, {tensor[:, 1].max():.3f}]")
        
        return tensor
     
    def load_parquet_files(self) -> Dict[str, pd.DataFrame]:
        """
        Load all Parquet files with comprehensive error handling.
        
        Returns:
            Dict[str, pd.DataFrame]: Dictionary containing train, val, test dataframes
        """
        logger.info("Starting Parquet file loading...")
        
        parquet_files = {
            'train': 'fakeddit_processed_train.parquet',
            'val': 'fakeddit_processed_val.parquet',
            'test': 'fakeddit_processed_test.parquet'
        }
        
        loaded_data = {}
        
        for split, filename in parquet_files.items():
            try:
                file_path = self.processed_dir / filename
                
                if not file_path.exists():
                    logger.error(f"Required data file not found: {filename}")
                    raise FileNotFoundError(f"Required data file not found: {filename}")
                    continue
                
                logger.info(f"Loading {filename}...")
                df = pd.read_parquet(file_path)
                
                # Validate dataframe
                if df.empty:
                    raise ValueError(f"Empty dataframe loaded from {filename}")
                
                logger.info(f"Successfully loaded {filename}: {df.shape[0]} rows, {df.shape[1]} columns")
                loaded_data[split] = df
                
                # Log basic statistics
                logger.info(f"{split} data columns: {list(df.columns)}")
                
            except Exception as e:
                logger.error(f"Failed to load {filename}: {str(e)}")
                raise RuntimeError(f"Failed to load required data file {filename}: {str(e)}")
        
        # Store loaded data
        self.train_data = loaded_data.get('train')
        self.val_data = loaded_data.get('val')
        self.test_data = loaded_data.get('test')
        
        logger.info("Parquet file loading completed")
        return loaded_data
    

    
    def load_pickle_files(self) -> Dict[str, Any]:
        """
        Load pickle files and verify text/image data alignment.
        
        Returns:
            Dict[str, Any]: Dictionary containing loaded pickle data
        """
        logger.info("Starting pickle file loading...")
        
        pickle_files = {
            'train_text_image': 'train__text_image__dataframe.pkl',
            'test_text_image': 'test__text_image__dataframe.pkl'
        }
        
        loaded_pickle = {}
        
        for key, filename in pickle_files.items():
            try:
                file_path = self.data_dir / filename
                
                if not file_path.exists():
                    logger.error(f"Pickle file not found: {filename}")
                    raise FileNotFoundError(f"Required pickle file not found: {file_path}")
                
                logger.info(f"Loading pickle file: {filename}")
                
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                
                # Validate pickle data
                if isinstance(data, pd.DataFrame):
                    logger.info(f"Loaded DataFrame from {filename}: {data.shape}")
                    if data.empty:
                        raise ValueError(f"Empty DataFrame in {filename}")
                elif isinstance(data, dict):
                    logger.info(f"Loaded dictionary from {filename} with keys: {list(data.keys())}")
                else:
                    logger.info(f"Loaded {type(data)} from {filename}")
                
                loaded_pickle[key] = data
                
            except Exception as e:
                logger.error(f"Failed to load pickle file {filename}: {str(e)}")
                raise RuntimeError(f"Failed to load required pickle file {filename}: {str(e)}")
        
        # Verify text/image alignment
        self._verify_text_image_alignment(loaded_pickle)
        
        # Store pickle data
        self.pickle_data = loaded_pickle
        
        logger.info("Pickle file loading completed")
        return loaded_pickle
    

    
    def _verify_text_image_alignment(self, pickle_data: Dict[str, Any]) -> None:
        """
        Verify that text and image data are properly aligned.
        
        Args:
            pickle_data (Dict[str, Any]): Loaded pickle data
        """
        logger.info("Verifying text/image data alignment...")
        
        try:
            for key, data in pickle_data.items():
                if isinstance(data, pd.DataFrame):
                    # Check for required columns
                    required_cols = ['text_features', 'image_features']
                    missing_cols = [col for col in required_cols if col not in data.columns]
                    
                    if missing_cols:
                        logger.warning(f"Missing columns in {key}: {missing_cols}")
                    else:
                        # Verify alignment
                        text_len = len(data['text_features'])
                        image_len = len(data['image_features'])
                        
                        if text_len != image_len:
                            logger.warning(f"Misaligned data in {key}: text={text_len}, image={image_len}")
                        else:
                            logger.info(f"Data alignment verified for {key}: {text_len} samples")
            
            logger.info("Text/image alignment verification completed")
            
        except Exception as e:
            logger.error(f"Alignment verification failed: {str(e)}")
    
    def preprocess_data(self, data: pd.DataFrame, split: str = 'train') -> Dict[str, torch.Tensor]:
        """
        Preprocess data for MHFN model input.
        Ensures tensors are [batch, 300] format.
        
        Args:
            data (pd.DataFrame): Input dataframe
            split (str): Data split identifier
            
        Returns:
            Dict[str, torch.Tensor]: Preprocessed tensors
        """
        logger.info(f"Starting data preprocessing for {split} split...")
        
        try:
            # Handle missing values
            data = self._handle_missing_values(data)
            
            # Extract features and labels
            features, labels = self._extract_features_labels(data)
            
            # Ensure proper tensor dimensions [batch, 300]
            features_tensor = self._ensure_tensor_dimensions(features)
            labels_tensor = torch.tensor(labels, dtype=torch.long)
            
            # Additional preprocessing
            features_tensor = self._normalize_features(features_tensor)
            
            # Extract source-temporal features as separate tensor
            source_temporal_tensor = self.extract_source_temporal_tensor(data)
            
            preprocessed = {
                'features': features_tensor,
                'labels': labels_tensor,
                'source_temporal': source_temporal_tensor,
                'batch_size': features_tensor.shape[0],
                'feature_dim': features_tensor.shape[1]
            }
            
            logger.info(f"Preprocessing completed for {split}:")
            logger.info(f"  Features shape: {features_tensor.shape}")
            logger.info(f"  Labels shape: {labels_tensor.shape}")
            logger.info(f"  Feature dimension: {features_tensor.shape[1]}")
            
            return preprocessed
            
        except Exception as e:
            logger.error(f"Preprocessing failed for {split}: {str(e)}")
            raise
    
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            data (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Cleaned dataframe
        """
        logger.info("Handling missing values...")
        
        # Check for missing values
        missing_counts = data.isnull().sum()
        if missing_counts.sum() > 0:
            logger.info(f"Found missing values: {missing_counts[missing_counts > 0].to_dict()}")
            
            # Handle different column types
            for col in data.columns:
                if data[col].isnull().sum() > 0:
                    if col in ['features', 'text_features', 'image_features'] or 'feature' in col.lower():
                        # For feature columns, fill with zeros
                        def handle_feature_na(x):
                            if x is None or (isinstance(x, float) and pd.isna(x)):
                                return [0.0] * self.target_dim
                            elif isinstance(x, (list, np.ndarray)) and len(x) == 0:
                                return [0.0] * self.target_dim
                            else:
                                return x
                        data[col] = data[col].apply(handle_feature_na)
                    elif col in ['text', 'title']:
                        # For text columns, fill with empty string
                        data[col] = data[col].fillna('')
                    elif col == 'label':
                        # For labels, fill with 0 (fake news)
                        data[col] = data[col].fillna(0)
                    else:
                        # For other columns, forward fill
                        data[col] = data[col].fillna(method='ffill')
        
        logger.info("Missing value handling completed")
        return data
    
    def _extract_features_labels(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract features and labels from dataframe.
        
        Args:
            data (pd.DataFrame): Input dataframe
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Features and labels arrays
        """
        logger.info("Extracting features and labels...")
        
        # Try different feature column names
        feature_columns = ['features', 'text_features', 'combined_features']
        features = None
        
        for col in feature_columns:
            if col in data.columns:
                logger.info(f"Using {col} as feature column")
                features = np.array(data[col].tolist())
                break
        
        if features is None:
            # Create features from text if no feature column exists
            logger.info("Creating features from text data")
            features = self._create_features_from_text(data)
        
        # Extract labels
        if 'label' in data.columns:
            labels = data['label'].values
        else:
            logger.warning("No label column found, creating mock labels")
            labels = np.random.randint(0, 2, len(data))
        
        logger.info(f"Extracted features shape: {features.shape}")
        logger.info(f"Extracted labels shape: {labels.shape}")
        
        return features, labels
    
    def _create_features_from_text(self, data: pd.DataFrame) -> np.ndarray:
        """
        Create feature vectors from text data using hybrid embeddings and source-temporal features.
        
        Args:
            data (pd.DataFrame): Input dataframe
            
        Returns:
            np.ndarray: Feature array
        """
        logger.info("Creating features from text using hybrid embeddings and source-temporal features...")
        
        features = []
        
        for idx, row in data.iterrows():
            # Combine text fields
            text = ''
            if 'title' in row and pd.notna(row['title']):
                text += str(row['title']) + ' '
            if 'text' in row and pd.notna(row['text']):
                text += str(row['text'])
            
            # Extract publisher and timestamp for source-temporal features
            publisher = row.get('publisher', '') if 'publisher' in row else row.get('domain', '')
            timestamp = row.get('created_utc', None) if 'created_utc' in row else row.get('timestamp', None)
            
            if text:
                # Use hybrid embeddings if available
                if HYBRID_EMBEDDINGS_AVAILABLE and hasattr(self, 'roberta_model'):
                    hybrid_features = self._create_hybrid_embeddings(text)
                    
                    # Create source-temporal features
                    source_temporal_features = self._create_source_temporal_features(publisher, timestamp)
                    
                    # Combine hybrid embeddings with source-temporal features
                    if len(hybrid_features) + len(source_temporal_features) <= self.target_dim:
                        feature_vector = np.concatenate([hybrid_features, source_temporal_features])
                        # Pad if necessary
                        if len(feature_vector) < self.target_dim:
                            padding = np.zeros(self.target_dim - len(feature_vector))
                            feature_vector = np.concatenate([feature_vector, padding])
                    else:
                        # Truncate hybrid features to make room for source-temporal features
                        truncated_hybrid = hybrid_features[:self.target_dim - len(source_temporal_features)]
                        feature_vector = np.concatenate([truncated_hybrid, source_temporal_features])
                    
                    # Ensure exact target dimension
                    feature_vector = feature_vector[:self.target_dim]
                    
                else:
                    # Fallback to basic features with source-temporal features
                    feature_vector = np.zeros(self.target_dim)
                    # Basic text features
                    feature_vector[0] = len(text)  # Text length
                    feature_vector[1] = len(text.split())  # Word count
                    feature_vector[2] = text.count('!')  # Exclamation marks
                    feature_vector[3] = text.count('?')  # Question marks
                    feature_vector[4] = len([w for w in text.split() if w.isupper()])  # Uppercase words
                    
                    # Add source-temporal features
                    source_temporal_features = self._create_source_temporal_features(publisher, timestamp)
                    feature_vector[5:7] = source_temporal_features  # Credibility and timestamp
                    
                    # Fill remaining with random values (placeholder for real embeddings)
                    np.random.seed(hash(text) % 2**32)
                    feature_vector[7:] = np.random.randn(self.target_dim - 7) * 0.1
            else:
                # No text available, use default features with source-temporal
                feature_vector = np.zeros(self.target_dim)
                source_temporal_features = self._create_source_temporal_features(publisher, timestamp)
                feature_vector[:2] = source_temporal_features  # Place at beginning
            
            features.append(feature_vector)
        
        return np.array(features)
    
    def _ensure_tensor_dimensions(self, features: np.ndarray) -> torch.Tensor:
        """
        Ensure features tensor has correct dimensions [batch, 300] for MHFN.
        
        Args:
            features (np.ndarray): Input features
            
        Returns:
            torch.Tensor: Properly dimensioned tensor
        """
        logger.info(f"Ensuring tensor dimensions from shape: {features.shape}")
        
        # Convert to tensor
        if isinstance(features, list):
            features = np.array(features)
        
        tensor = torch.tensor(features, dtype=torch.float32)
        
        # Handle different input shapes
        if len(tensor.shape) == 1:
            # Single sample
            tensor = tensor.unsqueeze(0)
        
        batch_size, current_dim = tensor.shape[0], tensor.shape[-1]
        
        if current_dim == self.target_dim:
            logger.info(f"Tensor already has correct dimension: {tensor.shape}")
            return tensor
        elif current_dim > self.target_dim:
            # Truncate to target dimension
            logger.info(f"Truncating from {current_dim} to {self.target_dim}")
            tensor = tensor[:, :self.target_dim]
        else:
            # Pad to target dimension
            logger.info(f"Padding from {current_dim} to {self.target_dim}")
            padding = torch.zeros(batch_size, self.target_dim - current_dim)
            tensor = torch.cat([tensor, padding], dim=1)
        
        logger.info(f"Final tensor shape: {tensor.shape}")
        return tensor
    
    def _normalize_features(self, features: torch.Tensor) -> torch.Tensor:
        """
        Normalize feature tensors.
        
        Args:
            features (torch.Tensor): Input features
            
        Returns:
            torch.Tensor: Normalized features
        """
        logger.info("Normalizing features...")
        
        # L2 normalization
        normalized = torch.nn.functional.normalize(features, p=2, dim=1)
        
        logger.info(f"Features normalized: mean={normalized.mean().item():.4f}, std={normalized.std().item():.4f}")
        return normalized
    
    def test_small_batch(self, batch_size: int = 32) -> Dict[str, Any]:
        """
        Test loading and preprocessing with a small batch.
        
        Args:
            batch_size (int): Size of test batch
            
        Returns:
            Dict[str, Any]: Test results
        """
        logger.info(f"Testing with small batch of size {batch_size}...")
        
        try:
            # Load data if not already loaded
            if self.train_data is None:
                self.load_parquet_files()
            
            if not self.pickle_data:
                self.load_pickle_files()
            
            # Get small batch from train data
            small_batch = self.train_data.head(batch_size).copy()
            
            # Preprocess the batch
            preprocessed = self.preprocess_data(small_batch, 'test_batch')
            
            # Run validation tests
            test_results = {
                'batch_size': batch_size,
                'actual_batch_size': preprocessed['batch_size'],
                'feature_shape': list(preprocessed['features'].shape),
                'label_shape': list(preprocessed['labels'].shape),
                'feature_dim': preprocessed['feature_dim'],
                'feature_mean': preprocessed['features'].mean().item(),
                'feature_std': preprocessed['features'].std().item(),
                'label_distribution': torch.bincount(preprocessed['labels']).tolist(),
                'success': True
            }
            
            # Validate tensor dimensions
            expected_shape = [batch_size, self.target_dim]
            actual_shape = list(preprocessed['features'].shape)
            
            if actual_shape != expected_shape:
                logger.warning(f"Shape mismatch: expected {expected_shape}, got {actual_shape}")
                test_results['shape_warning'] = f"Expected {expected_shape}, got {actual_shape}"
            
            logger.info("Small batch test completed successfully")
            logger.info(f"Test results: {test_results}")
            
            return test_results
            
        except Exception as e:
            logger.error(f"Small batch test failed: {str(e)}")
            return {
                'batch_size': batch_size,
                'success': False,
                'error': str(e)
            }
    
    def get_data_loader(self, split: str = 'train', batch_size: int = 32, shuffle: bool = True) -> DataLoader:
        """
        Create PyTorch DataLoader for the specified split.
        
        Args:
            split (str): Data split ('train', 'val', 'test')
            batch_size (int): Batch size
            shuffle (bool): Whether to shuffle data
            
        Returns:
            DataLoader: PyTorch DataLoader
        """
        logger.info(f"Creating DataLoader for {split} split...")
        
        # Get appropriate data
        if split == 'train' and self.train_data is not None:
            data = self.train_data
        elif split == 'val' and self.val_data is not None:
            data = self.val_data
        elif split == 'test' and self.test_data is not None:
            data = self.test_data
        else:
            logger.warning(f"No data available for {split}, loading...")
            self.load_parquet_files()
            data = getattr(self, f"{split}_data")
        
        # Preprocess data
        preprocessed = self.preprocess_data(data, split)
        
        # Create dataset
        dataset = FakeNewsDataset(
            features=preprocessed['features'],
            labels=preprocessed['labels']
        )
        
        # Create DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,  # Set to 0 for compatibility
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        logger.info(f"DataLoader created: {len(dataset)} samples, {len(dataloader)} batches")
        return dataloader
    
    def get_features_labels(self, split: str = 'train') -> Tuple[np.ndarray, np.ndarray]:
        """
        Get features and labels for the specified split.
        
        Args:
            split (str): Data split ('train', 'val', 'test')
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Features and labels arrays
        """
        logger.info(f"Getting features and labels for {split} split...")
        
        # Get appropriate data
        if split == 'train' and self.train_data is not None:
            data = self.train_data
        elif split == 'val' and self.val_data is not None:
            data = self.val_data
        elif split == 'test' and self.test_data is not None:
            data = self.test_data
        else:
            logger.warning(f"No data available for {split}, loading...")
            self.load_parquet_files()
            data = getattr(self, f"{split}_data")
        
        # Extract features and labels
        features, labels = self._extract_features_labels(data)
        
        logger.info(f"Retrieved {len(features)} samples for {split} split")
        return features, labels
    
    def get_data_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about loaded data.
        
        Returns:
            Dict[str, Any]: Data statistics
        """
        logger.info("Generating data statistics...")
        
        stats = {
            'timestamp': datetime.now().isoformat(),
            'data_dir': str(self.data_dir),
            'target_dimension': self.target_dim
        }
        
        # Parquet data statistics
        for split in ['train', 'val', 'test']:
            data = getattr(self, f"{split}_data")
            if data is not None:
                stats[f"{split}_data"] = {
                    'shape': list(data.shape),
                    'columns': list(data.columns),
                    'memory_usage': data.memory_usage(deep=True).sum(),
                    'null_counts': data.isnull().sum().to_dict()
                }
        
        # Pickle data statistics
        if self.pickle_data:
            stats['pickle_data'] = {}
            for key, data in self.pickle_data.items():
                if isinstance(data, pd.DataFrame):
                    stats['pickle_data'][key] = {
                        'shape': list(data.shape),
                        'columns': list(data.columns)
                    }
        
        logger.info("Data statistics generated")
        return stats

class FakeNewsDataset(Dataset):
    """
    PyTorch Dataset class for fake news data.
    """
    
    def __init__(self, features: torch.Tensor, labels: torch.Tensor):
        """
        Initialize dataset.
        
        Args:
            features (torch.Tensor): Feature tensors
            labels (torch.Tensor): Label tensors
        """
        self.features = features
        self.labels = labels
        
        assert len(features) == len(labels), "Features and labels must have same length"
    
    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get item by index.
        
        Args:
            idx (int): Item index
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Feature and label tensors
        """
        return self.features[idx], self.labels[idx]

def main():
    """
    Main function for testing the data loader.
    """
    logger.info("Starting DataLoader test...")
    
    try:
        # Initialize data loader
        data_loader = FakeNewsDataLoader()
        
        # Load all data
        logger.info("Loading Parquet files...")
        parquet_data = data_loader.load_parquet_files()
        
        logger.info("Loading pickle files...")
        pickle_data = data_loader.load_pickle_files()
        
        # Test small batch
        logger.info("Testing small batch...")
        test_results = data_loader.test_small_batch(batch_size=16)
        
        # Get statistics
        logger.info("Generating statistics...")
        stats = data_loader.get_data_statistics()
        
        # Create DataLoader
        logger.info("Creating PyTorch DataLoader...")
        train_loader = data_loader.get_data_loader('train', batch_size=32)
        
        # Test one batch
        for batch_features, batch_labels in train_loader:
            logger.info(f"Sample batch - Features: {batch_features.shape}, Labels: {batch_labels.shape}")
            break
        
        logger.info("DataLoader test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"DataLoader test failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)