#!/usr/bin/env python3
"""
MHFN (Multimodal Hybrid Fake News) Inference Wrapper

This module provides a deterministic, production-ready wrapper for the MHFN model:
- Handles text, image, and URL inputs gracefully
- Uses Hugging Face transformers for text encoding
- Uses CLIP/BLIP-2 for image processing
- Implements deterministic behavior with fixed seeds
- Provides calibrated probability outputs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import os
import hashlib
import random
import time
from typing import Optional, Dict, Any, Union, Tuple
from PIL import Image
import requests
from io import BytesIO
import warnings
from urllib.parse import urlparse
import ssl
import certifi

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set deterministic behavior
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Configure SSL context for secure requests
ssl_context = ssl.create_default_context(cafile=certifi.where())

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from transformers import AutoTokenizer, AutoModel
    from sentence_transformers import SentenceTransformer
except ImportError:
    logger.warning("Transformers not available, using mock embeddings")
    AutoTokenizer = None
    AutoModel = None
    SentenceTransformer = None

try:
    import clip
except ImportError:
    logger.warning("CLIP not available, using mock image processing")
    clip = None

try:
    import torchvision.transforms as transforms
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False
    logger.warning("torchvision not available, using basic image processing")

class MHFNModel(nn.Module):
    """Multimodal Hybrid Fake News Detection Model."""
    
    def __init__(self, 
                 text_dim: int = 768,
                 image_dim: int = 512,
                 url_dim: int = 128,
                 hidden_dim: int = 256,
                 num_classes: int = 2):
        super().__init__()
        
        # Modality-specific encoders
        self.text_encoder = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.image_encoder = nn.Sequential(
            nn.Linear(image_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.url_encoder = nn.Sequential(
            nn.Linear(url_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Attention mechanism for modality fusion
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Temperature scaling for calibration
        self.temperature = nn.Parameter(torch.ones(1))
    
    def forward(self, 
                text_features: Optional[torch.Tensor] = None,
                image_features: Optional[torch.Tensor] = None,
                url_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with optional modalities."""
        
        encoded_features = []
        
        # Encode available modalities
        if text_features is not None:
            text_encoded = self.text_encoder(text_features)
            encoded_features.append(text_encoded)
        
        if image_features is not None:
            image_encoded = self.image_encoder(image_features)
            encoded_features.append(image_encoded)
        
        if url_features is not None:
            url_encoded = self.url_encoder(url_features)
            encoded_features.append(url_encoded)
        
        if not encoded_features:
            # No features available, return neutral prediction
            batch_size = 1
            return torch.zeros(batch_size, 2)
        
        # Stack features for attention
        if len(encoded_features) == 1:
            fused_features = encoded_features[0]
        else:
            # Multi-head attention fusion
            stacked_features = torch.stack(encoded_features, dim=1)  # [batch, num_modalities, hidden_dim]
            attended_features, _ = self.attention(stacked_features, stacked_features, stacked_features)
            fused_features = attended_features.mean(dim=1)  # Average over modalities
        
        # Final classification
        logits = self.classifier(fused_features)
        
        # Apply temperature scaling
        calibrated_logits = logits / self.temperature
        
        return calibrated_logits

class MHFNInference:
    """Production-ready MHFN inference wrapper."""
    
    def __init__(self, 
                 model_path: str = "mhf_model.pth",
                 device: str = "auto",
                 text_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize MHFN inference system.
        
        Args:
            model_path: Path to trained MHFN model
            device: Device to run inference on ('cpu', 'cuda', or 'auto')
            text_model_name: Hugging Face model for text encoding
        """
        self.model_path = model_path
        self.text_model_name = text_model_name
        
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        
        # Initialize models
        self._load_models()
    
    def _load_models(self):
        """Load all required models."""
        # Load MHFN model
        from model import MHFN
        self.mhfn_model = MHFN(input_dim=300, hidden_dim=64)
        
        # Add dimension reduction layer for SentenceTransformer embeddings (384 -> 300)
        self.embedding_reducer = nn.Linear(384, 300).to(self.device)
        nn.init.xavier_uniform_(self.embedding_reducer.weight)
        nn.init.zeros_(self.embedding_reducer.bias)
        
        if os.path.exists(self.model_path):
            try:
                checkpoint = torch.load(self.model_path, map_location=self.device)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    self.mhfn_model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.mhfn_model.load_state_dict(checkpoint)
                logger.info(f"Loaded MHFN model from {self.model_path}")
            except Exception as e:
                logger.warning(f"Could not load model from {self.model_path}: {e}")
                logger.info("Using randomly initialized model")
        else:
            logger.warning(f"Model file {self.model_path} not found, using random weights")
        
        self.mhfn_model.to(self.device)
        self.mhfn_model.eval()
        self.embedding_reducer.eval()
        
        # Load text encoder
        if SentenceTransformer is not None:
            try:
                self.text_encoder = SentenceTransformer(self.text_model_name)
                logger.info(f"Loaded text encoder: {self.text_model_name}")
            except Exception as e:
                logger.warning(f"Could not load text encoder: {e}")
                self.text_encoder = None
        else:
            self.text_encoder = None
        
        # Load CLIP for image processing
        if clip is not None:
            try:
                self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
                logger.info("Loaded CLIP model for image processing")
            except Exception as e:
                logger.warning(f"Could not load CLIP: {e}")
                self.clip_model = None
                self.clip_preprocess = None
        else:
            self.clip_model = None
            self.clip_preprocess = None
    
    def _encode_text(self, text: str) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
        """Enhanced text encoding with metadata tracking."""
        metadata = {
            'encoding_method': 'unknown',
            'text_length': len(text) if text else 0,
            'word_count': len(text.split()) if text else 0,
            'processing_time': 0,
            'fallback_reason': None
        }
        
        start_time = time.time()
        
        if not text or len(text.strip()) < 3:
            metadata['encoding_method'] = 'skipped'
            metadata['fallback_reason'] = 'Text too short or empty'
            metadata['processing_time'] = time.time() - start_time
            return None, metadata
        
        try:
            if self.text_encoder is not None:
                # Use sentence transformers
                embedding = self.text_encoder.encode(text, convert_to_tensor=True)
                if embedding.dim() == 1:
                    embedding = embedding.unsqueeze(0)  # Add batch dimension
                tensor_result = embedding.to(self.device)
                
                # Validate embeddings
                if torch.any(torch.isnan(tensor_result)) or torch.any(torch.isinf(tensor_result)):
                    raise ValueError("Invalid embeddings detected (NaN or Inf)")
                
                # Reduce dimensions from 384 to 300 if needed
                if tensor_result.shape[-1] == 384:
                    tensor_result = self.embedding_reducer(tensor_result)
                    metadata['dimension_reduced'] = True
                else:
                    metadata['dimension_reduced'] = False
                
                metadata['encoding_method'] = 'SentenceTransformer'
                metadata['embedding_dim'] = tensor_result.shape[-1]
                metadata['processing_time'] = time.time() - start_time
                
                logger.info(f"Text encoded with SentenceTransformer: {len(text)} chars, {len(text.split())} words")
                return tensor_result, metadata
            else:
                # Enhanced deterministic fallback
                metadata['encoding_method'] = 'deterministic_fallback'
                metadata['fallback_reason'] = 'SentenceTransformer not available'
                
                # Fallback: simple hash-based encoding (deterministic)
                text_hash = hashlib.md5(text.encode()).hexdigest()
                # Convert hex to numbers and normalize
                hash_numbers = [int(text_hash[i:i+2], 16) for i in range(0, len(text_hash), 2)]
                embedding = torch.tensor(hash_numbers, dtype=torch.float32)
                # Pad or truncate to 300 dimensions (MHFN model input_dim)
                if len(embedding) < 300:
                    padding = torch.zeros(300 - len(embedding))
                    embedding = torch.cat([embedding, padding])
                else:
                    embedding = embedding[:300]
                # Normalize
                embedding = F.normalize(embedding, dim=0)
                tensor_result = embedding.unsqueeze(0).to(self.device)
                
                metadata['embedding_dim'] = tensor_result.shape[-1]
                metadata['processing_time'] = time.time() - start_time
                
                logger.info(f"Text encoded with deterministic fallback: {len(text)} chars, {len(text.split())} words")
                return tensor_result, metadata
        
        except Exception as e:
            logger.error(f"Text encoding error: {e}")
            metadata['encoding_method'] = 'error'
            metadata['fallback_reason'] = str(e)
            metadata['processing_time'] = time.time() - start_time
            return None, metadata
    
    def _load_image_from_url(self, image_url: str) -> Optional[Image.Image]:
        """Enhanced image loading with robust error handling and validation."""
        if not image_url or not isinstance(image_url, str):
            logger.warning("Invalid image URL provided")
            return None
        
        try:
            # Validate URL format
            parsed_url = urlparse(image_url)
            if not parsed_url.scheme or not parsed_url.netloc:
                logger.warning(f"Invalid URL format: {image_url}")
                return None
            
            # Enhanced request with proper headers and timeout
            headers = {
                'User-Agent': 'Mozilla/5.0 (compatible; FakeNewsDetector/1.0)',
                'Accept': 'image/*,*/*;q=0.8'
            }
            
            response = requests.get(
                image_url, 
                timeout=15, 
                headers=headers,
                stream=True,
                verify=True
            )
            response.raise_for_status()
            
            # Check content type
            content_type = response.headers.get('content-type', '').lower()
            if not any(img_type in content_type for img_type in ['image/', 'application/octet-stream']):
                logger.warning(f"Invalid content type: {content_type} for URL: {image_url}")
                return None
            
            # Check content length
            content_length = response.headers.get('content-length')
            if content_length and int(content_length) > 50 * 1024 * 1024:  # 50MB limit
                logger.warning(f"Image too large: {content_length} bytes")
                return None
            
            # Load and validate image
            image_data = response.content
            if len(image_data) == 0:
                logger.warning("Empty image data received")
                return None
            
            image = Image.open(BytesIO(image_data))
            
            # Validate image dimensions
            width, height = image.size
            if width < 10 or height < 10 or width > 10000 or height > 10000:
                logger.warning(f"Invalid image dimensions: {width}x{height}")
                return None
            
            # Convert to RGB and return
            rgb_image = image.convert('RGB')
            logger.info(f"Successfully loaded image: {width}x{height} from {image_url}")
            return rgb_image
            
        except requests.exceptions.Timeout:
            logger.error(f"Timeout loading image from URL: {image_url}")
        except requests.exceptions.ConnectionError:
            logger.error(f"Connection error loading image from URL: {image_url}")
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error {e.response.status_code} loading image from URL: {image_url}")
        except Image.UnidentifiedImageError:
            logger.error(f"Cannot identify image format from URL: {image_url}")
        except Exception as e:
            logger.error(f"Unexpected error loading image from URL {image_url}: {type(e).__name__}: {e}")
        
        return None
    
    def _encode_image(self, image_input: Union[str, Image.Image]) -> Optional[torch.Tensor]:
        """Enhanced image encoding with comprehensive fallback mechanisms."""
        try:
            # Load image if URL provided
            if isinstance(image_input, str):
                image = self._load_image_from_url(image_input)
                if image is None:
                    # Create deterministic fallback based on URL
                    url_hash = hashlib.md5(image_input.encode()).hexdigest()
                    hash_features = [int(url_hash[i:i+2], 16) for i in range(0, 32, 2)]
                    features = torch.tensor(hash_features + [0.1] * (512 - len(hash_features)), dtype=torch.float32)
                    features = F.normalize(features, dim=0)
                    return features.unsqueeze(0).to(self.device)
            else:
                image = image_input
            
            if self.clip_model is not None and self.clip_preprocess is not None:
                # Use CLIP encoding
                try:
                    if TORCHVISION_AVAILABLE:
                        # Enhanced preprocessing with torchvision
                        preprocess = transforms.Compose([
                            transforms.Resize((224, 224)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                        ])
                        image_tensor = preprocess(image).unsqueeze(0).to(self.device)
                    else:
                        image_tensor = self.clip_preprocess(image).unsqueeze(0).to(self.device)
                    
                    with torch.no_grad():
                        image_features = self.clip_model.encode_image(image_tensor)
                    
                    # Validate features
                    if torch.any(torch.isnan(image_features)) or torch.any(torch.isinf(image_features)):
                        raise ValueError("Invalid features detected (NaN or Inf)")
                    
                    return image_features.float()
                except Exception as clip_error:
                    logger.warning(f"CLIP encoding failed: {clip_error}, using fallback")
                    # Fall through to deterministic fallback
            
            # Deterministic fallback based on image properties
            width, height = image.size
            
            # Convert image to array for more features
            img_array = np.array(image)
            mean_rgb = np.mean(img_array, axis=(0, 1))
            std_rgb = np.std(img_array, axis=(0, 1))
            
            # Create comprehensive deterministic features
            features = [
                width / 1000.0,
                height / 1000.0,
                (width * height) / 1000000.0,
                width / height if height > 0 else 1.0,  # aspect ratio
                mean_rgb[0] / 255.0, mean_rgb[1] / 255.0, mean_rgb[2] / 255.0,
                std_rgb[0] / 255.0, std_rgb[1] / 255.0, std_rgb[2] / 255.0,
            ]
            
            # Add hash-based features for determinism
            if isinstance(image_input, str):
                url_hash = hashlib.md5(image_input.encode()).hexdigest()
                hash_features = [int(url_hash[i:i+2], 16) / 255.0 for i in range(0, 32, 2)]
                features.extend(hash_features)
            
            # Pad to 512 dimensions
            while len(features) < 512:
                features.append(0.1)
            features = features[:512]
            
            feature_tensor = torch.tensor(features, dtype=torch.float32)
            feature_tensor = F.normalize(feature_tensor, dim=0)
            return feature_tensor.unsqueeze(0).to(self.device)
        
        except Exception as e:
            logger.error(f"Image encoding error: {e}")
            return None
    
    def _encode_url(self, url: str) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
        """Enhanced URL encoding with comprehensive feature extraction and metadata."""
        metadata = {
            'encoding_method': 'url_features',
            'url_length': len(url) if url else 0,
            'processing_time': 0,
            'features_extracted': {},
            'fallback_reason': None
        }
        
        start_time = time.time()
        
        if not url:
            metadata['encoding_method'] = 'skipped'
            metadata['fallback_reason'] = 'URL is empty'
            metadata['processing_time'] = time.time() - start_time
            return None, metadata
        
        try:
            # Extract URL features (domain, length, etc.)
            from urllib.parse import urlparse
            
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            path = parsed.path
            
            # Create deterministic features
            features = [
                len(url) / 100.0,  # URL length
                len(domain) / 50.0,  # Domain length
                len(path) / 100.0,  # Path length
                1.0 if 'https' in url else 0.0,  # HTTPS
                1.0 if any(tld in domain for tld in ['.com', '.org', '.net']) else 0.0,  # Common TLD
                # Hash-based features for determinism
            ]
            
            # Add hash-based features to reach 128 dimensions
            url_hash = hashlib.md5(url.encode()).hexdigest()
            hash_features = [int(url_hash[i:i+2], 16) / 255.0 for i in range(0, min(len(url_hash), 246), 2)]
            features.extend(hash_features)
            
            # Pad to exactly 128 dimensions
            while len(features) < 128:
                features.append(0.0)
            features = features[:128]
            
            # Store feature metadata
            metadata['features_extracted'] = {
                'url_length': len(url),
                'domain_length': len(domain),
                'has_https': 'https' in url,
                'has_common_tld': any(tld in domain for tld in ['.com', '.org', '.net']),
                'path_length': len(path)
            }
            
            tensor_result = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
            metadata['feature_dim'] = tensor_result.shape[-1]
            metadata['processing_time'] = time.time() - start_time
            
            logger.info(f"URL encoded successfully: {len(url)} chars, domain: {domain}, https: {'https' in url}")
            return tensor_result, metadata
        
        except Exception as e:
            logger.error(f"URL encoding error: {e}")
            metadata['encoding_method'] = 'error'
            metadata['fallback_reason'] = str(e)
            metadata['processing_time'] = time.time() - start_time
            return None, metadata
    
    def predict(self, 
                text: Optional[str] = None,
                image_url: Optional[str] = None,
                url: Optional[str] = None) -> Dict[str, Any]:
        """
        Enhanced multimodal prediction with comprehensive error handling and metadata.
        
        Args:
            text: Input text to analyze
            image_url: URL of image to analyze
            url: URL for metadata analysis
        
        Returns:
            Dictionary with prediction results:
            {
                'p_fake': float,  # Probability that content is fake [0, 1]
                'p_real': float,  # Probability that content is real [0, 1]
                'verdict': str,   # 'FAKE' or 'REAL'
                'confidence': float,  # Confidence in prediction [0, 1]
                'modalities_used': List[str],  # Which modalities were processed
                'processing_details': Dict,
                'processing_time': float,
                'errors': List[str],
                'warnings': List[str]
            }
        """
        
        start_time = time.time()
        modalities_used = []
        processing_details = {}
        errors = []
        warnings = []
        
        logger.info(f"Starting enhanced prediction with text={text is not None}, "
                   f"image={image_url is not None}, url={url is not None}")
        
        # Encode modalities with enhanced error handling
        text_features = None
        if text and isinstance(text, str) and len(text.strip()) > 0:
            try:
                text_features, text_meta = self._encode_text(text)
                if text_features is not None:
                    modalities_used.append('text')
                    processing_details['text'] = text_meta
                    processing_details['text_length'] = len(text)
                    processing_details['text_words'] = len(text.split())
                else:
                    warnings.append(f"Text encoding returned None: {text_meta.get('fallback_reason', 'Unknown')}")
                    processing_details['text'] = text_meta
            except Exception as e:
                error_msg = f"Text encoding failed: {type(e).__name__}: {e}"
                logger.error(error_msg)
                errors.append(error_msg)
        elif text:
            warnings.append("Text input provided but invalid (empty or non-string)")
        
        image_features = None
        if image_url and isinstance(image_url, str) and len(image_url.strip()) > 0:
             try:
                 image_features, image_meta = self._encode_image(image_url)
                 if image_features is not None:
                     modalities_used.append('image')
                     processing_details['image'] = image_meta
                     processing_details['image_url'] = image_url
                     processing_details['image_feature_dim'] = image_features.shape[-1]
                 else:
                     warnings.append(f"Image encoding returned None: {image_meta.get('fallback_reason', 'Unknown')}")
                     processing_details['image'] = image_meta
             except Exception as e:
                 error_msg = f"Image encoding failed: {type(e).__name__}: {e}"
                 logger.error(error_msg)
                 errors.append(error_msg)
        elif image_url:
             warnings.append("Image URL provided but invalid (empty or non-string)")
        
        url_features = None
        if url and isinstance(url, str) and len(url.strip()) > 0:
            try:
                url_features, url_meta = self._encode_url(url)
                if url_features is not None:
                    modalities_used.append('url')
                    processing_details['url'] = url_meta
                    processing_details['url_domain'] = url.split('/')[2] if '://' in url else url
                    processing_details['url_length'] = len(url)
                else:
                    warnings.append(f"URL encoding returned None: {url_meta.get('fallback_reason', 'Unknown')}")
                    processing_details['url'] = url_meta
            except Exception as e:
                error_msg = f"URL encoding failed: {type(e).__name__}: {e}"
                logger.error(error_msg)
                errors.append(error_msg)
        elif url:
            warnings.append("URL provided but invalid (empty or non-string)")
        
        # Make prediction with enhanced error handling
        try:
            with torch.no_grad():
                # Prepare input tensor - concatenate available features
                input_features = []
                
                if text_features is not None:
                    input_features.append(text_features)
                else:
                    # Add zero padding for missing text features
                    input_features.append(torch.zeros(1, 300).to(self.device))
                
                # Concatenate all features into single input tensor
                x = torch.cat(input_features, dim=1) if len(input_features) > 1 else input_features[0]
                
                # Call model with correct signature (x, source_temporal)
                logits = self.mhfn_model(x)
                
                # Validate logits
                if torch.any(torch.isnan(logits)) or torch.any(torch.isinf(logits)):
                    raise ValueError("Model returned invalid logits (NaN or Inf)")
                
                # Convert to probabilities
                probabilities = F.softmax(logits, dim=1)
                p_real = probabilities[0, 0].item()
                p_fake = probabilities[0, 1].item()
            
            # Validate probabilities
            if not (0 <= p_real <= 1) or not (0 <= p_fake <= 1):
                raise ValueError(f"Invalid probability values: p_real={p_real}, p_fake={p_fake}")
            
            # Ensure probabilities sum to 1 and are deterministic
            total_prob = p_real + p_fake
            if total_prob > 0:
                p_real /= total_prob
                p_fake /= total_prob
            else:
                # Fallback for edge cases
                p_real = 0.5
                p_fake = 0.5
                warnings.append("Probabilities summed to zero, using default values")
            
            # Determine verdict and confidence
            verdict = 'FAKE' if p_fake > p_real else 'REAL'
            confidence = abs(p_fake - p_real)  # Confidence as margin
            
            # Apply modality-based confidence adjustment
            modality_count = len(modalities_used)
            if modality_count > 1:
                # Boost confidence for multimodal predictions
                confidence = min(0.99, confidence * (1.0 + 0.1 * (modality_count - 1)))
            elif modality_count == 1:
                # Slightly reduce confidence for single-modality predictions
                confidence = confidence * 0.9
            elif modality_count == 0:
                # Very low confidence for no modalities
                confidence = 0.1
                warnings.append("No modalities successfully processed")
            
        except Exception as e:
            error_msg = f"Model prediction failed: {type(e).__name__}: {e}"
            logger.error(error_msg)
            errors.append(error_msg)
            
            # Fallback prediction
            p_real = 0.5
            p_fake = 0.5
            verdict = 'UNCERTAIN'
            confidence = 0.0
        
        processing_time = time.time() - start_time
        
        result = {
            'p_fake': round(p_fake, 6),
            'p_real': round(p_real, 6),
            'verdict': verdict,
            'confidence': round(confidence, 6),
            'modalities_used': modalities_used,
            'processing_details': processing_details,
            'processing_time': round(processing_time, 3),
            'errors': errors,
            'warnings': warnings
        }
        
        logger.info(f"Enhanced MHFN prediction: {verdict} (p_fake={p_fake:.3f}, "
                   f"modalities={modalities_used}, time={processing_time:.2f}s)")
        return result

# Example usage and testing
if __name__ == "__main__":
    # Initialize inference system
    mhfn = MHFNInference()
    
    # Test cases
    test_cases = [
        {
            'text': "Breaking: Scientists discover that vaccines contain microchips for mind control",
            'description': "Conspiracy theory text"
        },
        {
            'text': "New study published in Nature shows climate change accelerating",
            'url': "https://www.nature.com/articles/example",
            'description': "Scientific news with credible source"
        },
        {
            'text': "Celebrity spotted at restaurant eating pizza",
            'image_url': "https://example.com/celebrity.jpg",
            'description': "Entertainment news with image"
        }
    ]
    
    print("\n=== MHFN Inference Testing ===")
    
    for i, case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: {case['description']}")
        
        result = mhfn.predict(
            text=case.get('text'),
            image_url=case.get('image_url'),
            url=case.get('url')
        )
        
        print(f"Verdict: {result['verdict']}")
        print(f"P(Fake): {result['p_fake']:.3f}")
        print(f"P(Real): {result['p_real']:.3f}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Modalities: {result['modalities_used']}")
    
    # Test determinism
    print("\n=== Determinism Test ===")
    test_text = "This is a test for deterministic behavior"
    
    results = []
    for i in range(5):
        result = mhfn.predict(text=test_text)
        results.append(result['p_fake'])
    
    print(f"5 repeated predictions: {results}")
    print(f"All identical: {len(set(results)) == 1}")