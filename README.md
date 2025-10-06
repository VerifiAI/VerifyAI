# Hybrid Deep Learning with Explainable AI for Fake News Detection

## Project Overview

This project implements a Multi-modal Hybrid Fusion Network (MHFN) that combines text and image features for fake news detection using deep learning techniques with explainable AI capabilities.

## Dataset Information

### Fakeddit Dataset Statistics

**Total Dataset Size:** 824 samples across all splits
- **Training Set:** 501 samples (60.8%)
- **Validation Set:** 108 samples (13.1%)
- **Test Set:** 215 samples (26.1%)

### Data Types and Content

#### Text Data
- **Total Text Samples:** 824 (100% coverage)
- **Average Text Length:** 
  - Training: 50.6 characters
  - Validation: 50.1 characters
  - Test: 52.3 characters
- **Text Features:** Clean titles, original titles, and preprocessed text embeddings
- **Text Processing:** Hybrid embeddings using RoBERTa and DeBERTa models with PCA dimensionality reduction

#### Image Data
- **Total Images:** 2,141 images (JPEG format)
- **Image Distribution:**
  - Training images: Available in `data/fakeddit/subset/images/train/`
  - Validation images: Available in `data/fakeddit/subset/images/validate/`
  - Test images: Available in `data/fakeddit/subset/images/test/`
- **Image Processing:** ResNet-50 based feature extraction with 2048-dimensional feature vectors
- **Image Features:** Pre-extracted image features stored as binary embeddings

#### Metadata
- **Metadata Columns (13 total):**
  - `author`: Reddit post author
  - `id`: Unique post identifier
  - `linked_submission_id`: Related submission ID
  - `num_comments`: Number of comments on the post
  - `score`: Reddit post score
  - `subreddit`: Source subreddit
  - `upvote_ratio`: Upvote to downvote ratio
  - Additional temporal and user-based features

### Label Distribution

#### Binary Classification (2-way labels)
- **Class 0 (Real News):** 177 samples in training (35.3%)
- **Class 1 (Fake News):** 324 samples in training (64.7%)

#### Multi-class Classification (3-way labels)
- **Class 0:** 324 samples (64.7%)
- **Class 1:** 16 samples (3.2%)
- **Class 2:** 161 samples (32.1%)

#### Fine-grained Classification (6-way labels)
- **Class 0:** 324 samples (64.7%)
- **Class 1:** 45 samples (9.0%)
- **Class 2:** 93 samples (18.6%)
- **Class 3:** 14 samples (2.8%)
- **Class 4:** 1 sample (0.2%)
- **Class 5:** 24 samples (4.8%)

## Model Architecture

### Multi-modal Hybrid Fusion Network (MHFN)

**Core Components:**
- **Text Processing:** LSTM layers with hybrid embeddings (RoBERTa + DeBERTa)
- **Image Processing:** ResNet-50 feature extraction
- **Fusion Layer:** Multi-modal feature integration
- **Classification:** Dense layers with dropout regularization

**Model Configurations:**
- **BERT Model:** 12 layers, 768 hidden units, 12 attention heads
- **RoBERTa Model:** Base configuration with 125M parameters
- **ResNet-50:** Pre-trained on ImageNet, 2048-dimensional output features
- **Batch Size:** 16-32 (configurable)
- **Learning Rate:** 2e-5 to 5e-5 (adaptive)
- **Training Epochs:** 10-50 (with early stopping)

## Data Preprocessing Pipeline

### Text Preprocessing
1. **Text Cleaning:** Removal of special characters, URLs, and noise
2. **Tokenization:** BERT/RoBERTa tokenizer with max sequence length of 512
3. **Embedding Generation:** Hybrid embeddings using multiple transformer models
4. **Dimensionality Reduction:** PCA to target dimensions (configurable)

### Image Preprocessing
1. **Image Loading:** JPEG images from Fakeddit subset
2. **Preprocessing:** Resize, normalize, and augment images
3. **Feature Extraction:** ResNet-50 based feature extraction
4. **Feature Storage:** Pre-computed features stored as binary embeddings

### Data Splitting Strategy
- **Temporal Split:** 70% train, 15% validation, 15% test
- **Stratified Sampling:** Maintains label distribution across splits
- **User-based Split:** Prevents data leakage across user posts

## Evaluation Metrics

**Primary Metrics:**
- **Accuracy:** Overall classification accuracy
- **Precision:** Per-class and macro-averaged precision
- **Recall:** Per-class and macro-averaged recall
- **F1-Score:** Harmonic mean of precision and recall
- **AUC-ROC:** Area under the receiver operating characteristic curve

**Performance Monitoring:**
- Real-time training metrics tracking
- Validation loss monitoring with early stopping
- Cross-validation for robust evaluation

## Configuration Files

### Environment Configuration (`config_env`)
- Google API credentials
- Flask application settings
- RapidAPI and News API configurations
- Database connection parameters

### Model Configuration
- Hyperparameter settings in `research_methodology.md`
- Training configurations for different model variants
- Ensemble pipeline parameters

## Testing and Validation

**Test Suites:**
- `test_chunk7_validation.py`: Core model validation tests
- `test_ensemble_pipeline.py`: Ensemble model pipeline tests
- `test_chunk8_validation.py`: Training functionality validation

**Validation Procedures:**
- Model initialization testing
- Data loader functionality verification
- Feature extraction validation
- End-to-end pipeline testing

## Installation and Usage

### Prerequisites
```bash
pip install torch transformers pandas numpy scikit-learn
pip install pyarrow fastparquet  # For parquet file support
```

### Data Setup
1. Download Fakeddit dataset text/metadata
2. Download Fakeddit subset images
3. Place data in `data/fakeddit/` directory structure
4. Run preprocessing pipeline to generate processed parquet files

### Model Training
```bash
python FakeNewsBackend/model.py  # Train MHFN model
python test_ensemble_pipeline.py  # Test ensemble pipeline
```

## Project Structure

```
├── data/
│   ├── fakeddit/
│   │   ├── subset/images/  # 2,141 JPEG images
│   │   └── README.md       # Dataset documentation
│   └── processed/          # Processed parquet files (824 samples)
├── FakeNewsBackend/
│   ├── model.py           # MHFN model implementation
│   └── data_loader.py     # Data preprocessing pipeline
├── test_*.py              # Validation and testing suites
└── config_env             # Environment configuration
```

## Key Features

- **Multi-modal Learning:** Combines text and image features for enhanced detection
- **Hybrid Embeddings:** Uses multiple transformer models for robust text representation
- **Explainable AI:** Provides interpretable results for fake news detection
- **Scalable Architecture:** Supports various classification granularities (2-way, 3-way, 6-way)
- **Comprehensive Evaluation:** Multiple metrics and validation strategies
- **Production Ready:** Complete testing suite and configuration management

## Research Methodology

This implementation follows established research practices for fake news detection:
- Temporal data splitting to prevent data leakage
- Multi-modal fusion for improved accuracy
- Ensemble methods for robust predictions
- Comprehensive evaluation across multiple metrics
- Explainable AI techniques for transparency

## Performance Notes

The current model achieves varying performance across different classification tasks:
- Binary classification shows the most balanced performance
- Multi-class classification requires careful handling of class imbalance
- Fine-grained classification (6-way) presents challenges due to limited samples in some classes

## Contributing

This project implements state-of-the-art techniques for fake news detection with a focus on explainability and multi-modal learning. The codebase is designed for research and educational purposes with comprehensive testing and validation procedures.