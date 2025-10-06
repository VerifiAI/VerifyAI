#!/usr/bin/env python3
"""
Test Suite for Source-Temporal Feature Integration
Tests publisher credibility encoding and timestamp normalization in MHFN model

Author: FakeNewsBackend Team
Date: August 25, 2025
Chunk: 17
"""

import pytest
import torch
import numpy as np
import pandas as pd
from datetime import datetime, timezone
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import FakeNewsDataLoader
from model import MHFN

class TestSourceTemporalIntegration:
    """Test class for source-temporal feature integration"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.data_loader = FakeNewsDataLoader()
        self.model = MHFN(input_dim=300, hidden_dim=64, source_temporal_dim=2)
        
        # Create mock data for testing
        self.mock_data = pd.DataFrame({
            'text': ['This is a test news article', 'Another test article'],
            'publisher': ['BBC', 'Unknown Source'],
            'pubDate': ['2025-08-25T10:00:00Z', '2025-08-24T15:30:00Z'],
            'label': [0, 1]
        })
    
    def test_publisher_credibility_encoding(self):
        """Test publisher credibility encoding (0-1 scale)"""
        # Test known publishers
        bbc_credibility = self.data_loader._get_publisher_credibility('BBC')
        nyt_credibility = self.data_loader._get_publisher_credibility('New York Times')
        ap_credibility = self.data_loader._get_publisher_credibility('Associated Press')
        
        # Verify credibility scores are in valid range
        assert 0.0 <= bbc_credibility <= 1.0, f"BBC credibility {bbc_credibility} not in [0,1]"
        assert 0.0 <= nyt_credibility <= 1.0, f"NYT credibility {nyt_credibility} not in [0,1]"
        assert 0.0 <= ap_credibility <= 1.0, f"AP credibility {ap_credibility} not in [0,1]"
        
        # Test expected values (based on actual mapping)
        assert bbc_credibility == 0.95, f"Expected BBC=0.95, got {bbc_credibility}"
        assert nyt_credibility == 0.90, f"Expected NYT=0.90, got {nyt_credibility}"
        assert ap_credibility == 0.93, f"Expected AP=0.93, got {ap_credibility}"
        
        # Test unknown publisher (should get default)
        unknown_credibility = self.data_loader._get_publisher_credibility('Unknown Source')
        assert unknown_credibility == 0.5, f"Expected unknown=0.5, got {unknown_credibility}"
        
        print("✓ Publisher credibility encoding test passed")
    
    def test_timestamp_normalization(self):
        """Test timestamp normalization (days since epoch/365)"""
        # Test with known timestamp
        test_timestamp = '2025-08-25T10:00:00Z'
        normalized = self.data_loader._normalize_timestamp(test_timestamp)
        
        # Verify normalization is in reasonable range
        assert isinstance(normalized, float), f"Expected float, got {type(normalized)}"
        assert normalized > 0, f"Normalized timestamp should be positive, got {normalized}"
        
        # Test with different timestamp formats
        timestamps = [
            '2025-08-25T10:00:00Z',
            '2025-08-24T15:30:00+00:00',
            '2025-08-23 12:00:00'
        ]
        
        for ts in timestamps:
            norm_ts = self.data_loader._normalize_timestamp(ts)
            assert norm_ts > 0, f"Failed to normalize timestamp: {ts}"
        
        # Test invalid timestamp (should get current time normalized)
        invalid_norm = self.data_loader._normalize_timestamp('invalid')
        # Should be a reasonable current time value (around 55+ for 2025)
        assert 50 < invalid_norm < 60, f"Expected current time normalized (50-60), got {invalid_norm}"
        
        print("✓ Timestamp normalization test passed")
    
    def test_source_temporal_tensor_creation(self):
        """Test creation of [batch, 2] source-temporal tensor"""
        # Test tensor extraction
        source_temporal_tensor = self.data_loader.extract_source_temporal_tensor(self.mock_data)
        
        # Verify tensor shape
        assert isinstance(source_temporal_tensor, torch.Tensor), "Expected torch.Tensor"
        assert source_temporal_tensor.shape == (2, 2), f"Expected shape (2, 2), got {source_temporal_tensor.shape}"
        
        # Verify tensor values are in valid ranges
        credibility_values = source_temporal_tensor[:, 0]
        timestamp_values = source_temporal_tensor[:, 1]
        
        assert torch.all(credibility_values >= 0.0) and torch.all(credibility_values <= 1.0), \
            f"Credibility values out of range: {credibility_values}"
        assert torch.all(timestamp_values >= 0.0), f"Timestamp values should be positive: {timestamp_values}"
        
        # Test specific values
        assert torch.isclose(credibility_values[0], torch.tensor(0.95)), "BBC credibility should be 0.95"
        assert torch.isclose(credibility_values[1], torch.tensor(0.5)), "Unknown source should be 0.5"
        
        print("✓ Source-temporal tensor creation test passed")
    
    def test_model_forward_with_source_temporal(self):
        """Test MHFN model forward pass with source-temporal features"""
        # Create mock input tensors
        batch_size = 2
        input_features = torch.randn(batch_size, 300)  # Main features
        source_temporal = torch.tensor([[0.95, 0.8], [0.5, 0.6]], dtype=torch.float32)  # [credibility, timestamp]
        
        # Test forward pass with source-temporal features
        output_with_st = self.model.forward(input_features, source_temporal)
        
        # Test forward pass without source-temporal features
        output_without_st = self.model.forward(input_features)
        
        # Verify output shapes
        assert output_with_st.shape == (batch_size, 1), f"Expected shape ({batch_size}, 1), got {output_with_st.shape}"
        assert output_without_st.shape == (batch_size, 1), f"Expected shape ({batch_size}, 1), got {output_without_st.shape}"
        
        # Verify outputs are probabilities (0-1 range)
        assert torch.all(output_with_st >= 0.0) and torch.all(output_with_st <= 1.0), \
            f"Output with ST not in [0,1]: {output_with_st}"
        assert torch.all(output_without_st >= 0.0) and torch.all(output_without_st <= 1.0), \
            f"Output without ST not in [0,1]: {output_without_st}"
        
        # Verify that source-temporal features influence predictions
        assert not torch.allclose(output_with_st, output_without_st, atol=1e-6), \
            "Source-temporal features should influence predictions"
        
        print("✓ Model forward pass with source-temporal test passed")
    
    def test_model_predict_with_source_temporal(self):
        """Test MHFN model predict method with source-temporal features"""
        # Create single input
        input_features = torch.randn(300)
        source_temporal = torch.tensor([0.95, 0.8], dtype=torch.float32)
        
        # Test prediction with source-temporal features
        pred_with_st = self.model.predict(input_features, source_temporal)
        
        # Test prediction without source-temporal features
        pred_without_st = self.model.predict(input_features)
        
        # Verify predictions are probabilities
        assert 0.0 <= pred_with_st <= 1.0, f"Prediction with ST not in [0,1]: {pred_with_st}"
        assert 0.0 <= pred_without_st <= 1.0, f"Prediction without ST not in [0,1]: {pred_without_st}"
        
        # Verify predictions are different (source-temporal influence)
        assert abs(pred_with_st - pred_without_st) > 1e-6, \
            "Source-temporal features should influence single predictions"
        
        print("✓ Model predict with source-temporal test passed")
    
    def test_end_to_end_integration(self):
        """Test end-to-end integration of source-temporal features"""
        # Preprocess mock data
        preprocessed = self.data_loader.preprocess_data(self.mock_data)
        
        # Verify preprocessed data contains source-temporal tensor
        assert 'source_temporal' in preprocessed, "Preprocessed data missing source_temporal"
        assert 'features' in preprocessed, "Preprocessed data missing features"
        
        features = preprocessed['features']
        source_temporal = preprocessed['source_temporal']
        
        # Test model prediction with preprocessed data
        predictions = self.model.forward(features, source_temporal)
        
        # Verify predictions
        assert predictions.shape[0] == len(self.mock_data), "Batch size mismatch"
        assert torch.all(predictions >= 0.0) and torch.all(predictions <= 1.0), \
            "Predictions not in valid range"
        
        print("✓ End-to-end integration test passed")
    
    def test_credibility_influence_on_predictions(self):
        """Test that different credibility scores influence predictions"""
        input_features = torch.randn(1, 300)
        
        # Test with high credibility (BBC)
        high_cred_st = torch.tensor([[0.9, 0.8]], dtype=torch.float32)
        pred_high_cred = self.model.forward(input_features, high_cred_st)
        
        # Test with low credibility (unknown source)
        low_cred_st = torch.tensor([[0.3, 0.8]], dtype=torch.float32)
        pred_low_cred = self.model.forward(input_features, low_cred_st)
        
        # Verify different credibility scores produce different predictions
        assert not torch.allclose(pred_high_cred, pred_low_cred, atol=1e-6), \
            "Different credibility scores should produce different predictions"
        
        print("✓ Credibility influence test passed")
    
    def test_timestamp_influence_on_predictions(self):
        """Test that different timestamps influence predictions"""
        input_features = torch.randn(1, 300)
        
        # Test with recent timestamp
        recent_st = torch.tensor([[0.8, 0.9]], dtype=torch.float32)
        pred_recent = self.model.forward(input_features, recent_st)
        
        # Test with older timestamp
        old_st = torch.tensor([[0.8, 0.3]], dtype=torch.float32)
        pred_old = self.model.forward(input_features, old_st)
        
        # Verify different timestamps produce different predictions
        assert not torch.allclose(pred_recent, pred_old, atol=1e-6), \
            "Different timestamps should produce different predictions"
        
        print("✓ Timestamp influence test passed")

def run_all_tests():
    """Run all source-temporal integration tests"""
    print("=" * 60)
    print("Source-Temporal Feature Integration Tests - Chunk 17")
    print("=" * 60)
    
    test_instance = TestSourceTemporalIntegration()
    test_instance.setup_method()
    
    try:
        test_instance.test_publisher_credibility_encoding()
        test_instance.test_timestamp_normalization()
        test_instance.test_source_temporal_tensor_creation()
        test_instance.test_model_forward_with_source_temporal()
        test_instance.test_model_predict_with_source_temporal()
        test_instance.test_end_to_end_integration()
        test_instance.test_credibility_influence_on_predictions()
        test_instance.test_timestamp_influence_on_predictions()
        
        print("\n" + "=" * 60)
        print("✓ ALL SOURCE-TEMPORAL TESTS PASSED - 100% SUCCESS!")
        print("✓ Publisher credibility encoding: WORKING")
        print("✓ Timestamp normalization: WORKING")
        print("✓ [batch, 2] tensor creation: WORKING")
        print("✓ MHFN integration: WORKING")
        print("✓ Prediction influence: VERIFIED")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_all_tests()
    if not success:
        exit(1)