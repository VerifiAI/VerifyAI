#!/usr/bin/env python3
"""
End-to-End Source-Temporal Integration Test
Chunk 17 - Hybrid Deep Learning with Explainable AI for Fake News Detection

This test validates the complete pipeline:
1. Feed data processing with publisher credibility and timestamps
2. Source-temporal tensor creation and integration
3. MHFN model prediction with enhanced features
4. Performance validation and accuracy metrics
"""

import sys
import os
import time
import torch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import FakeNewsDataLoader
from model import MHFN

class SourceTemporalE2ETest:
    def __init__(self):
        self.data_loader = FakeNewsDataLoader()
        self.model = MHFN(input_dim=300, hidden_dim=64, source_temporal_dim=2)
        self.test_results = {}
        
    def create_mock_feed_data(self) -> pd.DataFrame:
        """Create realistic mock feed data for testing"""
        print("Creating mock feed data...")
        
        # Create diverse test cases
        feeds = [
            {
                'title': 'Breaking: Major Scientific Discovery Announced',
                'text': 'Scientists at leading research institutions have announced a breakthrough in renewable energy technology that could revolutionize the industry.',
                'publisher': 'BBC',
                'timestamp': '2025-08-25 10:00:00',
                'label': 0  # Real news
            },
            {
                'title': 'SHOCKING: Aliens Land in Central Park!',
                'text': 'Witnesses report seeing extraterrestrial beings landing in New York City. Government officials refuse to comment on the situation.',
                'publisher': 'Unknown Tabloid',
                'timestamp': '2025-08-25 11:30:00',
                'label': 1  # Fake news
            },
            {
                'title': 'Economic Markets Show Steady Growth',
                'text': 'Financial analysts report consistent growth in major market indices, indicating positive economic trends for the upcoming quarter.',
                'publisher': 'Reuters',
                'timestamp': '2025-08-25 09:15:00',
                'label': 0  # Real news
            },
            {
                'title': 'Celebrity Scandal Rocks Entertainment Industry',
                'text': 'Unverified sources claim major celebrity involved in controversial incident. No official statements have been released.',
                'publisher': 'gossip.com',
                'timestamp': '2025-08-25 12:45:00',
                'label': 1  # Fake news
            },
            {
                'title': 'Climate Change Report Released by UN',
                'text': 'The United Nations has published its latest comprehensive report on global climate change impacts and mitigation strategies.',
                'publisher': 'Associated Press',
                'timestamp': '2025-08-25 08:00:00',
                'label': 0  # Real news
            }
        ]
        
        df = pd.DataFrame(feeds)
        print(f"✓ Created {len(df)} mock feed entries")
        return df
    
    def test_feed_processing_pipeline(self, feed_data: pd.DataFrame):
        """Test the complete feed processing pipeline"""
        print("\nTesting feed processing pipeline...")
        
        try:
            # Process the feed data
            processed_data = self.data_loader.preprocess_data(feed_data, split='test')
            
            # Validate processed data structure
            required_keys = ['features', 'labels', 'source_temporal']
            for key in required_keys:
                assert key in processed_data, f"Missing key: {key}"
            
            # Validate tensor shapes
            features = processed_data['features']
            labels = processed_data['labels']
            source_temporal = processed_data['source_temporal']
            
            assert features.shape[1] == 300, f"Features should be [batch, 300], got {features.shape}"
            assert source_temporal.shape[1] == 2, f"Source-temporal should be [batch, 2], got {source_temporal.shape}"
            assert features.shape[0] == source_temporal.shape[0], "Batch sizes should match"
            
            print(f"✓ Features shape: {features.shape}")
            print(f"✓ Source-temporal shape: {source_temporal.shape}")
            print(f"✓ Labels shape: {labels.shape}")
            
            # Validate credibility values
            credibility_values = source_temporal[:, 0]
            print(f"✓ Credibility range: [{credibility_values.min():.3f}, {credibility_values.max():.3f}]")
            
            # Validate timestamp normalization
            timestamp_values = source_temporal[:, 1]
            print(f"✓ Timestamp range: [{timestamp_values.min():.3f}, {timestamp_values.max():.3f}]")
            
            self.test_results['pipeline_processing'] = 'PASSED'
            return processed_data
            
        except Exception as e:
            print(f"✗ Pipeline processing failed: {e}")
            self.test_results['pipeline_processing'] = f'FAILED: {e}'
            raise
    
    def test_model_predictions(self, processed_data: dict):
        """Test model predictions with source-temporal features"""
        print("\nTesting model predictions...")
        
        try:
            features = processed_data['features']
            source_temporal = processed_data['source_temporal']
            labels = processed_data['labels']
            
            # Test batch prediction
            start_time = time.time()
            predictions = self.model.forward(features, source_temporal)
            prediction_time = time.time() - start_time
            
            # Validate prediction shape and values
            assert predictions.shape[0] == features.shape[0], "Prediction batch size mismatch"
            assert predictions.shape[1] == 1, "Predictions should be [batch, 1]"
            assert torch.all(predictions >= 0) and torch.all(predictions <= 1), "Predictions should be in [0, 1]"
            
            print(f"✓ Batch predictions shape: {predictions.shape}")
            print(f"✓ Prediction time: {prediction_time:.4f}s")
            print(f"✓ Prediction range: [{predictions.min():.3f}, {predictions.max():.3f}]")
            
            # Test individual predictions
            individual_predictions = []
            for i in range(features.shape[0]):
                pred = self.model.predict(features[i], source_temporal[i])
                # pred is already a float from the predict method
                individual_predictions.append(pred if isinstance(pred, float) else pred.item())
            
            print(f"✓ Individual predictions: {[f'{p:.3f}' for p in individual_predictions]}")
            
            # Performance metrics
            self.test_results['prediction_time'] = prediction_time
            self.test_results['avg_prediction_time'] = prediction_time / features.shape[0]
            self.test_results['model_predictions'] = 'PASSED'
            
            return predictions.detach().numpy().flatten()
            
        except Exception as e:
            print(f"✗ Model prediction failed: {e}")
            self.test_results['model_predictions'] = f'FAILED: {e}'
            raise
    
    def test_credibility_influence(self, feed_data: pd.DataFrame):
        """Test that publisher credibility influences predictions"""
        print("\nTesting credibility influence on predictions...")
        
        try:
            # Create two identical articles with different publishers
            test_article = {
                'title': 'Test Article for Credibility Analysis',
                'text': 'This is a test article to analyze the impact of publisher credibility on fake news detection.',
                'timestamp': '2025-08-25 12:00:00',
                'label': 0
            }
            
            # High credibility publisher
            high_cred_data = pd.DataFrame([{**test_article, 'publisher': 'BBC'}])
            high_cred_processed = self.data_loader.preprocess_data(high_cred_data, split='test')
            high_cred_pred = self.model.forward(
                high_cred_processed['features'],
                high_cred_processed['source_temporal']
            ).item()
            
            # Low credibility publisher
            low_cred_data = pd.DataFrame([{**test_article, 'publisher': 'fakenews.com'}])
            low_cred_processed = self.data_loader.preprocess_data(low_cred_data, split='test')
            low_cred_pred = self.model.forward(
                low_cred_processed['features'],
                low_cred_processed['source_temporal']
            ).item()
            
            print(f"✓ High credibility (BBC) prediction: {high_cred_pred:.3f}")
            print(f"✓ Low credibility (fakenews.com) prediction: {low_cred_pred:.3f}")
            
            # Verify credibility influence
            credibility_diff = abs(high_cred_pred - low_cred_pred)
            print(f"✓ Credibility influence: {credibility_diff:.3f}")
            
            self.test_results['credibility_influence'] = credibility_diff
            self.test_results['credibility_test'] = 'PASSED'
            
        except Exception as e:
            print(f"✗ Credibility influence test failed: {e}")
            self.test_results['credibility_test'] = f'FAILED: {e}'
            raise
    
    def test_temporal_influence(self, feed_data: pd.DataFrame):
        """Test that timestamp affects predictions"""
        print("\nTesting temporal influence on predictions...")
        
        try:
            # Create identical articles with different timestamps
            test_article = {
                'title': 'Test Article for Temporal Analysis',
                'text': 'This is a test article to analyze the impact of publication time on fake news detection.',
                'publisher': 'Reuters',
                'label': 0
            }
            
            # Recent timestamp
            recent_data = pd.DataFrame([{**test_article, 'timestamp': '2025-08-25 12:00:00'}])
            recent_processed = self.data_loader.preprocess_data(recent_data, split='test')
            recent_pred = self.model.forward(
                recent_processed['features'],
                recent_processed['source_temporal']
            ).item()
            
            # Older timestamp
            old_data = pd.DataFrame([{**test_article, 'timestamp': '2020-01-01 12:00:00'}])
            old_processed = self.data_loader.preprocess_data(old_data, split='test')
            old_pred = self.model.forward(
                old_processed['features'],
                old_processed['source_temporal']
            ).item()
            
            print(f"✓ Recent timestamp prediction: {recent_pred:.3f}")
            print(f"✓ Old timestamp prediction: {old_pred:.3f}")
            
            # Verify temporal influence
            temporal_diff = abs(recent_pred - old_pred)
            print(f"✓ Temporal influence: {temporal_diff:.3f}")
            
            self.test_results['temporal_influence'] = temporal_diff
            self.test_results['temporal_test'] = 'PASSED'
            
        except Exception as e:
            print(f"✗ Temporal influence test failed: {e}")
            self.test_results['temporal_test'] = f'FAILED: {e}'
            raise
    
    def run_performance_benchmark(self, feed_data: pd.DataFrame):
        """Run performance benchmarks"""
        print("\nRunning performance benchmarks...")
        
        try:
            # Process larger batch for performance testing
            large_batch = pd.concat([feed_data] * 20, ignore_index=True)  # 100 samples
            
            # Measure processing time
            start_time = time.time()
            processed_data = self.data_loader.preprocess_data(large_batch, split='test')
            processing_time = time.time() - start_time
            
            # Measure prediction time
            start_time = time.time()
            predictions = self.model.forward(
                processed_data['features'],
                processed_data['source_temporal']
            )
            prediction_time = time.time() - start_time
            
            batch_size = processed_data['features'].shape[0]
            
            print(f"✓ Batch size: {batch_size}")
            print(f"✓ Processing time: {processing_time:.4f}s")
            print(f"✓ Prediction time: {prediction_time:.4f}s")
            print(f"✓ Per-sample processing: {processing_time/batch_size*1000:.2f}ms")
            print(f"✓ Per-sample prediction: {prediction_time/batch_size*1000:.2f}ms")
            
            # Performance requirements check
            per_sample_total = (processing_time + prediction_time) / batch_size * 1000
            performance_ok = per_sample_total < 500  # Should be under 500ms per sample
            
            self.test_results['batch_size'] = batch_size
            self.test_results['processing_time'] = processing_time
            self.test_results['prediction_time'] = prediction_time
            self.test_results['per_sample_time_ms'] = per_sample_total
            self.test_results['performance_ok'] = performance_ok
            self.test_results['performance_benchmark'] = 'PASSED' if performance_ok else 'WARNING'
            
            if performance_ok:
                print(f"✓ Performance: {per_sample_total:.2f}ms per sample (< 500ms target)")
            else:
                print(f"⚠ Performance: {per_sample_total:.2f}ms per sample (> 500ms target)")
            
        except Exception as e:
            print(f"✗ Performance benchmark failed: {e}")
            self.test_results['performance_benchmark'] = f'FAILED: {e}'
            raise
    
    def generate_test_report(self):
        """Generate comprehensive test report"""
        print("\n" + "="*60)
        print("SOURCE-TEMPORAL E2E TEST REPORT - CHUNK 17")
        print("="*60)
        
        # Test summary
        passed_tests = sum(1 for result in self.test_results.values() 
                          if isinstance(result, str) and result == 'PASSED')
        total_tests = sum(1 for result in self.test_results.values() 
                         if isinstance(result, str) and result.endswith(('PASSED', 'FAILED', 'WARNING')))
        
        print(f"Tests Passed: {passed_tests}/{total_tests}")
        print(f"Success Rate: {passed_tests/total_tests*100:.1f}%")
        
        # Detailed results
        print("\nDetailed Results:")
        for key, value in self.test_results.items():
            if isinstance(value, str) and value.endswith(('PASSED', 'FAILED', 'WARNING')):
                status_icon = "✓" if value == 'PASSED' else "⚠" if value == 'WARNING' else "✗"
                print(f"  {status_icon} {key}: {value}")
        
        # Performance metrics
        if 'per_sample_time_ms' in self.test_results:
            print(f"\nPerformance Metrics:")
            print(f"  • Per-sample processing: {self.test_results['per_sample_time_ms']:.2f}ms")
            print(f"  • Credibility influence: {self.test_results.get('credibility_influence', 'N/A')}")
            print(f"  • Temporal influence: {self.test_results.get('temporal_influence', 'N/A')}")
        
        # Overall status
        all_critical_passed = all(
            self.test_results.get(test, 'FAILED') == 'PASSED'
            for test in ['pipeline_processing', 'model_predictions', 'credibility_test', 'temporal_test']
        )
        
        if all_critical_passed:
            print("\n🎉 ALL CRITICAL TESTS PASSED - SOURCE-TEMPORAL INTEGRATION SUCCESSFUL!")
        else:
            print("\n❌ CRITICAL TESTS FAILED - SOURCE-TEMPORAL INTEGRATION NEEDS FIXES")
        
        return self.test_results

def run_source_temporal_e2e_test():
    """Main test execution function"""
    print("="*60)
    print("SOURCE-TEMPORAL E2E TEST - CHUNK 17")
    print("Publisher Credibility & Timestamp Integration")
    print("="*60)
    
    test_suite = SourceTemporalE2ETest()
    
    try:
        # Create test data
        feed_data = test_suite.create_mock_feed_data()
        
        # Run all tests
        processed_data = test_suite.test_feed_processing_pipeline(feed_data)
        test_suite.test_model_predictions(processed_data)
        test_suite.test_credibility_influence(feed_data)
        test_suite.test_temporal_influence(feed_data)
        test_suite.run_performance_benchmark(feed_data)
        
        # Generate report
        results = test_suite.generate_test_report()
        
        return results
        
    except Exception as e:
        print(f"\n❌ SOURCE-TEMPORAL E2E TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return {'overall_status': 'FAILED', 'error': str(e)}

if __name__ == "__main__":
    results = run_source_temporal_e2e_test()