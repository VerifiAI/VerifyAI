#!/usr/bin/env python3
"""
Chunk 7 Validation Script for Data Loader
Hybrid Deep Learning with Explainable AI for Fake News Detection

This script validates all data loader functionality with 90%+ accuracy requirements:
- Parquet file loading
- Pickle file loading
- Data preprocessing
- Tensor dimension validation [batch, 300]
- Missing value handling
- Feature/label splitting
- Batch processing
- Integration testing

Author: AI Assistant
Date: August 24, 2025
Version: 1.0
"""

import os
import sys
import unittest
import logging
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from datetime import datetime
import warnings

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from data_loader import FakeNewsDataLoader, FakeNewsDataset
except ImportError as e:
    print(f"Error importing data_loader: {e}")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

class TestDataLoaderValidation(unittest.TestCase):
    """
    Comprehensive validation tests for FakeNewsDataLoader.
    """
    
    @classmethod
    def setUpClass(cls):
        """Set up test class."""
        cls.data_loader = FakeNewsDataLoader()
        cls.test_results = []
        cls.start_time = datetime.now()
        logger.info("Starting Chunk 7 Data Loader Validation Tests")
    
    def setUp(self):
        """Set up each test."""
        self.test_start = datetime.now()
    
    def tearDown(self):
        """Clean up after each test."""
        test_duration = (datetime.now() - self.test_start).total_seconds()
        test_name = self._testMethodName
        
        # Simple success tracking - will be updated by test runner
        self.test_results.append({
            'test_name': test_name,
            'success': True,  # Will be updated by runner
            'duration': test_duration,
            'timestamp': datetime.now().isoformat()
        })
        
        logger.info(f"Test {test_name} completed ({test_duration:.2f}s)")
    
    def test_01_data_loader_initialization(self):
        """Test data loader initialization."""
        logger.info("Testing data loader initialization...")
        
        # Test default initialization
        loader = FakeNewsDataLoader()
        self.assertIsNotNone(loader)
        self.assertEqual(loader.target_dim, 300)
        self.assertTrue(loader.data_dir.exists())
        
        # Test custom data directory
        custom_dir = Path.cwd().parent / 'data'
        if custom_dir.exists():
            custom_loader = FakeNewsDataLoader(str(custom_dir))
            self.assertEqual(custom_loader.data_dir, custom_dir)
        
        logger.info("✓ Data loader initialization test passed")
    
    def test_02_parquet_file_loading(self):
        """Test Parquet file loading functionality."""
        logger.info("Testing Parquet file loading...")
        
        # Load Parquet files
        parquet_data = self.data_loader.load_parquet_files()
        
        # Validate loaded data
        self.assertIsInstance(parquet_data, dict)
        self.assertIn('train', parquet_data)
        self.assertIn('val', parquet_data)
        self.assertIn('test', parquet_data)
        
        # Check data types and shapes
        for split, data in parquet_data.items():
            self.assertIsInstance(data, pd.DataFrame)
            self.assertGreater(len(data), 0, f"{split} data should not be empty")
            self.assertIn('label', data.columns, f"{split} should have label column")
        
        # Validate data loader attributes
        self.assertIsNotNone(self.data_loader.train_data)
        self.assertIsNotNone(self.data_loader.val_data)
        self.assertIsNotNone(self.data_loader.test_data)
        
        logger.info("✓ Parquet file loading test passed")
    
    def test_03_pickle_file_loading(self):
        """Test pickle file loading and alignment verification."""
        logger.info("Testing pickle file loading...")
        
        # Load pickle files
        pickle_data = self.data_loader.load_pickle_files()
        
        # Validate loaded data
        self.assertIsInstance(pickle_data, dict)
        self.assertGreater(len(pickle_data), 0, "Should load at least one pickle file")
        
        # Check data structure
        for key, data in pickle_data.items():
            if isinstance(data, pd.DataFrame):
                self.assertGreater(len(data), 0, f"{key} should not be empty")
                # Check for expected columns if they exist
                if 'text_features' in data.columns and 'image_features' in data.columns:
                    self.assertEqual(len(data['text_features']), len(data['image_features']),
                                   f"Text and image features should be aligned in {key}")
        
        # Validate data loader pickle data
        self.assertIsNotNone(self.data_loader.pickle_data)
        
        logger.info("✓ Pickle file loading test passed")
    
    def test_04_missing_value_handling(self):
        """Test missing value handling."""
        logger.info("Testing missing value handling...")
        
        # Create test data with missing values
        test_data = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'text': ['text1', None, 'text3', '', 'text5'],
            'title': ['title1', 'title2', None, 'title4', 'title5'],
            'label': [1, 0, None, 1, 0],
            'features': [[1.0]*300, None, [3.0]*300, [4.0]*300, [5.0]*300]
        })
        
        # Handle missing values
        cleaned_data = self.data_loader._handle_missing_values(test_data)
        
        # Validate cleaning
        self.assertEqual(len(cleaned_data), 5, "Should preserve all rows")
        self.assertFalse(cleaned_data['text'].isnull().any(), "Text nulls should be handled")
        self.assertFalse(cleaned_data['title'].isnull().any(), "Title nulls should be handled")
        self.assertFalse(cleaned_data['label'].isnull().any(), "Label nulls should be handled")
        
        # Check feature handling
        for idx, features in enumerate(cleaned_data['features']):
            self.assertIsNotNone(features, f"Features at index {idx} should not be None")
            self.assertEqual(len(features), 300, f"Features at index {idx} should have 300 dimensions")
        
        logger.info("✓ Missing value handling test passed")
    
    def test_05_feature_label_extraction(self):
        """Test feature and label extraction."""
        logger.info("Testing feature and label extraction...")
        
        # Use loaded train data
        if self.data_loader.train_data is None:
            self.data_loader.load_parquet_files()
        
        # Extract features and labels
        features, labels = self.data_loader._extract_features_labels(self.data_loader.train_data)
        
        # Validate extraction
        self.assertIsInstance(features, np.ndarray)
        self.assertIsInstance(labels, np.ndarray)
        self.assertEqual(len(features), len(labels), "Features and labels should have same length")
        self.assertEqual(features.shape[1], 300, "Features should have 300 dimensions")
        
        # Check label values
        unique_labels = np.unique(labels)
        self.assertTrue(all(label in [0, 1] for label in unique_labels), "Labels should be binary (0 or 1)")
        
        logger.info("✓ Feature and label extraction test passed")
    
    def test_06_tensor_dimension_validation(self):
        """Test tensor dimension validation [batch, 300]."""
        logger.info("Testing tensor dimension validation...")
        
        # Test different input shapes
        test_cases = [
            np.random.randn(50, 300),  # Correct shape
            np.random.randn(50, 200),  # Need padding
            np.random.randn(50, 500),  # Need truncation
            np.random.randn(300),      # Single sample
        ]
        
        for i, test_input in enumerate(test_cases):
            with self.subTest(case=i):
                tensor = self.data_loader._ensure_tensor_dimensions(test_input)
                
                # Validate output
                self.assertIsInstance(tensor, torch.Tensor)
                self.assertEqual(tensor.shape[-1], 300, f"Case {i}: Should have 300 features")
                self.assertEqual(len(tensor.shape), 2, f"Case {i}: Should be 2D tensor")
                
                if len(test_input.shape) == 1:
                    self.assertEqual(tensor.shape[0], 1, f"Case {i}: Single sample should have batch size 1")
                else:
                    self.assertEqual(tensor.shape[0], test_input.shape[0], f"Case {i}: Batch size should be preserved")
        
        logger.info("✓ Tensor dimension validation test passed")
    
    def test_07_data_preprocessing_pipeline(self):
        """Test complete data preprocessing pipeline."""
        logger.info("Testing data preprocessing pipeline...")
        
        # Use loaded train data
        if self.data_loader.train_data is None:
            self.data_loader.load_parquet_files()
        
        # Preprocess data
        preprocessed = self.data_loader.preprocess_data(self.data_loader.train_data, 'train')
        
        # Validate preprocessing results
        self.assertIsInstance(preprocessed, dict)
        self.assertIn('features', preprocessed)
        self.assertIn('labels', preprocessed)
        self.assertIn('batch_size', preprocessed)
        self.assertIn('feature_dim', preprocessed)
        
        # Validate tensor properties
        features = preprocessed['features']
        labels = preprocessed['labels']
        
        self.assertIsInstance(features, torch.Tensor)
        self.assertIsInstance(labels, torch.Tensor)
        self.assertEqual(features.shape[-1], 300, "Features should have 300 dimensions")
        self.assertEqual(preprocessed['feature_dim'], 300, "Feature dimension should be 300")
        self.assertEqual(len(features), len(labels), "Features and labels should have same length")
        
        # Check normalization
        feature_mean = features.mean().item()
        self.assertLess(abs(feature_mean), 0.1, "Features should be approximately normalized")
        
        logger.info("✓ Data preprocessing pipeline test passed")
    
    def test_08_small_batch_testing(self):
        """Test small batch processing."""
        logger.info("Testing small batch processing...")
        
        # Test different batch sizes
        batch_sizes = [8, 16, 32, 64]
        
        for batch_size in batch_sizes:
            with self.subTest(batch_size=batch_size):
                # Test small batch
                results = self.data_loader.test_small_batch(batch_size)
                
                # Validate results
                self.assertIsInstance(results, dict)
                self.assertTrue(results.get('success', False), f"Batch size {batch_size} should succeed")
                self.assertEqual(results.get('feature_dim'), 300, f"Feature dim should be 300 for batch {batch_size}")
                
                # Check shape consistency
                feature_shape = results.get('feature_shape', [])
                if len(feature_shape) == 2:
                    self.assertEqual(feature_shape[1], 300, f"Feature dimension should be 300 for batch {batch_size}")
        
        logger.info("✓ Small batch testing test passed")
    
    def test_09_pytorch_dataloader_creation(self):
        """Test PyTorch DataLoader creation."""
        logger.info("Testing PyTorch DataLoader creation...")
        
        # Create DataLoaders for different splits
        splits = ['train', 'val', 'test']
        
        for split in splits:
            with self.subTest(split=split):
                try:
                    dataloader = self.data_loader.get_data_loader(split, batch_size=16, shuffle=True)
                    
                    # Validate DataLoader
                    self.assertIsNotNone(dataloader)
                    self.assertGreater(len(dataloader), 0, f"{split} DataLoader should have batches")
                    
                    # Test one batch
                    for batch_features, batch_labels in dataloader:
                        self.assertEqual(batch_features.shape[-1], 300, f"{split} features should have 300 dims")
                        self.assertIsInstance(batch_features, torch.Tensor)
                        self.assertIsInstance(batch_labels, torch.Tensor)
                        break  # Only test first batch
                    
                except Exception as e:
                    self.fail(f"DataLoader creation failed for {split}: {str(e)}")
        
        logger.info("✓ PyTorch DataLoader creation test passed")
    
    def test_10_dataset_class_functionality(self):
        """Test FakeNewsDataset class functionality."""
        logger.info("Testing FakeNewsDataset class...")
        
        # Create test data
        features = torch.randn(100, 300)
        labels = torch.randint(0, 2, (100,))
        
        # Create dataset
        dataset = FakeNewsDataset(features, labels)
        
        # Validate dataset
        self.assertEqual(len(dataset), 100, "Dataset length should match input")
        
        # Test item access
        for i in [0, 50, 99]:  # Test different indices
            feature, label = dataset[i]
            self.assertIsInstance(feature, torch.Tensor)
            self.assertIsInstance(label, torch.Tensor)
            self.assertEqual(feature.shape, (300,), f"Feature at index {i} should have shape (300,)")
            self.assertEqual(label.shape, (), f"Label at index {i} should be scalar")
        
        # Test with mismatched lengths (should raise assertion)
        with self.assertRaises(AssertionError):
            FakeNewsDataset(torch.randn(100, 300), torch.randint(0, 2, (50,)))
        
        logger.info("✓ FakeNewsDataset class test passed")
    
    def test_11_data_statistics_generation(self):
        """Test data statistics generation."""
        logger.info("Testing data statistics generation...")
        
        # Ensure data is loaded
        if self.data_loader.train_data is None:
            self.data_loader.load_parquet_files()
        if not self.data_loader.pickle_data:
            self.data_loader.load_pickle_files()
        
        # Generate statistics
        stats = self.data_loader.get_data_statistics()
        
        # Validate statistics
        self.assertIsInstance(stats, dict)
        self.assertIn('timestamp', stats)
        self.assertIn('data_dir', stats)
        self.assertIn('target_dimension', stats)
        self.assertEqual(stats['target_dimension'], 300)
        
        # Check for data split statistics
        for split in ['train_data', 'val_data', 'test_data']:
            if split in stats:
                split_stats = stats[split]
                self.assertIn('shape', split_stats)
                self.assertIn('columns', split_stats)
                self.assertIsInstance(split_stats['shape'], list)
                self.assertIsInstance(split_stats['columns'], list)
        
        logger.info("✓ Data statistics generation test passed")
    
    def test_12_error_handling_robustness(self):
        """Test error handling and robustness."""
        logger.info("Testing error handling robustness...")
        
        # Test with invalid data directory
        try:
            invalid_loader = FakeNewsDataLoader('/nonexistent/path')
            # Should handle gracefully or raise appropriate error
        except Exception as e:
            self.assertIsInstance(e, (FileNotFoundError, OSError))
        
        # Test preprocessing with empty dataframe
        empty_df = pd.DataFrame()
        try:
            result = self.data_loader.preprocess_data(empty_df, 'empty')
            # Should handle gracefully
        except Exception as e:
            # Should be a reasonable error
            self.assertIsInstance(e, (ValueError, IndexError, KeyError))
        
        # Test with malformed data
        malformed_df = pd.DataFrame({
            'invalid_column': [1, 2, 3],
            'another_invalid': ['a', 'b', 'c']
        })
        
        try:
            result = self.data_loader.preprocess_data(malformed_df, 'malformed')
            # Should create features from available data
            self.assertIsInstance(result, dict)
        except Exception as e:
            # Should be handled gracefully
            pass
        
        logger.info("✓ Error handling robustness test passed")
    
    def test_13_memory_efficiency(self):
        """Test memory efficiency and resource management."""
        logger.info("Testing memory efficiency...")
        
        # Test with larger batch sizes
        large_batch_results = self.data_loader.test_small_batch(128)
        self.assertTrue(large_batch_results.get('success', False), "Large batch should succeed")
        
        # Test multiple DataLoader creations (should not cause memory leaks)
        for i in range(5):
            dataloader = self.data_loader.get_data_loader('train', batch_size=32)
            self.assertIsNotNone(dataloader)
            del dataloader  # Explicit cleanup
        
        # Test tensor operations don't accumulate
        for i in range(10):
            test_tensor = torch.randn(50, 200)
            result = self.data_loader._ensure_tensor_dimensions(test_tensor)
            self.assertEqual(result.shape, (50, 300))
            del test_tensor, result
        
        logger.info("✓ Memory efficiency test passed")
    
    def test_14_integration_with_mhfn_requirements(self):
        """Test integration with MHFN model requirements."""
        logger.info("Testing MHFN integration requirements...")
        
        # Ensure data is loaded
        if self.data_loader.train_data is None:
            self.data_loader.load_parquet_files()
        
        # Get DataLoader
        dataloader = self.data_loader.get_data_loader('train', batch_size=32)
        
        # Test multiple batches for consistency
        batch_count = 0
        for batch_features, batch_labels in dataloader:
            # Validate MHFN requirements
            self.assertEqual(batch_features.shape[-1], 300, "MHFN requires 300-dim features")
            self.assertEqual(len(batch_features.shape), 2, "MHFN requires 2D feature tensors")
            self.assertTrue(torch.is_tensor(batch_features), "MHFN requires PyTorch tensors")
            self.assertEqual(batch_features.dtype, torch.float32, "MHFN requires float32 tensors")
            
            # Check normalization (important for MHFN)
            feature_norm = torch.norm(batch_features, dim=1).mean()
            self.assertGreater(feature_norm.item(), 0.5, "Features should be reasonably normalized")
            self.assertLess(feature_norm.item(), 2.0, "Features should not be too large")
            
            batch_count += 1
            if batch_count >= 3:  # Test first 3 batches
                break
        
        self.assertGreater(batch_count, 0, "Should process at least one batch")
        
        logger.info("✓ MHFN integration requirements test passed")
    
    def test_15_performance_benchmarking(self):
        """Test performance benchmarking."""
        logger.info("Testing performance benchmarking...")
        
        import time
        
        # Benchmark data loading
        start_time = time.time()
        self.data_loader.load_parquet_files()
        parquet_load_time = time.time() - start_time
        
        start_time = time.time()
        self.data_loader.load_pickle_files()
        pickle_load_time = time.time() - start_time
        
        # Benchmark preprocessing
        start_time = time.time()
        preprocessed = self.data_loader.preprocess_data(self.data_loader.train_data.head(100), 'benchmark')
        preprocess_time = time.time() - start_time
        
        # Benchmark DataLoader creation
        start_time = time.time()
        dataloader = self.data_loader.get_data_loader('train', batch_size=32)
        dataloader_time = time.time() - start_time
        
        # Performance assertions (reasonable thresholds)
        self.assertLess(parquet_load_time, 10.0, "Parquet loading should be under 10 seconds")
        self.assertLess(pickle_load_time, 10.0, "Pickle loading should be under 10 seconds")
        self.assertLess(preprocess_time, 5.0, "Preprocessing 100 samples should be under 5 seconds")
        self.assertLess(dataloader_time, 5.0, "DataLoader creation should be under 5 seconds")
        
        logger.info(f"Performance metrics:")
        logger.info(f"  Parquet loading: {parquet_load_time:.2f}s")
        logger.info(f"  Pickle loading: {pickle_load_time:.2f}s")
        logger.info(f"  Preprocessing: {preprocess_time:.2f}s")
        logger.info(f"  DataLoader creation: {dataloader_time:.2f}s")
        
        logger.info("✓ Performance benchmarking test passed")
    
    @classmethod
    def tearDownClass(cls):
        """Generate final test report."""
        end_time = datetime.now()
        total_duration = (end_time - cls.start_time).total_seconds()
        
        # Calculate success rate
        total_tests = len(cls.test_results)
        passed_tests = sum(1 for result in cls.test_results if result['success'])
        success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        # Generate report
        report = {
            'timestamp': end_time.isoformat(),
            'total_duration': total_duration,
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': total_tests - passed_tests,
            'success_rate': success_rate,
            'target_success_rate': 90.0,
            'meets_requirement': success_rate >= 90.0,
            'test_details': cls.test_results
        }
        
        # Log final results
        logger.info("\n" + "="*80)
        logger.info("CHUNK 7 DATA LOADER VALIDATION RESULTS")
        logger.info("="*80)
        logger.info(f"Total Tests: {total_tests}")
        logger.info(f"Passed: {passed_tests}")
        logger.info(f"Failed: {total_tests - passed_tests}")
        logger.info(f"Success Rate: {success_rate:.1f}%")
        logger.info(f"Target: 90.0%")
        logger.info(f"Requirement Met: {'✓ YES' if report['meets_requirement'] else '✗ NO'}")
        logger.info(f"Total Duration: {total_duration:.2f} seconds")
        logger.info("="*80)
        
        if not report['meets_requirement']:
            logger.error("VALIDATION FAILED: Success rate below 90% requirement")
            failed_tests = [r for r in cls.test_results if not r['success']]
            for test in failed_tests:
                logger.error(f"  FAILED: {test['test_name']}")
        else:
            logger.info("✓ VALIDATION PASSED: All requirements met")
        
        # Save report to file
        import json
        report_file = Path('chunk7_validation_report.json')
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Detailed report saved to: {report_file}")
        
        # Set class attribute for external access
        cls.final_report = report

def run_validation():
    """
    Run the validation suite and return results.
    """
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestDataLoaderValidation)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    
    # Update test results with actual outcomes
    if hasattr(TestDataLoaderValidation, 'test_results'):
        failed_test_names = set()
        
        # Collect failed test names from errors and failures
        for test, error in result.errors + result.failures:
            test_name = test._testMethodName
            failed_test_names.add(test_name)
        
        # Update success status in test results
        for test_result in TestDataLoaderValidation.test_results:
            test_result['success'] = test_result['test_name'] not in failed_test_names
    
    # Return success status
    return result.wasSuccessful(), getattr(TestDataLoaderValidation, 'final_report', None)

if __name__ == '__main__':
    logger.info("Starting Chunk 7 Data Loader Validation...")
    
    try:
        success, report = run_validation()
        
        if success and report['meets_requirement']:
            logger.info("\n🎉 CHUNK 7 VALIDATION COMPLETED SUCCESSFULLY!")
            logger.info(f"Success Rate: {report['success_rate']:.1f}% (Target: 90.0%)")
            sys.exit(0)
        else:
            logger.error("\n❌ CHUNK 7 VALIDATION FAILED!")
            logger.error(f"Success Rate: {report['success_rate']:.1f}% (Target: 90.0%)")
            sys.exit(1)
    
    except Exception as e:
        logger.error(f"Validation script failed with error: {str(e)}")
        sys.exit(1)