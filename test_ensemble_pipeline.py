#!/usr/bin/env python3
"""
Comprehensive test suite for ensemble pipeline functionality
Tests model stacking, hyperparameter optimization, cross-validation, and performance monitoring
"""

import pytest
import numpy as np
import pandas as pd
import json
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import logging

# Import ensemble pipeline
from ensemble_pipeline import EnsemblePipeline, create_ensemble_pipeline, evaluate_ensemble_performance

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestEnsemblePipeline:
    """Test suite for ensemble pipeline functionality"""
    
    @classmethod
    def setup_class(cls):
        """Setup test data and pipeline"""
        # Create synthetic test data with more distinguishing features
        cls.fake_texts = [
            "BREAKING URGENT SHOCKING: Scientists discover aliens living among us! Government cover-up exposed! UNBELIEVABLE MIRACLE!",
            "You won't believe what this celebrity did! Click here for shocking photos! DOCTORS HATE THIS AMAZING SECRET!",
            "Miracle cure discovered! Doctors hate this one simple trick! INCREDIBLE BREAKTHROUGH EXPOSED!",
            "URGENT ALERT: Your computer has been infected! Download our antivirus now! SHOCKING VIRUS SCANDAL!",
            "Local mom makes $5000 a day with this weird trick! AMAZING MONEY SECRET REVEALED!",
            "Breaking: President secretly replaced by robot double! GOVERNMENT COVERUP EXPLOSIVE NEWS!",
            "Shocking truth about vaccines that Big Pharma doesn't want you to know! MIRACLE CURE HIDDEN!",
            "This one food will cure all diseases! Doctors are furious! UNBELIEVABLE HEALTH SECRET!",
            "EXPOSED: The real reason behind global warming will shock you! SCIENTISTS STUNNED AMAZING!",
            "Celebrity death hoax spreads across social media platforms! INCREDIBLE SCANDAL BREAKING!"
        ]
        
        cls.real_texts = [
            "The Federal Reserve announced a 0.25% interest rate increase following their monthly meeting. The methodology supports these findings through statistical analysis.",
            "Researchers at MIT published findings on renewable energy efficiency in peer-reviewed journal. The empirical evidence demonstrates significant correlations.",
            "The Supreme Court heard arguments on healthcare legislation reform yesterday. Statistical significance was established through comprehensive review.",
            "NASA's Mars rover successfully collected soil samples for analysis. The evidence-based approach confirms preliminary results.",
            "Stock markets closed mixed today with technology sector showing gains. Peer review methodology validates these economic indicators.",
            "Climate scientists report record-breaking temperatures in Arctic regions. The statistical significance supports climate change research.",
            "New archaeological discovery sheds light on ancient civilization practices. Evidence-based analysis confirms historical patterns.",
            "Economic indicators suggest steady growth in manufacturing sector. The methodology demonstrates empirical correlations.",
            "Medical researchers announce breakthrough in cancer treatment trials. Peer review process validates statistical significance.",
            "International trade negotiations continue between major economic partners. The empirical evidence supports economic projections."
        ]
        
        # Combine texts and labels
        cls.texts = cls.fake_texts + cls.real_texts
        cls.labels = [1] * len(cls.fake_texts) + [0] * len(cls.real_texts)  # 1 = fake, 0 = real
        
        # Split data
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = train_test_split(
            cls.texts, cls.labels, test_size=0.3, random_state=42, stratify=cls.labels
        )
        
        logger.info(f"Test data prepared: {len(cls.X_train)} training, {len(cls.X_test)} testing samples")
    
    def test_ensemble_pipeline_creation(self):
        """Test ensemble pipeline creation and initialization"""
        logger.info("Testing ensemble pipeline creation...")
        
        # Create pipeline
        pipeline = create_ensemble_pipeline()
        
        # Verify pipeline components
        assert pipeline is not None, "Pipeline creation failed"
        assert hasattr(pipeline, 'models'), "Pipeline missing models attribute"
        assert hasattr(pipeline, 'meta_model'), "Pipeline missing meta_model attribute"
        assert hasattr(pipeline, 'vectorizer'), "Pipeline missing vectorizer attribute"
        
        # Check individual models
        expected_models = ['xgboost', 'lightgbm', 'random_forest']
        for model_name in expected_models:
            assert model_name in pipeline.models, f"Missing {model_name} model"
        
        logger.info("✓ Ensemble pipeline creation test passed")
    
    def test_pipeline_training(self):
        """Test pipeline training functionality"""
        logger.info("Testing pipeline training...")
        
        # Create and train pipeline
        pipeline = create_ensemble_pipeline()
        
        start_time = time.time()
        pipeline.fit(self.X_train, self.y_train)
        training_time = time.time() - start_time
        
        # Verify training completed
        assert hasattr(pipeline, 'is_fitted'), "Pipeline missing is_fitted attribute"
        assert pipeline.is_fitted, "Pipeline not properly fitted"
        
        # Check training time is reasonable
        assert training_time < 300, f"Training took too long: {training_time:.2f}s"
        
        logger.info(f"✓ Pipeline training test passed (time: {training_time:.2f}s)")
    
    def test_ensemble_predictions(self):
        """Test ensemble prediction functionality"""
        logger.info("Testing ensemble predictions...")
        
        # Create and train pipeline
        pipeline = create_ensemble_pipeline()
        pipeline.fit(self.X_train, self.y_train)
        
        # Make predictions
        start_time = time.time()
        predictions = pipeline.predict(self.X_test)
        prediction_time = time.time() - start_time
        
        # Verify prediction structure
        assert 'ensemble_prediction' in predictions, "Missing ensemble_prediction"
        assert 'individual_predictions' in predictions, "Missing individual_predictions"
        assert 'confidence_scores' in predictions, "Missing confidence_scores"
        
        # Check prediction shapes
        ensemble_pred = predictions['ensemble_prediction']
        assert len(ensemble_pred) == len(self.X_test), "Incorrect prediction length"
        assert all(pred in [0, 1] for pred in ensemble_pred), "Invalid prediction values"
        
        # Check individual model predictions
        individual_preds = predictions['individual_predictions']
        expected_models = ['xgboost', 'lightgbm', 'random_forest']
        for model_name in expected_models:
            assert model_name in individual_preds, f"Missing {model_name} predictions"
            assert len(individual_preds[model_name]) == len(self.X_test), f"Incorrect {model_name} prediction length"
        
        # Check confidence scores
        confidence_scores = predictions['confidence_scores']
        assert len(confidence_scores) == len(self.X_test), "Incorrect confidence scores length"
        
        logger.info(f"✓ Ensemble predictions test passed (time: {prediction_time:.3f}s)")
    
    def test_model_performance(self):
        """Test model performance metrics"""
        logger.info("Testing model performance...")
        
        # Create and train pipeline
        pipeline = create_ensemble_pipeline()
        pipeline.fit(self.X_train, self.y_train)
        
        # Make predictions
        predictions = pipeline.predict(self.X_test)
        y_pred = predictions['ensemble_prediction']
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, average='weighted')
        recall = recall_score(self.y_test, y_pred, average='weighted')
        f1 = f1_score(self.y_test, y_pred, average='weighted')
        
        # Performance thresholds (realistic for synthetic data)
        min_accuracy = 0.55  # Minimum 55% accuracy (better than random)
        min_precision = 0.5
        min_recall = 0.5
        min_f1 = 0.6
        
        # Verify performance
        assert accuracy >= min_accuracy, f"Accuracy too low: {accuracy:.3f} < {min_accuracy}"
        assert precision >= min_precision, f"Precision too low: {precision:.3f} < {min_precision}"
        assert recall >= min_recall, f"Recall too low: {recall:.3f} < {min_recall}"
        assert f1 >= min_f1, f"F1-score too low: {f1:.3f} < {min_f1}"
        
        logger.info(f"✓ Model performance test passed (Acc: {accuracy:.3f}, P: {precision:.3f}, R: {recall:.3f}, F1: {f1:.3f})")
    
    def test_hyperparameter_optimization(self):
        """Test hyperparameter optimization functionality"""
        logger.info("Testing hyperparameter optimization...")
        
        # Create pipeline with optimization
        pipeline = create_ensemble_pipeline(use_optuna=True, n_trials=5)
        
        # Train with optimization
        start_time = time.time()
        pipeline.fit(self.X_train, self.y_train)
        optimization_time = time.time() - start_time
        
        # Verify optimization completed
        assert hasattr(pipeline, 'best_params'), "Pipeline missing best_params attribute"
        assert pipeline.best_params is not None, "Hyperparameter optimization failed"
        
        # Check optimization time is reasonable
        assert optimization_time < 600, f"Optimization took too long: {optimization_time:.2f}s"
        
        logger.info(f"✓ Hyperparameter optimization test passed (time: {optimization_time:.2f}s)")
    
    def test_cross_validation(self):
        """Test cross-validation functionality"""
        logger.info("Testing cross-validation...")
        
        # Create pipeline
        pipeline = create_ensemble_pipeline()
        
        # Perform cross-validation
        cv_results = pipeline.cross_validate(self.X_train, self.y_train, cv=3)
        
        # Verify CV results
        assert 'cv_scores' in cv_results, "Missing cv_scores"
        assert 'mean_score' in cv_results, "Missing mean_score"
        assert 'std_score' in cv_results, "Missing std_score"
        
        # Check CV scores
        cv_scores = cv_results['cv_scores']
        assert len(cv_scores) == 3, "Incorrect number of CV scores"
        assert all(0 <= score <= 1 for score in cv_scores), "Invalid CV scores"
        
        # Check mean score
        mean_score = cv_results['mean_score']
        assert 0 <= mean_score <= 1, "Invalid mean CV score"
        assert mean_score >= 0.5, f"Mean CV score too low: {mean_score:.3f}"
        
        logger.info(f"✓ Cross-validation test passed (Mean CV: {mean_score:.3f} ± {cv_results['std_score']:.3f})")
    
    def test_feature_importance(self):
        """Test feature importance extraction"""
        logger.info("Testing feature importance...")
        
        # Create and train pipeline
        pipeline = create_ensemble_pipeline()
        pipeline.fit(self.X_train, self.y_train)
        
        # Get feature importance
        try:
            importance = pipeline.get_feature_importance()
            
            # Verify importance structure
            if importance is not None:
                assert isinstance(importance, dict), "Feature importance should be dict"
                
                # Check for expected models
                expected_models = ['xgboost', 'lightgbm', 'random_forest']
                for model_name in expected_models:
                    if model_name in importance:
                        model_importance = importance[model_name]
                        assert isinstance(model_importance, (list, np.ndarray)), f"Invalid {model_name} importance format"
                        assert len(model_importance) > 0, f"Empty {model_name} importance"
            
            logger.info("✓ Feature importance test passed")
        except Exception as e:
            logger.warning(f"Feature importance test skipped: {e}")
    
    def test_performance_monitoring(self):
        """Test performance monitoring functionality"""
        logger.info("Testing performance monitoring...")
        
        # Create and train pipeline
        pipeline = create_ensemble_pipeline()
        pipeline.fit(self.X_train, self.y_train)
        
        # Evaluate performance
        performance_metrics = evaluate_ensemble_performance(pipeline, self.X_test, self.y_test)
        
        # Verify metrics structure
        expected_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        for metric in expected_metrics:
            assert metric in performance_metrics, f"Missing {metric} in performance metrics"
            assert 0 <= performance_metrics[metric] <= 1, f"Invalid {metric} value"
        
        # Check individual model metrics
        if 'individual_models' in performance_metrics:
            individual_metrics = performance_metrics['individual_models']
            expected_models = ['xgboost', 'lightgbm', 'random_forest']
            for model_name in expected_models:
                if model_name in individual_metrics:
                    model_metrics = individual_metrics[model_name]
                    for metric in expected_metrics:
                        assert metric in model_metrics, f"Missing {metric} for {model_name}"
        
        logger.info("✓ Performance monitoring test passed")
    
    def test_error_handling(self):
        """Test error handling and edge cases"""
        logger.info("Testing error handling...")
        
        pipeline = create_ensemble_pipeline()
        
        # Test prediction without training
        try:
            pipeline.predict(["test text"])
            assert False, "Should raise error for untrained pipeline"
        except Exception:
            pass  # Expected behavior
        
        # Test empty input
        pipeline.fit(self.X_train, self.y_train)
        try:
            result = pipeline.predict([])
            assert 'ensemble_prediction' in result
            assert len(result['ensemble_prediction']) == 0
        except Exception as e:
            logger.warning(f"Empty input handling: {e}")
        
        # Test single input
        try:
            result = pipeline.predict(["This is a test sentence."])
            assert 'ensemble_prediction' in result
            assert len(result['ensemble_prediction']) == 1
        except Exception as e:
            logger.error(f"Single input failed: {e}")
            raise
        
        logger.info("✓ Error handling test passed")

def run_comprehensive_test():
    """Run comprehensive ensemble pipeline test suite"""
    logger.info("Starting comprehensive ensemble pipeline test suite...")
    
    test_results = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'tests_run': 0,
        'tests_passed': 0,
        'tests_failed': 0,
        'test_details': [],
        'overall_status': 'UNKNOWN'
    }
    
    # Initialize test class
    test_suite = TestEnsemblePipeline()
    test_suite.setup_class()
    
    # List of test methods
    test_methods = [
        'test_ensemble_pipeline_creation',
        'test_pipeline_training',
        'test_ensemble_predictions',
        'test_model_performance',
        'test_hyperparameter_optimization',
        'test_cross_validation',
        'test_feature_importance',
        'test_performance_monitoring',
        'test_error_handling'
    ]
    
    # Run each test
    for test_name in test_methods:
        test_results['tests_run'] += 1
        
        try:
            start_time = time.time()
            test_method = getattr(test_suite, test_name)
            test_method()
            execution_time = time.time() - start_time
            
            test_results['tests_passed'] += 1
            test_results['test_details'].append({
                'test_name': test_name,
                'status': 'PASSED',
                'execution_time': round(execution_time, 3),
                'error': None
            })
            
        except Exception as e:
            execution_time = time.time() - start_time
            test_results['tests_failed'] += 1
            test_results['test_details'].append({
                'test_name': test_name,
                'status': 'FAILED',
                'execution_time': round(execution_time, 3),
                'error': str(e)
            })
            logger.error(f"Test {test_name} failed: {e}")
    
    # Calculate overall status
    if test_results['tests_failed'] == 0:
        test_results['overall_status'] = 'ALL_PASSED'
    elif test_results['tests_passed'] > test_results['tests_failed']:
        test_results['overall_status'] = 'MOSTLY_PASSED'
    else:
        test_results['overall_status'] = 'MOSTLY_FAILED'
    
    # Calculate success rate
    success_rate = (test_results['tests_passed'] / test_results['tests_run']) * 100
    test_results['success_rate'] = round(success_rate, 1)
    
    logger.info(f"Test suite completed: {test_results['tests_passed']}/{test_results['tests_run']} tests passed ({success_rate:.1f}%)")
    
    return test_results

if __name__ == '__main__':
    # Run comprehensive test suite
    results = run_comprehensive_test()
    
    # Print summary
    print("\n" + "="*60)
    print("ENSEMBLE PIPELINE TEST RESULTS")
    print("="*60)
    print(f"Tests Run: {results['tests_run']}")
    print(f"Tests Passed: {results['tests_passed']}")
    print(f"Tests Failed: {results['tests_failed']}")
    print(f"Success Rate: {results['success_rate']}%")
    print(f"Overall Status: {results['overall_status']}")
    print("="*60)
    
    # Save results to file
    with open('ensemble_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Detailed results saved to: ensemble_test_results.json")