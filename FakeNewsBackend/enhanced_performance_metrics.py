#!/usr/bin/env python3
"""
Enhanced Performance Metrics System
Provides comprehensive evaluation metrics for fake news detection with real data

Features:
- Accuracy, Precision, Recall, F1-score calculations
- Per-class metrics for imbalanced datasets
- AUC-ROC with confidence intervals
- Cross-validation with stratified k-fold
- Hold-out testing with statistical reporting
- Real-time performance monitoring
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, roc_curve, auc, confusion_matrix,
    classification_report, precision_recall_curve
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import label_binarize
import scipy.stats as stats
from typing import Dict, List, Tuple, Any, Optional, Union
import logging
import time
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class EnhancedPerformanceMetrics:
    """
    Comprehensive performance evaluation system for fake news detection
    """
    
    def __init__(self, class_names: List[str] = None):
        """
        Initialize the performance metrics calculator
        
        Args:
            class_names: List of class names ['Real', 'Fake']
        """
        self.class_names = class_names or ['Real', 'Fake']
        self.metrics_history = []
        self.cv_results = {}
        self.holdout_results = {}
        
    def calculate_comprehensive_metrics(self, 
                                      y_true: np.ndarray, 
                                      y_pred: np.ndarray, 
                                      y_proba: np.ndarray = None,
                                      sample_weight: np.ndarray = None) -> Dict[str, Any]:
        """
        Calculate comprehensive performance metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities (for AUC-ROC)
            sample_weight: Sample weights for weighted metrics
            
        Returns:
            Dictionary containing all performance metrics
        """
        start_time = time.time()
        
        try:
            # Basic metrics
            accuracy = accuracy_score(y_true, y_pred, sample_weight=sample_weight)
            
            # Per-class metrics
            precision_macro = precision_score(y_true, y_pred, average='macro', sample_weight=sample_weight, zero_division=0)
            recall_macro = recall_score(y_true, y_pred, average='macro', sample_weight=sample_weight, zero_division=0)
            f1_macro = f1_score(y_true, y_pred, average='macro', sample_weight=sample_weight, zero_division=0)
            
            precision_weighted = precision_score(y_true, y_pred, average='weighted', sample_weight=sample_weight, zero_division=0)
            recall_weighted = recall_score(y_true, y_pred, average='weighted', sample_weight=sample_weight, zero_division=0)
            f1_weighted = f1_score(y_true, y_pred, average='weighted', sample_weight=sample_weight, zero_division=0)
            
            # Per-class detailed metrics
            per_class_precision = precision_score(y_true, y_pred, average=None, sample_weight=sample_weight, zero_division=0)
            per_class_recall = recall_score(y_true, y_pred, average=None, sample_weight=sample_weight, zero_division=0)
            per_class_f1 = f1_score(y_true, y_pred, average=None, sample_weight=sample_weight, zero_division=0)
            
            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred, sample_weight=sample_weight)
            
            # AUC-ROC calculation
            auc_roc = None
            auc_confidence_interval = None
            if y_proba is not None:
                try:
                    if len(np.unique(y_true)) == 2:  # Binary classification
                        auc_roc = roc_auc_score(y_true, y_proba[:, 1] if y_proba.ndim > 1 else y_proba, sample_weight=sample_weight)
                        # Calculate confidence interval using DeLong method approximation
                        auc_confidence_interval = self._calculate_auc_confidence_interval(y_true, y_proba[:, 1] if y_proba.ndim > 1 else y_proba)
                    else:  # Multi-class
                        auc_roc = roc_auc_score(y_true, y_proba, multi_class='ovr', sample_weight=sample_weight)
                except Exception as e:
                    logger.warning(f"Could not calculate AUC-ROC: {e}")
                    auc_roc = None
            
            # Class distribution analysis
            class_distribution = np.bincount(y_true) / len(y_true)
            
            # Performance by class
            per_class_metrics = {}
            for i, class_name in enumerate(self.class_names[:len(per_class_precision)]):
                per_class_metrics[class_name] = {
                    'precision': float(per_class_precision[i]),
                    'recall': float(per_class_recall[i]),
                    'f1_score': float(per_class_f1[i]),
                    'support': int(np.sum(y_true == i)),
                    'distribution': float(class_distribution[i])
                }
            
            # Compile comprehensive results
            metrics = {
                'timestamp': datetime.now().isoformat(),
                'computation_time_ms': round((time.time() - start_time) * 1000, 2),
                
                # Overall metrics
                'accuracy': float(accuracy),
                'macro_avg_precision': float(precision_macro),
                'macro_avg_recall': float(recall_macro),
                'macro_avg_f1_score': float(f1_macro),
                'weighted_avg_precision': float(precision_weighted),
                'weighted_avg_recall': float(recall_weighted),
                'weighted_avg_f1_score': float(f1_weighted),
                
                # AUC-ROC
                'auc_roc': float(auc_roc) if auc_roc is not None else None,
                'auc_confidence_interval': auc_confidence_interval,
                
                # Per-class metrics
                'per_class_metrics': per_class_metrics,
                
                # Confusion matrix
                'confusion_matrix': cm.tolist(),
                
                # Dataset characteristics
                'total_samples': int(len(y_true)),
                'class_distribution': class_distribution.tolist(),
                'is_balanced': bool(np.std(class_distribution) < 0.1),  # Consider balanced if std < 0.1
                
                # Classification report
                'classification_report': classification_report(y_true, y_pred, target_names=self.class_names, output_dict=True, zero_division=0)
            }
            
            # Store in history
            self.metrics_history.append(metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating comprehensive metrics: {e}")
            raise
    
    def _calculate_auc_confidence_interval(self, y_true: np.ndarray, y_scores: np.ndarray, confidence: float = 0.95) -> Tuple[float, float]:
        """
        Calculate confidence interval for AUC using bootstrap method
        """
        try:
            n_bootstraps = 1000
            rng = np.random.RandomState(42)
            bootstrapped_scores = []
            
            for _ in range(n_bootstraps):
                # Bootstrap sample
                indices = rng.randint(0, len(y_scores), len(y_scores))
                if len(np.unique(y_true[indices])) < 2:
                    continue
                    
                score = roc_auc_score(y_true[indices], y_scores[indices])
                bootstrapped_scores.append(score)
            
            sorted_scores = np.array(bootstrapped_scores)
            sorted_scores.sort()
            
            # Calculate confidence interval
            alpha = 1.0 - confidence
            lower_bound = np.percentile(sorted_scores, (alpha/2) * 100)
            upper_bound = np.percentile(sorted_scores, (1 - alpha/2) * 100)
            
            return (float(lower_bound), float(upper_bound))
            
        except Exception as e:
            logger.warning(f"Could not calculate AUC confidence interval: {e}")
            return None
    
    def cross_validate_performance(self, 
                                 model: Any, 
                                 X: np.ndarray, 
                                 y: np.ndarray, 
                                 cv_folds: int = 5,
                                 stratify: bool = True,
                                 random_state: int = 42) -> Dict[str, Any]:
        """
        Perform stratified k-fold cross-validation
        
        Args:
            model: Trained model with predict and predict_proba methods
            X: Feature matrix
            y: Target labels
            cv_folds: Number of cross-validation folds
            stratify: Whether to use stratified sampling
            random_state: Random state for reproducibility
            
        Returns:
            Cross-validation results with mean and std
        """
        logger.info(f"Starting {cv_folds}-fold cross-validation...")
        
        try:
            if stratify:
                cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
            else:
                from sklearn.model_selection import KFold
                cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
            
            fold_results = []
            
            for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
                logger.info(f"Processing fold {fold_idx + 1}/{cv_folds}...")
                
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Train model on fold
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_val)
                y_proba = None
                if hasattr(model, 'predict_proba'):
                    try:
                        y_proba = model.predict_proba(X_val)
                    except:
                        pass
                
                # Calculate metrics for this fold
                fold_metrics = self.calculate_comprehensive_metrics(y_val, y_pred, y_proba)
                fold_metrics['fold'] = fold_idx + 1
                fold_results.append(fold_metrics)
            
            # Aggregate results across folds
            cv_summary = self._aggregate_cv_results(fold_results)
            
            # Store results
            self.cv_results = {
                'summary': cv_summary,
                'fold_details': fold_results,
                'cv_config': {
                    'n_folds': cv_folds,
                    'stratified': stratify,
                    'random_state': random_state
                }
            }
            
            logger.info(f"Cross-validation completed. Mean accuracy: {cv_summary['accuracy']['mean']:.4f} ± {cv_summary['accuracy']['std']:.4f}")
            
            return self.cv_results
            
        except Exception as e:
            logger.error(f"Error in cross-validation: {e}")
            raise
    
    def _aggregate_cv_results(self, fold_results: List[Dict]) -> Dict[str, Any]:
        """
        Aggregate cross-validation results across folds
        """
        metrics_to_aggregate = [
            'accuracy', 'macro_avg_precision', 'macro_avg_recall', 'macro_avg_f1_score',
            'weighted_avg_precision', 'weighted_avg_recall', 'weighted_avg_f1_score', 'auc_roc'
        ]
        
        aggregated = {}
        
        for metric in metrics_to_aggregate:
            values = [fold[metric] for fold in fold_results if fold[metric] is not None]
            if values:
                aggregated[metric] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'values': values
                }
        
        # Aggregate per-class metrics
        per_class_aggregated = {}
        for class_name in self.class_names:
            per_class_aggregated[class_name] = {}
            for metric in ['precision', 'recall', 'f1_score']:
                values = [fold['per_class_metrics'].get(class_name, {}).get(metric, 0) 
                         for fold in fold_results]
                values = [v for v in values if v is not None]
                if values:
                    per_class_aggregated[class_name][metric] = {
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values))
                    }
        
        aggregated['per_class_metrics'] = per_class_aggregated
        
        return aggregated
    
    def holdout_test_evaluation(self, 
                              model: Any, 
                              X: np.ndarray, 
                              y: np.ndarray, 
                              test_size: float = 0.2,
                              n_iterations: int = 10,
                              random_state: int = 42) -> Dict[str, Any]:
        """
        Perform multiple hold-out test evaluations
        
        Args:
            model: Model to evaluate
            X: Feature matrix
            y: Target labels
            test_size: Proportion of data to use for testing
            n_iterations: Number of hold-out iterations
            random_state: Base random state
            
        Returns:
            Hold-out test results with statistics
        """
        logger.info(f"Starting {n_iterations} hold-out test iterations...")
        
        try:
            iteration_results = []
            
            for i in range(n_iterations):
                # Create train-test split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, 
                    stratify=y, random_state=random_state + i
                )
                
                # Train model
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test)
                y_proba = None
                if hasattr(model, 'predict_proba'):
                    try:
                        y_proba = model.predict_proba(X_test)
                    except:
                        pass
                
                # Calculate metrics
                iteration_metrics = self.calculate_comprehensive_metrics(y_test, y_pred, y_proba)
                iteration_metrics['iteration'] = i + 1
                iteration_results.append(iteration_metrics)
            
            # Aggregate results
            holdout_summary = self._aggregate_cv_results(iteration_results)  # Same aggregation logic
            
            # Store results
            self.holdout_results = {
                'summary': holdout_summary,
                'iteration_details': iteration_results,
                'config': {
                    'test_size': test_size,
                    'n_iterations': n_iterations,
                    'base_random_state': random_state
                }
            }
            
            logger.info(f"Hold-out testing completed. Mean accuracy: {holdout_summary['accuracy']['mean']:.4f} ± {holdout_summary['accuracy']['std']:.4f}")
            
            return self.holdout_results
            
        except Exception as e:
            logger.error(f"Error in hold-out testing: {e}")
            raise
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive performance summary
        """
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_evaluations': len(self.metrics_history),
            'cross_validation_available': bool(self.cv_results),
            'holdout_testing_available': bool(self.holdout_results)
        }
        
        if self.cv_results:
            summary['cross_validation_summary'] = {
                'mean_accuracy': self.cv_results['summary']['accuracy']['mean'],
                'std_accuracy': self.cv_results['summary']['accuracy']['std'],
                'mean_f1_macro': self.cv_results['summary']['macro_avg_f1_score']['mean'],
                'std_f1_macro': self.cv_results['summary']['macro_avg_f1_score']['std']
            }
        
        if self.holdout_results:
            summary['holdout_testing_summary'] = {
                'mean_accuracy': self.holdout_results['summary']['accuracy']['mean'],
                'std_accuracy': self.holdout_results['summary']['accuracy']['std'],
                'mean_f1_macro': self.holdout_results['summary']['macro_avg_f1_score']['mean'],
                'std_f1_macro': self.holdout_results['summary']['macro_avg_f1_score']['std']
            }
        
        if self.metrics_history:
            latest_metrics = self.metrics_history[-1]
            summary['latest_evaluation'] = {
                'accuracy': latest_metrics['accuracy'],
                'macro_f1': latest_metrics['macro_avg_f1_score'],
                'auc_roc': latest_metrics['auc_roc'],
                'per_class_f1': {name: metrics['f1_score'] 
                               for name, metrics in latest_metrics['per_class_metrics'].items()}
            }
        
        return summary
    
    def export_results(self, filepath: str):
        """
        Export all results to JSON file
        """
        results = {
            'metrics_history': self.metrics_history,
            'cross_validation_results': self.cv_results,
            'holdout_test_results': self.holdout_results,
            'performance_summary': self.get_performance_summary()
        }
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results exported to {filepath}")


def create_performance_evaluator(class_names: List[str] = None) -> EnhancedPerformanceMetrics:
    """
    Factory function to create performance evaluator
    """
    return EnhancedPerformanceMetrics(class_names=class_names)


if __name__ == "__main__":
    # Example usage
    logger.info("Enhanced Performance Metrics System initialized")
    logger.info("Available classes: EnhancedPerformanceMetrics")
    logger.info("Available functions: create_performance_evaluator")