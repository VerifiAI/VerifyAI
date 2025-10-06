#!/usr/bin/env python3
"""
Enhanced Explainability System
Provides comprehensive AI explainability features for fake news detection

Features:
- SHAP (SHapley Additive exPlanations) for global and local explanations
- LIME (Local Interpretable Model-agnostic Explanations) for instance-level explanations
- Feature importance analysis
- Decision reasoning and rationale
- Confidence score analysis
- Multimodal feature explanations
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
import logging
import time
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# SHAP imports
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logging.warning("SHAP not available. Install with: pip install shap")

# LIME imports
try:
    from lime import lime_tabular
    from lime.lime_text import LimeTextExplainer
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    logging.warning("LIME not available. Install with: pip install lime")

# Additional imports for text processing
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import Pipeline
except ImportError:
    pass

logger = logging.getLogger(__name__)

class EnhancedExplainabilitySystem:
    """
    Comprehensive explainability system for fake news detection models
    """
    
    def __init__(self, 
                 model: Any = None,
                 feature_names: List[str] = None,
                 class_names: List[str] = None,
                 text_features: List[str] = None):
        """
        Initialize the explainability system
        
        Args:
            model: Trained model with predict and predict_proba methods
            feature_names: List of feature names
            class_names: List of class names ['Real', 'Fake']
            text_features: List of text feature column names
        """
        self.model = model
        self.feature_names = feature_names or []
        self.class_names = class_names or ['Real', 'Fake']
        self.text_features = text_features or []
        
        # Initialize explainers
        self.shap_explainer = None
        self.lime_explainer = None
        self.lime_text_explainer = None
        
        # Cache for explanations
        self.explanation_cache = {}
        
        # Check availability
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not available. Some features will be limited.")
        if not LIME_AVAILABLE:
            logger.warning("LIME not available. Some features will be limited.")
    
    def initialize_explainers(self, 
                            X_background: np.ndarray,
                            text_data: List[str] = None,
                            mode: str = 'auto'):
        """
        Initialize SHAP and LIME explainers with background data
        
        Args:
            X_background: Background dataset for SHAP (representative sample)
            text_data: Text data for LIME text explainer
            mode: Explainer mode ('auto', 'tree', 'linear', 'kernel')
        """
        logger.info("Initializing explainers...")
        
        try:
            # Initialize SHAP explainer
            if SHAP_AVAILABLE and self.model is not None:
                self._initialize_shap_explainer(X_background, mode)
            
            # Initialize LIME explainer
            if LIME_AVAILABLE:
                self._initialize_lime_explainer(X_background)
                
                # Initialize LIME text explainer if text data provided
                if text_data:
                    self._initialize_lime_text_explainer(text_data)
            
            logger.info("Explainers initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing explainers: {e}")
            raise
    
    def _initialize_shap_explainer(self, X_background: np.ndarray, mode: str):
        """
        Initialize SHAP explainer based on model type
        """
        try:
            # Determine explainer type
            if mode == 'auto':
                # Try to detect model type
                model_type = str(type(self.model)).lower()
                if any(tree_type in model_type for tree_type in ['tree', 'forest', 'xgb', 'lgb', 'catboost']):
                    mode = 'tree'
                elif any(linear_type in model_type for linear_type in ['linear', 'logistic', 'svm']):
                    mode = 'linear'
                else:
                    mode = 'kernel'
            
            # Initialize appropriate explainer
            if mode == 'tree':
                self.shap_explainer = shap.TreeExplainer(self.model)
            elif mode == 'linear':
                self.shap_explainer = shap.LinearExplainer(self.model, X_background)
            elif mode == 'kernel':
                # Use a smaller background sample for kernel explainer (computationally expensive)
                background_sample = shap.sample(X_background, min(100, len(X_background)))
                self.shap_explainer = shap.KernelExplainer(self.model.predict_proba, background_sample)
            else:
                # Default to Explainer (works with most models)
                self.shap_explainer = shap.Explainer(self.model.predict_proba, X_background)
            
            logger.info(f"SHAP explainer initialized with mode: {mode}")
            
        except Exception as e:
            logger.warning(f"Could not initialize SHAP explainer: {e}")
            self.shap_explainer = None
    
    def _initialize_lime_explainer(self, X_background: np.ndarray):
        """
        Initialize LIME tabular explainer
        """
        try:
            self.lime_explainer = lime_tabular.LimeTabularExplainer(
                X_background,
                feature_names=self.feature_names,
                class_names=self.class_names,
                mode='classification',
                discretize_continuous=True,
                random_state=42
            )
            logger.info("LIME tabular explainer initialized")
            
        except Exception as e:
            logger.warning(f"Could not initialize LIME explainer: {e}")
            self.lime_explainer = None
    
    def _initialize_lime_text_explainer(self, text_data: List[str]):
        """
        Initialize LIME text explainer
        """
        try:
            self.lime_text_explainer = LimeTextExplainer(
                class_names=self.class_names,
                mode='classification',
                random_state=42
            )
            logger.info("LIME text explainer initialized")
            
        except Exception as e:
            logger.warning(f"Could not initialize LIME text explainer: {e}")
            self.lime_text_explainer = None
    
    def explain_prediction(self, 
                         X_instance: np.ndarray,
                         text_instance: str = None,
                         include_shap: bool = True,
                         include_lime: bool = True,
                         num_features: int = 10) -> Dict[str, Any]:
        """
        Generate comprehensive explanation for a single prediction
        
        Args:
            X_instance: Feature vector for the instance
            text_instance: Text content (if available)
            include_shap: Whether to include SHAP explanations
            include_lime: Whether to include LIME explanations
            num_features: Number of top features to explain
            
        Returns:
            Comprehensive explanation dictionary
        """
        start_time = time.time()
        
        try:
            # Ensure X_instance is 2D
            if X_instance.ndim == 1:
                X_instance = X_instance.reshape(1, -1)
            
            # Get model prediction and confidence
            prediction = self.model.predict(X_instance)[0]
            prediction_proba = self.model.predict_proba(X_instance)[0]
            confidence = float(np.max(prediction_proba))
            
            explanation = {
                'timestamp': datetime.now().isoformat(),
                'computation_time_ms': 0,  # Will be updated at the end
                
                # Basic prediction info
                'prediction': {
                    'class': self.class_names[prediction] if prediction < len(self.class_names) else str(prediction),
                    'class_index': int(prediction),
                    'confidence': confidence,
                    'probabilities': {
                        self.class_names[i] if i < len(self.class_names) else f'Class_{i}': float(prob)
                        for i, prob in enumerate(prediction_proba)
                    }
                },
                
                # Explanation components
                'shap_explanation': None,
                'lime_explanation': None,
                'lime_text_explanation': None,
                'feature_importance': None,
                'decision_reasoning': None
            }
            
            # Generate SHAP explanation
            if include_shap and self.shap_explainer is not None:
                explanation['shap_explanation'] = self._generate_shap_explanation(
                    X_instance, num_features
                )
            
            # Generate LIME explanation
            if include_lime and self.lime_explainer is not None:
                explanation['lime_explanation'] = self._generate_lime_explanation(
                    X_instance[0], num_features
                )
            
            # Generate LIME text explanation
            if text_instance and self.lime_text_explainer is not None:
                explanation['lime_text_explanation'] = self._generate_lime_text_explanation(
                    text_instance, num_features
                )
            
            # Generate feature importance summary
            explanation['feature_importance'] = self._generate_feature_importance_summary(
                explanation, num_features
            )
            
            # Generate decision reasoning
            explanation['decision_reasoning'] = self._generate_decision_reasoning(
                explanation, X_instance[0], text_instance
            )
            
            # Update computation time
            explanation['computation_time_ms'] = round((time.time() - start_time) * 1000, 2)
            
            return explanation
            
        except Exception as e:
            logger.error(f"Error generating explanation: {e}")
            raise
    
    def _generate_shap_explanation(self, X_instance: np.ndarray, num_features: int) -> Dict[str, Any]:
        """
        Generate SHAP explanation for the instance
        """
        try:
            # Calculate SHAP values
            shap_values = self.shap_explainer.shap_values(X_instance)
            
            # Handle different SHAP value formats
            if isinstance(shap_values, list):
                # Multi-class case - use values for predicted class
                prediction_idx = self.model.predict(X_instance)[0]
                shap_vals = shap_values[prediction_idx][0] if len(shap_values) > prediction_idx else shap_values[0][0]
            else:
                # Binary case or single output
                shap_vals = shap_values[0] if shap_values.ndim > 1 else shap_values
            
            # Get feature contributions
            feature_contributions = []
            for i, (feature_name, shap_val) in enumerate(zip(self.feature_names, shap_vals)):
                if i >= len(self.feature_names):
                    feature_name = f"Feature_{i}"
                
                feature_contributions.append({
                    'feature_name': feature_name,
                    'feature_index': i,
                    'shap_value': float(shap_val),
                    'feature_value': float(X_instance[0][i]) if i < len(X_instance[0]) else 0.0,
                    'abs_shap_value': float(abs(shap_val))
                })
            
            # Sort by absolute SHAP value
            feature_contributions.sort(key=lambda x: x['abs_shap_value'], reverse=True)
            
            # Get top features
            top_features = feature_contributions[:num_features]
            
            # Calculate base value (expected value)
            base_value = getattr(self.shap_explainer, 'expected_value', 0)
            if isinstance(base_value, (list, np.ndarray)):
                base_value = base_value[0] if len(base_value) > 0 else 0
            
            return {
                'base_value': float(base_value),
                'total_shap_value': float(np.sum(shap_vals)),
                'top_features': top_features,
                'all_features': feature_contributions,
                'explanation_type': 'SHAP (SHapley Additive exPlanations)',
                'interpretation': self._interpret_shap_values(top_features)
            }
            
        except Exception as e:
            logger.warning(f"Error generating SHAP explanation: {e}")
            return None
    
    def _generate_lime_explanation(self, X_instance: np.ndarray, num_features: int) -> Dict[str, Any]:
        """
        Generate LIME explanation for the instance
        """
        try:
            # Generate LIME explanation
            lime_exp = self.lime_explainer.explain_instance(
                X_instance,
                self.model.predict_proba,
                num_features=num_features,
                num_samples=1000
            )
            
            # Extract feature contributions
            feature_contributions = []
            for feature_idx, contribution in lime_exp.as_list():
                feature_name = self.feature_names[feature_idx] if feature_idx < len(self.feature_names) else f"Feature_{feature_idx}"
                
                feature_contributions.append({
                    'feature_name': feature_name,
                    'feature_index': feature_idx,
                    'lime_contribution': float(contribution),
                    'feature_value': float(X_instance[feature_idx]) if feature_idx < len(X_instance) else 0.0,
                    'abs_contribution': float(abs(contribution))
                })
            
            # Sort by absolute contribution
            feature_contributions.sort(key=lambda x: x['abs_contribution'], reverse=True)
            
            # Get local prediction from LIME
            local_pred = lime_exp.local_pred[0] if hasattr(lime_exp, 'local_pred') else None
            
            return {
                'local_prediction': float(local_pred) if local_pred is not None else None,
                'feature_contributions': feature_contributions,
                'explanation_type': 'LIME (Local Interpretable Model-agnostic Explanations)',
                'interpretation': self._interpret_lime_values(feature_contributions),
                'model_fidelity': getattr(lime_exp, 'score', None)
            }
            
        except Exception as e:
            logger.warning(f"Error generating LIME explanation: {e}")
            return None
    
    def _generate_lime_text_explanation(self, text_instance: str, num_features: int) -> Dict[str, Any]:
        """
        Generate LIME text explanation
        """
        try:
            # Create a wrapper function for text prediction
            def predict_text_proba(texts):
                # This would need to be adapted based on your text processing pipeline
                # For now, return dummy probabilities
                return np.array([[0.5, 0.5] for _ in texts])
            
            # Generate explanation
            lime_text_exp = self.lime_text_explainer.explain_instance(
                text_instance,
                predict_text_proba,
                num_features=num_features,
                num_samples=500
            )
            
            # Extract word contributions
            word_contributions = []
            for word, contribution in lime_text_exp.as_list():
                word_contributions.append({
                    'word': word,
                    'contribution': float(contribution),
                    'abs_contribution': float(abs(contribution))
                })
            
            # Sort by absolute contribution
            word_contributions.sort(key=lambda x: x['abs_contribution'], reverse=True)
            
            return {
                'text_length': len(text_instance.split()),
                'word_contributions': word_contributions,
                'explanation_type': 'LIME Text Explanation',
                'interpretation': self._interpret_text_contributions(word_contributions)
            }
            
        except Exception as e:
            logger.warning(f"Error generating LIME text explanation: {e}")
            return None
    
    def _generate_feature_importance_summary(self, explanation: Dict, num_features: int) -> Dict[str, Any]:
        """
        Generate consolidated feature importance summary
        """
        try:
            feature_summary = {}
            
            # Combine SHAP and LIME results
            if explanation.get('shap_explanation'):
                for feature in explanation['shap_explanation']['top_features']:
                    name = feature['feature_name']
                    feature_summary[name] = {
                        'feature_name': name,
                        'shap_importance': feature['abs_shap_value'],
                        'shap_direction': 'positive' if feature['shap_value'] > 0 else 'negative',
                        'feature_value': feature['feature_value']
                    }
            
            if explanation.get('lime_explanation'):
                for feature in explanation['lime_explanation']['feature_contributions']:
                    name = feature['feature_name']
                    if name in feature_summary:
                        feature_summary[name]['lime_importance'] = feature['abs_contribution']
                        feature_summary[name]['lime_direction'] = 'positive' if feature['lime_contribution'] > 0 else 'negative'
                    else:
                        feature_summary[name] = {
                            'feature_name': name,
                            'lime_importance': feature['abs_contribution'],
                            'lime_direction': 'positive' if feature['lime_contribution'] > 0 else 'negative',
                            'feature_value': feature['feature_value']
                        }
            
            # Calculate consensus importance
            for name, summary in feature_summary.items():
                shap_imp = summary.get('shap_importance', 0)
                lime_imp = summary.get('lime_importance', 0)
                
                # Average importance (could use other aggregation methods)
                summary['consensus_importance'] = (shap_imp + lime_imp) / 2 if shap_imp and lime_imp else max(shap_imp, lime_imp)
                
                # Direction consensus
                shap_dir = summary.get('shap_direction')
                lime_dir = summary.get('lime_direction')
                if shap_dir and lime_dir:
                    summary['consensus_direction'] = shap_dir if shap_dir == lime_dir else 'mixed'
                else:
                    summary['consensus_direction'] = shap_dir or lime_dir or 'unknown'
            
            # Sort by consensus importance
            sorted_features = sorted(feature_summary.values(), 
                                   key=lambda x: x['consensus_importance'], 
                                   reverse=True)[:num_features]
            
            return {
                'top_features': sorted_features,
                'total_features_analyzed': len(feature_summary),
                'explanation_methods_used': [
                    method for method in ['SHAP', 'LIME'] 
                    if explanation.get(f'{method.lower()}_explanation') is not None
                ]
            }
            
        except Exception as e:
            logger.warning(f"Error generating feature importance summary: {e}")
            return None
    
    def _generate_decision_reasoning(self, explanation: Dict, X_instance: np.ndarray, text_instance: str = None) -> Dict[str, Any]:
        """
        Generate human-readable decision reasoning
        """
        try:
            prediction = explanation['prediction']
            confidence = prediction['confidence']
            predicted_class = prediction['class']
            
            # Generate reasoning based on top features
            reasoning_parts = []
            
            # Confidence assessment
            if confidence > 0.8:
                confidence_level = "high"
            elif confidence > 0.6:
                confidence_level = "moderate"
            else:
                confidence_level = "low"
            
            reasoning_parts.append(
                f"The model predicts this content is '{predicted_class}' with {confidence_level} confidence ({confidence:.2%})."
            )
            
            # Feature-based reasoning
            if explanation.get('feature_importance') and explanation['feature_importance'].get('top_features'):
                top_features = explanation['feature_importance']['top_features'][:3]  # Top 3 features
                
                reasoning_parts.append("Key factors influencing this decision:")
                
                for i, feature in enumerate(top_features, 1):
                    direction = feature['consensus_direction']
                    importance = feature['consensus_importance']
                    name = feature['feature_name']
                    value = feature.get('feature_value', 'N/A')
                    
                    if direction == 'positive':
                        effect = "supports"
                    elif direction == 'negative':
                        effect = "contradicts"
                    else:
                        effect = "influences"
                    
                    reasoning_parts.append(
                        f"{i}. {name} (value: {value:.3f if isinstance(value, (int, float)) else value}) "
                        f"{effect} the '{predicted_class}' classification (importance: {importance:.3f})"
                    )
            
            # Text-based reasoning (if available)
            if explanation.get('lime_text_explanation'):
                text_contrib = explanation['lime_text_explanation']['word_contributions'][:3]
                if text_contrib:
                    reasoning_parts.append("Most influential words in the text:")
                    for word_info in text_contrib:
                        word = word_info['word']
                        contrib = word_info['contribution']
                        direction = "supports" if contrib > 0 else "contradicts"
                        reasoning_parts.append(f"- '{word}' {direction} the prediction")
            
            # Risk assessment
            risk_factors = []
            if confidence < 0.7:
                risk_factors.append("Low prediction confidence")
            
            # Check for conflicting signals
            if explanation.get('feature_importance'):
                top_features = explanation['feature_importance']['top_features'][:5]
                positive_count = sum(1 for f in top_features if f['consensus_direction'] == 'positive')
                negative_count = sum(1 for f in top_features if f['consensus_direction'] == 'negative')
                
                if abs(positive_count - negative_count) <= 1:
                    risk_factors.append("Mixed signals from different features")
            
            if risk_factors:
                reasoning_parts.append(f"Potential concerns: {', '.join(risk_factors)}")
            
            return {
                'summary': reasoning_parts[0],
                'detailed_reasoning': reasoning_parts,
                'confidence_level': confidence_level,
                'risk_factors': risk_factors,
                'recommendation': self._generate_recommendation(predicted_class, confidence, risk_factors)
            }
            
        except Exception as e:
            logger.warning(f"Error generating decision reasoning: {e}")
            return None
    
    def _generate_recommendation(self, predicted_class: str, confidence: float, risk_factors: List[str]) -> str:
        """
        Generate recommendation based on prediction and risk factors
        """
        if predicted_class.lower() == 'fake':
            if confidence > 0.8 and not risk_factors:
                return "High confidence fake news detection. Consider flagging or fact-checking."
            elif confidence > 0.6:
                return "Moderate confidence fake news detection. Recommend additional verification."
            else:
                return "Low confidence fake news detection. Manual review strongly recommended."
        else:  # Real news
            if confidence > 0.8 and not risk_factors:
                return "High confidence legitimate news detection. Content appears trustworthy."
            elif confidence > 0.6:
                return "Moderate confidence legitimate news detection. Generally appears reliable."
            else:
                return "Low confidence legitimate news detection. Consider additional verification."
    
    def _interpret_shap_values(self, top_features: List[Dict]) -> str:
        """
        Generate interpretation of SHAP values
        """
        if not top_features:
            return "No significant feature contributions found."
        
        positive_features = [f for f in top_features if f['shap_value'] > 0]
        negative_features = [f for f in top_features if f['shap_value'] < 0]
        
        interpretation = []
        
        if positive_features:
            interpretation.append(
                f"Features pushing towards positive prediction: {', '.join([f['feature_name'] for f in positive_features[:3]])}"
            )
        
        if negative_features:
            interpretation.append(
                f"Features pushing towards negative prediction: {', '.join([f['feature_name'] for f in negative_features[:3]])}"
            )
        
        return " | ".join(interpretation)
    
    def _interpret_lime_values(self, feature_contributions: List[Dict]) -> str:
        """
        Generate interpretation of LIME values
        """
        if not feature_contributions:
            return "No significant local feature contributions found."
        
        top_positive = [f for f in feature_contributions if f['lime_contribution'] > 0][:2]
        top_negative = [f for f in feature_contributions if f['lime_contribution'] < 0][:2]
        
        interpretation = []
        
        if top_positive:
            interpretation.append(
                f"Locally important positive features: {', '.join([f['feature_name'] for f in top_positive])}"
            )
        
        if top_negative:
            interpretation.append(
                f"Locally important negative features: {', '.join([f['feature_name'] for f in top_negative])}"
            )
        
        return " | ".join(interpretation)
    
    def _interpret_text_contributions(self, word_contributions: List[Dict]) -> str:
        """
        Generate interpretation of text contributions
        """
        if not word_contributions:
            return "No significant word contributions found."
        
        positive_words = [w['word'] for w in word_contributions if w['contribution'] > 0][:3]
        negative_words = [w['word'] for w in word_contributions if w['contribution'] < 0][:3]
        
        interpretation = []
        
        if positive_words:
            interpretation.append(f"Words supporting prediction: {', '.join(positive_words)}")
        
        if negative_words:
            interpretation.append(f"Words contradicting prediction: {', '.join(negative_words)}")
        
        return " | ".join(interpretation)
    
    def batch_explain(self, 
                     X_batch: np.ndarray,
                     text_batch: List[str] = None,
                     num_features: int = 10) -> List[Dict[str, Any]]:
        """
        Generate explanations for a batch of instances
        
        Args:
            X_batch: Batch of feature vectors
            text_batch: Batch of text instances (optional)
            num_features: Number of top features to explain
            
        Returns:
            List of explanation dictionaries
        """
        explanations = []
        
        for i, X_instance in enumerate(X_batch):
            text_instance = text_batch[i] if text_batch and i < len(text_batch) else None
            
            try:
                explanation = self.explain_prediction(
                    X_instance, text_instance, num_features=num_features
                )
                explanation['batch_index'] = i
                explanations.append(explanation)
                
            except Exception as e:
                logger.warning(f"Error explaining instance {i}: {e}")
                explanations.append({
                    'batch_index': i,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
        
        return explanations
    
    def get_global_feature_importance(self, 
                                    X_sample: np.ndarray,
                                    sample_size: int = 100) -> Dict[str, Any]:
        """
        Calculate global feature importance using SHAP
        
        Args:
            X_sample: Sample of data for global analysis
            sample_size: Number of samples to use
            
        Returns:
            Global feature importance analysis
        """
        if not SHAP_AVAILABLE or self.shap_explainer is None:
            logger.warning("SHAP not available for global feature importance")
            return None
        
        try:
            # Sample data if needed
            if len(X_sample) > sample_size:
                indices = np.random.choice(len(X_sample), sample_size, replace=False)
                X_sample = X_sample[indices]
            
            # Calculate SHAP values
            shap_values = self.shap_explainer.shap_values(X_sample)
            
            # Handle different formats
            if isinstance(shap_values, list):
                # Use first class for global importance
                shap_vals = shap_values[0]
            else:
                shap_vals = shap_values
            
            # Calculate mean absolute SHAP values
            mean_abs_shap = np.mean(np.abs(shap_vals), axis=0)
            
            # Create feature importance ranking
            feature_importance = []
            for i, importance in enumerate(mean_abs_shap):
                feature_name = self.feature_names[i] if i < len(self.feature_names) else f"Feature_{i}"
                feature_importance.append({
                    'feature_name': feature_name,
                    'feature_index': i,
                    'mean_abs_shap': float(importance),
                    'mean_shap': float(np.mean(shap_vals[:, i]))
                })
            
            # Sort by importance
            feature_importance.sort(key=lambda x: x['mean_abs_shap'], reverse=True)
            
            return {
                'timestamp': datetime.now().isoformat(),
                'sample_size': len(X_sample),
                'feature_importance_ranking': feature_importance,
                'top_10_features': feature_importance[:10],
                'total_features': len(feature_importance)
            }
            
        except Exception as e:
            logger.error(f"Error calculating global feature importance: {e}")
            return None


def create_explainability_system(model: Any = None,
                               feature_names: List[str] = None,
                               class_names: List[str] = None,
                               text_features: List[str] = None) -> EnhancedExplainabilitySystem:
    """
    Factory function to create explainability system
    """
    return EnhancedExplainabilitySystem(
        model=model,
        feature_names=feature_names,
        class_names=class_names,
        text_features=text_features
    )


if __name__ == "__main__":
    # Example usage
    logger.info("Enhanced Explainability System initialized")
    logger.info("Available classes: EnhancedExplainabilitySystem")
    logger.info("Available functions: create_explainability_system")
    logger.info(f"SHAP available: {SHAP_AVAILABLE}")
    logger.info(f"LIME available: {LIME_AVAILABLE}")