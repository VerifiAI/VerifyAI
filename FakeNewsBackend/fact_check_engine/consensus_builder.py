import asyncio
import logging
from typing import List, Dict, Optional, Any, Tuple, Set
from dataclasses import dataclass
from datetime import datetime
import statistics
import math
from collections import defaultdict, Counter
from enum import Enum

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    logging.warning("NumPy not available. Some advanced consensus features will be limited.")

from ..proof_validation.proof_validator import ValidationResult
from ..proof_validation.scoring import ConsensusResult, SourceScore
from .verdict_engine import VerdictType, FinalVerdict

logger = logging.getLogger(__name__)


class ConsensusMethod(Enum):
    """Enumeration of consensus building methods."""
    MAJORITY_VOTE = "majority_vote"
    WEIGHTED_AVERAGE = "weighted_average"
    BAYESIAN_INFERENCE = "bayesian_inference"
    DELPHI_METHOD = "delphi_method"
    EVIDENCE_ACCUMULATION = "evidence_accumulation"


@dataclass
class ConsensusInput:
    """Input for consensus building."""
    validation_results: List[ValidationResult]
    source_scores: List[SourceScore]
    evidence_weights: Dict[str, float]
    method_preferences: List[ConsensusMethod]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "validation_count": len(self.validation_results),
            "source_count": len(self.source_scores),
            "evidence_weights": self.evidence_weights,
            "methods": [method.value for method in self.method_preferences]
        }


@dataclass
class ConsensusMetrics:
    """Metrics for consensus quality assessment."""
    agreement_score: float
    confidence_interval: Tuple[float, float]
    uncertainty_measure: float
    stability_score: float
    robustness_score: float
    method_convergence: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "agreement_score": self.agreement_score,
            "confidence_interval": self.confidence_interval,
            "uncertainty_measure": self.uncertainty_measure,
            "stability_score": self.stability_score,
            "robustness_score": self.robustness_score,
            "method_convergence": self.method_convergence
        }


@dataclass
class EnhancedConsensusResult:
    """Enhanced consensus result with detailed analysis."""
    claim: str
    consensus_verdict: VerdictType
    consensus_confidence: float
    consensus_metrics: ConsensusMetrics
    method_results: Dict[str, Dict[str, Any]]
    disagreement_analysis: Dict[str, Any]
    sensitivity_analysis: Dict[str, Any]
    recommendation_strength: str
    calculated_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "claim": self.claim,
            "consensus_verdict": self.consensus_verdict.value,
            "consensus_confidence": self.consensus_confidence,
            "consensus_metrics": self.consensus_metrics.to_dict(),
            "method_results": self.method_results,
            "disagreement_analysis": self.disagreement_analysis,
            "sensitivity_analysis": self.sensitivity_analysis,
            "recommendation_strength": self.recommendation_strength,
            "calculated_at": self.calculated_at.isoformat()
        }


class ConsensusBuilderError(Exception):
    """Custom exception for consensus builder errors."""
    pass


class AdvancedConsensusBuilder:
    """Advanced consensus building system for fact-checking decisions."""
    
    def __init__(self):
        """Initialize advanced consensus builder."""
        # Consensus parameters
        self.consensus_params = {
            "min_agreement_threshold": 0.6,
            "high_agreement_threshold": 0.8,
            "uncertainty_threshold": 0.3,
            "stability_threshold": 0.7,
            "robustness_threshold": 0.6,
            "confidence_level": 0.95
        }
        
        # Method weights (can be adjusted based on performance)
        self.method_weights = {
            ConsensusMethod.MAJORITY_VOTE: 0.2,
            ConsensusMethod.WEIGHTED_AVERAGE: 0.3,
            ConsensusMethod.BAYESIAN_INFERENCE: 0.25,
            ConsensusMethod.DELPHI_METHOD: 0.15,
            ConsensusMethod.EVIDENCE_ACCUMULATION: 0.1
        }
        
        # Prior probabilities for Bayesian inference
        self.priors = {
            VerdictType.TRUE: 0.3,
            VerdictType.FALSE: 0.3,
            VerdictType.MISLEADING: 0.15,
            VerdictType.MIXED: 0.15,
            VerdictType.UNPROVEN: 0.08,
            VerdictType.INSUFFICIENT_EVIDENCE: 0.02
        }
        
        logger.info("AdvancedConsensusBuilder initialized")
    
    async def build_enhanced_consensus(
        self,
        consensus_input: ConsensusInput,
        claim: str
    ) -> EnhancedConsensusResult:
        """
        Build enhanced consensus using multiple methods.
        
        Args:
            consensus_input: Input data for consensus building
            claim: The claim being fact-checked
            
        Returns:
            EnhancedConsensusResult with comprehensive analysis
        """
        start_time = datetime.now()
        
        try:
            # Apply multiple consensus methods
            method_results = {}
            
            for method in consensus_input.method_preferences:
                try:
                    result = await self._apply_consensus_method(method, consensus_input)
                    method_results[method.value] = result
                except Exception as e:
                    logger.warning(f"Consensus method {method.value} failed: {e}")
                    continue
            
            if not method_results:
                raise ConsensusBuilderError("All consensus methods failed")
            
            # Combine method results
            combined_verdict, combined_confidence = self._combine_method_results(
                method_results, consensus_input
            )
            
            # Calculate consensus metrics
            consensus_metrics = self._calculate_consensus_metrics(
                method_results, consensus_input
            )
            
            # Analyze disagreements
            disagreement_analysis = self._analyze_disagreements(
                method_results, consensus_input
            )
            
            # Perform sensitivity analysis
            sensitivity_analysis = await self._perform_sensitivity_analysis(
                consensus_input, method_results
            )
            
            # Determine recommendation strength
            recommendation_strength = self._determine_recommendation_strength(
                combined_confidence, consensus_metrics
            )
            
            result = EnhancedConsensusResult(
                claim=claim,
                consensus_verdict=combined_verdict,
                consensus_confidence=combined_confidence,
                consensus_metrics=consensus_metrics,
                method_results=method_results,
                disagreement_analysis=disagreement_analysis,
                sensitivity_analysis=sensitivity_analysis,
                recommendation_strength=recommendation_strength,
                calculated_at=datetime.now()
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Enhanced consensus built in {processing_time:.2f}s: {combined_verdict.value} ({combined_confidence:.2f})")
            
            return result
            
        except Exception as e:
            logger.error(f"Enhanced consensus building failed: {e}")
            raise ConsensusBuilderError(f"Failed to build consensus: {e}")
    
    async def _apply_consensus_method(
        self,
        method: ConsensusMethod,
        consensus_input: ConsensusInput
    ) -> Dict[str, Any]:
        """Apply a specific consensus method."""
        if method == ConsensusMethod.MAJORITY_VOTE:
            return self._majority_vote_consensus(consensus_input)
        elif method == ConsensusMethod.WEIGHTED_AVERAGE:
            return self._weighted_average_consensus(consensus_input)
        elif method == ConsensusMethod.BAYESIAN_INFERENCE:
            return self._bayesian_inference_consensus(consensus_input)
        elif method == ConsensusMethod.DELPHI_METHOD:
            return self._delphi_method_consensus(consensus_input)
        elif method == ConsensusMethod.EVIDENCE_ACCUMULATION:
            return self._evidence_accumulation_consensus(consensus_input)
        else:
            raise ValueError(f"Unknown consensus method: {method}")
    
    def _majority_vote_consensus(self, consensus_input: ConsensusInput) -> Dict[str, Any]:
        """Simple majority vote consensus."""
        if not consensus_input.validation_results:
            return {"verdict": VerdictType.INSUFFICIENT_EVIDENCE, "confidence": 0.0}
        
        # Count verdicts
        verdict_counts = Counter()
        for result in consensus_input.validation_results:
            # Map validation verdicts to final verdicts
            if result.verdict == "REAL":
                verdict_counts[VerdictType.TRUE] += 1
            elif result.verdict == "FAKE":
                verdict_counts[VerdictType.FALSE] += 1
            else:
                verdict_counts[VerdictType.UNPROVEN] += 1
        
        # Find majority
        total_votes = len(consensus_input.validation_results)
        majority_verdict = verdict_counts.most_common(1)[0][0]
        majority_count = verdict_counts[majority_verdict]
        
        confidence = majority_count / total_votes
        
        return {
            "verdict": majority_verdict,
            "confidence": confidence,
            "vote_distribution": dict(verdict_counts),
            "total_votes": total_votes
        }
    
    def _weighted_average_consensus(self, consensus_input: ConsensusInput) -> Dict[str, Any]:
        """Weighted average consensus based on source quality."""
        if not consensus_input.validation_results or not consensus_input.source_scores:
            return {"verdict": VerdictType.INSUFFICIENT_EVIDENCE, "confidence": 0.0}
        
        # Calculate weighted votes
        weighted_votes = defaultdict(float)
        total_weight = 0.0
        
        for i, result in enumerate(consensus_input.validation_results):
            # Get corresponding source weight
            weight = 1.0  # Default weight
            if i < len(consensus_input.source_scores):
                weight = consensus_input.source_scores[i].weight
            
            # Map verdict and add weighted vote
            if result.verdict == "REAL":
                weighted_votes[VerdictType.TRUE] += weight * result.confidence
            elif result.verdict == "FAKE":
                weighted_votes[VerdictType.FALSE] += weight * result.confidence
            else:
                weighted_votes[VerdictType.UNPROVEN] += weight * result.confidence
            
            total_weight += weight
        
        # Normalize weights
        if total_weight > 0:
            for verdict in weighted_votes:
                weighted_votes[verdict] /= total_weight
        
        # Find highest weighted verdict
        if weighted_votes:
            consensus_verdict = max(weighted_votes, key=weighted_votes.get)
            consensus_confidence = weighted_votes[consensus_verdict]
        else:
            consensus_verdict = VerdictType.INSUFFICIENT_EVIDENCE
            consensus_confidence = 0.0
        
        return {
            "verdict": consensus_verdict,
            "confidence": consensus_confidence,
            "weighted_distribution": dict(weighted_votes),
            "total_weight": total_weight
        }
    
    def _bayesian_inference_consensus(self, consensus_input: ConsensusInput) -> Dict[str, Any]:
        """Bayesian inference consensus using prior probabilities."""
        if not consensus_input.validation_results:
            return {"verdict": VerdictType.INSUFFICIENT_EVIDENCE, "confidence": 0.0}
        
        # Start with prior probabilities
        posterior_probs = self.priors.copy()
        
        # Update with evidence
        for result in consensus_input.validation_results:
            # Calculate likelihood of observing this result given each verdict
            likelihoods = self._calculate_likelihoods(result)
            
            # Update posterior probabilities
            for verdict in posterior_probs:
                posterior_probs[verdict] *= likelihoods.get(verdict, 0.5)
        
        # Normalize probabilities
        total_prob = sum(posterior_probs.values())
        if total_prob > 0:
            for verdict in posterior_probs:
                posterior_probs[verdict] /= total_prob
        
        # Find most probable verdict
        consensus_verdict = max(posterior_probs, key=posterior_probs.get)
        consensus_confidence = posterior_probs[consensus_verdict]
        
        return {
            "verdict": consensus_verdict,
            "confidence": consensus_confidence,
            "posterior_probabilities": {v.value: p for v, p in posterior_probs.items()},
            "prior_probabilities": {v.value: p for v, p in self.priors.items()}
        }
    
    def _calculate_likelihoods(self, result: ValidationResult) -> Dict[VerdictType, float]:
        """Calculate likelihood of observing result given each possible verdict."""
        likelihoods = {}
        
        # Base likelihood on result confidence and verdict
        base_confidence = result.confidence
        
        if result.verdict == "REAL":
            likelihoods[VerdictType.TRUE] = base_confidence
            likelihoods[VerdictType.FALSE] = 1.0 - base_confidence
            likelihoods[VerdictType.MISLEADING] = 0.3
            likelihoods[VerdictType.MIXED] = 0.4
            likelihoods[VerdictType.UNPROVEN] = 0.2
            likelihoods[VerdictType.INSUFFICIENT_EVIDENCE] = 0.1
        elif result.verdict == "FAKE":
            likelihoods[VerdictType.TRUE] = 1.0 - base_confidence
            likelihoods[VerdictType.FALSE] = base_confidence
            likelihoods[VerdictType.MISLEADING] = 0.6
            likelihoods[VerdictType.MIXED] = 0.4
            likelihoods[VerdictType.UNPROVEN] = 0.3
            likelihoods[VerdictType.INSUFFICIENT_EVIDENCE] = 0.1
        else:  # AMBIGUOUS
            likelihoods[VerdictType.TRUE] = 0.3
            likelihoods[VerdictType.FALSE] = 0.3
            likelihoods[VerdictType.MISLEADING] = 0.4
            likelihoods[VerdictType.MIXED] = 0.6
            likelihoods[VerdictType.UNPROVEN] = 0.7
            likelihoods[VerdictType.INSUFFICIENT_EVIDENCE] = 0.5
        
        return likelihoods
    
    def _delphi_method_consensus(self, consensus_input: ConsensusInput) -> Dict[str, Any]:
        """Delphi method consensus with iterative refinement."""
        if not consensus_input.validation_results:
            return {"verdict": VerdictType.INSUFFICIENT_EVIDENCE, "confidence": 0.0}
        
        # Simulate Delphi rounds (simplified version)
        current_estimates = []
        
        # Round 1: Initial estimates
        for result in consensus_input.validation_results:
            if result.verdict == "REAL":
                current_estimates.append(result.confidence)
            elif result.verdict == "FAKE":
                current_estimates.append(-result.confidence)
            else:
                current_estimates.append(0.0)
        
        # Round 2: Adjust based on group feedback
        if current_estimates:
            group_mean = statistics.mean(current_estimates)
            group_std = statistics.stdev(current_estimates) if len(current_estimates) > 1 else 0
            
            # Adjust estimates toward group consensus
            adjusted_estimates = []
            for estimate in current_estimates:
                # Move estimate 30% toward group mean
                adjusted = estimate * 0.7 + group_mean * 0.3
                adjusted_estimates.append(adjusted)
            
            final_consensus = statistics.mean(adjusted_estimates)
            consensus_std = statistics.stdev(adjusted_estimates) if len(adjusted_estimates) > 1 else 0
        else:
            final_consensus = 0.0
            consensus_std = 0.0
        
        # Convert to verdict and confidence
        if final_consensus > 0.3:
            verdict = VerdictType.TRUE
            confidence = min(1.0, abs(final_consensus))
        elif final_consensus < -0.3:
            verdict = VerdictType.FALSE
            confidence = min(1.0, abs(final_consensus))
        else:
            verdict = VerdictType.UNPROVEN
            confidence = 1.0 - abs(final_consensus)
        
        return {
            "verdict": verdict,
            "confidence": confidence,
            "final_consensus": final_consensus,
            "consensus_std": consensus_std,
            "rounds": 2
        }
    
    def _evidence_accumulation_consensus(self, consensus_input: ConsensusInput) -> Dict[str, Any]:
        """Evidence accumulation consensus based on evidence strength."""
        if not consensus_input.validation_results:
            return {"verdict": VerdictType.INSUFFICIENT_EVIDENCE, "confidence": 0.0}
        
        # Accumulate evidence scores
        supporting_evidence = 0.0
        refuting_evidence = 0.0
        neutral_evidence = 0.0
        
        for result in consensus_input.validation_results:
            evidence_strength = result.evidence_score * result.credibility_score
            
            if result.verdict == "REAL":
                supporting_evidence += evidence_strength * result.confidence
            elif result.verdict == "FAKE":
                refuting_evidence += evidence_strength * result.confidence
            else:
                neutral_evidence += evidence_strength * result.confidence
        
        total_evidence = supporting_evidence + refuting_evidence + neutral_evidence
        
        if total_evidence == 0:
            return {"verdict": VerdictType.INSUFFICIENT_EVIDENCE, "confidence": 0.0}
        
        # Determine verdict based on evidence accumulation
        if supporting_evidence > refuting_evidence * 1.5:
            verdict = VerdictType.TRUE
            confidence = supporting_evidence / total_evidence
        elif refuting_evidence > supporting_evidence * 1.5:
            verdict = VerdictType.FALSE
            confidence = refuting_evidence / total_evidence
        elif abs(supporting_evidence - refuting_evidence) / total_evidence < 0.2:
            verdict = VerdictType.MIXED
            confidence = max(supporting_evidence, refuting_evidence) / total_evidence
        else:
            verdict = VerdictType.UNPROVEN
            confidence = neutral_evidence / total_evidence
        
        return {
            "verdict": verdict,
            "confidence": confidence,
            "evidence_accumulation": {
                "supporting": supporting_evidence,
                "refuting": refuting_evidence,
                "neutral": neutral_evidence,
                "total": total_evidence
            }
        }
    
    def _combine_method_results(
        self,
        method_results: Dict[str, Dict[str, Any]],
        consensus_input: ConsensusInput
    ) -> Tuple[VerdictType, float]:
        """Combine results from multiple consensus methods."""
        if not method_results:
            return VerdictType.INSUFFICIENT_EVIDENCE, 0.0
        
        # Weight method results
        weighted_verdicts = defaultdict(float)
        total_weight = 0.0
        
        for method_name, result in method_results.items():
            method_enum = ConsensusMethod(method_name)
            weight = self.method_weights.get(method_enum, 0.1)
            
            verdict = result["verdict"]
            confidence = result["confidence"]
            
            weighted_verdicts[verdict] += weight * confidence
            total_weight += weight
        
        # Normalize weights
        if total_weight > 0:
            for verdict in weighted_verdicts:
                weighted_verdicts[verdict] /= total_weight
        
        # Find consensus verdict
        if weighted_verdicts:
            consensus_verdict = max(weighted_verdicts, key=weighted_verdicts.get)
            consensus_confidence = weighted_verdicts[consensus_verdict]
        else:
            consensus_verdict = VerdictType.INSUFFICIENT_EVIDENCE
            consensus_confidence = 0.0
        
        return consensus_verdict, consensus_confidence
    
    def _calculate_consensus_metrics(
        self,
        method_results: Dict[str, Dict[str, Any]],
        consensus_input: ConsensusInput
    ) -> ConsensusMetrics:
        """Calculate comprehensive consensus metrics."""
        if not method_results:
            return ConsensusMetrics(0.0, (0.0, 0.0), 1.0, 0.0, 0.0, {})
        
        # Calculate agreement score
        verdicts = [result["verdict"] for result in method_results.values()]
        verdict_counts = Counter(verdicts)
        most_common_count = verdict_counts.most_common(1)[0][1]
        agreement_score = most_common_count / len(verdicts)
        
        # Calculate confidence interval (simplified)
        confidences = [result["confidence"] for result in method_results.values()]
        if confidences:
            mean_confidence = statistics.mean(confidences)
            std_confidence = statistics.stdev(confidences) if len(confidences) > 1 else 0
            margin = 1.96 * std_confidence / math.sqrt(len(confidences))  # 95% CI
            confidence_interval = (
                max(0.0, mean_confidence - margin),
                min(1.0, mean_confidence + margin)
            )
        else:
            confidence_interval = (0.0, 0.0)
        
        # Calculate uncertainty measure
        if confidences:
            uncertainty_measure = statistics.stdev(confidences) if len(confidences) > 1 else 0
        else:
            uncertainty_measure = 1.0
        
        # Calculate stability score (how consistent are the methods)
        stability_score = 1.0 - uncertainty_measure
        
        # Calculate robustness score (based on agreement and confidence)
        robustness_score = agreement_score * statistics.mean(confidences) if confidences else 0.0
        
        # Method convergence
        method_convergence = {}
        for method_name, result in method_results.items():
            method_convergence[method_name] = result["confidence"]
        
        return ConsensusMetrics(
            agreement_score=agreement_score,
            confidence_interval=confidence_interval,
            uncertainty_measure=uncertainty_measure,
            stability_score=stability_score,
            robustness_score=robustness_score,
            method_convergence=method_convergence
        )
    
    def _analyze_disagreements(
        self,
        method_results: Dict[str, Dict[str, Any]],
        consensus_input: ConsensusInput
    ) -> Dict[str, Any]:
        """Analyze disagreements between methods."""
        if len(method_results) < 2:
            return {"disagreement_level": "none", "conflicting_methods": []}
        
        verdicts = [result["verdict"] for result in method_results.values()]
        unique_verdicts = set(verdicts)
        
        disagreement_analysis = {
            "disagreement_level": "none",
            "conflicting_methods": [],
            "verdict_distribution": dict(Counter(verdicts)),
            "confidence_variance": 0.0
        }
        
        if len(unique_verdicts) > 1:
            # Calculate disagreement level
            verdict_counts = Counter(verdicts)
            max_count = max(verdict_counts.values())
            disagreement_ratio = 1.0 - (max_count / len(verdicts))
            
            if disagreement_ratio > 0.5:
                disagreement_analysis["disagreement_level"] = "high"
            elif disagreement_ratio > 0.3:
                disagreement_analysis["disagreement_level"] = "moderate"
            else:
                disagreement_analysis["disagreement_level"] = "low"
            
            # Identify conflicting methods
            majority_verdict = verdict_counts.most_common(1)[0][0]
            conflicting_methods = [
                method for method, result in method_results.items()
                if result["verdict"] != majority_verdict
            ]
            disagreement_analysis["conflicting_methods"] = conflicting_methods
        
        # Calculate confidence variance
        confidences = [result["confidence"] for result in method_results.values()]
        if len(confidences) > 1:
            disagreement_analysis["confidence_variance"] = statistics.variance(confidences)
        
        return disagreement_analysis
    
    async def _perform_sensitivity_analysis(
        self,
        consensus_input: ConsensusInput,
        method_results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Perform sensitivity analysis on consensus results."""
        sensitivity_analysis = {
            "parameter_sensitivity": {},
            "method_sensitivity": {},
            "robustness_test": {}
        }
        
        # Test sensitivity to method weights
        original_weights = self.method_weights.copy()
        
        # Test with equal weights
        equal_weight = 1.0 / len(self.method_weights)
        self.method_weights = {method: equal_weight for method in self.method_weights}
        
        equal_weight_verdict, equal_weight_confidence = self._combine_method_results(
            method_results, consensus_input
        )
        
        # Restore original weights
        self.method_weights = original_weights
        
        # Compare results
        original_verdict, original_confidence = self._combine_method_results(
            method_results, consensus_input
        )
        
        sensitivity_analysis["method_sensitivity"] = {
            "original_result": {"verdict": original_verdict.value, "confidence": original_confidence},
            "equal_weights_result": {"verdict": equal_weight_verdict.value, "confidence": equal_weight_confidence},
            "verdict_changed": original_verdict != equal_weight_verdict,
            "confidence_change": abs(original_confidence - equal_weight_confidence)
        }
        
        # Test robustness by removing one method at a time
        robustness_results = {}
        for method_to_remove in method_results:
            reduced_results = {k: v for k, v in method_results.items() if k != method_to_remove}
            if reduced_results:
                reduced_verdict, reduced_confidence = self._combine_method_results(
                    reduced_results, consensus_input
                )
                robustness_results[method_to_remove] = {
                    "verdict": reduced_verdict.value,
                    "confidence": reduced_confidence,
                    "verdict_changed": original_verdict != reduced_verdict
                }
        
        sensitivity_analysis["robustness_test"] = robustness_results
        
        return sensitivity_analysis
    
    def _determine_recommendation_strength(
        self,
        confidence: float,
        metrics: ConsensusMetrics
    ) -> str:
        """Determine the strength of the recommendation."""
        if (confidence > 0.8 and 
            metrics.agreement_score > self.consensus_params["high_agreement_threshold"] and
            metrics.robustness_score > self.consensus_params["robustness_threshold"]):
            return "STRONG"
        elif (confidence > 0.6 and 
              metrics.agreement_score > self.consensus_params["min_agreement_threshold"]):
            return "MODERATE"
        elif confidence > 0.4:
            return "WEAK"
        else:
            return "INSUFFICIENT"
    
    def get_consensus_summary(self, result: EnhancedConsensusResult) -> Dict[str, Any]:
        """Generate a summary of the consensus analysis."""
        return {
            "consensus_verdict": result.consensus_verdict.value,
            "confidence": result.consensus_confidence,
            "recommendation_strength": result.recommendation_strength,
            "agreement_level": result.consensus_metrics.agreement_score,
            "uncertainty": result.consensus_metrics.uncertainty_measure,
            "methods_used": list(result.method_results.keys()),
            "disagreement_level": result.disagreement_analysis.get("disagreement_level", "unknown"),
            "robustness": result.consensus_metrics.robustness_score,
            "key_insights": self._extract_key_insights(result)
        }
    
    def _extract_key_insights(self, result: EnhancedConsensusResult) -> List[str]:
        """Extract key insights from consensus analysis."""
        insights = []
        
        if result.consensus_metrics.agreement_score > 0.8:
            insights.append("High agreement across all methods")
        elif result.consensus_metrics.agreement_score < 0.5:
            insights.append("Significant disagreement between methods")
        
        if result.consensus_metrics.uncertainty_measure < 0.2:
            insights.append("Low uncertainty in the consensus")
        elif result.consensus_metrics.uncertainty_measure > 0.5:
            insights.append("High uncertainty in the consensus")
        
        if result.disagreement_analysis.get("disagreement_level") == "high":
            insights.append("Methods show conflicting assessments")
        
        if result.recommendation_strength == "STRONG":
            insights.append("Strong evidence supports the verdict")
        elif result.recommendation_strength == "INSUFFICIENT":
            insights.append("Insufficient evidence for confident assessment")
        
        return insights


## Suggestions for Upgrade:
# 1. Implement advanced machine learning ensemble methods for consensus building
# 2. Add support for temporal consensus tracking to monitor how verdicts change over time
# 3. Integrate with external expert systems for human-in-the-loop consensus validation
# 4. Add support for multi-criteria decision analysis (MCDA) methods for complex trade-offs