import asyncio
import logging
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import statistics
import json

from ..proof_validation.proof_validator import ValidationResult
from ..proof_validation.scoring import ConsensusResult, SourceScore, AdvancedScoring
from ..proof_validation.proofs_aggregator import AggregatedProofs

logger = logging.getLogger(__name__)


class VerdictType(Enum):
    """Enumeration of possible verdict types."""
    TRUE = "TRUE"
    FALSE = "FALSE"
    MISLEADING = "MISLEADING"
    UNPROVEN = "UNPROVEN"
    MIXED = "MIXED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class ConfidenceLevel(Enum):
    """Enumeration of confidence levels."""
    VERY_HIGH = "VERY_HIGH"  # 90-100%
    HIGH = "HIGH"           # 75-89%
    MEDIUM = "MEDIUM"       # 60-74%
    LOW = "LOW"            # 40-59%
    VERY_LOW = "VERY_LOW"  # 0-39%


@dataclass
class VerdictExplanation:
    """Detailed explanation of verdict reasoning."""
    primary_factors: List[str]
    supporting_evidence_summary: str
    refuting_evidence_summary: str
    key_sources: List[str]
    limitations: List[str]
    methodology_notes: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "primary_factors": self.primary_factors,
            "supporting_evidence_summary": self.supporting_evidence_summary,
            "refuting_evidence_summary": self.refuting_evidence_summary,
            "key_sources": self.key_sources,
            "limitations": self.limitations,
            "methodology_notes": self.methodology_notes
        }


@dataclass
class FinalVerdict:
    """Final fact-checking verdict with comprehensive details."""
    claim: str
    verdict: VerdictType
    confidence_score: float
    confidence_level: ConfidenceLevel
    explanation: VerdictExplanation
    evidence_summary: Dict[str, Any]
    source_analysis: Dict[str, Any]
    methodology_details: Dict[str, Any]
    quality_indicators: Dict[str, Any]
    timestamp: datetime
    processing_time: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "claim": self.claim,
            "verdict": self.verdict.value,
            "confidence_score": self.confidence_score,
            "confidence_level": self.confidence_level.value,
            "explanation": self.explanation.to_dict(),
            "evidence_summary": self.evidence_summary,
            "source_analysis": self.source_analysis,
            "methodology_details": self.methodology_details,
            "quality_indicators": self.quality_indicators,
            "timestamp": self.timestamp.isoformat(),
            "processing_time": self.processing_time
        }


class VerdictEngineError(Exception):
    """Custom exception for verdict engine errors."""
    pass


class VerdictEngine:
    """Advanced verdict determination engine for fact-checking."""
    
    def __init__(self):
        """Initialize verdict engine."""
        # Decision thresholds
        self.thresholds = {
            "high_confidence": 0.8,
            "medium_confidence": 0.6,
            "low_confidence": 0.4,
            "min_evidence_threshold": 3,
            "consensus_threshold": 0.7,
            "mixed_verdict_threshold": 0.3,
            "misleading_threshold": 0.6
        }
        
        # Verdict mapping rules
        self.verdict_rules = {
            "strong_supporting": VerdictType.TRUE,
            "strong_refuting": VerdictType.FALSE,
            "mixed_evidence": VerdictType.MIXED,
            "insufficient_evidence": VerdictType.INSUFFICIENT_EVIDENCE,
            "misleading_context": VerdictType.MISLEADING,
            "unproven_claims": VerdictType.UNPROVEN
        }
        
        # Quality requirements for different confidence levels
        self.quality_requirements = {
            ConfidenceLevel.VERY_HIGH: {
                "min_sources": 5,
                "min_credible_sources": 3,
                "min_fact_check_sources": 1,
                "min_consensus": 0.85,
                "max_bias_score": 0.2
            },
            ConfidenceLevel.HIGH: {
                "min_sources": 4,
                "min_credible_sources": 2,
                "min_fact_check_sources": 1,
                "min_consensus": 0.75,
                "max_bias_score": 0.3
            },
            ConfidenceLevel.MEDIUM: {
                "min_sources": 3,
                "min_credible_sources": 2,
                "min_fact_check_sources": 0,
                "min_consensus": 0.65,
                "max_bias_score": 0.4
            },
            ConfidenceLevel.LOW: {
                "min_sources": 2,
                "min_credible_sources": 1,
                "min_fact_check_sources": 0,
                "min_consensus": 0.55,
                "max_bias_score": 0.5
            },
            ConfidenceLevel.VERY_LOW: {
                "min_sources": 1,
                "min_credible_sources": 0,
                "min_fact_check_sources": 0,
                "min_consensus": 0.0,
                "max_bias_score": 1.0
            }
        }
        
        logger.info("VerdictEngine initialized")
    
    async def determine_final_verdict(
        self,
        proofs: AggregatedProofs,
        validation_result: ValidationResult,
        consensus_result: ConsensusResult
    ) -> FinalVerdict:
        """
        Determine final verdict combining all analysis results.
        
        Args:
            proofs: AggregatedProofs containing evidence
            validation_result: ValidationResult from proof validation
            consensus_result: ConsensusResult from consensus building
            
        Returns:
            FinalVerdict with comprehensive analysis
        """
        start_time = datetime.now()
        
        try:
            # Analyze evidence patterns
            evidence_analysis = self._analyze_evidence_patterns(proofs, validation_result)
            
            # Determine primary verdict type
            primary_verdict = self._determine_primary_verdict(
                validation_result, consensus_result, evidence_analysis
            )
            
            # Calculate confidence score
            confidence_score = self._calculate_final_confidence(
                validation_result, consensus_result, evidence_analysis
            )
            
            # Determine confidence level
            confidence_level = self._determine_confidence_level(
                confidence_score, proofs, consensus_result
            )
            
            # Generate explanation
            explanation = self._generate_verdict_explanation(
                primary_verdict, proofs, validation_result, consensus_result, evidence_analysis
            )
            
            # Compile evidence summary
            evidence_summary = self._compile_evidence_summary(proofs, validation_result)
            
            # Analyze sources
            source_analysis = self._analyze_sources(consensus_result.source_scores)
            
            # Document methodology
            methodology_details = self._document_methodology(
                validation_result, consensus_result
            )
            
            # Calculate quality indicators
            quality_indicators = self._calculate_quality_indicators(
                proofs, validation_result, consensus_result
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            final_verdict = FinalVerdict(
                claim=proofs.claim,
                verdict=primary_verdict,
                confidence_score=confidence_score,
                confidence_level=confidence_level,
                explanation=explanation,
                evidence_summary=evidence_summary,
                source_analysis=source_analysis,
                methodology_details=methodology_details,
                quality_indicators=quality_indicators,
                timestamp=datetime.now(),
                processing_time=processing_time
            )
            
            logger.info(f"Final verdict determined: {primary_verdict.value} (confidence: {confidence_score:.2f})")
            return final_verdict
            
        except Exception as e:
            logger.error(f"Verdict determination failed: {e}")
            raise VerdictEngineError(f"Failed to determine verdict: {e}")
    
    def _analyze_evidence_patterns(
        self,
        proofs: AggregatedProofs,
        validation_result: ValidationResult
    ) -> Dict[str, Any]:
        """Analyze patterns in evidence to inform verdict."""
        supporting_count = len(proofs.supporting_evidence)
        refuting_count = len(proofs.refuting_evidence)
        neutral_count = len(proofs.neutral_evidence)
        total_evidence = supporting_count + refuting_count + neutral_count
        
        if total_evidence == 0:
            return {
                "pattern": "no_evidence",
                "strength": 0.0,
                "direction": "neutral",
                "consistency": 0.0
            }
        
        # Calculate evidence ratios
        support_ratio = supporting_count / total_evidence
        refute_ratio = refuting_count / total_evidence
        neutral_ratio = neutral_count / total_evidence
        
        # Determine dominant pattern
        if support_ratio > 0.7:
            pattern = "strong_supporting"
            strength = support_ratio
            direction = "supporting"
        elif refute_ratio > 0.7:
            pattern = "strong_refuting"
            strength = refute_ratio
            direction = "refuting"
        elif abs(support_ratio - refute_ratio) < 0.2:
            pattern = "mixed_evidence"
            strength = max(support_ratio, refute_ratio)
            direction = "mixed"
        elif neutral_ratio > 0.5:
            pattern = "mostly_neutral"
            strength = neutral_ratio
            direction = "neutral"
        else:
            pattern = "unclear"
            strength = max(support_ratio, refute_ratio, neutral_ratio)
            direction = "unclear"
        
        # Calculate consistency
        consistency = validation_result.consistency_score
        
        return {
            "pattern": pattern,
            "strength": strength,
            "direction": direction,
            "consistency": consistency,
            "support_ratio": support_ratio,
            "refute_ratio": refute_ratio,
            "neutral_ratio": neutral_ratio,
            "evidence_distribution": {
                "supporting": supporting_count,
                "refuting": refuting_count,
                "neutral": neutral_count
            }
        }
    
    def _determine_primary_verdict(
        self,
        validation_result: ValidationResult,
        consensus_result: ConsensusResult,
        evidence_analysis: Dict[str, Any]
    ) -> VerdictType:
        """Determine the primary verdict type."""
        # Check for insufficient evidence first
        total_evidence = sum(evidence_analysis["evidence_distribution"].values())
        if total_evidence < self.thresholds["min_evidence_threshold"]:
            return VerdictType.INSUFFICIENT_EVIDENCE
        
        # Use consensus result as primary indicator
        consensus_verdict = consensus_result.consensus_verdict
        consensus_confidence = consensus_result.consensus_confidence
        
        # Map validation verdicts to final verdicts
        if consensus_verdict == "REAL":
            if consensus_confidence > self.thresholds["high_confidence"]:
                return VerdictType.TRUE
            elif evidence_analysis["pattern"] == "mixed_evidence":
                return VerdictType.MIXED
            else:
                return VerdictType.TRUE
        
        elif consensus_verdict == "FAKE":
            if consensus_confidence > self.thresholds["high_confidence"]:
                return VerdictType.FALSE
            elif evidence_analysis["pattern"] == "mixed_evidence":
                return VerdictType.MIXED
            else:
                return VerdictType.FALSE
        
        else:  # AMBIGUOUS
            # Further analysis for ambiguous cases
            if evidence_analysis["pattern"] == "mixed_evidence":
                return VerdictType.MIXED
            elif evidence_analysis["pattern"] == "mostly_neutral":
                return VerdictType.UNPROVEN
            elif validation_result.bias_score > 0.6:
                return VerdictType.MISLEADING
            else:
                return VerdictType.UNPROVEN
    
    def _calculate_final_confidence(
        self,
        validation_result: ValidationResult,
        consensus_result: ConsensusResult,
        evidence_analysis: Dict[str, Any]
    ) -> float:
        """Calculate final confidence score."""
        # Base confidence from consensus
        base_confidence = consensus_result.consensus_confidence
        
        # Adjust based on validation factors
        validation_adjustment = (
            validation_result.evidence_score * 0.2 +
            validation_result.credibility_score * 0.2 +
            validation_result.consistency_score * 0.15 +
            (1.0 - validation_result.bias_score) * 0.1  # Lower bias = higher confidence
        )
        
        # Evidence pattern adjustment
        pattern_adjustment = 0.0
        if evidence_analysis["pattern"] in ["strong_supporting", "strong_refuting"]:
            pattern_adjustment = 0.1 * evidence_analysis["strength"]
        elif evidence_analysis["pattern"] == "mixed_evidence":
            pattern_adjustment = -0.1  # Reduce confidence for mixed evidence
        
        # Agreement level adjustment
        agreement_adjustment = (consensus_result.agreement_level - 0.5) * 0.2
        
        # Source quality adjustment
        source_quality = self._calculate_source_quality_factor(consensus_result.source_scores)
        source_adjustment = (source_quality - 0.5) * 0.15
        
        # Combine adjustments
        final_confidence = (
            base_confidence * 0.4 +
            validation_adjustment +
            pattern_adjustment +
            agreement_adjustment +
            source_adjustment
        )
        
        # Ensure confidence is within bounds
        return max(0.0, min(1.0, final_confidence))
    
    def _calculate_source_quality_factor(self, source_scores: List[SourceScore]) -> float:
        """Calculate overall source quality factor."""
        if not source_scores:
            return 0.5
        
        # Average final scores
        avg_score = statistics.mean([score.final_score for score in source_scores])
        
        # Bonus for high-authority sources
        high_authority_count = sum(1 for score in source_scores if score.authority_score > 0.8)
        authority_bonus = min(0.2, high_authority_count * 0.05)
        
        # Penalty for high-bias sources
        high_bias_count = sum(1 for score in source_scores if score.bias_score > 0.6)
        bias_penalty = min(0.2, high_bias_count * 0.05)
        
        return max(0.0, min(1.0, avg_score + authority_bonus - bias_penalty))
    
    def _determine_confidence_level(
        self,
        confidence_score: float,
        proofs: AggregatedProofs,
        consensus_result: ConsensusResult
    ) -> ConfidenceLevel:
        """Determine confidence level based on score and quality requirements."""
        # Start with score-based level
        if confidence_score >= 0.9:
            target_level = ConfidenceLevel.VERY_HIGH
        elif confidence_score >= 0.75:
            target_level = ConfidenceLevel.HIGH
        elif confidence_score >= 0.6:
            target_level = ConfidenceLevel.MEDIUM
        elif confidence_score >= 0.4:
            target_level = ConfidenceLevel.LOW
        else:
            target_level = ConfidenceLevel.VERY_LOW
        
        # Check if quality requirements are met
        requirements = self.quality_requirements[target_level]
        
        # Count high-credibility sources
        credible_sources = sum(
            1 for score in consensus_result.source_scores 
            if score.credibility_score > 0.7
        )
        
        # Check requirements
        checks = [
            len(consensus_result.source_scores) >= requirements["min_sources"],
            credible_sources >= requirements["min_credible_sources"],
            proofs.fact_check_sources_count >= requirements["min_fact_check_sources"],
            consensus_result.agreement_level >= requirements["min_consensus"]
        ]
        
        # If requirements not met, downgrade confidence level
        if not all(checks):
            if target_level == ConfidenceLevel.VERY_HIGH:
                return ConfidenceLevel.HIGH
            elif target_level == ConfidenceLevel.HIGH:
                return ConfidenceLevel.MEDIUM
            elif target_level == ConfidenceLevel.MEDIUM:
                return ConfidenceLevel.LOW
            elif target_level == ConfidenceLevel.LOW:
                return ConfidenceLevel.VERY_LOW
        
        return target_level
    
    def _generate_verdict_explanation(
        self,
        verdict: VerdictType,
        proofs: AggregatedProofs,
        validation_result: ValidationResult,
        consensus_result: ConsensusResult,
        evidence_analysis: Dict[str, Any]
    ) -> VerdictExplanation:
        """Generate detailed explanation for the verdict."""
        primary_factors = []
        limitations = []
        methodology_notes = []
        
        # Primary factors based on verdict
        if verdict == VerdictType.TRUE:
            primary_factors.append(f"Strong supporting evidence from {len(proofs.supporting_evidence)} sources")
            if consensus_result.consensus_confidence > 0.8:
                primary_factors.append("High consensus among sources")
        elif verdict == VerdictType.FALSE:
            primary_factors.append(f"Strong refuting evidence from {len(proofs.refuting_evidence)} sources")
            if consensus_result.consensus_confidence > 0.8:
                primary_factors.append("High consensus among sources")
        elif verdict == VerdictType.MIXED:
            primary_factors.append("Conflicting evidence from multiple sources")
            primary_factors.append(f"Supporting: {len(proofs.supporting_evidence)}, Refuting: {len(proofs.refuting_evidence)}")
        elif verdict == VerdictType.INSUFFICIENT_EVIDENCE:
            primary_factors.append("Insufficient evidence available for verification")
            primary_factors.append(f"Only {proofs.total_sources} sources found")
        
        # Add credibility factors
        if validation_result.credibility_score > 0.8:
            primary_factors.append("High-credibility sources")
        elif validation_result.credibility_score < 0.5:
            limitations.append("Limited source credibility")
        
        # Add bias considerations
        if validation_result.bias_score > 0.6:
            limitations.append("Potential bias detected in sources")
        
        # Generate evidence summaries
        supporting_summary = self._generate_evidence_summary(proofs.supporting_evidence, "supporting")
        refuting_summary = self._generate_evidence_summary(proofs.refuting_evidence, "refuting")
        
        # Key sources
        key_sources = [
            score.source_name for score in consensus_result.source_scores[:5]
            if score.final_score > 0.7
        ]
        
        # Methodology notes
        methodology_notes.extend([
            f"Analysis based on {proofs.total_sources} sources",
            f"Consensus algorithm used with {len(consensus_result.source_scores)} weighted sources",
            f"Evidence validation using multiple quality metrics"
        ])
        
        if proofs.fact_check_sources_count > 0:
            methodology_notes.append(f"Includes {proofs.fact_check_sources_count} dedicated fact-checking sources")
        
        return VerdictExplanation(
            primary_factors=primary_factors,
            supporting_evidence_summary=supporting_summary,
            refuting_evidence_summary=refuting_summary,
            key_sources=key_sources,
            limitations=limitations,
            methodology_notes=methodology_notes
        )
    
    def _generate_evidence_summary(self, evidence_list: List[Any], evidence_type: str) -> str:
        """Generate summary of evidence."""
        if not evidence_list:
            return f"No {evidence_type} evidence found."
        
        count = len(evidence_list)
        
        # Get top sources
        top_sources = []
        for evidence in evidence_list[:3]:  # Top 3
            if hasattr(evidence, 'source_name') and evidence.source_name:
                top_sources.append(evidence.source_name)
        
        summary = f"{count} {evidence_type} source{'s' if count != 1 else ''} found"
        
        if top_sources:
            summary += f", including {', '.join(top_sources[:2])}"
            if len(top_sources) > 2:
                summary += f" and {len(top_sources) - 2} other{'s' if len(top_sources) > 3 else ''}"
        
        return summary + "."
    
    def _compile_evidence_summary(
        self,
        proofs: AggregatedProofs,
        validation_result: ValidationResult
    ) -> Dict[str, Any]:
        """Compile comprehensive evidence summary."""
        return {
            "total_evidence": len(proofs.supporting_evidence) + len(proofs.refuting_evidence) + len(proofs.neutral_evidence),
            "supporting_evidence": len(proofs.supporting_evidence),
            "refuting_evidence": len(proofs.refuting_evidence),
            "neutral_evidence": len(proofs.neutral_evidence),
            "evidence_quality_score": validation_result.evidence_score,
            "source_credibility_score": validation_result.credibility_score,
            "consistency_score": validation_result.consistency_score,
            "temporal_relevance": validation_result.temporal_score
        }
    
    def _analyze_sources(self, source_scores: List[SourceScore]) -> Dict[str, Any]:
        """Analyze source characteristics."""
        if not source_scores:
            return {
                "total_sources": 0,
                "average_quality": 0.0,
                "high_quality_sources": 0,
                "authoritative_sources": 0,
                "low_bias_sources": 0
            }
        
        return {
            "total_sources": len(source_scores),
            "average_quality": statistics.mean([score.final_score for score in source_scores]),
            "quality_std_dev": statistics.stdev([score.final_score for score in source_scores]) if len(source_scores) > 1 else 0,
            "high_quality_sources": len([s for s in source_scores if s.final_score > 0.7]),
            "authoritative_sources": len([s for s in source_scores if s.authority_score > 0.8]),
            "low_bias_sources": len([s for s in source_scores if s.bias_score < 0.3]),
            "top_sources": [
                {"name": score.source_name, "score": score.final_score}
                for score in source_scores[:5]
            ]
        }
    
    def _document_methodology(
        self,
        validation_result: ValidationResult,
        consensus_result: ConsensusResult
    ) -> Dict[str, Any]:
        """Document the methodology used."""
        return {
            "validation_method": "comprehensive_multi_factor_analysis",
            "consensus_method": "weighted_source_voting",
            "factors_considered": [
                "source_credibility",
                "evidence_quality",
                "temporal_relevance",
                "bias_detection",
                "consistency_analysis"
            ],
            "validation_details": validation_result.validation_details,
            "consensus_factors": consensus_result.consensus_factors,
            "quality_thresholds": self.thresholds
        }
    
    def _calculate_quality_indicators(
        self,
        proofs: AggregatedProofs,
        validation_result: ValidationResult,
        consensus_result: ConsensusResult
    ) -> Dict[str, Any]:
        """Calculate various quality indicators."""
        return {
            "evidence_diversity": len(set(
                evidence.source_name for evidence in 
                proofs.supporting_evidence + proofs.refuting_evidence + proofs.neutral_evidence
            )),
            "source_agreement": consensus_result.agreement_level,
            "fact_check_coverage": proofs.fact_check_sources_count > 0,
            "temporal_freshness": validation_result.temporal_score,
            "bias_level": validation_result.bias_score,
            "outlier_sources": len(consensus_result.outlier_sources),
            "processing_quality": {
                "aggregation_confidence": proofs.aggregation_confidence,
                "validation_confidence": validation_result.confidence,
                "consensus_confidence": consensus_result.consensus_confidence
            }
        }
    
    def get_verdict_summary(self, verdict: FinalVerdict) -> Dict[str, Any]:
        """Generate a concise summary of the verdict."""
        return {
            "claim": verdict.claim,
            "verdict": verdict.verdict.value,
            "confidence": {
                "score": verdict.confidence_score,
                "level": verdict.confidence_level.value
            },
            "key_points": verdict.explanation.primary_factors[:3],
            "evidence_count": verdict.evidence_summary["total_evidence"],
            "source_count": verdict.source_analysis["total_sources"],
            "fact_check_sources": verdict.evidence_summary.get("fact_check_sources", 0),
            "limitations": verdict.explanation.limitations,
            "timestamp": verdict.timestamp.isoformat()
        }


## Suggestions for Upgrade:
# 1. Implement machine learning models for verdict prediction based on historical fact-checking data
# 2. Add support for temporal claim analysis to track how claims evolve over time
# 3. Integrate with external fact-checking databases for cross-validation of verdicts
# 4. Add explainable AI features to provide more detailed reasoning for verdict decisions