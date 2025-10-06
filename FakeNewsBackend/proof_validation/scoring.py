"""
Scoring Module

Author: Trae AI
Date: 2024-01-15
Purpose: Advanced scoring system for credibility weighting and consensus building
License: MIT

This module provides sophisticated scoring algorithms for weighting evidence
credibility, building consensus across sources, and generating confidence scores
for fact-checking verdicts.
"""

import asyncio
import logging
from typing import List, Dict, Optional, Any, Tuple, Set
from dataclasses import dataclass
from datetime import datetime, timedelta
import statistics
import math
from collections import defaultdict, Counter

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    logging.warning("NumPy not available. Some advanced scoring features will be limited.")

from .proof_validator import ValidationResult, EvidenceAnalysis
from .proofs_aggregator import AggregatedProofs, ProofEvidence

logger = logging.getLogger(__name__)


@dataclass
class SourceScore:
    """Score for an individual source."""
    source_url: str
    source_name: str
    credibility_score: float
    authority_score: float
    bias_score: float
    consistency_score: float
    temporal_score: float
    final_score: float
    weight: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "source_url": self.source_url,
            "source_name": self.source_name,
            "credibility_score": self.credibility_score,
            "authority_score": self.authority_score,
            "bias_score": self.bias_score,
            "consistency_score": self.consistency_score,
            "temporal_score": self.temporal_score,
            "final_score": self.final_score,
            "weight": self.weight
        }


@dataclass
class ConsensusResult:
    """Result of consensus building analysis."""
    claim: str
    consensus_verdict: str
    consensus_confidence: float
    agreement_level: float
    source_scores: List[SourceScore]
    weighted_evidence: Dict[str, float]
    outlier_sources: List[str]
    consensus_factors: Dict[str, Any]
    calculated_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "claim": self.claim,
            "consensus_verdict": self.consensus_verdict,
            "consensus_confidence": self.consensus_confidence,
            "agreement_level": self.agreement_level,
            "source_scores": [score.to_dict() for score in self.source_scores],
            "weighted_evidence": self.weighted_evidence,
            "outlier_sources": self.outlier_sources,
            "consensus_factors": self.consensus_factors,
            "calculated_at": self.calculated_at.isoformat()
        }


@dataclass
class CredibilityWeights:
    """Weights for different credibility factors."""
    source_authority: float = 0.25
    historical_accuracy: float = 0.20
    bias_penalty: float = 0.15
    temporal_relevance: float = 0.15
    content_quality: float = 0.15
    consensus_alignment: float = 0.10
    
    def normalize(self) -> 'CredibilityWeights':
        """Normalize weights to sum to 1.0."""
        total = (
            self.source_authority + self.historical_accuracy + self.bias_penalty +
            self.temporal_relevance + self.content_quality + self.consensus_alignment
        )
        
        if total == 0:
            return CredibilityWeights()
        
        return CredibilityWeights(
            source_authority=self.source_authority / total,
            historical_accuracy=self.historical_accuracy / total,
            bias_penalty=self.bias_penalty / total,
            temporal_relevance=self.temporal_relevance / total,
            content_quality=self.content_quality / total,
            consensus_alignment=self.consensus_alignment / total
        )


class ScoringError(Exception):
    """Custom exception for scoring errors."""
    pass


class AdvancedScoring:
    """Advanced scoring system for credibility weighting and consensus building."""
    
    def __init__(self):
        """Initialize advanced scoring system."""
        # Default credibility weights
        self.default_weights = CredibilityWeights().normalize()
        
        # Source reputation database (would be loaded from external source in production)
        self.source_reputation = self._load_source_reputation()
        
        # Scoring parameters
        self.scoring_params = {
            "min_sources_for_consensus": 3,
            "outlier_threshold": 2.0,  # Standard deviations
            "agreement_threshold": 0.7,
            "high_confidence_threshold": 0.8,
            "temporal_decay_factor": 0.1,
            "bias_penalty_factor": 0.2
        }
        
        # Evidence type weights
        self.evidence_weights = {
            "fact_check": 1.0,
            "news": 0.8,
            "web": 0.6,
            "social": 0.4
        }
        
        logger.info("AdvancedScoring initialized")
    
    def _load_source_reputation(self) -> Dict[str, Dict[str, float]]:
        """Load source reputation scores (mock data for demonstration)."""
        return {
            "reuters.com": {
                "accuracy": 0.95,
                "bias": 0.1,
                "reliability": 0.9
            },
            "apnews.com": {
                "accuracy": 0.94,
                "bias": 0.1,
                "reliability": 0.9
            },
            "bbc.com": {
                "accuracy": 0.88,
                "bias": 0.2,
                "reliability": 0.85
            },
            "cnn.com": {
                "accuracy": 0.82,
                "bias": 0.3,
                "reliability": 0.8
            },
            "foxnews.com": {
                "accuracy": 0.75,
                "bias": 0.4,
                "reliability": 0.7
            },
            "snopes.com": {
                "accuracy": 0.92,
                "bias": 0.15,
                "reliability": 0.9
            },
            "politifact.com": {
                "accuracy": 0.90,
                "bias": 0.2,
                "reliability": 0.85
            }
        }
    
    async def calculate_source_scores(
        self,
        proofs: AggregatedProofs,
        evidence_analyses: List[EvidenceAnalysis],
        weights: Optional[CredibilityWeights] = None
    ) -> List[SourceScore]:
        """
        Calculate comprehensive scores for all sources.
        
        Args:
            proofs: AggregatedProofs containing evidence
            evidence_analyses: List of evidence analyses
            weights: Custom credibility weights
            
        Returns:
            List of SourceScore objects
        """
        if weights is None:
            weights = self.default_weights
        
        try:
            source_scores = []
            
            # Group evidence by source
            evidence_by_source = defaultdict(list)
            for analysis in evidence_analyses:
                source_url = analysis.evidence.source_url
                evidence_by_source[source_url].append(analysis)
            
            # Calculate scores for each source
            for source_url, analyses in evidence_by_source.items():
                if not analyses:
                    continue
                
                # Get representative analysis (first one for source info)
                representative = analyses[0]
                
                # Calculate component scores
                authority_score = self._calculate_authority_score(source_url)
                historical_accuracy = self._get_historical_accuracy(source_url)
                bias_score = self._calculate_source_bias_score(analyses)
                temporal_score = self._calculate_source_temporal_score(analyses)
                content_quality = self._calculate_source_content_quality(analyses)
                
                # Calculate weighted final score
                final_score = (
                    authority_score * weights.source_authority +
                    historical_accuracy * weights.historical_accuracy +
                    (1.0 - bias_score) * weights.bias_penalty +  # Invert bias (lower bias = higher score)
                    temporal_score * weights.temporal_relevance +
                    content_quality * weights.content_quality
                )
                
                # Calculate source weight based on final score and evidence count
                evidence_count = len(analyses)
                weight = final_score * math.log(1 + evidence_count)
                
                source_score = SourceScore(
                    source_url=source_url,
                    source_name=representative.evidence.source_name,
                    credibility_score=representative.evidence.credibility_score,
                    authority_score=authority_score,
                    bias_score=bias_score,
                    consistency_score=self._calculate_source_consistency(analyses),
                    temporal_score=temporal_score,
                    final_score=final_score,
                    weight=weight
                )
                
                source_scores.append(source_score)
            
            # Normalize weights
            if source_scores:
                total_weight = sum(score.weight for score in source_scores)
                if total_weight > 0:
                    for score in source_scores:
                        score.weight = score.weight / total_weight
            
            # Sort by final score (highest first)
            source_scores.sort(key=lambda x: x.final_score, reverse=True)
            
            logger.info(f"Calculated scores for {len(source_scores)} sources")
            return source_scores
            
        except Exception as e:
            logger.error(f"Source scoring failed: {e}")
            raise ScoringError(f"Failed to calculate source scores: {e}")
    
    def _calculate_authority_score(self, source_url: str) -> float:
        """Calculate authority score for a source."""
        if not source_url:
            return 0.5
        
        try:
            from urllib.parse import urlparse
            domain = urlparse(source_url).netloc.lower()
            
            # Remove www. prefix
            if domain.startswith("www."):
                domain = domain[4:]
            
            # Check reputation database
            if domain in self.source_reputation:
                return self.source_reputation[domain].get("reliability", 0.5)
            
            # Domain-based scoring
            if domain.endswith('.edu') or domain.endswith('.gov'):
                return 0.9
            elif domain.endswith('.org'):
                return 0.7
            elif any(fact_checker in domain for fact_checker in ['snopes', 'politifact', 'factcheck']):
                return 0.85
            elif any(news_org in domain for news_org in ['reuters', 'ap', 'bbc', 'npr']):
                return 0.8
            else:
                return 0.5
                
        except Exception:
            return 0.5
    
    def _get_historical_accuracy(self, source_url: str) -> float:
        """Get historical accuracy score for a source."""
        if not source_url:
            return 0.5
        
        try:
            from urllib.parse import urlparse
            domain = urlparse(source_url).netloc.lower()
            
            if domain.startswith("www."):
                domain = domain[4:]
            
            if domain in self.source_reputation:
                return self.source_reputation[domain].get("accuracy", 0.5)
            
            return 0.5  # Default for unknown sources
            
        except Exception:
            return 0.5
    
    def _calculate_source_bias_score(self, analyses: List[EvidenceAnalysis]) -> float:
        """Calculate bias score for a source (0 = no bias, 1 = high bias)."""
        if not analyses:
            return 0.5
        
        # Average bias indicators across all evidence from this source
        total_bias_indicators = sum(len(analysis.bias_indicators) for analysis in analyses)
        avg_bias_indicators = total_bias_indicators / len(analyses)
        
        # Convert to bias score (more indicators = higher bias)
        bias_score = min(1.0, avg_bias_indicators * 0.1)
        
        # Check reputation database
        if analyses:
            source_url = analyses[0].evidence.source_url
            try:
                from urllib.parse import urlparse
                domain = urlparse(source_url).netloc.lower()
                if domain.startswith("www."):
                    domain = domain[4:]
                
                if domain in self.source_reputation:
                    reputation_bias = self.source_reputation[domain].get("bias", 0.5)
                    # Combine with calculated bias
                    bias_score = (bias_score + reputation_bias) / 2
            except Exception:
                pass
        
        return bias_score
    
    def _calculate_source_temporal_score(self, analyses: List[EvidenceAnalysis]) -> float:
        """Calculate temporal relevance score for a source."""
        if not analyses:
            return 0.5
        
        temporal_scores = [analysis.temporal_relevance for analysis in analyses]
        return statistics.mean(temporal_scores)
    
    def _calculate_source_content_quality(self, analyses: List[EvidenceAnalysis]) -> float:
        """Calculate content quality score for a source."""
        if not analyses:
            return 0.5
        
        quality_scores = [analysis.quality_score for analysis in analyses]
        return statistics.mean(quality_scores)
    
    def _calculate_source_consistency(self, analyses: List[EvidenceAnalysis]) -> float:
        """Calculate consistency score for a source."""
        if not analyses:
            return 0.5
        
        # Check if all evidence from this source points in the same direction
        evidence_types = [analysis.evidence.evidence_type for analysis in analyses]
        type_counts = Counter(evidence_types)
        
        if len(type_counts) == 1:
            return 1.0  # Perfect consistency
        
        # Calculate consistency based on dominant type
        max_count = max(type_counts.values())
        total_count = len(analyses)
        
        return max_count / total_count
    
    async def build_consensus(
        self,
        proofs: AggregatedProofs,
        source_scores: List[SourceScore],
        validation_result: ValidationResult
    ) -> ConsensusResult:
        """
        Build consensus across sources using weighted voting.
        
        Args:
            proofs: AggregatedProofs containing evidence
            source_scores: List of source scores
            validation_result: Initial validation result
            
        Returns:
            ConsensusResult with consensus verdict and confidence
        """
        try:
            # Group evidence by verdict
            evidence_by_verdict = defaultdict(list)
            
            all_evidence = (
                proofs.supporting_evidence + 
                proofs.refuting_evidence + 
                proofs.neutral_evidence
            )
            
            # Map evidence to verdicts
            for evidence in all_evidence:
                if evidence.evidence_type == "supporting":
                    evidence_by_verdict["REAL"].append(evidence)
                elif evidence.evidence_type == "refuting":
                    evidence_by_verdict["FAKE"].append(evidence)
                else:
                    evidence_by_verdict["AMBIGUOUS"].append(evidence)
            
            # Calculate weighted votes
            weighted_votes = {"REAL": 0.0, "FAKE": 0.0, "AMBIGUOUS": 0.0}
            
            # Create source score lookup
            source_score_lookup = {score.source_url: score for score in source_scores}
            
            for verdict, evidence_list in evidence_by_verdict.items():
                for evidence in evidence_list:
                    source_score = source_score_lookup.get(evidence.source_url)
                    if source_score:
                        # Weight vote by source score and evidence quality
                        vote_weight = (
                            source_score.weight * 0.7 +
                            evidence.relevance_score * 0.2 +
                            evidence.credibility_score * 0.1
                        )
                        weighted_votes[verdict] += vote_weight
            
            # Normalize votes
            total_votes = sum(weighted_votes.values())
            if total_votes > 0:
                for verdict in weighted_votes:
                    weighted_votes[verdict] = weighted_votes[verdict] / total_votes
            
            # Determine consensus verdict
            consensus_verdict = max(weighted_votes, key=weighted_votes.get)
            consensus_confidence = weighted_votes[consensus_verdict]
            
            # Calculate agreement level
            agreement_level = self._calculate_agreement_level(weighted_votes)
            
            # Identify outlier sources
            outlier_sources = self._identify_outliers(source_scores, consensus_verdict)
            
            # Calculate consensus factors
            consensus_factors = {
                "total_sources": len(source_scores),
                "high_quality_sources": len([s for s in source_scores if s.final_score > 0.7]),
                "agreement_threshold_met": agreement_level > self.scoring_params["agreement_threshold"],
                "outlier_count": len(outlier_sources),
                "weighted_distribution": weighted_votes,
                "dominant_verdict_strength": consensus_confidence
            }
            
            # Adjust confidence based on agreement level
            if agreement_level < self.scoring_params["agreement_threshold"]:
                consensus_confidence *= 0.8  # Reduce confidence for low agreement
            
            result = ConsensusResult(
                claim=proofs.claim,
                consensus_verdict=consensus_verdict,
                consensus_confidence=consensus_confidence,
                agreement_level=agreement_level,
                source_scores=source_scores,
                weighted_evidence=weighted_votes,
                outlier_sources=outlier_sources,
                consensus_factors=consensus_factors,
                calculated_at=datetime.now()
            )
            
            logger.info(f"Consensus built: {consensus_verdict} (confidence: {consensus_confidence:.2f}, agreement: {agreement_level:.2f})")
            return result
            
        except Exception as e:
            logger.error(f"Consensus building failed: {e}")
            raise ScoringError(f"Failed to build consensus: {e}")
    
    def _calculate_agreement_level(self, weighted_votes: Dict[str, float]) -> float:
        """Calculate agreement level among sources."""
        if not weighted_votes:
            return 0.0
        
        # Agreement is higher when votes are concentrated in fewer categories
        vote_values = list(weighted_votes.values())
        
        # Calculate entropy (lower entropy = higher agreement)
        entropy = 0.0
        for vote in vote_values:
            if vote > 0:
                entropy -= vote * math.log2(vote)
        
        # Convert entropy to agreement score (0 = no agreement, 1 = perfect agreement)
        max_entropy = math.log2(len(weighted_votes))
        if max_entropy == 0:
            return 1.0
        
        agreement = 1.0 - (entropy / max_entropy)
        return max(0.0, min(1.0, agreement))
    
    def _identify_outliers(self, source_scores: List[SourceScore], consensus_verdict: str) -> List[str]:
        """Identify outlier sources that disagree with consensus."""
        if not source_scores:
            return []
        
        outliers = []
        
        # Calculate mean and standard deviation of source scores
        scores = [score.final_score for score in source_scores]
        if len(scores) < 3:  # Need at least 3 sources to identify outliers
            return outliers
        
        mean_score = statistics.mean(scores)
        std_score = statistics.stdev(scores) if len(scores) > 1 else 0
        
        if std_score == 0:
            return outliers
        
        # Identify sources that are statistical outliers
        threshold = self.scoring_params["outlier_threshold"]
        
        for score in source_scores:
            z_score = abs(score.final_score - mean_score) / std_score
            if z_score > threshold:
                outliers.append(score.source_url)
        
        return outliers
    
    def calculate_confidence_intervals(
        self,
        consensus_result: ConsensusResult,
        confidence_level: float = 0.95
    ) -> Dict[str, Tuple[float, float]]:
        """
        Calculate confidence intervals for verdict probabilities.
        
        Args:
            consensus_result: ConsensusResult object
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            
        Returns:
            Dictionary with confidence intervals for each verdict
        """
        if not NUMPY_AVAILABLE:
            logger.warning("NumPy not available. Confidence intervals will be approximate.")
            # Simple approximation without NumPy
            intervals = {}
            for verdict, prob in consensus_result.weighted_evidence.items():
                margin = 0.1 * (1 - consensus_result.agreement_level)  # Rough approximation
                intervals[verdict] = (max(0, prob - margin), min(1, prob + margin))
            return intervals
        
        try:
            # Use bootstrap method for confidence intervals
            n_sources = len(consensus_result.source_scores)
            if n_sources < 3:
                # Not enough sources for meaningful intervals
                return {
                    verdict: (prob, prob) 
                    for verdict, prob in consensus_result.weighted_evidence.items()
                }
            
            # Bootstrap sampling
            n_bootstrap = 1000
            bootstrap_results = {verdict: [] for verdict in consensus_result.weighted_evidence}
            
            for _ in range(n_bootstrap):
                # Sample sources with replacement
                sampled_indices = np.random.choice(n_sources, size=n_sources, replace=True)
                sampled_scores = [consensus_result.source_scores[i] for i in sampled_indices]
                
                # Recalculate weighted votes for this sample
                sample_votes = {"REAL": 0.0, "FAKE": 0.0, "AMBIGUOUS": 0.0}
                total_weight = sum(score.weight for score in sampled_scores)
                
                if total_weight > 0:
                    for score in sampled_scores:
                        # Simplified vote assignment based on source score
                        if score.final_score > 0.6:
                            sample_votes["REAL"] += score.weight / total_weight
                        elif score.final_score < 0.4:
                            sample_votes["FAKE"] += score.weight / total_weight
                        else:
                            sample_votes["AMBIGUOUS"] += score.weight / total_weight
                
                # Store results
                for verdict, vote in sample_votes.items():
                    bootstrap_results[verdict].append(vote)
            
            # Calculate confidence intervals
            alpha = 1 - confidence_level
            intervals = {}
            
            for verdict, values in bootstrap_results.items():
                if values:
                    lower_percentile = (alpha / 2) * 100
                    upper_percentile = (1 - alpha / 2) * 100
                    
                    lower_bound = np.percentile(values, lower_percentile)
                    upper_bound = np.percentile(values, upper_percentile)
                    
                    intervals[verdict] = (lower_bound, upper_bound)
                else:
                    intervals[verdict] = (0.0, 0.0)
            
            return intervals
            
        except Exception as e:
            logger.warning(f"Confidence interval calculation failed: {e}")
            # Fallback to simple intervals
            return {
                verdict: (max(0, prob - 0.1), min(1, prob + 0.1))
                for verdict, prob in consensus_result.weighted_evidence.items()
            }
    
    def get_scoring_summary(self, consensus_result: ConsensusResult) -> Dict[str, Any]:
        """Generate comprehensive scoring summary."""
        source_scores = consensus_result.source_scores
        
        summary = {
            "consensus_overview": {
                "verdict": consensus_result.consensus_verdict,
                "confidence": consensus_result.consensus_confidence,
                "agreement_level": consensus_result.agreement_level
            },
            "source_analysis": {
                "total_sources": len(source_scores),
                "high_quality_sources": len([s for s in source_scores if s.final_score > 0.7]),
                "low_bias_sources": len([s for s in source_scores if s.bias_score < 0.3]),
                "authoritative_sources": len([s for s in source_scores if s.authority_score > 0.8])
            },
            "evidence_distribution": consensus_result.weighted_evidence,
            "quality_metrics": {
                "avg_source_score": statistics.mean([s.final_score for s in source_scores]) if source_scores else 0,
                "score_std_dev": statistics.stdev([s.final_score for s in source_scores]) if len(source_scores) > 1 else 0,
                "top_source_score": max([s.final_score for s in source_scores]) if source_scores else 0,
                "bottom_source_score": min([s.final_score for s in source_scores]) if source_scores else 0
            },
            "reliability_indicators": {
                "outlier_count": len(consensus_result.outlier_sources),
                "consensus_strength": consensus_result.consensus_factors.get("dominant_verdict_strength", 0),
                "source_diversity": len(set(s.source_name for s in source_scores))
            }
        }
        
        return summary


## Suggestions for Upgrade:
# 1. Implement machine learning models for dynamic source credibility scoring based on historical performance
# 2. Add network analysis to detect coordinated inauthentic behavior and source clustering
# 3. Integrate with real-time fact-checking databases for cross-validation of source reliability
# 4. Implement Bayesian inference methods for more sophisticated consensus building and uncertainty quantification