import asyncio
import logging
from typing import List, Dict, Optional, Any, Tuple, Set
from dataclasses import dataclass
from datetime import datetime, timedelta
import re
import json
from collections import Counter, defaultdict
import statistics

# ML and NLP libraries
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    logging.warning("NumPy not available. Some statistical features will be limited.")

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    logging.warning("TextBlob not available. Sentiment analysis will be limited.")

from .proofs_aggregator import AggregatedProofs, ProofEvidence

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of proof validation analysis."""
    claim: str
    verdict: str  # 'REAL', 'FAKE', 'AMBIGUOUS'
    confidence: float
    evidence_score: float
    credibility_score: float
    consistency_score: float
    bias_score: float
    temporal_score: float
    supporting_strength: float
    refuting_strength: float
    validation_details: Dict[str, Any]
    validated_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "claim": self.claim,
            "verdict": self.verdict,
            "confidence": self.confidence,
            "evidence_score": self.evidence_score,
            "credibility_score": self.credibility_score,
            "consistency_score": self.consistency_score,
            "bias_score": self.bias_score,
            "temporal_score": self.temporal_score,
            "supporting_strength": self.supporting_strength,
            "refuting_strength": self.refuting_strength,
            "validation_details": self.validation_details,
            "validated_at": self.validated_at.isoformat()
        }


@dataclass
class EvidenceAnalysis:
    """Analysis of individual evidence piece."""
    evidence: ProofEvidence
    quality_score: float
    bias_indicators: List[str]
    sentiment_score: float
    factual_indicators: List[str]
    temporal_relevance: float
    source_authority: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "evidence_url": self.evidence.source_url,
            "quality_score": self.quality_score,
            "bias_indicators": self.bias_indicators,
            "sentiment_score": self.sentiment_score,
            "factual_indicators": self.factual_indicators,
            "temporal_relevance": self.temporal_relevance,
            "source_authority": self.source_authority
        }


class ProofValidatorError(Exception):
    """Custom exception for proof validator errors."""
    pass


class ProofValidator:
    """Advanced proof validation system for fact-checking."""
    
    def __init__(self):
        """Initialize proof validator."""
        # Load validation patterns and rules
        self.bias_indicators = self._load_bias_indicators()
        self.factual_indicators = self._load_factual_indicators()
        self.authority_domains = self._load_authority_domains()
        self.unreliable_indicators = self._load_unreliable_indicators()
        
        # Validation thresholds
        self.thresholds = {
            "high_confidence": 0.8,
            "medium_confidence": 0.6,
            "low_confidence": 0.4,
            "min_evidence_count": 3,
            "credibility_threshold": 0.7,
            "consistency_threshold": 0.6
        }
        
        # Scoring weights
        self.weights = {
            "evidence_quality": 0.25,
            "source_credibility": 0.20,
            "consistency": 0.20,
            "temporal_relevance": 0.15,
            "bias_penalty": 0.10,
            "authority_bonus": 0.10
        }
        
        logger.info("ProofValidator initialized")
    
    def _load_bias_indicators(self) -> List[str]:
        """Load patterns that indicate potential bias."""
        return [
            r'\b(?:always|never|all|none|every|completely|totally|absolutely)\b',
            r'\b(?:obviously|clearly|undoubtedly|certainly|definitely)\b',
            r'\b(?:shocking|amazing|incredible|unbelievable|outrageous)\b',
            r'\b(?:they say|some people|many believe|it is said)\b',
            r'\b(?:mainstream media|fake news|cover.?up|conspiracy)\b',
            r'[!]{2,}|[?]{2,}|[A-Z]{5,}',  # Excessive punctuation/caps
            r'\b(?:liberal|conservative|leftist|rightist)\s+(?:media|agenda|bias)\b'
        ]
    
    def _load_factual_indicators(self) -> List[str]:
        """Load patterns that indicate factual content."""
        return [
            r'\b(?:according to|based on|research shows|study finds|data indicates)\b',
            r'\b(?:published in|peer.?reviewed|journal|academic|scientific)\b',
            r'\b(?:statistics|data|evidence|research|analysis|report)\b',
            r'\b(?:professor|doctor|expert|researcher|scientist|analyst)\b',
            r'\b(?:university|institute|organization|agency|department)\b',
            r'\d{4}.*(?:study|research|report|survey|analysis)',
            r'\b(?:methodology|sample size|margin of error|confidence interval)\b'
        ]
    
    def _load_authority_domains(self) -> Dict[str, float]:
        """Load authoritative domains with authority scores."""
        return {
            # Academic and research institutions
            "edu": 0.9,
            "ac.uk": 0.9,
            "harvard.edu": 0.95,
            "mit.edu": 0.95,
            "stanford.edu": 0.95,
            "oxford.ac.uk": 0.95,
            "cambridge.ac.uk": 0.95,
            
            # Government and official sources
            "gov": 0.9,
            "who.int": 0.95,
            "cdc.gov": 0.95,
            "fda.gov": 0.9,
            "nih.gov": 0.95,
            "nasa.gov": 0.9,
            
            # Reputable news organizations
            "reuters.com": 0.9,
            "apnews.com": 0.9,
            "bbc.com": 0.85,
            "npr.org": 0.85,
            "pbs.org": 0.85,
            
            # Fact-checking organizations
            "snopes.com": 0.9,
            "politifact.com": 0.9,
            "factcheck.org": 0.9,
            "fullfact.org": 0.85,
            
            # Scientific journals and databases
            "nature.com": 0.95,
            "science.org": 0.95,
            "pubmed.ncbi.nlm.nih.gov": 0.95,
            "scholar.google.com": 0.8
        }
    
    def _load_unreliable_indicators(self) -> List[str]:
        """Load patterns that indicate unreliable content."""
        return [
            r'\b(?:click here|you won\'t believe|doctors hate|one weird trick)\b',
            r'\b(?:miracle cure|secret|hidden truth|they don\'t want you to know)\b',
            r'\b(?:breaking|urgent|alert|warning).*[!]{2,}\b',
            r'\b(?:share if you agree|like and share|viral|trending)\b',
            r'\b(?:anonymous source|unnamed official|insider claims)\b'
        ]
    
    async def validate_proofs(self, proofs: AggregatedProofs) -> ValidationResult:
        """
        Validate aggregated proofs and determine claim verdict.
        
        Args:
            proofs: AggregatedProofs object containing evidence
            
        Returns:
            ValidationResult with verdict and detailed analysis
        """
        start_time = datetime.now()
        
        try:
            # Analyze individual evidence pieces
            evidence_analyses = await self._analyze_evidence_pieces(proofs)
            
            # Calculate component scores
            evidence_score = self._calculate_evidence_score(evidence_analyses)
            credibility_score = self._calculate_credibility_score(proofs, evidence_analyses)
            consistency_score = self._calculate_consistency_score(proofs, evidence_analyses)
            bias_score = self._calculate_bias_score(evidence_analyses)
            temporal_score = self._calculate_temporal_score(proofs)
            
            # Calculate supporting vs refuting strength
            supporting_strength = self._calculate_supporting_strength(proofs, evidence_analyses)
            refuting_strength = self._calculate_refuting_strength(proofs, evidence_analyses)
            
            # Determine verdict and confidence
            verdict, confidence = self._determine_verdict(
                supporting_strength, refuting_strength, evidence_score,
                credibility_score, consistency_score, bias_score
            )
            
            # Compile validation details
            validation_details = {
                "evidence_count": {
                    "supporting": len(proofs.supporting_evidence),
                    "refuting": len(proofs.refuting_evidence),
                    "neutral": len(proofs.neutral_evidence)
                },
                "source_analysis": {
                    "total_sources": proofs.total_sources,
                    "credible_sources": proofs.credible_sources_count,
                    "fact_check_sources": proofs.fact_check_sources_count
                },
                "quality_metrics": {
                    "avg_evidence_quality": statistics.mean([a.quality_score for a in evidence_analyses]) if evidence_analyses else 0,
                    "high_quality_count": len([a for a in evidence_analyses if a.quality_score > 0.7]),
                    "bias_detected_count": len([a for a in evidence_analyses if a.bias_indicators])
                },
                "temporal_analysis": {
                    "recent_evidence_count": len([
                        e for e in (proofs.supporting_evidence + proofs.refuting_evidence)
                        if e.extracted_at > datetime.now() - timedelta(days=30)
                    ]),
                    "temporal_consistency": temporal_score
                },
                "validation_method": "comprehensive_analysis",
                "processing_time": (datetime.now() - start_time).total_seconds()
            }
            
            result = ValidationResult(
                claim=proofs.claim,
                verdict=verdict,
                confidence=confidence,
                evidence_score=evidence_score,
                credibility_score=credibility_score,
                consistency_score=consistency_score,
                bias_score=bias_score,
                temporal_score=temporal_score,
                supporting_strength=supporting_strength,
                refuting_strength=refuting_strength,
                validation_details=validation_details,
                validated_at=datetime.now()
            )
            
            logger.info(f"Proof validation completed: {verdict} (confidence: {confidence:.2f})")
            return result
            
        except Exception as e:
            logger.error(f"Proof validation failed: {e}")
            raise ProofValidatorError(f"Validation failed: {e}")
    
    async def _analyze_evidence_pieces(self, proofs: AggregatedProofs) -> List[EvidenceAnalysis]:
        """Analyze individual pieces of evidence."""
        analyses = []
        
        all_evidence = (
            proofs.supporting_evidence + 
            proofs.refuting_evidence + 
            proofs.neutral_evidence
        )
        
        for evidence in all_evidence:
            analysis = await self._analyze_single_evidence(evidence)
            analyses.append(analysis)
        
        return analyses
    
    async def _analyze_single_evidence(self, evidence: ProofEvidence) -> EvidenceAnalysis:
        """Analyze a single piece of evidence."""
        content = evidence.content.lower()
        
        # Quality score based on content characteristics
        quality_score = self._calculate_content_quality(evidence.content)
        
        # Detect bias indicators
        bias_indicators = []
        for pattern in self.bias_indicators:
            if re.search(pattern, content, re.IGNORECASE):
                bias_indicators.append(pattern)
        
        # Detect factual indicators
        factual_indicators = []
        for pattern in self.factual_indicators:
            if re.search(pattern, content, re.IGNORECASE):
                factual_indicators.append(pattern)
        
        # Sentiment analysis
        sentiment_score = self._analyze_sentiment(evidence.content)
        
        # Temporal relevance
        temporal_relevance = self._calculate_temporal_relevance(evidence)
        
        # Source authority
        source_authority = self._calculate_source_authority(evidence.source_url)
        
        return EvidenceAnalysis(
            evidence=evidence,
            quality_score=quality_score,
            bias_indicators=bias_indicators,
            sentiment_score=sentiment_score,
            factual_indicators=factual_indicators,
            temporal_relevance=temporal_relevance,
            source_authority=source_authority
        )
    
    def _calculate_content_quality(self, content: str) -> float:
        """Calculate quality score for content."""
        if not content:
            return 0.0
        
        quality_factors = []
        
        # Length factor (not too short, not too long)
        length = len(content)
        if 100 <= length <= 2000:
            quality_factors.append(0.8)
        elif 50 <= length < 100 or 2000 < length <= 5000:
            quality_factors.append(0.6)
        else:
            quality_factors.append(0.3)
        
        # Sentence structure (presence of proper sentences)
        sentences = re.split(r'[.!?]+', content)
        avg_sentence_length = sum(len(s.split()) for s in sentences) / max(1, len(sentences))
        if 10 <= avg_sentence_length <= 30:
            quality_factors.append(0.8)
        else:
            quality_factors.append(0.5)
        
        # Capitalization (proper use of capitals)
        caps_ratio = sum(1 for c in content if c.isupper()) / max(1, len(content))
        if caps_ratio < 0.1:  # Not too many caps
            quality_factors.append(0.8)
        else:
            quality_factors.append(0.4)
        
        # Punctuation (reasonable use)
        punct_ratio = sum(1 for c in content if c in '!?') / max(1, len(content))
        if punct_ratio < 0.02:  # Not excessive punctuation
            quality_factors.append(0.8)
        else:
            quality_factors.append(0.5)
        
        # Spelling and grammar (basic check)
        words = content.split()
        if len(words) > 10:
            # Simple heuristic: check for common misspellings
            misspelling_indicators = ['teh', 'recieve', 'seperate', 'definately']
            misspelling_count = sum(1 for word in words if word.lower() in misspelling_indicators)
            if misspelling_count == 0:
                quality_factors.append(0.8)
            else:
                quality_factors.append(0.4)
        
        return statistics.mean(quality_factors) if quality_factors else 0.5
    
    def _analyze_sentiment(self, content: str) -> float:
        """Analyze sentiment of content."""
        if not TEXTBLOB_AVAILABLE:
            # Simple sentiment analysis without TextBlob
            positive_words = ['good', 'great', 'excellent', 'positive', 'true', 'correct', 'accurate']
            negative_words = ['bad', 'terrible', 'false', 'wrong', 'incorrect', 'fake', 'misleading']
            
            content_lower = content.lower()
            positive_count = sum(1 for word in positive_words if word in content_lower)
            negative_count = sum(1 for word in negative_words if word in content_lower)
            
            if positive_count + negative_count == 0:
                return 0.0
            
            return (positive_count - negative_count) / (positive_count + negative_count)
        
        try:
            blob = TextBlob(content)
            return blob.sentiment.polarity  # Returns value between -1 and 1
        except Exception:
            return 0.0
    
    def _calculate_temporal_relevance(self, evidence: ProofEvidence) -> float:
        """Calculate temporal relevance of evidence."""
        if not evidence.extracted_at:
            return 0.5  # Default for unknown dates
        
        days_old = (datetime.now() - evidence.extracted_at).days
        
        # More recent evidence is generally more relevant
        if days_old <= 7:
            return 1.0
        elif days_old <= 30:
            return 0.8
        elif days_old <= 90:
            return 0.6
        elif days_old <= 365:
            return 0.4
        else:
            return 0.2
    
    def _calculate_source_authority(self, url: str) -> float:
        """Calculate authority score for source URL."""
        if not url:
            return 0.5
        
        try:
            from urllib.parse import urlparse
            domain = urlparse(url).netloc.lower()
            
            # Remove www. prefix
            if domain.startswith("www."):
                domain = domain[4:]
            
            # Check for exact matches
            for auth_domain, score in self.authority_domains.items():
                if domain == auth_domain or domain.endswith('.' + auth_domain):
                    return score
            
            # Check for domain extensions
            if domain.endswith('.edu') or domain.endswith('.gov'):
                return 0.8
            elif domain.endswith('.org'):
                return 0.6
            
            return 0.5  # Default authority
            
        except Exception:
            return 0.5
    
    def _calculate_evidence_score(self, analyses: List[EvidenceAnalysis]) -> float:
        """Calculate overall evidence quality score."""
        if not analyses:
            return 0.0
        
        # Average quality scores
        quality_scores = [a.quality_score for a in analyses]
        avg_quality = statistics.mean(quality_scores)
        
        # Bonus for factual indicators
        factual_bonus = sum(len(a.factual_indicators) for a in analyses) / len(analyses) * 0.1
        
        # Penalty for unreliable indicators
        unreliable_penalty = 0.0
        for analysis in analyses:
            content = analysis.evidence.content.lower()
            for pattern in self.unreliable_indicators:
                if re.search(pattern, content, re.IGNORECASE):
                    unreliable_penalty += 0.05
        
        unreliable_penalty = min(0.3, unreliable_penalty)  # Cap penalty
        
        return max(0.0, min(1.0, avg_quality + factual_bonus - unreliable_penalty))
    
    def _calculate_credibility_score(
        self, 
        proofs: AggregatedProofs, 
        analyses: List[EvidenceAnalysis]
    ) -> float:
        """Calculate overall credibility score."""
        if not analyses:
            return 0.0
        
        # Source credibility from proofs
        source_credibility = proofs.credible_sources_count / max(1, proofs.total_sources)
        
        # Authority scores from analyses
        authority_scores = [a.source_authority for a in analyses]
        avg_authority = statistics.mean(authority_scores)
        
        # Fact-checking source bonus
        fact_check_bonus = proofs.fact_check_sources_count / max(1, proofs.total_sources) * 0.2
        
        return min(1.0, source_credibility * 0.5 + avg_authority * 0.4 + fact_check_bonus * 0.1)
    
    def _calculate_consistency_score(
        self, 
        proofs: AggregatedProofs, 
        analyses: List[EvidenceAnalysis]
    ) -> float:
        """Calculate consistency score across evidence."""
        supporting_count = len(proofs.supporting_evidence)
        refuting_count = len(proofs.refuting_evidence)
        total_directional = supporting_count + refuting_count
        
        if total_directional == 0:
            return 0.5  # Neutral when no directional evidence
        
        # Consistency is higher when evidence points in the same direction
        max_direction = max(supporting_count, refuting_count)
        consistency = max_direction / total_directional
        
        # Bonus for high-quality consistent evidence
        if analyses:
            directional_analyses = [
                a for a in analyses 
                if a.evidence.evidence_type in ['supporting', 'refuting']
            ]
            
            if directional_analyses:
                avg_quality = statistics.mean([a.quality_score for a in directional_analyses])
                consistency = consistency * (0.7 + avg_quality * 0.3)
        
        return min(1.0, consistency)
    
    def _calculate_bias_score(self, analyses: List[EvidenceAnalysis]) -> float:
        """Calculate bias score (lower is better)."""
        if not analyses:
            return 0.5
        
        total_bias_indicators = sum(len(a.bias_indicators) for a in analyses)
        avg_bias_indicators = total_bias_indicators / len(analyses)
        
        # Convert to score where 0 = high bias, 1 = low bias
        bias_score = max(0.0, 1.0 - (avg_bias_indicators * 0.1))
        
        return bias_score
    
    def _calculate_temporal_score(self, proofs: AggregatedProofs) -> float:
        """Calculate temporal relevance score."""
        all_evidence = (
            proofs.supporting_evidence + 
            proofs.refuting_evidence + 
            proofs.neutral_evidence
        )
        
        if not all_evidence:
            return 0.5
        
        # Calculate average temporal relevance
        temporal_scores = []
        for evidence in all_evidence:
            if evidence.extracted_at:
                days_old = (datetime.now() - evidence.extracted_at).days
                if days_old <= 30:
                    temporal_scores.append(1.0)
                elif days_old <= 90:
                    temporal_scores.append(0.8)
                elif days_old <= 365:
                    temporal_scores.append(0.6)
                else:
                    temporal_scores.append(0.3)
            else:
                temporal_scores.append(0.5)
        
        return statistics.mean(temporal_scores)
    
    def _calculate_supporting_strength(
        self, 
        proofs: AggregatedProofs, 
        analyses: List[EvidenceAnalysis]
    ) -> float:
        """Calculate strength of supporting evidence."""
        supporting_analyses = [
            a for a in analyses 
            if a.evidence.evidence_type == 'supporting'
        ]
        
        if not supporting_analyses:
            return 0.0
        
        # Weight by quality and credibility
        weighted_strength = 0.0
        total_weight = 0.0
        
        for analysis in supporting_analyses:
            weight = (
                analysis.quality_score * 0.4 +
                analysis.evidence.credibility_score * 0.3 +
                analysis.source_authority * 0.2 +
                analysis.temporal_relevance * 0.1
            )
            weighted_strength += weight
            total_weight += 1.0
        
        return weighted_strength / total_weight if total_weight > 0 else 0.0
    
    def _calculate_refuting_strength(
        self, 
        proofs: AggregatedProofs, 
        analyses: List[EvidenceAnalysis]
    ) -> float:
        """Calculate strength of refuting evidence."""
        refuting_analyses = [
            a for a in analyses 
            if a.evidence.evidence_type == 'refuting'
        ]
        
        if not refuting_analyses:
            return 0.0
        
        # Weight by quality and credibility
        weighted_strength = 0.0
        total_weight = 0.0
        
        for analysis in refuting_analyses:
            weight = (
                analysis.quality_score * 0.4 +
                analysis.evidence.credibility_score * 0.3 +
                analysis.source_authority * 0.2 +
                analysis.temporal_relevance * 0.1
            )
            weighted_strength += weight
            total_weight += 1.0
        
        return weighted_strength / total_weight if total_weight > 0 else 0.0
    
    def _determine_verdict(
        self,
        supporting_strength: float,
        refuting_strength: float,
        evidence_score: float,
        credibility_score: float,
        consistency_score: float,
        bias_score: float
    ) -> Tuple[str, float]:
        """Determine final verdict and confidence."""
        # Calculate overall score
        overall_score = (
            evidence_score * self.weights["evidence_quality"] +
            credibility_score * self.weights["source_credibility"] +
            consistency_score * self.weights["consistency"] +
            bias_score * self.weights["bias_penalty"]
        )
        
        # Determine direction based on evidence strength
        strength_difference = supporting_strength - refuting_strength
        
        # Determine verdict
        if abs(strength_difference) < 0.2:  # Very close
            verdict = "AMBIGUOUS"
            confidence = overall_score * 0.6  # Lower confidence for ambiguous
        elif strength_difference > 0.2:  # Supporting evidence stronger
            verdict = "REAL"
            confidence = overall_score * (0.7 + supporting_strength * 0.3)
        else:  # Refuting evidence stronger
            verdict = "FAKE"
            confidence = overall_score * (0.7 + refuting_strength * 0.3)
        
        # Apply minimum thresholds
        if confidence < self.thresholds["low_confidence"]:
            verdict = "AMBIGUOUS"
        
        # Ensure confidence is within bounds
        confidence = max(0.0, min(1.0, confidence))
        
        return verdict, confidence
    
    def get_validation_explanation(self, result: ValidationResult) -> Dict[str, Any]:
        """Generate human-readable explanation of validation result."""
        explanation = {
            "verdict_summary": f"Claim classified as {result.verdict} with {result.confidence:.1%} confidence",
            "key_factors": [],
            "evidence_breakdown": {
                "supporting_evidence": result.validation_details["evidence_count"]["supporting"],
                "refuting_evidence": result.validation_details["evidence_count"]["refuting"],
                "neutral_evidence": result.validation_details["evidence_count"]["neutral"]
            },
            "quality_assessment": {
                "evidence_quality": f"{result.evidence_score:.1%}",
                "source_credibility": f"{result.credibility_score:.1%}",
                "consistency": f"{result.consistency_score:.1%}"
            },
            "recommendations": []
        }
        
        # Add key factors
        if result.supporting_strength > 0.7:
            explanation["key_factors"].append("Strong supporting evidence found")
        if result.refuting_strength > 0.7:
            explanation["key_factors"].append("Strong refuting evidence found")
        if result.credibility_score > 0.8:
            explanation["key_factors"].append("High-credibility sources")
        if result.bias_score < 0.5:
            explanation["key_factors"].append("Potential bias detected in sources")
        
        # Add recommendations
        if result.confidence < 0.6:
            explanation["recommendations"].append("Seek additional sources for verification")
        if result.validation_details["source_analysis"]["fact_check_sources"] == 0:
            explanation["recommendations"].append("Consult dedicated fact-checking sources")
        if result.temporal_score < 0.5:
            explanation["recommendations"].append("Look for more recent information")
        
        return explanation


## Suggestions for Upgrade:
# 1. Integrate advanced ML models (BERT, RoBERTa) for better content analysis and bias detection
# 2. Add support for cross-referencing claims against known fact-checking databases
# 3. Implement network analysis to detect coordinated inauthentic behavior across sources
# 4. Add multimodal validation capabilities for images, videos, and audio evidence