#!/usr/bin/env python3
"""
Advanced Cross-Verification Engine for Fake News Detection
Analyzes multiple sources and evidence to determine content authenticity
"""

import re
import json
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from urllib.parse import urlparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SourceCredibilityAnalyzer:
    """Analyzes source credibility based on multiple factors"""
    
    # Trusted news sources with credibility scores (0.0-1.0)
    TRUSTED_SOURCES = {
        'reuters.com': 0.95,
        'ap.org': 0.95,
        'bbc.com': 0.92,
        'cnn.com': 0.85,
        'nytimes.com': 0.90,
        'washingtonpost.com': 0.88,
        'theguardian.com': 0.87,
        'npr.org': 0.90,
        'factcheck.org': 0.95,
        'snopes.com': 0.92,
        'politifact.com': 0.93,
        'mediabiasfactcheck.com': 0.88,
        'nature.com': 0.98,
        'science.org': 0.98,
        'who.int': 0.95,
        'cdc.gov': 0.95,
        'gov.uk': 0.90,
        'europa.eu': 0.88
    }
    
    # Known unreliable sources with low credibility scores
    UNRELIABLE_SOURCES = {
        'infowars.com': 0.15,
        'breitbart.com': 0.35,
        'naturalnews.com': 0.20,
        'beforeitsnews.com': 0.10,
        'worldnewsdailyreport.com': 0.05,
        'theonion.com': 0.05,  # Satire
        'clickhole.com': 0.05,  # Satire
    }
    
    def __init__(self):
        self.domain_cache = {}
    
    def analyze_source_credibility(self, url: str, source_name: str = "") -> Dict:
        """Analyze credibility of a source URL"""
        try:
            domain = urlparse(url).netloc.lower()
            domain = domain.replace('www.', '')
            
            # Check trusted sources
            if domain in self.TRUSTED_SOURCES:
                credibility_score = self.TRUSTED_SOURCES[domain]
                credibility_level = "High"
            elif domain in self.UNRELIABLE_SOURCES:
                credibility_score = self.UNRELIABLE_SOURCES[domain]
                credibility_level = "Low"
            else:
                # Analyze domain characteristics
                credibility_score = self._analyze_domain_characteristics(domain, source_name)
                if credibility_score >= 0.7:
                    credibility_level = "High"
                elif credibility_score >= 0.4:
                    credibility_level = "Medium"
                else:
                    credibility_level = "Low"
            
            return {
                'domain': domain,
                'credibility_score': credibility_score,
                'credibility_level': credibility_level,
                'is_trusted': credibility_score >= 0.7,
                'is_unreliable': credibility_score <= 0.3
            }
            
        except Exception as e:
            logger.error(f"Error analyzing source credibility: {e}")
            return {
                'domain': 'unknown',
                'credibility_score': 0.5,
                'credibility_level': 'Medium',
                'is_trusted': False,
                'is_unreliable': False
            }
    
    def _analyze_domain_characteristics(self, domain: str, source_name: str) -> float:
        """Analyze domain characteristics for credibility scoring"""
        score = 0.5  # Base score
        
        # Government domains
        if domain.endswith('.gov') or domain.endswith('.gov.uk'):
            score += 0.3
        
        # Educational domains
        elif domain.endswith('.edu') or domain.endswith('.ac.uk'):
            score += 0.25
        
        # Organization domains
        elif domain.endswith('.org'):
            score += 0.1
        
        # News-related keywords
        news_keywords = ['news', 'times', 'post', 'herald', 'tribune', 'journal', 'gazette']
        if any(keyword in domain for keyword in news_keywords):
            score += 0.1
        
        # Suspicious characteristics
        suspicious_patterns = [
            r'\d+',  # Numbers in domain
            r'(fake|hoax|conspiracy|truth|real|exposed)',  # Suspicious keywords
            r'(blog|wordpress|blogspot)',  # Blog platforms
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, domain, re.IGNORECASE):
                score -= 0.15
        
        # Very short domains (likely suspicious)
        if len(domain.split('.')[0]) <= 4:
            score -= 0.1
        
        # Multiple hyphens (suspicious)
        if domain.count('-') >= 2:
            score -= 0.1
        
        return max(0.0, min(1.0, score))

class EvidenceAnalyzer:
    """Analyzes evidence quality and relevance"""
    
    def __init__(self):
        self.credibility_analyzer = SourceCredibilityAnalyzer()
    
    def analyze_evidence_quality(self, evidence: Dict) -> Dict:
        """Analyze quality of individual evidence"""
        try:
            # Source credibility analysis
            source_analysis = self.credibility_analyzer.analyze_source_credibility(
                evidence.get('link', ''), 
                evidence.get('source', '')
            )
            
            # Content quality analysis
            content_score = self._analyze_content_quality(
                evidence.get('title', ''),
                evidence.get('snippet', '')
            )
            
            # Recency analysis
            recency_score = self._analyze_recency(evidence.get('date', ''))
            
            # Calculate overall quality score
            quality_score = (
                source_analysis['credibility_score'] * 0.5 +
                content_score * 0.3 +
                recency_score * 0.2
            )
            
            return {
                'quality_score': quality_score,
                'source_credibility': source_analysis,
                'content_quality': content_score,
                'recency_score': recency_score,
                'is_high_quality': quality_score >= 0.7,
                'is_low_quality': quality_score <= 0.3
            }
            
        except Exception as e:
            logger.error(f"Error analyzing evidence quality: {e}")
            return {
                'quality_score': 0.5,
                'source_credibility': {'credibility_score': 0.5},
                'content_quality': 0.5,
                'recency_score': 0.5,
                'is_high_quality': False,
                'is_low_quality': False
            }
    
    def _analyze_content_quality(self, title: str, snippet: str) -> float:
        """Analyze content quality based on text characteristics"""
        score = 0.5
        content = f"{title} {snippet}".lower()
        
        # Positive indicators
        quality_indicators = [
            r'(study|research|analysis|investigation|report)',
            r'(according to|based on|data shows|statistics)',
            r'(expert|professor|scientist|researcher)',
            r'(published|peer.reviewed|journal)'
        ]
        
        for indicator in quality_indicators:
            if re.search(indicator, content):
                score += 0.1
        
        # Negative indicators
        poor_quality_indicators = [
            r'(shocking|amazing|unbelievable|incredible)',
            r'(click here|you won\'t believe|must see)',
            r'(conspiracy|cover.up|they don\'t want)',
            r'(!!!|\?\?\?)',  # Excessive punctuation
            r'(ALL CAPS|[A-Z]{10,})'  # Excessive caps
        ]
        
        for indicator in poor_quality_indicators:
            if re.search(indicator, content):
                score -= 0.15
        
        # Length analysis
        if len(content) < 50:  # Too short
            score -= 0.1
        elif len(content) > 200:  # Good length
            score += 0.05
        
        return max(0.0, min(1.0, score))
    
    def _analyze_recency(self, date_str: str) -> float:
        """Analyze recency of the source"""
        try:
            if not date_str:
                return 0.5  # Unknown date
            
            # Parse date (simplified - would need more robust parsing)
            # For now, return moderate score
            return 0.6
            
        except Exception:
            return 0.5

class CrossVerificationEngine:
    """Main cross-verification engine"""
    
    def __init__(self):
        self.evidence_analyzer = EvidenceAnalyzer()
        self.credibility_analyzer = SourceCredibilityAnalyzer()
    
    def cross_verify_content(self, serper_report: Dict) -> Dict:
        """Perform comprehensive cross-verification of content"""
        try:
            # Extract evidence from report
            supporting_evidence = serper_report.get('analysis', {}).get('supportingEvidence', [])
            contradicting_evidence = serper_report.get('analysis', {}).get('contradictingEvidence', [])
            neutral_evidence = serper_report.get('analysis', {}).get('neutralEvidence', [])
            
            # Analyze each type of evidence
            supporting_analysis = self._analyze_evidence_group(supporting_evidence, 'supporting')
            contradicting_analysis = self._analyze_evidence_group(contradicting_evidence, 'contradicting')
            neutral_analysis = self._analyze_evidence_group(neutral_evidence, 'neutral')
            
            # Calculate overall verification metrics
            verification_result = self._calculate_verification_result(
                supporting_analysis, contradicting_analysis, neutral_analysis
            )
            
            # Generate final verdict
            final_verdict = self._generate_final_verdict(verification_result)
            
            return {
                'verification_result': verification_result,
                'final_verdict': final_verdict,
                'evidence_analysis': {
                    'supporting': supporting_analysis,
                    'contradicting': contradicting_analysis,
                    'neutral': neutral_analysis
                },
                'confidence_score': final_verdict['confidence'],
                'is_fake': final_verdict['verdict'] in ['FAKE', 'LIKELY FAKE'],
                'is_real': final_verdict['verdict'] in ['REAL', 'LIKELY REAL'],
                'is_mixed': final_verdict['verdict'] == 'MIXED EVIDENCE',
                'is_inconclusive': 'INCONCLUSIVE' in final_verdict['verdict'],
                'has_insufficient_evidence': final_verdict['verdict'] == 'INSUFFICIENT EVIDENCE',
                'reasoning': final_verdict['reasoning'],
                'source_quality_summary': final_verdict.get('source_analysis', {})
            }
            
        except Exception as e:
            logger.error(f"Error in cross-verification: {e}")
            return self._get_default_result()
    
    def _analyze_evidence_group(self, evidence_list: List[Dict], evidence_type: str) -> Dict:
        """Analyze a group of evidence (supporting/contradicting/neutral)"""
        if not evidence_list:
            return {
                'count': 0,
                'average_quality': 0.0,
                'high_quality_count': 0,
                'trusted_sources_count': 0,
                'total_credibility_score': 0.0,
                'weight': 0.0
            }
        
        total_quality = 0.0
        high_quality_count = 0
        trusted_sources_count = 0
        total_credibility = 0.0
        
        for evidence in evidence_list:
            analysis = self.evidence_analyzer.analyze_evidence_quality(evidence)
            
            total_quality += analysis['quality_score']
            total_credibility += analysis['source_credibility']['credibility_score']
            
            if analysis['is_high_quality']:
                high_quality_count += 1
            
            if analysis['source_credibility']['is_trusted']:
                trusted_sources_count += 1
        
        count = len(evidence_list)
        average_quality = total_quality / count
        average_credibility = total_credibility / count
        
        # Calculate weight based on quality and credibility
        weight = (average_quality * 0.6 + average_credibility * 0.4) * count
        
        return {
            'count': count,
            'average_quality': average_quality,
            'average_credibility': average_credibility,
            'high_quality_count': high_quality_count,
            'trusted_sources_count': trusted_sources_count,
            'total_credibility_score': total_credibility,
            'weight': weight
        }
    
    def _calculate_verification_result(self, supporting: Dict, contradicting: Dict, neutral: Dict) -> Dict:
        """Calculate overall verification metrics"""
        # Calculate weighted scores
        supporting_weight = supporting['weight']
        contradicting_weight = contradicting['weight']
        neutral_weight = neutral['weight'] * 0.3  # Neutral evidence has less impact
        
        total_weight = supporting_weight + contradicting_weight + neutral_weight
        
        if total_weight == 0:
            return {
                'supporting_score': 0.0,
                'contradicting_score': 0.0,
                'neutral_score': 0.0,
                'confidence': 0.1,
                'evidence_strength': 'Very Weak'
            }
        
        # Normalize scores
        supporting_score = supporting_weight / total_weight
        contradicting_score = contradicting_weight / total_weight
        neutral_score = neutral_weight / total_weight
        
        # Calculate confidence based on evidence quality and quantity
        total_sources = supporting['count'] + contradicting['count'] + neutral['count']
        quality_factor = (
            supporting['average_quality'] * supporting['count'] +
            contradicting['average_quality'] * contradicting['count'] +
            neutral['average_quality'] * neutral['count']
        ) / max(total_sources, 1)
        
        # Confidence increases with more high-quality sources
        confidence = min(0.95, max(0.1, quality_factor * 0.7 + (total_sources / 20) * 0.3))
        
        # Determine evidence strength
        if total_sources >= 10 and quality_factor >= 0.7:
            evidence_strength = 'Very Strong'
        elif total_sources >= 5 and quality_factor >= 0.6:
            evidence_strength = 'Strong'
        elif total_sources >= 3 and quality_factor >= 0.5:
            evidence_strength = 'Moderate'
        elif total_sources >= 1:
            evidence_strength = 'Weak'
        else:
            evidence_strength = 'Very Weak'
        
        return {
            'supporting_score': supporting_score,
            'contradicting_score': contradicting_score,
            'neutral_score': neutral_score,
            'confidence': confidence,
            'evidence_strength': evidence_strength,
            'total_sources': total_sources,
            'quality_factor': quality_factor
        }
    
    def _generate_final_verdict(self, verification_result: Dict) -> Dict:
        """Generate final FAKE/REAL verdict with enhanced reasoning and source analysis"""
        supporting_score = verification_result['supporting_score']
        contradicting_score = verification_result['contradicting_score']
        confidence = verification_result['confidence']
        evidence_strength = verification_result['evidence_strength']
        total_sources = verification_result['total_sources']
        quality_factor = verification_result['quality_factor']
        
        # Enhanced decision thresholds based on source quality
        STRONG_THRESHOLD = 0.7
        MODERATE_THRESHOLD = 0.55
        WEAK_THRESHOLD = 0.35
        
        reasoning_parts = []
        
        # Calculate evidence balance for better decision making
        evidence_balance = supporting_score - contradicting_score
        
        # Determine verdict based on enhanced evidence analysis
        if contradicting_score >= STRONG_THRESHOLD and evidence_balance < -0.3:
            verdict = 'FAKE'
            verdict_confidence = min(0.95, confidence + 0.15)
            reasoning_parts.append(f"Strong contradicting evidence ({contradicting_score:.1%}) with clear evidence imbalance")
            
        elif supporting_score >= STRONG_THRESHOLD and evidence_balance > 0.3:
            verdict = 'REAL'
            verdict_confidence = min(0.95, confidence + 0.15)
            reasoning_parts.append(f"Strong supporting evidence ({supporting_score:.1%}) with clear evidence imbalance")
            
        elif contradicting_score >= MODERATE_THRESHOLD and evidence_balance < -0.2:
            verdict = 'LIKELY FAKE'
            verdict_confidence = min(0.85, confidence + 0.1)
            reasoning_parts.append(f"Moderate contradicting evidence ({contradicting_score:.1%}) outweighs support")
            
        elif supporting_score >= MODERATE_THRESHOLD and evidence_balance > 0.2:
            verdict = 'LIKELY REAL'
            verdict_confidence = min(0.85, confidence + 0.1)
            reasoning_parts.append(f"Moderate supporting evidence ({supporting_score:.1%}) outweighs contradiction")
            
        elif abs(evidence_balance) <= 0.2 and (supporting_score + contradicting_score) > 0.4:
            verdict = 'MIXED EVIDENCE'
            verdict_confidence = confidence * 0.8
            reasoning_parts.append(f"Conflicting evidence with balanced scores (S:{supporting_score:.1%}, C:{contradicting_score:.1%})")
            
        elif total_sources < 3 or quality_factor < 0.4:
            verdict = 'INSUFFICIENT EVIDENCE'
            verdict_confidence = max(0.1, confidence * 0.6)
            reasoning_parts.append(f"Insufficient high-quality sources ({total_sources} sources, {quality_factor:.1%} avg quality)")
            
        else:
            # Default to inconclusive with safety bias
            verdict = 'INCONCLUSIVE (LEAN FAKE)'
            verdict_confidence = max(0.1, confidence - 0.2)
            reasoning_parts.append("Unclear evidence pattern - applying safety bias")
        
        # Add detailed evidence analysis to reasoning
        reasoning_parts.append(f"Evidence strength: {evidence_strength}")
        reasoning_parts.append(f"Source quality: {quality_factor:.1%} avg from {total_sources} sources")
        
        # Determine confidence level with more granular categories
        if verdict_confidence >= 0.85:
            confidence_level = 'Very High'
        elif verdict_confidence >= 0.7:
            confidence_level = 'High'
        elif verdict_confidence >= 0.55:
            confidence_level = 'Moderate'
        elif verdict_confidence >= 0.35:
            confidence_level = 'Low'
        else:
            confidence_level = 'Very Low'
        
        reasoning_parts.append(f"Confidence level: {confidence_level}")
        
        return {
            'verdict': verdict,
            'confidence': round(verdict_confidence, 3),
            'confidence_level': confidence_level,
            'reasoning': ' | '.join(reasoning_parts),
            'is_conclusive': verdict in ['FAKE', 'REAL'] and verdict_confidence >= 0.7,
            'evidence_balance': round(evidence_balance, 3),
            'source_analysis': {
                'total_sources': total_sources,
                'quality_factor': round(quality_factor, 3),
                'evidence_strength': evidence_strength
            }
        }
    
    def _generate_enhanced_reasoning(self, verification_result: Dict, verdict: str, evidence_balance: float, total_evidence_strength: float) -> str:
        """Generate enhanced human-readable reasoning for the verdict"""
        reasoning_parts = []
        
        # Evidence summary with percentages
        supporting = verification_result['supporting_score']
        contradicting = verification_result['contradicting_score']
        neutral = verification_result['neutral_score']
        
        reasoning_parts.append(f"Evidence Analysis: {supporting:.1%} Supporting, {contradicting:.1%} Contradicting, {neutral:.1%} Neutral")
        
        # Detailed balance analysis
        if evidence_balance > 0.4:
            reasoning_parts.append("Strong evidence favors authenticity")
        elif evidence_balance > 0.2:
            reasoning_parts.append("Moderate evidence supports authenticity")
        elif evidence_balance < -0.4:
            reasoning_parts.append("Strong evidence indicates falsification")
        elif evidence_balance < -0.2:
            reasoning_parts.append("Moderate evidence suggests falsification")
        elif abs(evidence_balance) <= 0.2:
            reasoning_parts.append("Evidence is conflicted or balanced")
        
        # Source quality assessment
        if total_evidence_strength > 0.7:
            reasoning_parts.append("High-quality source verification")
        elif total_evidence_strength > 0.5:
            reasoning_parts.append("Moderate source reliability")
        elif total_evidence_strength > 0.3:
            reasoning_parts.append("Limited source verification")
        else:
            reasoning_parts.append("Insufficient reliable sources")
        
        # Verdict-specific reasoning
        if 'FAKE' in verdict:
            reasoning_parts.append("Content appears to contain false information")
        elif 'REAL' in verdict:
            reasoning_parts.append("Content appears to be authentic")
        elif 'MIXED' in verdict:
            reasoning_parts.append("Content contains both accurate and questionable elements")
        elif 'INSUFFICIENT' in verdict:
            reasoning_parts.append("Not enough reliable sources for definitive assessment")
        
        return " | ".join(reasoning_parts)
    
    def _get_default_result(self) -> Dict:
        """Return default result for error cases"""
        return {
            'verification_result': {
                'supporting_score': 0.0,
                'contradicting_score': 0.0,
                'neutral_score': 0.0,
                'confidence': 0.1,
                'evidence_strength': 'Very Weak'
            },
            'final_verdict': {
                'verdict': 'UNKNOWN',
                'confidence': 0.1,
                'confidence_level': 'Low',
                'reasoning': 'Error in analysis',
                'is_conclusive': False
            },
            'evidence_analysis': {
                'supporting': {'count': 0, 'weight': 0.0},
                'contradicting': {'count': 0, 'weight': 0.0},
                'neutral': {'count': 0, 'weight': 0.0}
            },
            'confidence_score': 0.1,
            'is_fake': False,
            'is_real': False,
            'reasoning': 'Analysis error occurred'
        }

# Example usage and testing
if __name__ == "__main__":
    # Test the cross-verification engine
    engine = CrossVerificationEngine()
    
    # Sample SerperAPI report
    sample_report = {
        'analysis': {
            'supportingEvidence': [
                {
                    'title': 'Reuters confirms the news story',
                    'snippet': 'According to official sources and verified data...',
                    'link': 'https://reuters.com/article/123',
                    'source': 'Reuters'
                }
            ],
            'contradictingEvidence': [
                {
                    'title': 'Fact-check reveals false information',
                    'snippet': 'Investigation shows the claim is misleading...',
                    'link': 'https://factcheck.org/article/456',
                    'source': 'FactCheck.org'
                },
                {
                    'title': 'Snopes debunks the claim',
                    'snippet': 'Our research indicates this is false...',
                    'link': 'https://snopes.com/article/789',
                    'source': 'Snopes'
                }
            ],
            'neutralEvidence': []
        }
    }
    
    result = engine.cross_verify_content(sample_report)
    print(json.dumps(result, indent=2))