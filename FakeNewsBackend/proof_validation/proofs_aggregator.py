import asyncio
import logging
from typing import List, Dict, Optional, Any, Tuple, Set
from dataclasses import dataclass
from datetime import datetime, timedelta
import re
import json
from urllib.parse import urlparse
from collections import defaultdict

# Content extraction libraries
try:
    import aiohttp
    from aiohttp import ClientSession, ClientTimeout
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    logging.warning("aiohttp not available. Content extraction will be limited.")

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False
    logging.warning("BeautifulSoup not available. HTML parsing will be limited.")

logger = logging.getLogger(__name__)


@dataclass
class ProofEvidence:
    """Represents a piece of evidence supporting or refuting a claim."""
    content: str
    source_url: str
    source_name: str
    evidence_type: str  # 'supporting', 'refuting', 'neutral', 'unclear'
    confidence: float
    relevance_score: float
    credibility_score: float
    extracted_at: datetime
    context: Optional[str] = None
    quotes: List[str] = None
    
    def __post_init__(self):
        if self.quotes is None:
            self.quotes = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "content": self.content,
            "source_url": self.source_url,
            "source_name": self.source_name,
            "evidence_type": self.evidence_type,
            "confidence": self.confidence,
            "relevance_score": self.relevance_score,
            "credibility_score": self.credibility_score,
            "extracted_at": self.extracted_at.isoformat(),
            "context": self.context,
            "quotes": self.quotes
        }


@dataclass
class AggregatedProofs:
    """Container for all aggregated evidence about a claim."""
    claim: str
    supporting_evidence: List[ProofEvidence]
    refuting_evidence: List[ProofEvidence]
    neutral_evidence: List[ProofEvidence]
    total_sources: int
    credible_sources_count: int
    fact_check_sources_count: int
    aggregation_confidence: float
    aggregated_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "claim": self.claim,
            "supporting_evidence": [e.to_dict() for e in self.supporting_evidence],
            "refuting_evidence": [e.to_dict() for e in self.refuting_evidence],
            "neutral_evidence": [e.to_dict() for e in self.neutral_evidence],
            "total_sources": self.total_sources,
            "credible_sources_count": self.credible_sources_count,
            "fact_check_sources_count": self.fact_check_sources_count,
            "aggregation_confidence": self.aggregation_confidence,
            "aggregated_at": self.aggregated_at.isoformat()
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics of aggregated proofs."""
        return {
            "total_evidence": len(self.supporting_evidence) + len(self.refuting_evidence) + len(self.neutral_evidence),
            "supporting_count": len(self.supporting_evidence),
            "refuting_count": len(self.refuting_evidence),
            "neutral_count": len(self.neutral_evidence),
            "support_ratio": len(self.supporting_evidence) / max(1, len(self.supporting_evidence) + len(self.refuting_evidence)),
            "avg_credibility": self._calculate_average_credibility(),
            "fact_check_ratio": self.fact_check_sources_count / max(1, self.total_sources)
        }
    
    def _calculate_average_credibility(self) -> float:
        """Calculate average credibility across all evidence."""
        all_evidence = self.supporting_evidence + self.refuting_evidence + self.neutral_evidence
        if not all_evidence:
            return 0.0
        
        return sum(e.credibility_score for e in all_evidence) / len(all_evidence)


class ProofsAggregatorError(Exception):
    """Custom exception for proofs aggregator errors."""
    pass


class ProofsAggregator:
    """Aggregates and organizes evidence from multiple sources for fact-checking."""
    
    def __init__(self):
        """Initialize proofs aggregator."""
        self.session = None
        self.timeout = ClientTimeout(total=30)
        
        # Content extraction patterns
        self.fact_check_indicators = self._load_fact_check_indicators()
        self.supporting_patterns = self._load_supporting_patterns()
        self.refuting_patterns = self._load_refuting_patterns()
        
        # Configuration
        self.config = {
            "max_content_length": 5000,
            "min_content_length": 50,
            "extract_full_content": True,
            "extract_quotes": True,
            "parallel_extraction": True,
            "max_concurrent_requests": 10
        }
        
        logger.info("ProofsAggregator initialized")
    
    def _load_fact_check_indicators(self) -> List[str]:
        """Load patterns that indicate fact-checking content."""
        return [
            r'\b(?:fact.?check|verify|verification|debunk|myth|false|true|rating|verdict)\b',
            r'\b(?:snopes|politifact|factcheck\.org|fullfact|checkyourfact)\b',
            r'\b(?:claim|allegation|statement|assertion).*(?:true|false|misleading|unproven)\b',
            r'\b(?:we rate|our rating|verdict|conclusion|finding)\b'
        ]
    
    def _load_supporting_patterns(self) -> List[str]:
        """Load patterns that indicate supporting evidence."""
        return [
            r'\b(?:confirms?|verifies?|proves?|demonstrates?|shows?|establishes?)\b',
            r'\b(?:true|accurate|correct|valid|legitimate|authentic)\b',
            r'\b(?:evidence|data|research|study|report).*(?:supports?|confirms?)\b',
            r'\b(?:according to|based on|research shows|studies indicate)\b'
        ]
    
    def _load_refuting_patterns(self) -> List[str]:
        """Load patterns that indicate refuting evidence."""
        return [
            r'\b(?:false|incorrect|wrong|inaccurate|misleading|debunked)\b',
            r'\b(?:disproves?|refutes?|contradicts?|disputes?|denies?)\b',
            r'\b(?:no evidence|lacks evidence|unsubstantiated|unproven)\b',
            r'\b(?:myth|hoax|conspiracy|fabricated|fake)\b'
        ]
    
    async def aggregate_proofs(
        self,
        claim: str,
        search_sources: List[Any],  # SearchSource objects from search_orchestrator
        extract_content: bool = True
    ) -> AggregatedProofs:
        """
        Aggregate proofs from search sources.
        
        Args:
            claim: The claim being fact-checked
            search_sources: List of SearchSource objects
            extract_content: Whether to extract full content from URLs
            
        Returns:
            AggregatedProofs containing organized evidence
        """
        start_time = datetime.now()
        
        try:
            # Initialize session for content extraction
            if extract_content and AIOHTTP_AVAILABLE:
                self.session = aiohttp.ClientSession(timeout=self.timeout)
            
            # Extract evidence from sources
            all_evidence = []
            
            if self.config["parallel_extraction"] and extract_content:
                # Parallel content extraction
                extraction_tasks = [
                    self._extract_evidence_from_source(source, claim, extract_content)
                    for source in search_sources
                ]
                
                # Limit concurrent requests
                semaphore = asyncio.Semaphore(self.config["max_concurrent_requests"])
                
                async def bounded_extraction(task):
                    async with semaphore:
                        return await task
                
                evidence_results = await asyncio.gather(
                    *[bounded_extraction(task) for task in extraction_tasks],
                    return_exceptions=True
                )
                
                for result in evidence_results:
                    if isinstance(result, Exception):
                        logger.warning(f"Evidence extraction failed: {result}")
                    elif result:
                        all_evidence.append(result)
            else:
                # Sequential extraction
                for source in search_sources:
                    try:
                        evidence = await self._extract_evidence_from_source(
                            source, claim, extract_content
                        )
                        if evidence:
                            all_evidence.append(evidence)
                    except Exception as e:
                        logger.warning(f"Evidence extraction failed for {source.url}: {e}")
            
            # Categorize evidence
            supporting_evidence = []
            refuting_evidence = []
            neutral_evidence = []
            
            for evidence in all_evidence:
                if evidence.evidence_type == "supporting":
                    supporting_evidence.append(evidence)
                elif evidence.evidence_type == "refuting":
                    refuting_evidence.append(evidence)
                else:
                    neutral_evidence.append(evidence)
            
            # Calculate statistics
            total_sources = len(search_sources)
            credible_sources_count = sum(
                1 for source in search_sources 
                if source.credibility_score > 0.7
            )
            fact_check_sources_count = sum(
                1 for source in search_sources 
                if source.source_type == "fact_check"
            )
            
            # Calculate aggregation confidence
            aggregation_confidence = self._calculate_aggregation_confidence(
                all_evidence, total_sources, credible_sources_count
            )
            
            # Create aggregated proofs
            proofs = AggregatedProofs(
                claim=claim,
                supporting_evidence=supporting_evidence,
                refuting_evidence=refuting_evidence,
                neutral_evidence=neutral_evidence,
                total_sources=total_sources,
                credible_sources_count=credible_sources_count,
                fact_check_sources_count=fact_check_sources_count,
                aggregation_confidence=aggregation_confidence,
                aggregated_at=datetime.now()
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Aggregated {len(all_evidence)} pieces of evidence in {processing_time:.2f}s")
            
            return proofs
            
        except Exception as e:
            logger.error(f"Proof aggregation failed: {e}")
            raise ProofsAggregatorError(f"Failed to aggregate proofs: {e}")
        
        finally:
            if self.session:
                await self.session.close()
                self.session = None
    
    async def _extract_evidence_from_source(
        self,
        source: Any,  # SearchSource object
        claim: str,
        extract_content: bool
    ) -> Optional[ProofEvidence]:
        """Extract evidence from a single source."""
        try:
            content = source.snippet or ""
            context = None
            quotes = []
            
            # Extract full content if requested
            if extract_content and self.session:
                try:
                    full_content, extracted_quotes = await self._extract_full_content(source.url)
                    if full_content:
                        content = full_content
                        quotes = extracted_quotes
                except Exception as e:
                    logger.debug(f"Content extraction failed for {source.url}: {e}")
                    # Fall back to snippet
                    pass
            
            # Determine evidence type
            evidence_type = self._classify_evidence_type(content, claim)
            
            # Calculate relevance score
            relevance_score = self._calculate_relevance_score(content, claim)
            
            # Extract context if available
            if len(content) > 200:
                context = self._extract_context(content, claim)
            
            evidence = ProofEvidence(
                content=content[:self.config["max_content_length"]],
                source_url=source.url,
                source_name=source.source_name,
                evidence_type=evidence_type,
                confidence=0.7,  # Base confidence, will be refined later
                relevance_score=relevance_score,
                credibility_score=source.credibility_score,
                extracted_at=datetime.now(),
                context=context,
                quotes=quotes
            )
            
            return evidence
            
        except Exception as e:
            logger.warning(f"Failed to extract evidence from {source.url}: {e}")
            return None
    
    async def _extract_full_content(self, url: str) -> Tuple[str, List[str]]:
        """Extract full content and quotes from a URL."""
        if not self.session or not AIOHTTP_AVAILABLE:
            return "", []
        
        try:
            async with self.session.get(url) as response:
                if response.status != 200:
                    return "", []
                
                html_content = await response.text()
                
                if not BS4_AVAILABLE:
                    # Basic text extraction without BeautifulSoup
                    text = re.sub(r'<[^>]+>', ' ', html_content)
                    text = re.sub(r'\s+', ' ', text).strip()
                    return text[:self.config["max_content_length"]], []
                
                # Parse with BeautifulSoup
                soup = BeautifulSoup(html_content, 'html.parser')
                
                # Remove script and style elements
                for script in soup(["script", "style", "nav", "header", "footer"]):
                    script.decompose()
                
                # Extract main content
                content_selectors = [
                    'article', 'main', '.content', '.article-body',
                    '.post-content', '.entry-content', 'p'
                ]
                
                content_text = ""
                for selector in content_selectors:
                    elements = soup.select(selector)
                    if elements:
                        content_text = ' '.join(elem.get_text() for elem in elements)
                        break
                
                if not content_text:
                    content_text = soup.get_text()
                
                # Clean up text
                content_text = re.sub(r'\s+', ' ', content_text).strip()
                
                # Extract quotes
                quotes = self._extract_quotes_from_content(content_text)
                
                return content_text[:self.config["max_content_length"]], quotes
                
        except Exception as e:
            logger.debug(f"Content extraction failed for {url}: {e}")
            return "", []
    
    def _extract_quotes_from_content(self, content: str) -> List[str]:
        """Extract relevant quotes from content."""
        if not self.config["extract_quotes"]:
            return []
        
        quotes = []
        
        # Find quoted text
        quote_patterns = [
            r'"([^"]{20,200})"',  # Double quotes
            r"'([^']{20,200})'",  # Single quotes
            r'"([^"]{20,200})"',  # Smart quotes
        ]
        
        for pattern in quote_patterns:
            matches = re.findall(pattern, content)
            quotes.extend(matches)
        
        # Limit number of quotes
        return quotes[:5]
    
    def _classify_evidence_type(self, content: str, claim: str) -> str:
        """Classify evidence as supporting, refuting, or neutral."""
        content_lower = content.lower()
        claim_lower = claim.lower()
        
        # Check for fact-checking indicators first
        fact_check_score = 0
        for pattern in self.fact_check_indicators:
            if re.search(pattern, content_lower, re.IGNORECASE):
                fact_check_score += 1
        
        # Check for supporting patterns
        supporting_score = 0
        for pattern in self.supporting_patterns:
            matches = len(re.findall(pattern, content_lower, re.IGNORECASE))
            supporting_score += matches
        
        # Check for refuting patterns
        refuting_score = 0
        for pattern in self.refuting_patterns:
            matches = len(re.findall(pattern, content_lower, re.IGNORECASE))
            refuting_score += matches
        
        # Determine evidence type
        if fact_check_score > 0:
            # For fact-checking content, look for explicit verdicts
            if re.search(r'\b(?:false|incorrect|wrong|misleading|debunked)\b', content_lower):
                return "refuting"
            elif re.search(r'\b(?:true|accurate|correct|verified|confirmed)\b', content_lower):
                return "supporting"
        
        # General classification
        if refuting_score > supporting_score and refuting_score > 0:
            return "refuting"
        elif supporting_score > refuting_score and supporting_score > 0:
            return "supporting"
        else:
            return "neutral"
    
    def _calculate_relevance_score(self, content: str, claim: str) -> float:
        """Calculate how relevant the content is to the claim."""
        content_lower = content.lower()
        claim_lower = claim.lower()
        
        # Extract key terms from claim
        claim_words = set(re.findall(r'\b\w{3,}\b', claim_lower))
        content_words = set(re.findall(r'\b\w{3,}\b', content_lower))
        
        # Calculate word overlap
        if not claim_words:
            return 0.0
        
        overlap = len(claim_words.intersection(content_words))
        word_overlap_score = overlap / len(claim_words)
        
        # Boost score for exact phrase matches
        phrase_boost = 0.0
        claim_phrases = [claim_lower]
        
        # Add sub-phrases
        claim_parts = claim_lower.split()
        if len(claim_parts) > 3:
            for i in range(len(claim_parts) - 2):
                phrase = ' '.join(claim_parts[i:i+3])
                claim_phrases.append(phrase)
        
        for phrase in claim_phrases:
            if phrase in content_lower:
                phrase_boost += 0.2
        
        # Content length factor (longer content might be more comprehensive)
        length_factor = min(1.0, len(content) / 1000.0)
        
        # Combine scores
        relevance_score = (
            word_overlap_score * 0.5 +
            min(phrase_boost, 0.4) +
            length_factor * 0.1
        )
        
        return min(1.0, relevance_score)
    
    def _extract_context(self, content: str, claim: str) -> str:
        """Extract relevant context around claim-related content."""
        content_lower = content.lower()
        claim_lower = claim.lower()
        
        # Find sentences containing claim-related terms
        sentences = re.split(r'[.!?]+', content)
        relevant_sentences = []
        
        claim_words = set(re.findall(r'\b\w{3,}\b', claim_lower))
        
        for sentence in sentences:
            sentence_words = set(re.findall(r'\b\w{3,}\b', sentence.lower()))
            overlap = len(claim_words.intersection(sentence_words))
            
            if overlap >= 2:  # At least 2 words in common
                relevant_sentences.append(sentence.strip())
        
        # Return top 3 most relevant sentences
        context = '. '.join(relevant_sentences[:3])
        return context[:500] if context else None
    
    def _calculate_aggregation_confidence(
        self,
        evidence_list: List[ProofEvidence],
        total_sources: int,
        credible_sources_count: int
    ) -> float:
        """Calculate confidence in the aggregation process."""
        if not evidence_list:
            return 0.0
        
        # Source quantity factor
        quantity_factor = min(1.0, len(evidence_list) / 10.0)
        
        # Source quality factor
        quality_factor = credible_sources_count / max(1, total_sources)
        
        # Evidence diversity factor (different types of evidence)
        evidence_types = set(e.evidence_type for e in evidence_list)
        diversity_factor = len(evidence_types) / 3.0  # 3 possible types
        
        # Average relevance factor
        avg_relevance = sum(e.relevance_score for e in evidence_list) / len(evidence_list)
        
        # Combine factors
        confidence = (
            quantity_factor * 0.3 +
            quality_factor * 0.3 +
            diversity_factor * 0.2 +
            avg_relevance * 0.2
        )
        
        return min(1.0, confidence)
    
    def get_evidence_summary(self, proofs: AggregatedProofs) -> Dict[str, Any]:
        """Generate a comprehensive summary of evidence."""
        summary = proofs.get_summary()
        
        # Add detailed breakdowns
        summary.update({
            "source_breakdown": {
                "total_sources": proofs.total_sources,
                "credible_sources": proofs.credible_sources_count,
                "fact_check_sources": proofs.fact_check_sources_count
            },
            "evidence_quality": {
                "high_relevance_count": len([
                    e for e in (proofs.supporting_evidence + proofs.refuting_evidence + proofs.neutral_evidence)
                    if e.relevance_score > 0.7
                ]),
                "high_credibility_count": len([
                    e for e in (proofs.supporting_evidence + proofs.refuting_evidence + proofs.neutral_evidence)
                    if e.credibility_score > 0.7
                ])
            },
            "content_analysis": {
                "total_content_length": sum(
                    len(e.content) for e in (proofs.supporting_evidence + proofs.refuting_evidence + proofs.neutral_evidence)
                ),
                "quotes_extracted": sum(
                    len(e.quotes) for e in (proofs.supporting_evidence + proofs.refuting_evidence + proofs.neutral_evidence)
                )
            }
        })
        
        return summary


## Suggestions for Upgrade:
# 1. Implement advanced NLP techniques for better content analysis and evidence classification
# 2. Add machine learning models for automatic evidence quality assessment and relevance scoring
# 3. Integrate with content extraction services like Mercury or Readability for better text extraction
# 4. Add support for multimedia evidence extraction (images, videos) with OCR and speech-to-text